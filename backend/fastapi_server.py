import os
import re
import sys
import json
import time
import uuid
import queue
import torch
import uvicorn
import asyncio
import aiohttp
import logging
import requests
import threading
import numpy as np
import multiprocessing
import stream2sentence as s2s
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from collections import defaultdict
from collections.abc import Awaitable
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from supabase import create_client, Client
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from enum import Enum, auto

from backend.RealtimeSTT import AudioToTextRecorder
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.RealtimeTTS.threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

sys.path.append('/workspace/tts/Code')

class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool

@dataclass
class Voice:
    voice: str                  
    method: str
    speaker_desc: str
    scene_prompt: str
    audio_path: str = ""
    text_path: str = ""

@dataclass
class ConversationMessage:
    role: str
    name: str
    content: str

@dataclass
class ModelSettings:
    model: str
    temperature: float 
    top_p: float
    min_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

@dataclass 
class TextChunk:
    text: str
    message_id: str
    character_name: str
    chunk_index: int
    is_final: bool
    timestamp: float

@dataclass 
class ResponseChunk:
    text: str
    message_id: str
    character_name: str
    chunk_index: int
    is_final: bool
    timestamp: float

@dataclass
class TTSChunk:
    text: str
    message_id: str
    character_name: Character
    voice: Optional[Voice]
    chunk_index: int
    is_final: bool
    timestamp: float

@dataclass
class AudioChunk:
    """Represents a single audio chunk for streaming playback"""
    chunk_id: str              # Unique chunk identifier (e.g., "msg-001-chunk-0")
    message_id: str            # Parent message ID
    character_id: str          # Which character is speaking
    character_name: str        # Character name for display
    audio_data: bytes          # PCM16 @ 24kHz audio data
    chunk_index: int           # Position in message (0, 1, 2...)
    is_final: bool             # Last chunk in this message?
    timestamp: float = field(default_factory=time.time)

@dataclass
class STTCallbacks:
    """Callback functions for STT events"""
    # Transcription callbacks
    on_realtime_update: Optional[Callable[[str], Any]] = None
    on_realtime_stabilized: Optional[Callable[[str], Any]] = None
    on_final_transcription: Optional[Callable[[str], Any]] = None
    
    # VAD callbacks - useful for interrupt detection
    on_vad_start: Optional[Callable[[], Any]] = None
    on_vad_stop: Optional[Callable[[], Any]] = None
    on_vad_detect_start: Optional[Callable[[], Any]] = None
    on_vad_detect_stop: Optional[Callable[[], Any]] = None
    
    # Recording lifecycle callbacks
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None
    
    # Turn detection (silence during speech)
    on_turn_start: Optional[Callable[[], Any]] = None
    on_turn_end: Optional[Callable[[], Any]] = None

########################################
##--             Queues             --##
########################################

class Queues:
    """Queue Management for various pipeline stages"""

    def __init__(self):

        self.transcribe_queue = asyncio.Queue()

        self.response_queue = asyncio.Queue()

        self.sentence_queue = asyncio.Queue()

        self.audio_queue = asyncio.Queue()

########################################
##--           STT Service          --##
########################################

class STTService:
    """Speech-to-Text using RealtimeSTT"""

    def __init__(self):
        self.recorder: Optional[AudioToTextRecorder] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, Any] = {}
        self.is_listening = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def initialize(self):
        """Initialize the STT recorder"""
        logger.info("Initializing STT service...")

        try:
            self.recorder = AudioToTextRecorder(
                model="small.en",
                language="en",
                enable_realtime_transcription=True,
                realtime_processing_pause=0.1,
                realtime_model_type="small.en",
                on_realtime_transcription_update=self._on_realtime_update,
                on_realtime_transcription_stabilized=self._on_realtime_stabilized,
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,
                on_vad_detect_start=self._on_vad_detect_start,
                on_vad_detect_stop=self._on_vad_detect_stop,
                silero_sensitivity=0.4,
                webrtc_sensitivity=3,
                post_speech_silence_duration=0.7,
                min_length_of_recording=0.5,
                spinner=False,
                level=logging.WARNING,
                use_microphone=False
            )

            self.recording_thread = threading.Thread(target=self.transcription_loop, daemon=True)
            self.recording_thread.start()

            logger.info("STT service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            raise

    def transcription_loop(self):
        """Main recording/transcription loop running in separate thread"""
        logger.info("STT recording loop started")

        while True:
            if self.is_listening.is_set():
                try:
                    # Get transcription from recorder
                    text = self.recorder.text()

                    if text and text.strip():
                        logger.info(f"Final transcription: {text}")
                        self._on_final_transcription(text)
                    else:
                        logger.info("Empty transcription received")
                    
                    self.is_listening.clear()
                    
                except Exception as e:
                    self.is_listening.clear()

            else:
                time.sleep(0.05)
    
    def _on_realtime_update(self, text: str):
        """Called when realtime transcription updates"""
        if self.callbacks.on_realtime_update:
            self._schedule_callback(self.callbacks.on_realtime_update, text)
    
    def _on_realtime_stabilized(self, text: str):
        """Called when realtime transcription stabilizes"""
        if self.callbacks.on_realtime_stabilized:
            self._schedule_callback(self.callbacks.on_realtime_stabilized, text)

    def _on_final_transcription(self, user_message: str):
        if self.callbacks.on_final_transcription:
            self._schedule_callback(self.callbacks.on_final_transcription, user_message)
    
    def _on_vad_detect_start(self):
        """Called when system starts listening for voice activity."""
        if self.callbacks.on_vad_detect_start:
            self._schedule_callback(self.callbacks.on_vad_detect_start)
    
    def _on_vad_detect_stop(self):
        """Called when system stops listening for voice activity"""
        if self.callbacks.on_vad_detect_stop:
            self._schedule_callback(self.callbacks.on_vad_detect_stop)
    
    def _on_vad_start(self):
        """Called when actual voice activity is detected during recording."""
        if self.callbacks.on_vad_start:
            self._schedule_callback(self.callbacks.on_vad_start)
    
    def _on_vad_stop(self):
        """Called when voice activity stops during recording"""
        if self.callbacks.on_vad_stop:
            self._schedule_callback(self.callbacks.on_vad_stop)

    def start_listening(self):
        """Start listening for voice activity"""
        self.is_listening = True
        if self.recorder:
            self.recorder.listen()

    def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
    
    def _on_recording_start(self):
        """Called when recording starts"""
        if self.callbacks.on_recording_start:
            self._schedule_callback(self.callbacks.on_recording_start)
    
    def _on_recording_stop(self):
        """Called when recording stops"""
        if self.callbacks.on_recording_stop:
            self._schedule_callback(self.callbacks.on_recording_stop)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

########################################
##--       Callback Utilities       --##
########################################
    
    def _schedule_callback(self, callback: Callable, *args):
        """Schedule a callback to run, handling both sync and async callbacks"""
        if self.loop is None:
            return
            
        try:
            if asyncio.iscoroutinefunction(callback):
                # Schedule async callback
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(callback(*args))
                )

            else:
                # Schedule sync callback
                self.loop.call_soon_threadsafe(callback, *args)
        except Exception as e:
            logger.error(f"Error scheduling callback: {e}")
    
    async def _run_callback_async(self, callback: Callable, *args):
        """Run a callback that may be sync or async"""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)

########################################
##--   Text to Sentence Extractor   --##
########################################

class TextToSentence:
    """
    Bridges async LLM text stream to synchronous stream2sentence library.
    
    Architecture:
    - Async side: Receives text chunks, feeds to CharIterator
    - Sync side: stream2sentence extracts sentences in background thread
    - Output: Sentences are pushed to an async queue as they're detected
    
    This enables true streaming: first sentence starts TTS while LLM is still
    generating subsequent text.
    """
    
    def __init__(
        self,
        sentence_callback: Callable[[str, int], None] = None,
        minimum_sentence_length: int = 25,
        minimum_first_fragment_length: int = 10,
        quick_yield_single_sentence_fragment: bool = True,
        cleanup_text_links: bool = True,
        cleanup_text_emojis: bool = False,
        tokenize_sentences: bool = False,
    ):

        self.sentence_callback = sentence_callback
        
        self.s2s_config = {
            "minimum_sentence_length": minimum_sentence_length,
            "minimum_first_fragment_length": minimum_first_fragment_length,
            "quick_yield_single_sentence_fragment": quick_yield_single_sentence_fragment,
            "cleanup_text_links": cleanup_text_links,
            "cleanup_text_emojis": cleanup_text_emojis,
            "tokenize_sentences": tokenize_sentences,
        }
        
        # Thread-safe components
        self.char_iter: Optional[CharIterator] = None
        self.thread_safe_iter: Optional[AccumulatingThreadSafeGenerator] = None

        # Sentence output queue (thread-safe)
        self.sentence_queue = Queue()

        # Control
        self.sentences_thread: Optional[threading.Thread] = None
        self._is_running = False
        self._is_complete = False
        self._sentence_count = 0
        self._accumulated_text = ""
        
        # Thread pool for non-blocking operations
        self._executor = ThreadPoolExecutor(max_workers=1)
    
    def start(self):
        """Initialize and start the sentence extraction pipeline"""
        self.char_iter = CharIterator()
        self.thread_safe_iter = AccumulatingThreadSafeGenerator(self.char_iter)
        self.sentence_queue = Queue()
        
        self._is_running = True
        self._is_complete = False
        self._sentence_count = 0
        self._accumulated_text = ""
        
        # Start sentence extraction thread
        self.sentences_thread = threading.Thread(target=self.run_sentence_generator, daemon=True)
        self.sentences_thread.start()
    
    def feed_text(self, text: str):
        """
        Feed text chunk from LLM stream.
        Thread-safe, can be called from async context.
        """
        if not self._is_running:
            return
            
        self._accumulated_text += text
        self.char_iter.add(text)

    def get_accumulated_text(self) -> str:
        """Get all text that was fed to the sentence extractor"""
        return self._accumulated_text
    
    def finish(self):
        """
        Signal that LLM stream is complete.
        Flushes any remaining text as final sentence.
        """
        if not self._is_running:
            return
            
        self._is_complete = True
        
        # Signal end to CharIterator
        if self.char_iter:
            self.char_iter.add("")  # Empty string can signal completion
    
    def run_sentence_generator(self):
        """
        Background thread: Extract sentences using stream2sentence.
        Runs until stream is complete and all sentences are extracted.
        """
        try:
            sentence_generator = s2s.generate_sentences(
                self.thread_safe_iter,
                **self.s2s_config
            )
            
            for sentence in sentence_generator:
                if not self._is_running:
                    break
                    
                sentence = sentence.strip()
                if sentence:
                    # Push to queue
                    self.sentence_queue.put((sentence, self._sentence_count))
                    
                    # Callback if provided
                    if self.sentence_callback:
                        self.sentence_callback(sentence, self._sentence_count)
                    
                    self._sentence_count += 1
                    
        except Exception as e:
            print(f"Sentence extraction error: {e}")
        finally:
            # Signal completion
            self.sentence_queue.put(None)  # Sentinel value
            self._is_running = False
    
    async def get_sentences(self) -> AsyncIterator[tuple[str, int]]:
        """
        Async generator that yields sentences as they become available.
        Yields: (sentence_text, index)
        """
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Non-blocking check with short timeout
                sentence = await loop.run_in_executor(self._executor, lambda: self.sentence_queue.get(timeout=0.02))
                
                if sentence is None:  # Sentinel
                    break
                    
                yield sentence
                
            except Empty:
                # No sentence ready yet, yield control
                await asyncio.sleep(0.005)
                
                # Check if we should exit
                if not self._is_running and self.sentence_queue.empty():
                    break
    
    def shutdown(self):
        """Clean shutdown"""
        self._is_running = False
        if self.sentences_thread and self.sentences_thread.is_alive():
            self.sentences_thread.join(timeout=1.0)
        self._executor.shutdown(wait=False)

########################################
##--           LLM Service          --##
########################################

class LLMService:
    """LLM Service for multi-character conversation loop"""

    def __init__(self, queues: Queues, api_key: str):

        self.is_initialized = False
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        # Active characters in the conversation
        self.active_characters: List[Character] = []

        # Conversation history - shared across all characters
        # Format: List of dicts with role, name, content
        self.conversation_history: List[Dict[str, str]] = []

        # Current model settings (can be updated per request)
        self.model_settings: Optional[ModelSettings] = None

        # Per-response tracking (reset for each character response)
        self.sentence_extractor: Optional[TextToSentence] = None
        self.chunk_index = 0
        self.index = 0
        self.response_text = ""
        self.is_complete = False

        # Interrupt handling
        self.interrupt_event = asyncio.Event()

    async def initialize(self):
        self.is_initialized = True
        logger.info("LLMService initialized")

    def strip_character_tags(self, text: str) -> str:
        """Strip character tags from text for display/TTS purposes"""
        return re.sub(r'<[^>]+>', '', text).strip()

    def add_user_message(self, content: str, name: str = "User"):
        """Add user message to conversation history"""
        self.conversation_history.append({
            "role": "user",
            "name": name,
            "content": content
        })

    def add_character_message(self, character: Character, content: str):
        """Add character response to conversation history"""
        self.conversation_history.append({
            "role": "assistant",
            "name": character.name,
            "content": content
        })

    def add_message_to_conversation_history(self, role: str, name: str, content: str):
        """Add (user or character) message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "name": name,
            "content": content
        })

    def set_active_characters(self, characters: List[Character]):
        """Set the active characters for the conversation"""
        self.active_characters = characters

    async def load_active_characters_from_db(self):
        """Load active characters from Supabase database"""
        try:
            logger.info("Loading active characters from database...")

            response = supabase.table("characters") \
                .select("*") \
                .eq("is_active", True) \
                .execute()

            if response.data:
                characters = [
                    Character(
                        id=char.get("id", str(uuid.uuid4())),
                        name=char.get("name", ""),
                        voice=char.get("voice", ""),
                        system_prompt=char.get("system_prompt", ""),
                        image_url=char.get("image_url", ""),
                        images=char.get("images", []),
                        is_active=char.get("is_active", True)
                    )
                    for char in response.data
                ]

                self.set_active_characters(characters)
                logger.info(f"✅ Loaded {len(characters)} active characters: {[c.name for c in characters]}")
                return characters
            else:
                logger.info("No active characters found in database")
                self.set_active_characters([])
                return []

        except Exception as e:
            logger.error(f"Failed to load active characters from database: {e}")
            return []

    def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests"""
        self.model_settings = model_settings

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    def reset_response_tracking(self):
        """Reset per-response tracking variables"""
        self.chunk_index = 0
        self.index = 0
        self.response_text = ""
        self.is_complete = False

    def create_character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""
        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else. Wrap your entire response in <{character.name}></{character.name}> tags.'
        }

    def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""
        mentioned_characters = []
        processed_characters = set()

        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        name_mentions.sort(key=lambda x: x['position'])

        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters
    
    def get_model_settings(self) -> ModelSettings:
        """Get current model settings for the LLM request"""
        if self.model_settings is None:
            # Return default settings if not set
            return ModelSettings(
                model="meta-llama/llama-3.1-8b-instruct",
                temperature=0.7,
                top_p=0.9,
                min_p=0.0,
                top_k=40,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0
            )
        return self.model_settings

    async def conversation_loop(self, user_name: str = "Jay"):
        """Main LLM conversation loop for multi-character conversations."""

        logger.info("Starting main LLM loop")

        while True:
            try:

                user_message = await self.queues.transcribe_queue.get()

                if not user_message or not user_message.strip():
                    continue

                self.conversation_history.append({"role": "user", "name": user_name, "content": user_message})

                mentioned_characters = self.parse_character_mentions(message=user_message, active_characters=self.active_characters)

                for character in mentioned_characters:
                    if self.interrupt_event.is_set():
                        break

                    messages = []

                    messages.append({"role": "system", "name": character.name, "content": character.system_prompt})

                    messages.extend(self.conversation_history)

                    messages.append(self.create_character_instruction_message(character))

                    model_settings = self.get_model_settings()

                    text_stream = await self.client.chat.completions.create(
                        model=model_settings.model,
                        messages=messages,
                        temperature=model_settings.temperature,
                        top_p=model_settings.top_p,
                        frequency_penalty=model_settings.frequency_penalty,
                        presence_penalty=model_settings.presence_penalty,
                        stream=True
                    )

                    response_text = await self.character_response_stream(character=character, text_stream=text_stream)

                    if response_text:
                        self.conversation_history.append({"role": "assistant", "name": character.name, "content": response_text})

            except Exception as e:
                logger.error(f"Error in LLM loop: {e}")

    async def character_response_stream(self, character: Character, text_stream: AsyncIterator) -> str:
        """Generate and stream a single character's response."""

        message_id = f"msg-{character.id}-{int(time.time() * 1000)}"

        # Create new sentence extractor for this response
        self.sentence_extractor = TextToSentence()
        self.sentence_extractor.start()
        
        # Create task for sentence-to-TTS processing
        sentence_task = asyncio.create_task(self.process_sentences_for_tts(character=character, message_id=message_id))

        try:
            # Stream from LLM
            async for chunk in text_stream:
                # Check for interrupt
                if self.interrupt_event.is_set():
                    logger.info(f"Interrupt detected during {character.name}'s response")
                    break

                # Extract content from chunk
                content = chunk.choices[0].delta.content
                if content:
                    self.response_text += content

                    # Feed to sentence extractor (non-blocking)
                    self.sentence_extractor.feed_text(content)

                    # Stream to UI immediately
                    response_chunk = TextChunk(text=content, message_id=message_id, character_name=character.name, 
                                               chunk_index=self.chunk_index, is_final=False, timestamp=time.time())
                    
                    await self.queues.response_queue.put(response_chunk)
                    self.chunk_index += 1

            # Signal LLM stream complete
            self.sentence_extractor.finish()

            # Send final text chunk to UI
            response_text = TextChunk(text="", message_id=message_id, character_name=character.name, 
                                      chunk_index=self.chunk_index, is_final=True, timestamp=time.time())
            
            await self.queues.response_queue.put(response_text)

            # Wait for sentence processing to complete
            await sentence_task

        except Exception as e:
            logger.error(f"Error in character_response_stream for {character.name}: {e}")
            raise
        finally:
            if self.sentence_extractor:
                self.sentence_extractor.shutdown()
            self.is_complete = True

        return self.response_text

    async def process_sentences_for_tts(self, character: Character, message_id: str):
        """
        Process sentences as they're extracted and queue for TTS.
        Runs concurrently with LLM text stream.

        Args:
            character: The character whose sentences are being processed
            message_id: Unique ID for the current message
        """
        sentences = []
        index = 0

        try:
            async for sentence, index in self.sentence_extractor.get_sentences():
                # Check for interrupt
                if self.interrupt_event.is_set():
                    break

                sentences.append(sentence)

                # Queue for TTS immediately
                sentence = TTSChunk(
                    text=sentence,
                    message_id=message_id,
                    character=character,
                    voice=Voice,
                    index=index,
                    is_final=False,
                    timestamp=time.time()
                )
                await self.queues.sentence_queue.put(sentence)

                logger.info(f"Sentence {index} queued for TTS ({character.name}): "
                           f"{sentence[:50]}{'...' if len(sentence) > 50 else ''}")

                index += 1

            # Send final marker for this character's TTS
            if sentences:
                final_marker = TTSChunk(
                    text="",
                    message_id=message_id,
                    character=character,
                    voice=Voice,
                    index=index,
                    is_final=True,
                    timestamp=time.time()
                )
                await self.queues.sentence_queue.put(final_marker)

        except Exception as e:
            logger.error(f"Error processing sentences for {character.name}: {e}")

########################################
##--           TTS Service          --##
########################################

def revert_delay_pattern(data: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Undo Higgs delay pattern so decoded frames line up."""
    if data.ndim != 2:
        raise ValueError('Expected 2D tensor from audio tokenizer')
    if data.shape[1] - data.shape[0] < start_idx:
        raise ValueError('Invalid start_idx for delay pattern reversion')

    out = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out.append(data[i:(i + 1), i + start_idx:(data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out, dim=0)

class TTSService:
    """Higgs Streaming"""

    def __init__(self):
        self.serve_engine = None
        self.voice_dir = "/workspace/tts/Code/backend/voices"
        self.sample_rate = 24000
        self._initialized = False
        self._engine_lock = Lock()
        self._chunk_overlap_duration = 0.04  # 40ms crossfade
        self._chunk_size = 16  # Tokens per streaming chunk
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def initialize(self):
        """Initialize Higgs Audio TTS engine"""
        if self._initialized:
            return

        logger.info("Initializing Higgs Audio TTS service...")

        try:
            from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            self.serve_engine = HiggsAudioServeEngine(
                model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
                audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
                device=device
            )

            self._initialized = True
            logger.info("Higgs Audio TTS service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Higgs Audio TTS: {e}")
            raise

    def _load_voice_reference(self, voice: str):
        """Load reference audio and text for voice cloning"""
        from backend.boson_multimodal.data_types import Message, AudioContent

        audio_path = os.path.join(self.voice_dir, f"{voice}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice}.txt")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Voice audio file not found: {audio_path}")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Voice text file not found: {text_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        # Build messages with reference audio (few-shot approach)
        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

        return messages

    async def stream_pcm_chunks(self, text: str, voice: str) -> AsyncIterator[bytes]:
        """Stream PCM16 audio chunks using delta token streaming"""
        from backend.boson_multimodal.data_types import ChatMLSample, Message

        if not self._initialized:
            raise RuntimeError("TTS Manager not initialized")

        # Load voice reference messages
        messages = self._load_voice_reference(voice)

        # Add user message with text to generate
        messages.append(Message(role="user", content=text))

        # Create ChatML sample
        chat_sample = ChatMLSample(messages=messages)

        # Stream generation with delta tokens
        try:
            output = self.serve_engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop_strings=['<|end_of_text|>', '<|eot_id|>'],
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                force_audio_gen=True,
            )

            # Initialize streaming state
            audio_tokens: List[torch.Tensor] = []
            audio_tensor: Optional[torch.Tensor] = None
            seq_len = 0

            # Crossfade setup
            cross_fade_samples = int(self._chunk_overlap_duration * self.sample_rate)
            fade_out = np.linspace(1, 0, cross_fade_samples) if cross_fade_samples > 0 else None
            fade_in = np.linspace(0, 1, cross_fade_samples) if cross_fade_samples > 0 else None
            prev_tail: Optional[np.ndarray] = None

            with torch.inference_mode():
                async for delta in output:
                    # Skip if no audio tokens
                    if delta.audio_tokens is None:
                        continue

                    # Check for end token (1025)
                    if torch.all(delta.audio_tokens == 1025):
                        break

                    # Accumulate audio tokens
                    audio_tokens.append(delta.audio_tokens[:, None])
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    # Count sequence length (skip padding token 1024)
                    if torch.all(delta.audio_tokens != 1024):
                        seq_len += 1

                    # Decode and yield when chunk size reached
                    if seq_len > 0 and seq_len % self._chunk_size == 0:
                        try:
                            # Revert delay pattern and decode
                            vq_code = (
                                revert_delay_pattern(audio_tensor, start_idx=seq_len - self._chunk_size + 1)
                                .clip(0, 1023)
                                .to(self._device)
                            )
                            waveform_tensor = self.serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                            # Convert to numpy
                            if isinstance(waveform_tensor, torch.Tensor):
                                waveform_np = waveform_tensor.detach().cpu().numpy()
                            else:
                                waveform_np = np.asarray(waveform_tensor, dtype=np.float32)

                            # Apply crossfade
                            if prev_tail is None:
                                # First chunk
                                if cross_fade_samples > 0 and waveform_np.size > cross_fade_samples:
                                    chunk_head = waveform_np[:-cross_fade_samples]
                                    prev_tail = waveform_np[-cross_fade_samples:]
                                else:
                                    chunk_head = waveform_np
                                    prev_tail = None

                                if chunk_head.size > 0:
                                    # Convert to PCM16 and yield
                                    pcm = np.clip(chunk_head, -1.0, 1.0)
                                    pcm16 = (pcm * 32767.0).astype(np.int16)
                                    yield pcm16.tobytes()
                            else:
                                # Subsequent chunks with crossfade
                                if cross_fade_samples > 0 and waveform_np.size >= cross_fade_samples:
                                    overlap = prev_tail * fade_out + waveform_np[:cross_fade_samples] * fade_in
                                    middle = (
                                        waveform_np[cross_fade_samples:-cross_fade_samples]
                                        if waveform_np.size > 2 * cross_fade_samples
                                        else np.array([], dtype=waveform_np.dtype)
                                    )
                                    to_send = overlap if middle.size == 0 else np.concatenate([overlap, middle])

                                    if to_send.size > 0:
                                        # Convert to PCM16 and yield
                                        pcm = np.clip(to_send, -1.0, 1.0)
                                        pcm16 = (pcm * 32767.0).astype(np.int16)
                                        yield pcm16.tobytes()

                                    prev_tail = waveform_np[-cross_fade_samples:]
                                else:
                                    # Convert to PCM16 and yield
                                    pcm = np.clip(waveform_np, -1.0, 1.0)
                                    pcm16 = (pcm * 32767.0).astype(np.int16)
                                    yield pcm16.tobytes()

                        except Exception as e:
                            # Skip errors and continue
                            logger.warning(f"Error decoding audio chunk: {e}")
                            continue

            # Flush remaining tokens
            if seq_len > 0 and seq_len % self._chunk_size != 0 and audio_tensor is not None:
                try:
                    vq_code = (
                        revert_delay_pattern(audio_tensor, start_idx=seq_len - seq_len % self._chunk_size + 1)
                        .clip(0, 1023)
                        .to(self._device)
                    )
                    waveform_tensor = self.serve_engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                    if isinstance(waveform_tensor, torch.Tensor):
                        waveform_np = waveform_tensor.detach().cpu().numpy()
                    else:
                        waveform_np = np.asarray(waveform_tensor, dtype=np.float32)

                    if prev_tail is None:
                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()
                    else:
                        if cross_fade_samples > 0 and waveform_np.size >= cross_fade_samples:
                            overlap = prev_tail * fade_out + waveform_np[:cross_fade_samples] * fade_in
                            rest = waveform_np[cross_fade_samples:]
                            to_send = overlap if rest.size == 0 else np.concatenate([overlap, rest])

                            pcm = np.clip(to_send, -1.0, 1.0)
                            pcm16 = (pcm * 32767.0).astype(np.int16)
                            yield pcm16.tobytes()
                        else:
                            pcm = np.clip(waveform_np, -1.0, 1.0)
                            pcm16 = (pcm * 32767.0).astype(np.int16)
                            yield pcm16.tobytes()
                except Exception as e:
                    logger.warning(f"Error flushing remaining audio: {e}")

            # Yield final tail if exists
            if prev_tail is not None and prev_tail.size > 0:
                pcm = np.clip(prev_tail, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

        except Exception as e:
            logger.error(f"Error in TTS streaming: {e}")
            raise

    def get_available_voices(self):
        """Get list of available voices in format expected by frontend"""
        if not os.path.exists(self.voice_dir):
            return []

        voices = []
        for file in os.listdir(self.voice_dir):
            if file.endswith('.wav'):
                voice = file[:-4]  # Remove .wav extension
                # Only include if matching .txt file exists
                if os.path.exists(os.path.join(self.voice_dir, f"{voice}.txt")):
                    # Format: {id: "voice", name: "Voice Name"}
                    display_name = voice.replace('_', ' ').title()
                    voices.append({
                        "id": voice,
                        "name": display_name
                    })

        # Sort by display name
        voices.sort(key=lambda v: v['name'])
        return voices

    def shutdown(self):
        """Cleanup resources"""
        logger.info('Shutting down TTS manager')
        self.serve_engine = None
        self._initialized = False

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and coordinates services"""

    def __init__(self):
        self.stt_service: Optional[STTService] = None
        self.llm_service: Optional[LLMService] = None
        self.tts_service: Optional[TTSService] = None
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None
        self.service_tasks: List[asyncio.Task] = []

        # API key from environment
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "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")

    async def initialize(self):
        """Initialize all services with proper callbacks"""

        self.queues = Queues()

        # Setup STT callbacks
        stt_callbacks = STTCallbacks(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )

        # Initialize STT service
        self.stt_service = STTService()
        self.stt_service.callbacks = stt_callbacks
        self.stt_service.loop = asyncio.get_event_loop()
        self.stt_service.initialize()

        # Initialize LLM service with queues and API key
        self.llm_service = LLMService(
            queues=self.queues,
            api_key=self.openrouter_api_key
        )
        await self.llm_service.initialize()

        # Initialize TTS Service Manager with queues
        self.tts_service = TTSService(queues=self.queues)
        await self.tts_service.initialize()

        logger.info("WebSocketManager initialized")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket connected")

        # Load active characters from database on connection
        if self.llm_service:
            await self.llm_service.load_active_characters_from_db()

        await self.start_service_tasks()

    async def start_service_tasks(self):
        """Start all services"""

        self.service_tasks = [
            asyncio.create_task(self.llm_service.conversation_loop()),
            asyncio.create_task(self.tts_service.main_tts_loop(send_audio_callback=self.stream_audio_to_client)),
            asyncio.create_task(self.stream_text_to_client())
        ]

    async def shutdown(self):
        """Shutdown all services gracefully"""
        logger.info("Shutting down WebSocket Manager services...")

        # Cancel all service tasks
        for task in self.service_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Shutdown TTS service
        if self.tts_service:
            await self.tts_service.shutdown()

        # Clear queues
        if self.queues:
            while not self.queues.sentence_queue.empty():
                try:
                    self.queues.sentence_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        logger.info("WebSocket Manager services shut down")
    
    async def handle_text_message(self, message: str):
        """Handle incoming text messages from WebSocket client"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            payload = data.get("data", {})

            if message_type == "user_message":
                user_message = payload.get("text", "")
                await self.handle_user_message(user_message)

            elif message_type == "start_listening":
                if self.stt_service:
                    self.stt_service.start_listening()

            elif message_type == "stop_listening":
                if self.stt_service:
                    self.stt_service.stop_listening()

            elif message_type == "model_settings":
                settings_data = payload
                model_settings = ModelSettings(
                    model=settings_data.get("model", "meta-llama/llama-3.1-8b-instruct"),
                    temperature=float(settings_data.get("temperature", 0.7)),
                    top_p=float(settings_data.get("top_p", 0.9)),
                    min_p=float(settings_data.get("min_p", 0.0)),
                    top_k=int(settings_data.get("top_k", 40)),
                    frequency_penalty=float(settings_data.get("frequency_penalty", 0.0)),
                    presence_penalty=float(settings_data.get("presence_penalty", 0.0)),
                    repetition_penalty=float(settings_data.get("repetition_penalty", 1.0))
                )
                if self.llm_service:
                    self.llm_service.set_model_settings(model_settings)
                logger.info(f"Model settings updated: {model_settings.model}")

            elif message_type == "set_characters":
                # Set active characters for the conversation
                characters_data = payload.get("characters", [])
                characters = [
                    Character(
                        id=char.get("id", str(uuid.uuid4())),
                        name=char.get("name", ""),
                        voice=char.get("voice", ""),
                        system_prompt=char.get("system_prompt", ""),
                        image_url=char.get("image_url", ""),
                        images=char.get("images", []),
                        is_active=char.get("is_active", True)
                    )
                    for char in characters_data
                ]
                if self.llm_service:
                    self.llm_service.set_active_characters(characters)
                logger.info(f"Active characters set: {[c.name for c in characters]}")

            elif message_type == "clear_history":
                # Clear conversation history
                if self.llm_service:
                    self.llm_service.clear_conversation_history()
                logger.info("Conversation history cleared")

            elif message_type == "interrupt":
                # Signal interrupt to stop current generation
                if self.llm_service:
                    self.llm_service.interrupt_event.set()
                logger.info("Interrupt signal sent")

            elif message_type == "refresh_active_characters":
                # Refresh active characters from database
                if self.llm_service:
                    await self.llm_service.load_active_characters_from_db()
                logger.info("Active characters refreshed from database")

            elif message_type == "ping":
                # Respond to ping with pong for connection health check
                await self.send_text_to_client({"type": "pong", "timestamp": time.time()})

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_data: bytes):
        """Feed audio for transcription"""
        if self.stt_service:
            self.stt_service.feed_audio(audio_data)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribe_queue.put(user_message)

    async def stream_text_to_client(self):
        """Stream response/text chunks from response_queue to WebSocket"""
        while True:
            try:
                response_chunk: TextChunk = await self.queues.response_queue.get()
                await self.send_text_to_client({
                    "type": "response_chunk",
                    "data": {
                        "text": response_chunk.text,
                        "character_name": response_chunk.character_name,
                        "message_id": response_chunk.message_id,
                        "is_final": response_chunk.is_final
                    }
                })
            except Exception as e:
                logger.error(f"Error streaming text: {e}")

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            await self.websocket.send_text(json.dumps(data))
    
    async def stream_audio_to_client(self, audio_data: bytes):
        """Send binary audio to client (TTS)"""
        if self.websocket:
            await self.websocket.send_bytes(audio_data)

    async def on_realtime_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})
    
    async def on_realtime_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})
    
    async def on_final_transcription(self, user_message: str):
        await self.queues.transcribe_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_final", "text": user_message})

    async def on_character_response_chunk(self, response_chunk: str):
        await self.queues.response_queue.put(response_chunk)
        await self.send_text_to_client({"type": "response_chunk", "text": response_chunk})

    async def on_character_response_text(self, response_text: str):
        await self.queues.response_queue.put(response_text)
        await self.send_text_to_client({"type": "response_text", "text": response_text})

    async def disconnect(self):
        """Handle WebSocket disconnection"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        logger.info("WebSocket disconnected")

########################################
##--           FastAPI App          --##
########################################

ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up services...")
    await ws_manager.initialize()
    print("All services initialized!")
    yield
    print("Shutting down services...")
    await ws_manager.shutdown()
    print("All services shut down!")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
##--       WebSocket Endpoint       --##
########################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                await ws_manager.handle_text_message(message["text"])
            
            elif "bytes" in message:
                await ws_manager.handle_audio_message(message["bytes"])
    
    except WebSocketDisconnect:
        await ws_manager.disconnect()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect()

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
