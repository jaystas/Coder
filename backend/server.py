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
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator, Awaitable
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

from backend.RealtimeSTT import AudioToTextRecorder
from backend.stream2sentence import generate_sentences_async
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# Database Director - centralized CRUD operations for Supabase
from backend.database_director import (db, Character, CharacterCreate, CharacterUpdate, Voice, VoiceCreate, VoiceUpdate, Conversation, ConversationCreate, ConversationUpdate, Message as ConversationMessage, MessageCreate)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################
##--           Data Classes         --##
########################################

@dataclass 
class ResponseChunk:
    text: str
    message_id: str
    character_name: str
    index: int
    is_final: bool
    timestamp: float

@dataclass
class Sentence:
    sentence: str
    index: int
    message_id: str
    character_name: str
    is_final: bool

@dataclass
class TTSSentence:
    sentence: str
    index: int
    message_id: str
    character_name: str
    voice: str
    is_final: bool

@dataclass
class AudioChunk:
    message_id: str
    chunk_id: str
    character_name: str
    audio_data: bytes
    index: int
    is_final: bool
    timestamp: float

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

Callback = Callable[..., Optional[Awaitable[None]]]

########################################
##--         Queue Management       --##
########################################

class Queues:
    """Queue Management for various pipeline stages"""

    def __init__(self):
        self.transcribe_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.sentence_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()

########################################
##--     Transcription Pipeline     --##
########################################

class Transcribe:
    """Transcription Pipeline"""

    def __init__(
        self,
        on_realtime_update: Optional[Callback] = None,
        on_realtime_stabilized: Optional[Callback] = None,
        on_final_transcription: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,
    ):
        self.recorder: Optional[AudioToTextRecorder] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.is_listening = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Interrupt detection state
        self._tts_playing = False
        self._interrupt_detected = False

        # Store callbacks in dictionary
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_realtime_update': on_realtime_update,
            'on_realtime_stabilized': on_realtime_stabilized,
            'on_final_transcription': on_final_transcription,
            'on_vad_start': on_vad_start,
            'on_vad_stop': on_vad_stop,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
        }

    def initialize(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initialize the STT recorder"""
        logger.info("Initializing STT service...")

        # Store event loop for async callback dispatch
        self.loop = loop or asyncio.get_event_loop()

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
                on_vad_detect_start=self._on_vad_start,
                on_vad_detect_stop=self._on_vad_stop,
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

    def transcription_loop(self):
        """Main transcription loop running in separate thread (synchronous)"""

        logger.info("STT transcription loop started")

        while True:
            if self.is_listening:
                try:
                    user_message = self.recorder.text()

                    if user_message and user_message.strip():
                        callback = self.callbacks.get('on_final_transcription')
                        if callback:
                            self.run_callback(callback, user_message)

                except Exception as e:
                    logger.error(f"Error in recording loop: {e}")

            time.sleep(0.1)

    def start_listening(self):
        """Start listening for audio input"""
        self.is_listening = True
        logger.info("Started listening for audio")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False
        logger.info("Stopped listening for audio")

    def set_tts_playing(self, playing: bool):
        """Set TTS playing state for interrupt detection"""
        self._tts_playing = playing

    def clear_interrupt(self):
        """Clear interrupt detected flag"""
        self._interrupt_detected = False

    def was_interrupted(self) -> bool:
        """Check if interrupt was detected"""
        return self._interrupt_detected

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    def run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run a user callback from a RealtimeSTT background thread."""

        if callback is None or self.loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)

        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def _on_realtime_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self.run_callback(self.callbacks.get('on_realtime_update'), text)

    def _on_realtime_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_realtime_stabilized'), text)

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        if self._tts_playing:
            self._interrupt_detected = True
            logger.info("Interrupt detected: user speaking during TTS")

        self.run_callback(self.callbacks.get('on_vad_start'))

    def _on_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self.run_callback(self.callbacks.get('on_vad_stop'))

    def _on_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self.run_callback(self.callbacks.get('on_recording_start'))

    def _on_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self.run_callback(self.callbacks.get('on_recording_stop'))

########################################
##--  Chat Pipeline and Management  --##
########################################

class Chat:
    """Chat text stream to audio stream pipeline"""

    def __init__(self, queues: Queues, api_key: str):
        """Initialize conversation"""
        self.conversation_history: List[Dict] = []
        self.conversation_id: Optional[str] = None
        self.queues = queues
        self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

        # Model and character state
        self.model_settings: Optional[ModelSettings] = None
        self.active_characters: List[Character] = []

        # Interrupt control
        self.interrupt_event = asyncio.Event()

        # UI streaming callbacks (set by WebSocketManager)
        self.on_character_response_chunk: Optional[Callback] = None
        self.on_character_response_full: Optional[Callback] = None

    async def initialize(self):
        """Initialize chat service"""
        pass

    async def start_new_chat(self):
        """Start a new chat session"""
        self.conversation_history = []
        self.conversation_id = str(uuid.uuid4())
        self.interrupt_event.clear()

    def interrupt(self):
        """Signal to interrupt current generation"""
        self.interrupt_event.set()
        logger.info("Chat interrupt signal set")

    def clear_interrupt(self):
        """Clear the interrupt signal"""
        self.interrupt_event.clear()

    async def get_active_characters(self) -> List[Character]:
        """Get active characters from database"""
        return await db.get_active_characters()


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

    def wrap_with_character_tags(self, text: str, character_name: str) -> str:
        """Wrap response text with character name XML tags for conversation history."""

        return f"<{character_name}>{text}</{character_name}>"

    def create_character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""

        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else.'
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
    

    async def send_conversation_prompt(self, messages: str, character: Character, user_name: str = "Jay",
                                         model_settings: ModelSettings = None, user_message=str):
        """Build message structure and initiate character response for single and multi-character conversations."""

        while True:
            try:
                user_message = await self.queues.transcribe_queue.get(user_message)

                if not user_message or not user_message.strip():
                    continue

                # Clear any previous interrupt signal before processing new message
                self.interrupt_event.clear()

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

                    response_full = await self.character_response_stream(character=character, text_stream=text_stream)

                    if response_full:

                        response_wrapped = self.wrap_with_character_tags(response_full, character.name)
                        self.conversation_history.append({"role": "assistant", "name": character.name, "content": response_wrapped})

            except Exception as e:
                logger.error(f"Error in LLM loop: {e}")

    async def character_response_stream(self, character: Character, text_stream: AsyncIterator) -> str:
        """
        Process LLM stream into sentences for TTS.

        1. Extract text chunks from LLM stream
        2. Fire UI callback immediately for each chunk
        3. Feed chunks to stream2sentence for sentence detection
        4. Put completed sentences in sentence_queue for TTS
        5. Return full response for conversation history
        """

        message_id = f"msg-{character.id}-{int(time.time() * 1000)}"
        response_full = ""
        chunk_index = 0
        sentence_index = 0

        async def chunk_generator() -> AsyncGenerator[str, None]:
            """Inner generator that extracts content and fires UI callbacks"""
            nonlocal response_full, chunk_index

            async for chunk in text_stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        response_full += content

                        # Fire UI callback immediately (lowest latency for display)
                        if self.on_character_response_chunk:
                            response_chunk = ResponseChunk(
                                text=content,
                                message_id=message_id,
                                character_name=character.name,
                                index=chunk_index,
                                is_final=False,
                                timestamp=time.time()
                            )
                            await self.on_character_response_chunk(response_chunk)

                        chunk_index += 1
                        yield content

        # Process chunks through stream2sentence
        async for sentence in generate_sentences_async(
            chunk_generator(),
            minimum_first_fragment_length=10,
            minimum_sentence_length=20,
        ):
            sentence_text = sentence.strip()
            if sentence_text:
                # Create TTSSentence with voice info and put in queue
                tts_sentence = TTSSentence(
                    sentence=sentence_text,
                    index=sentence_index,
                    message_id=message_id,
                    character_name=character.name,
                    voice=character.voice,
                    is_final=False
                )
                await self.queues.sentence_queue.put(tts_sentence)
                sentence_index += 1

        # Signal end of sentences for this message
        final_sentinel = TTSSentence(
            sentence="",
            index=sentence_index,
            message_id=message_id,
            character_name=character.name,
            voice=character.voice,
            is_final=True
        )
        await self.queues.sentence_queue.put(final_sentinel)

        # Fire full response callback
        if self.on_character_response_full:
            await self.on_character_response_full(response_full, message_id, character.name)

        return response_full

########################################
##--         TTS Pipeline           --##
########################################

class TTS:
    """TTS Pipeline - consumes sentences from queue, generates audio via Higgs"""

    def __init__(self, queues: Queues):
        self.queues = queues
        self.serve_engine: Optional[HiggsAudioServeEngine] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Initialization flag
        self._initialized = False

        # Audio processing settings
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.voice_dir = "voices"
        self.sample_rate = 24000  # Higgs Audio sample rate
        self._chunk_size = 20  # Audio tokens per chunk
        self._chunk_overlap_duration = 0.05  # Crossfade duration in seconds

        # Callbacks
        self.on_audio_stream_start: Optional[Callback] = None
        self.on_audio_chunk: Optional[Callback] = None
        self.on_audio_stream_stop: Optional[Callback] = None

    async def initialize(self):
        """Initialize Higgs Audio TTS engine"""
        if self._initialized:
            return

        logger.info("Initializing Higgs Audio TTS service...")

        try:
            from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine

            logger.info(f"Using device: {self._device}")

            self.serve_engine = HiggsAudioServeEngine(
                model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
                audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
                device=self._device
            )

            self._initialized = True
            logger.info("Higgs Audio TTS service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Higgs Audio TTS: {e}")

    async def start(self):
        """Start the TTS processing task"""
        self.is_running = True
        self._task = asyncio.create_task(self.process_sentences())
        logger.info("TTS processing task started")

    async def stop(self):
        """Stop the TTS processing task"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TTS processing task stopped")

    async def process_sentences(self):
        """Main TTS processing loop - consumes from sentence_queue"""
        current_message_id = None

        while self.is_running:
            try:
                # Wait for sentence with timeout to allow checking is_running
                tts_sentence: TTSSentence = await asyncio.wait_for(self.queues.sentence_queue.get(), timeout=0.1)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Track message transitions for audio stream callbacks
            if tts_sentence.message_id != current_message_id:
                # New message starting - fire stop for previous if exists
                if current_message_id is not None and self.on_audio_stream_stop:
                    await self.on_audio_stream_stop(current_message_id)

                current_message_id = tts_sentence.message_id

                # Fire start callback for new message
                if self.on_audio_stream_start:
                    await self.on_audio_stream_start(message_id=tts_sentence.message_id, character_name=tts_sentence.character_name)

            # Handle end-of-message sentinel
            if tts_sentence.is_final:
                if self.on_audio_stream_stop:
                    await self.on_audio_stream_stop(tts_sentence.message_id)
                current_message_id = None
                continue

            # Generate TTS audio for this sentence
            try:
                chunk_index = 0
                async for audio_bytes in self.generate_sentence_audio(tts_sentence):
                    # Create AudioChunk and fire callback
                    audio_chunk = AudioChunk(
                        message_id=tts_sentence.message_id,
                        chunk_id=f"{tts_sentence.message_id}-{tts_sentence.index}-{chunk_index}",
                        character_name=tts_sentence.character_name,
                        audio_data=audio_bytes,
                        index=chunk_index,
                        is_final=False,
                        timestamp=time.time()
                    )
                    if self.on_audio_chunk:
                        await self.on_audio_chunk(audio_chunk)
                    chunk_index += 1
            except Exception as e:
                logger.error(f"TTS generation failed for sentence: {e}")
                continue

    def _load_voice_reference(self, voice_name: str):
        """Load reference audio and text for voice cloning"""
        from backend.boson_multimodal.data_types import Message, AudioContent

        audio_path = os.path.join(self.voice_dir, f"{voice_name}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice_name}.txt")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        # Build messages with reference audio (few-shot approach)
        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

        return messages

    async def generate_sentence_audio(self, tts_sentence: TTSSentence) -> AsyncGenerator[bytes, None]:
        """Generate audio for a sentence using Higgs Audio (async generator yielding PCM chunks)"""

        messages = self._load_voice_reference(tts_sentence.voice)

        # Add user message with text to generate
        messages.append(Message(role="user", content=tts_sentence.sentence))

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
                voice_name = file[:-4]  # Remove .wav extension
                # Only include if matching .txt file exists
                if os.path.exists(os.path.join(self.voice_dir, f"{voice_name}.txt")):
                    # Format: {id: "voice_name", name: "Voice Name"}
                    display_name = voice_name.replace('_', ' ').title()
                    voices.append({
                        "id": voice_name,
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
    """Manages WebSocket and orchestrates service modules"""

    def __init__(self):
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None
        self.transcribe: Optional[Transcribe] = None
        self.chat: Optional[Chat] = None
        self.tts: Optional[TTS] = None
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "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")

        # Background task tracking
        self._llm_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize all services and start background tasks"""

        self.queues = Queues()

        # Initialize STT service with callbacks via constructor injection
        self.transcribe = Transcribe(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )
        # Initialize STT recorder with current event loop
        self.transcribe.initialize(loop=asyncio.get_running_loop())

        # Initialize LLM service with queues and API key
        self.chat = Chat(
            queues=self.queues,
            api_key=self.openrouter_api_key
        )
        await self.chat.initialize()

        # Wire up Chat UI streaming callbacks
        self.chat.on_character_response_chunk = self.on_character_response_chunk
        self.chat.on_character_response_full = self.on_character_response_full

        # Start LLM processing loop as background task
        self._llm_task = asyncio.create_task(self.chat.send_conversation_prompt(messages=str, character=Character))
        logger.info("LLM processing loop started")

        # Initialize TTS service
        self.tts = TTS(queues=self.queues)
        await self.tts.initialize()

        # Wire up TTS audio callbacks
        self.tts.on_audio_stream_start = self.on_audio_stream_start
        self.tts.on_audio_chunk = self.on_audio_chunk
        self.tts.on_audio_stream_stop = self.on_audio_stream_stop

        # Start TTS background task
        await self.tts.start()

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket connected")

        # Load active characters from database on connection
        if self.chat:
            await self.chat.load_active_characters_from_db()

    async def shutdown(self):
        """Shutdown all services gracefully"""
        # Cancel LLM processing task
        if self._llm_task:
            self._llm_task.cancel()
            try:
                await self._llm_task
            except asyncio.CancelledError:
                pass
            logger.info("LLM processing loop stopped")

        # Stop TTS service
        if self.tts:
            await self.tts.stop()

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
                if self.transcribe:
                    self.transcribe.start_listening()

            elif message_type == "stop_listening":
                if self.transcribe:
                    self.transcribe.stop_listening()

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
                if self.chat:
                    self.chat.set_model_settings(model_settings)
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
                if self.chat:
                    self.chat.set_active_characters(characters)
                logger.info(f"Active characters set: {[c.name for c in characters]}")

            elif message_type == "clear_history":
                if self.chat:
                    self.chat.clear_conversation_history()
                logger.info("Conversation history cleared")

            elif message_type == "interrupt":
                if self.chat:
                    self.chat.interrupt()
                logger.info("Interrupt signal sent to LLM and TTS")

            elif message_type == "refresh_active_characters":
                if self.chat:
                    await self.chat.load_active_characters_from_db()
                logger.info("Active characters refreshed from database")

            elif message_type == "ping":
                await self.send_text_to_client({"type": "pong", "timestamp": time.time()})

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_data: bytes):
        """Feed audio for transcription"""
        if self.transcribe:
            self.transcribe.feed_audio(audio_data)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribe_queue.put(user_message)

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

    async def on_character_response_chunk(self, response_chunk: ResponseChunk):
        """Send text chunk to client for UI display"""
        await self.send_text_to_client({
            "type": "response_chunk",
            "text": response_chunk.text,
            "message_id": response_chunk.message_id,
            "character_name": response_chunk.character_name,
            "index": response_chunk.index
        })

    async def on_character_response_full(self, response_full: str, message_id: str, character_name: str):
        """Send full response to client"""
        await self.send_text_to_client({
            "type": "response_full",
            "text": response_full,
            "message_id": message_id,
            "character_name": character_name
        })

    async def on_audio_stream_start(self, message_id: str, character_name: str):
        """Notify client that audio for a character is starting"""
        await self.send_text_to_client({
            "type": "audio_stream_start",
            "message_id": message_id,
            "character_name": character_name,
            "timestamp": time.time()
        })

    async def on_audio_chunk(self, audio_chunk: AudioChunk):
        """Send audio chunk to client"""
        # Send metadata first
        await self.send_text_to_client({
            "type": "audio_chunk_meta",
            "message_id": audio_chunk.message_id,
            "chunk_id": audio_chunk.chunk_id,
            "character_name": audio_chunk.character_name,
            "index": audio_chunk.index,
        })
        # Then send binary audio
        await self.stream_audio_to_client(audio_chunk.audio_data)

    async def on_audio_stream_stop(self, message_id: str):
        """Notify client that audio stream has ended"""
        await self.send_text_to_client({
            "type": "audio_stream_stop",
            "message_id": message_id,
            "timestamp": time.time()
        })

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
