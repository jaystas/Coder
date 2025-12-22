# Standard library
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

from backend.database_director import (
    db,
    Character,
    CharacterCreate,
    CharacterUpdate,
    Voice,
    VoiceCreate,
    VoiceUpdate,
    Conversation,
    ConversationCreate,
    ConversationUpdate,
    Message as ConversationMessage,
    MessageCreate,
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################
##--          Data Models           --##
########################################

@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis with character context"""
    text: str
    index: int
    session_id: str
    character_id: str
    character_name: str
    voice_id: str
    is_final: bool = False
    is_session_complete: bool = False

@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming with character context"""
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    character_id: str
    is_final: bool = False
    is_session_complete: bool = False

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

########################################
##--        Queue Management        --##
########################################

class PipeQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(self):
        # Queue for transcribed user messages (STT → Chat)
        self.transcribe_queue: asyncio.Queue[str] = asyncio.Queue()
        # Queue for sentences to be synthesized (Chat → TTS)
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
        # Queue for audio chunks to stream (TTS → WebSocket)
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

########################################
##--  Speech to Text Transcription  --##
########################################

Callback = Callable[..., Optional[Awaitable[None]]]

class Transcribe:
    """Realtime transcription of user's audio prompt"""

    def __init__(self,
        on_transcription_update: Optional[Callback] = None,
        on_transcription_stabilized: Optional[Callback] = None,
        on_transcription_finished: Optional[Callback] = None,
        on_vad_detect_start: Optional[Callback] = None,
        on_vad_detect_stop: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,
    ):
        # Event loop for async callback dispatching (set by WebSocketManager)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Transcription thread
        self._thread: Optional[threading.Thread] = None

        # Callback registry - keys match the internal callback method names
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_transcription_update': on_transcription_update,
            'on_transcription_stabilized': on_transcription_stabilized,
            'on_transcription_finished': on_transcription_finished,
            'on_vad_detect_start': on_vad_detect_start,
            'on_vad_detect_stop': on_vad_detect_stop,
            'on_vad_start': on_vad_start,
            'on_vad_stop': on_vad_stop,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
        }

        self.is_listening = False

        self.recorder = AudioToTextRecorder(
            model="small.en",
            language="en",
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            realtime_model_type="small.en",
            on_realtime_transcription_update=self._on_transcription_update,
            on_realtime_transcription_stabilized=self._on_transcription_stabilized,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_vad_detect_start=self._on_vad_detect_start,
            on_vad_detect_stop=self._on_vad_detect_stop,
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.5,
            spinner=False,
            level=logging.WARNING,
            use_microphone=False
        )

    def transcriber(self):
        """Transcribes in real-time from browser audio feed"""

        while self.is_listening:
            try:
                user_message = self.recorder.text()

                if user_message and user_message.strip():
                    callback = self.callbacks.get('on_transcription_finished')
                    if callback:
                        self.run_callback(callback, user_message)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")

        
    def run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run a user callback from a RealtimeSTT background thread."""

        if callback is None or self.loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)

        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    def start_listening(self):
        """Start listening for audio input"""
        if self.is_listening:
            return  # Already listening

        self.is_listening = True

        # Start transcription in background thread
        self._thread = threading.Thread(target=self.transcriber, daemon=True)
        self._thread.start()
        logger.info("Started listening for audio")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        logger.info("Stopped listening for audio")

    def _on_transcription_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self.run_callback(self.callbacks.get('on_transcription_update'), text)

    def _on_transcription_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_stabilized'), text)

    def _on_transcription_finished(self, user_message: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_finished'), user_message)

    def _on_vad_detect_start(self) -> None:
        """RealtimeSTT callback: started listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_start'))

    def _on_vad_detect_stop(self) -> None:
        """RealtimeSTT callback: stopped listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_stop'))

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
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
##--       LLM Chat Functions       --##
########################################

class ChatFunctions:
    """Manages LLM chat sessions and conversation history"""

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self.model_settings: Optional[ModelSettings] = None
        self.active_characters: List[Character] = []
        self.user_name: str = "Jay"

    async def start_new_chat(self):
        """Start a new chat session"""
        self.conversation_history = []
        self.conversation_id = str(uuid.uuid4())

    async def get_active_characters(self) -> List[Character]:
        """Get active characters from database"""
        return await db.get_active_characters()

    def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests"""

        self.model_settings = model_settings

    def clear_conversation_history(self):
        """Clear the conversation history"""

        self.conversation_history = []

    def wrap_character_tags(self, text: str, character_name: str) -> str:
        """Wrap response text with character name XML tags for conversation history."""

        return f"<{character_name}>{text}</{character_name}>"

    def character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create Character Instruction Message."""

        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else.'
        }
    
    def get_model_settings(self) -> ModelSettings:
        """Get current model settings for the LLM request"""
        if self.model_settings is None:

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
    
    def build_messages_for_character(self, character: Character) -> List[Dict[str, str]]:
        """Build the message list for OpenRouter API call."""

        messages = []

        # Character's system prompt
        if character.system_prompt:
            messages.append({"role": "system", "content": character.system_prompt})

        # Conversation history
        messages.extend(self.conversation_history)

        # Instruction for this character
        messages.append(self.character_instruction_message(character))

        return messages
    
    async def process_user_messages(self, queues: PipeQueues, user_message: str, user_name: str, llm_stream: 'LLMTextStream', sentence_queue: asyncio.Queue[TTSSentence], session_id: str,on_text_chunk: Optional[Callable[[str], Awaitable[None]]] = None) -> None:
            """Processes User Messages"""
            
            self.conversation_history.append({"role": "user", "name": user_name, "content": user_message})

            responding_characters = self.parse_character_mentions(message=user_message, active_characters=self.active_characters)

            for i, character in enumerate(responding_characters):
                is_last = (i == len(responding_characters) - 1)

                messages = self.build_messages_for_character(character)

                full_response = await llm_stream.stream_character_response(
                    messages=messages,
                    character=character,
                    session_id=session_id,
                    model_settings=self.get_model_settings(),
                    sentence_queue=sentence_queue,
                    on_text_chunk=on_text_chunk,
                    is_last_character=is_last
                )

                response_wrapped = self.wrap_character_tags(full_response, character.name)
                self.conversation_history.append({"role": "assistant", "name": character.name, "content": response_wrapped})

########################################
##--      Text Stream Processor     --##
########################################

class LLMTextStream:
    """Processes LLM stream into sentences for TTS"""

    def __init__(self, queues: PipeQueues, api_key: str):
        self.queues = queues
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    async def stream_character_response(
        self,
        messages: List[Dict[str, str]],
        character: Character,
        session_id: str,
        model_settings: ModelSettings,
        sentence_queue: asyncio.Queue[TTSSentence],
        on_text_chunk: Optional[Callable[[str], Awaitable[None]]] = None,
        is_last_character: bool = False
    ) -> str:
        """
        Stream LLM response for a character, extract sentences, queue for TTS.

        Returns full response text. Does NOT wait for TTS/audio - that runs concurrently.
        """
        sentence_index = 0
        full_response = ""

        try:
            stream = await self.client.chat.completions.create(
                model=model_settings.model,
                messages=messages,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
                stream=True,
                extra_body={
                    "top_k": model_settings.top_k,
                    "min_p": model_settings.min_p,
                    "repetition_penalty": model_settings.repetition_penalty,
                }
            )

            async def chunk_generator() -> AsyncGenerator[str, None]:
                nonlocal full_response
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        content = chunk.choices[0].delta.content
                        if content:
                            full_response += content
                            if on_text_chunk:
                                await on_text_chunk(content)
                            yield content

            async for sentence in generate_sentences_async(
                chunk_generator(),
                minimum_first_fragment_length=10,
                minimum_sentence_length=15,
                quick_yield_single_sentence_fragment=True,
                sentence_fragment_delimiters=".?!;:,\n…)]}。-",
                full_sentence_delimiters=".?!\n…。",
            ):
                sentence_text = sentence.strip()
                if sentence_text:
                    await sentence_queue.put(TTSSentence(
                        text=sentence_text,
                        index=sentence_index,
                        session_id=session_id,
                        character_id=character.id,
                        character_name=character.name,
                        voice_id=character.voice,
                        is_final=False,
                        is_session_complete=False
                    ))
                    logger.info(f"[LLM] {character.name} sentence {sentence_index}: {sentence_text[:50]}...")
                    sentence_index += 1

        except Exception as e:
            logger.error(f"[LLM] Error streaming for {character.name}: {e}")

        # Signal end of this character's text stream
        await sentence_queue.put(TTSSentence(
            text="",
            index=sentence_index,
            session_id=session_id,
            character_id=character.id,
            character_name=character.name,
            voice_id=character.voice,
            is_final=True,
            is_session_complete=is_last_character
        ))
        logger.info(f"[LLM] {character.name} complete: {sentence_index} sentences")

        return full_response

########################################
##--      Text to Speech Worker     --##
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

class TTSWorker:
    """Worker Synthesizes Sentences using Higgs Audio"""

    def __init__(self, queues: PipeQueues):
        self.queues = queues
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Audio settings
        self.sample_rate = 24000
        self._chunk_size = 14
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Voice reference
        self.voice_dir = "backend/voices"
        self.voice_name = "amelia"

    async def initialize(self):

        self.engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=self._device
        )

        logger.info("Higgs Audio TTS initialized")

    async def start(self):
        """Start TTS Worker"""
        self.is_running = True
        self._task = asyncio.create_task(self.process_sentences())

    async def stop(self):
        """Stop TTS Worker"""
        self.is_running = False
        if self._task:
            self._task.cancel()

            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def process_sentences(self):
        """Process Sentences - Synthesize Audio"""
        while self.is_running:
            try:
                sentence = await asyncio.wait_for(self.queues.sentence_queue.get(), timeout=0.1)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Pass through sentinels
            if sentence.is_final:
                await self.queues.audio_queue.put(AudioChunk(
                    audio_bytes=b"",
                    sentence_index=sentence.index,
                    chunk_index=0,
                    session_id=sentence.session_id,
                    character_id=sentence.character_id,
                    is_final=True,
                    is_session_complete=sentence.is_session_complete
                ))

                logger.info(f"[TTS] End sentinel passed through")
                continue

            # Generate audio for this sentence
            logger.info(f"[TTS] Generating audio for sentence {sentence.index}")

            chunk_index = 0
            try:
                async for pcm_bytes in self.generate_audio_for_sentence(sentence.text):
                    audio_chunk = AudioChunk(
                        audio_bytes=pcm_bytes,
                        sentence_index=sentence.index,
                        chunk_index=chunk_index,
                        session_id=sentence.session_id,
                        character_id=sentence.character_id,
                        is_final=False,
                        is_session_complete=False
                    )

                    await self.queues.audio_queue.put(audio_chunk)
                    chunk_index += 1

                logger.info(f"[TTS] {sentence.character_name} #{sentence.index}: {chunk_index} chunks")
            except Exception as e:
                logger.error(f"[TTS] Error generating audio: {e}")
                continue

    def load_voice_reference(self, voice: str):
        """Load reference audio and text for voice cloning"""

        audio_path = os.path.join(self.voice_dir, f"{voice}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice}.txt")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        messages = [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

        return messages

    async def generate_audio_for_sentence(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming"""

        messages = self.load_voice_reference(voice)
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                temperature=0.7,
                top_p=0.95,
                top_k=50,
                stop_strings=['<|end_of_text|>', '<|eot_id|>'],
                ras_win_len=7,
                ras_win_max_num_repeat=2,
                force_audio_gen=True,
            ):
                if delta.audio_tokens is None:
                    continue

                # Check for end token (1025)
                if torch.all(delta.audio_tokens == 1025):
                    break

                # Accumulate tokens
                audio_tokens.append(delta.audio_tokens[:, None])

                # Count non-padding tokens (1024 is padding)
                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1

                # Decode when chunk size reached
                if seq_len > 0 and seq_len % self._chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    try:
                        # Revert delay pattern and decode
                        vq_code = (
                            revert_delay_pattern(
                                audio_tensor,
                                start_idx=seq_len - self._chunk_size + 1
                            )
                            .clip(0, 1023)
                            .to(self._device)
                        )

                        waveform = self.engine.audio_tokenizer.decode(
                            vq_code.unsqueeze(0)
                        )[0, 0]

                        # Convert to numpy
                        if isinstance(waveform, torch.Tensor):
                            waveform_np = waveform.detach().cpu().numpy()
                        else:
                            waveform_np = np.asarray(waveform, dtype=np.float32)

                        # Convert to PCM16 bytes
                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()

                    except Exception as e:
                        logger.warning(f"Error decoding chunk: {e}")
                        continue

        # Flush remaining tokens
        if seq_len > 0 and seq_len % self._chunk_size != 0 and audio_tokens:
            audio_tensor = torch.cat(audio_tokens, dim=-1)
            remaining = seq_len % self._chunk_size

            try:
                vq_code = (
                    revert_delay_pattern(
                        audio_tensor,
                        start_idx=seq_len - remaining + 1
                    )
                    .clip(0, 1023)
                    .to(self._device)
                )

                waveform = self.engine.audio_tokenizer.decode(
                    vq_code.unsqueeze(0)
                )[0, 0]

                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.detach().cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform, dtype=np.float32)

                pcm = np.clip(waveform_np, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

            except Exception as e:
                logger.warning(f"Error flushing remaining audio: {e}")

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

########################################
##--      Conversation Pipeline     --##
########################################

class ConversationPipeline:
    """Orchestrates Complete LLM → TTS → Audio Pipeline"""

    def __init__(self, api_key: str):
        self.queues = PipeQueues()
        self.chat = ChatFunctions()
        self.llm_stream = LLMTextStream(self.queues, api_key)
        self.tts_worker = TTSWorker(self.queues)
        self.initialized = False


    async def start_pipeline(self):
        """Start the voice transcription → LLM → TTS pipeline as a background task"""
        self.consumer_task = asyncio.create_task(self.get_user_messages())

    async def get_user_messages(self):
        """Background task: get user message from transcribe queue and process."""

        while True:
            user_message: str = await self.queues.transcribe_queue.get()

            if user_message and user_message.strip():
                session_id = str(uuid.uuid4())
                
                await self.chat.process_user_message(
                    user_message=user_message,
                    llm_stream=self.llm_stream,
                    sentence_queue=self.queues.sentence_queue,
                    session_id=session_id
                )

    async def conversate(self, user_message: str, character: Character, websocket: WebSocket):
        """Main conversation flow: user message → LLM → TTS → audio stream"""

        # This is where calls will go (creates one easy to read top down flow).
        # Do not put whole functions here, just calls.

    async def audio_player(self):
        """Streams PCM Audio to Browser for Real-time Playback"""

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and routes messages"""

    def __init__(self):
        # WebSocket connection
        self.websocket: Optional[WebSocket] = None

        # Event loop reference for async callback dispatching
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Shared queues for pipeline
        self.queues = PipeQueues()

        # Chat functions for LLM interaction
        self.chat = ChatFunctions()

        # Transcription handler with callbacks
        self.transcribe = Transcribe(
            on_transcription_update=self.on_transcription_update,
            on_transcription_stabilized=self.on_transcription_stabilized,
            on_transcription_finished=self.on_transcription_finished,
        )

    async def initialize(self):
        """Initialize resources on app startup"""
        self.loop = asyncio.get_running_loop()
        # Pass event loop to transcribe for async callback dispatching
        self.transcribe.loop = self.loop
        logger.info("WebSocketManager initialized")

        self.chat = ChatFunctions(queues=self.queues, api_key=self.openrouter_api_key)
        
        await self.chat.initialize()

    async def shutdown(self):
        """Clean up resources on app shutdown"""
        if self.transcribe:
            self.transcribe.stop_listening()

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket client connected")

    async def disconnect(self):
        """Clean up on WebSocket disconnect"""
        if self.transcribe:
            self.transcribe.stop_listening()
        self.websocket = None
        logger.info("WebSocket client disconnected")

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

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_bytes: bytes):
        """Feed audio for transcription"""
        if self.transcribe:
            self.transcribe.feed_audio(audio_bytes)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribe_queue.put(user_message)

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to send text to client: {e}")

    async def stream_audio_to_client(self, audio_bytes: bytes):
        """Send binary audio to client (TTS)"""
        if self.websocket:
            try:
                await self.websocket.send_bytes(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to send audio to client: {e}")

    async def on_transcription_update(self, text: str):
        """Callback: realtime transcription update"""
        await self.send_text_to_client({"type": "stt_update", "text": text})

    async def on_transcription_stabilized(self, text: str):
        """Callback: stabilized transcription"""
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})

    async def on_transcription_finished(self, user_message: str):
        """Callback: final transcription complete"""
        await self.queues.transcribe_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_finished", "text": user_message})

    async def on_text_stream_start(self, message_id: str, character_name: str):
        """Notify client that text for a character is starting"""
        await self.send_text_to_client({
            "type": "text_stream_start",
            "message_id": message_id,
            "character_name": character_name,
            "timestamp": time.time()
        })

    async def on_text_stream_stop(self, message_id: str):
        """Notify client that text stream has ended"""
        await self.send_text_to_client({
            "type": "text_stream_stop",
            "message_id": message_id,
            "timestamp": time.time()
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

########################################
##--           FastAPI App          --##
########################################

convo_pipe = ConversationPipeline()
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
