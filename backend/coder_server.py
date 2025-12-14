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
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator, Awaitable
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

Callback = Callable[..., Optional[Awaitable[None]]]


@dataclass
class STTCallbacks:
    """Callback functions for STT events"""
    # Transcription callbacks
    on_realtime_update: Optional[Callable[[str], Any]] = None
    on_realtime_stabilized: Optional[Callable[[str], Any]] = None
    on_final_transcription: Optional[Callable[[str], Any]] = None

    # VAD callbacks
    on_vad_start: Optional[Callable[[], Any]] = None
    on_vad_stop: Optional[Callable[[], Any]] = None

    # Recording callbacks
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None


class STTPipeline:
    """
    Async-compatible Speech-to-Text service for voice chat applications.

    Wraps RealtimeSTT's AudioToTextRecorder and bridges its threaded callbacks
    to asyncio for use with FastAPI WebSocket handlers.

    Usage:
        stt = STTPipeline(
            on_realtime_update=lambda text: broadcast({"type": "partial", "text": text}),
            on_final_transcription=lambda text: llm_queue.put(text),
            on_vad_start=handle_interrupt,
        )

        await stt.start()
        stt.start_listening()  # User clicked mic

        # Audio flows in from WebSocket handler:
        async def handle_audio(data: bytes):
            stt.feed_audio(data)
    """

    def __init__(
        self,
        # Async callbacks - bridged from RealtimeSTT's threads
        on_realtime_update: Optional[Callback] = None,
        on_realtime_stabilized: Optional[Callback] = None,
        on_final_transcription: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,

        # Model configuration
        model: str = "small",
        realtime_model: str = "small",
        language: str = "en",
        device: str = "cuda",
        compute_type: str = "float16",

        # Real-time transcription settings
        enable_realtime_transcription: bool = True,
        realtime_processing_pause: float = 0.1,
        init_realtime_after_seconds: float = 0.15,

        # VAD settings
        silero_sensitivity: float = 0.4,
        webrtc_sensitivity: int = 3,
        post_speech_silence_duration: float = 0.5,
        min_length_of_recording: float = 0.3,
        pre_recording_buffer_duration: float = 0.5,

        # Additional recorder kwargs
        **recorder_kwargs
    ):
        """
        Initialize the STT service.

        Args:
            on_realtime_update: Called with partial transcription text (may change)
            on_realtime_stabilized: Called with stabilized text (more reliable)
            on_final_transcription: Called with final transcription after speech ends
            on_vad_start: Called when voice activity detected
            on_vad_stop: Called when voice activity ends
            on_recording_start: Called when recording begins
            on_recording_stop: Called when recording ends

            model: Main transcription model (for final transcription)
            realtime_model: Model for real-time transcription (faster)
            language: Language code for transcription
            device: Device to run models on ("cuda" or "cpu")
            compute_type: Compute type for models

            enable_realtime_transcription: Enable real-time updates
            realtime_processing_pause: Pause between real-time transcriptions
            init_realtime_after_seconds: Delay before starting realtime

            silero_sensitivity: Silero VAD sensitivity (0-1)
            webrtc_sensitivity: WebRTC VAD sensitivity (0-3)
            post_speech_silence_duration: Silence duration to end recording
            min_length_of_recording: Minimum recording length
            pre_recording_buffer_duration: Audio buffer before VAD triggers

            **recorder_kwargs: Additional kwargs for AudioToTextRecorder
        """
        # Store user callbacks in dataclass to avoid name collision with handler methods
        self.callbacks = STTCallbacks(
            on_realtime_update=on_realtime_update,
            on_realtime_stabilized=on_realtime_stabilized,
            on_final_transcription=on_final_transcription,
            on_vad_start=on_vad_start,
            on_vad_stop=on_vad_stop,
            on_recording_start=on_recording_start,
            on_recording_stop=on_recording_stop,
        )

        # Store config
        self._model = model
        self._realtime_model = realtime_model
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._enable_realtime = enable_realtime_transcription
        self._realtime_pause = realtime_processing_pause
        self._realtime_init_delay = init_realtime_after_seconds
        self._silero_sensitivity = silero_sensitivity
        self._webrtc_sensitivity = webrtc_sensitivity
        self._post_speech_silence = post_speech_silence_duration
        self._min_recording_length = min_length_of_recording
        self._pre_recording_buffer = pre_recording_buffer_duration
        self._recorder_kwargs = recorder_kwargs

        # Runtime state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._recorder: Optional[AudioToTextRecorder] = None
        self._is_running = False
        self._is_listening = False

        # Interrupt detection
        self._tts_playing = False
        self._interrupt_detected = False
        self._state_lock = threading.Lock()

        # Transcription loop task
        self._transcription_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_listening(self) -> bool:
        with self._state_lock:
            return self._is_listening

    @property
    def state(self) -> str:
        """Current recorder state: inactive, listening, recording, transcribing."""
        if self._recorder:
            return self._recorder.state
        return "inactive"

    def _create_recorder(self) -> AudioToTextRecorder:
        """Create and configure the AudioToTextRecorder."""
        return AudioToTextRecorder(
            # Model config
            model=self._model,
            realtime_model_type=self._realtime_model,
            language=self._language,
            device=self._device,
            compute_type=self._compute_type,

            # No microphone - audio fed via WebSocket
            use_microphone=False,

            # Real-time transcription with our bridged callbacks
            enable_realtime_transcription=self._enable_realtime,
            realtime_processing_pause=self._realtime_pause,
            init_realtime_after_seconds=self._realtime_init_delay,
            on_realtime_transcription_update=self._on_realtime_update,
            on_realtime_transcription_stabilized=self._on_realtime_stabilized,

            # VAD config with our bridged callbacks
            silero_sensitivity=self._silero_sensitivity,
            webrtc_sensitivity=self._webrtc_sensitivity,
            post_speech_silence_duration=self._post_speech_silence,
            min_length_of_recording=self._min_recording_length,
            pre_recording_buffer_duration=self._pre_recording_buffer,
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,

            # Recording callbacks
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,

            # Server-side settings
            spinner=False,
            no_log_file=True,

            **self._recorder_kwargs
        )

    async def start(self) -> None:
        """Start the STT service and initialize models."""
        if self._is_running:
            logger.warning("STT service already running")
            return

        self._loop = asyncio.get_running_loop()

        logger.info("Initializing STT with model: %s", self._model)
        self._recorder = await self._loop.run_in_executor(
            None, self._create_recorder
        )

        self._is_running = True
        logger.info("STT service started")

    async def stop(self) -> None:
        """Stop the STT service and clean up."""
        if not self._is_running:
            return

        self._is_running = False
        self._is_listening = False

        # Cancel transcription loop if running
        if self._transcription_task and not self._transcription_task.done():
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        # Shutdown recorder
        if self._recorder:
            await self._loop.run_in_executor(None, self._recorder.shutdown)
            self._recorder = None

        logger.info("STT service stopped")

    async def __aenter__(self) -> "STTPipeline":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    def start_listening(self) -> None:
        """Start listening for voice activity."""
        if not self._is_running or not self._recorder:
            logger.warning("Cannot start listening: STT not running")
            return

        with self._state_lock:
            self._is_listening = True
            self._interrupt_detected = False

        self._recorder.listen()
        logger.debug("Listening started")

    def stop_listening(self) -> None:
        """Stop listening and clear audio buffer."""
        if not self._is_running or not self._recorder:
            return

        with self._state_lock:
            self._is_listening = False

        self._recorder.clear_audio_queue()
        logger.debug("Listening stopped")

    def abort(self) -> None:
        """Abort current recording without transcribing."""
        if self._recorder:
            self._recorder.abort()
            logger.debug("Recording aborted")

    def set_tts_playing(self, playing: bool) -> None:
        """Set TTS playback state for interrupt detection.When TTS is playing and VAD detects speech, an interrupt is flagged."""
        with self._state_lock:
            self._tts_playing = playing
            if not playing:
                self._interrupt_detected = False

    def is_interrupt_detected(self) -> bool:
        """Check if user interrupted during TTS playback."""
        with self._state_lock:
            return self._interrupt_detected

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        with self._state_lock:
            self._interrupt_detected = False

    def _invoke_callback(self, callback: Optional[Callback], *args) -> None:
        """Invoke a user callback from a RealtimeSTT background thread. Handles both sync and async callbacks, bridging to the event loop when necessary."""

        if callback is None or self._loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            # Async callback - schedule on event loop
            asyncio.run_coroutine_threadsafe(callback(*args), self._loop)
        else:
            # Sync callback - run thread-safe on event loop
            self._loop.call_soon_threadsafe(callback, *args)

    def _on_realtime_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self._invoke_callback(self.callbacks.on_realtime_update, text)

    def _on_realtime_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self._invoke_callback(self.callbacks.on_realtime_stabilized, text)

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        with self._state_lock:
            if self._tts_playing:
                self._interrupt_detected = True
                logger.info("Interrupt detected: user speaking during TTS")

        self._invoke_callback(self.callbacks.on_vad_start)

    def _on_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self._invoke_callback(self.callbacks.on_vad_stop)

    def _on_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self._invoke_callback(self.callbacks.on_recording_start)

    def _on_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self._invoke_callback(self.callbacks.on_recording_stop)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self._recorder:
            try:
                self._recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    async def get_transcription(self) -> Optional[str]:
        """
        Wait for complete speech turn and return transcription.

        Pure data retrieval - no callbacks invoked. Use run_transcription_loop()
        for continuous transcription with callback handling.

        Returns:
            Final transcription text, or None if not running/error occurred.
        """
        if not self._is_running or not self._recorder:
            return None

        if not self._is_listening:
            self.start_listening()

        try:
            # text() blocks until speech complete
            user_message = await self._loop.run_in_executor(None, self._recorder.text)
            return user_message

        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    async def run_transcription_loop(self) -> None:
        """
        Run continuous transcription loop.

        Orchestration layer that handles callbacks. Continuously listens and
        transcribes, invoking on_final_transcription for each complete utterance.

        Run this as a background task via start_transcription_loop().
        """
        logger.info("Starting transcription loop")

        while self._is_running:
            try:
                if not self._is_listening:
                    self.start_listening()

                user_message = await self.get_transcription()

                if user_message:
                    # Invoke callback here (orchestration layer handles side effects)
                    if self.callbacks.on_final_transcription:
                        self._invoke_callback(self.callbacks.on_final_transcription, user_message)

                    logger.debug("Transcription: %s", user_message[:50] + "..." if len(user_message) > 50 else user_message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Transcription loop error: %s", e)
                await asyncio.sleep(0.1)

        logger.info("Transcription loop ended")

    def start_transcription_loop(self) -> asyncio.Task:
        """Start the transcription loop as a background task. Returns the task so it can be cancelled if needed."""

        if self._transcription_task and not self._transcription_task.done():
            logger.warning("Transcription loop already running")
            return self._transcription_task

        self._transcription_task = asyncio.create_task(self.run_transcription_loop())
        return self._transcription_task

    def stop_transcription_loop(self) -> None:
        """Stop the transcription loop."""
        if self._transcription_task:
            self._transcription_task.cancel()

########################################
##--           LLM Service          --##
########################################

class LLMPipeline:
    """LLM Service for multi-character conversation loop"""

    async def conversation_loop(self):
        """Main LLM conversation loop for multi-character conversations."""

########################################
##--           TTS Service          --##
########################################

class TTSPipeline:
    """Higgs Streaming"""

    async def speech_loop(self):
        """Main audio generation loop for character speech"""

########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and coordinates services"""

    def __init__(self):
        self.stt_pipeline: Optional[STTPipeline] = None
        self.llm_pipeline: Optional[LLMPipeline] = None
        self.tts_pipeline: Optional[TTSPipeline] = None
        self.websocket: Optional[WebSocket] = None
        self.queues: Optional[Queues] = None
        self.service_tasks: List[asyncio.Task] = []

        # API key from environment
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "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")

    async def initialize(self):
        """Initialize all services with proper callbacks"""

        self.queues = Queues()

        # Initialize STT service with callbacks via constructor injection
        self.stt_pipeline = STTPipeline(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )
        await self.stt_pipeline.start()

        # Initialize LLM service with queues and API key
        self.llm_pipeline = LLMPipeline(
            queues=self.queues,
            api_key=self.openrouter_api_key
        )
        await self.llm_pipeline.initialize()

        # Initialize TTS Service Manager with queues
        self.tts_pipeline = TTSPipeline(queues=self.queues)
        await self.tts_pipeline.initialize()

        logger.info("WebSocketManager initialized")

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket connected")

        # Load active characters from database on connection
        if self.llm_pipeline:
            await self.llm_pipeline.load_active_characters_from_db()

        await self.start_service_tasks()

    async def start_service_tasks(self):
        """Start all services"""

        self.service_tasks = [
            asyncio.create_task(self.stt_pipeline.run_transcription_loop()),
            asyncio.create_task(self.llm_pipeline.conversation_loop()),
            asyncio.create_task(self.tts_pipeline.speech_loop(send_audio_callback=self.stream_audio_to_client)),
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
        if self.tts_pipeline:
            await self.tts_pipeline.shutdown()

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
                if self.stt_pipeline:
                    self.stt_pipeline.start_listening()

            elif message_type == "stop_listening":
                if self.stt_pipeline:
                    self.stt_pipeline.stop_listening()

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
                if self.llm_pipeline:
                    self.llm_pipeline.set_model_settings(model_settings)
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
                if self.llm_pipeline:
                    self.llm_pipeline.set_active_characters(characters)
                logger.info(f"Active characters set: {[c.name for c in characters]}")

            elif message_type == "clear_history":
                # Clear conversation history
                if self.llm_pipeline:
                    self.llm_pipeline.clear_conversation_history()
                logger.info("Conversation history cleared")

            elif message_type == "interrupt":
                # Signal interrupt to stop current generation
                if self.llm_pipeline:
                    self.llm_pipeline.interrupt_event.set()
                logger.info("Interrupt signal sent")

            elif message_type == "refresh_active_characters":
                # Refresh active characters from database
                if self.llm_pipeline:
                    await self.llm_pipeline.load_active_characters_from_db()
                logger.info("Active characters refreshed from database")

            elif message_type == "ping":
                # Respond to ping with pong for connection health check
                await self.send_text_to_client({"type": "pong", "timestamp": time.time()})

        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_data: bytes):
        """Feed audio for transcription"""
        if self.stt_pipeline:
            self.stt_pipeline.feed_audio(audio_data)

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
