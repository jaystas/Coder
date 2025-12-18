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
from enum import Enum, auto
from datetime import datetime
from pydantic import BaseModel
from queue import Queue, Empty
from openai import AsyncOpenAI
from collections import defaultdict
from collections.abc import Awaitable
from threading import Thread, Event, Lock
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from supabase import Client
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from typing import Callable, Optional, Dict, List, Union, Any, AsyncIterator, AsyncGenerator, Awaitable
from stream2sentence import generate_sentences_async
from concurrent.futures import ThreadPoolExecutor

from backend.RealtimeSTT import AudioToTextRecorder
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# Database Director - centralized CRUD operations for Supabase
from backend.database_director import (db, Character, CharacterCreate, CharacterUpdate, Voice, VoiceCreate, VoiceUpdate, Conversation, ConversationCreate, ConversationUpdate, Message as ConversationMessage, MessageCreate)

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

    def __init__(self):
        self.recorder: Optional[AudioToTextRecorder] = None
        self.recording_thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, Any] = {}
        self.is_listening = False
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


    async def transcription_loop(self):
        """Main trascription loop running in separate thread"""

        logger.info("STT transcription loop started")

        while True:
            if self.is_listening:
                try:

                    user_message = self.recorder.text()
                    
                    if user_message and user_message.strip():

                        if self.callbacks.on_final_transcription:
                            self.run_callback(self.callbacks.on_final_transcription, user_message)

                except Exception as e:
                    logger.error(f"Error in recording loop: {e}")

            time.sleep(0.1)

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
        self.run_callback(self.callbacks.on_realtime_update, text)

    def _on_realtime_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.on_realtime_stabilized, text)

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        if self._tts_playing:
            self._interrupt_detected = True
            logger.info("Interrupt detected: user speaking during TTS")

        self.run_callback(self.callbacks.on_vad_start)

    def _on_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self.run_callback(self.callbacks.on_vad_stop)

    def _on_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self.run_callback(self.callbacks.on_recording_start)

    def _on_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self.run_callback(self.callbacks.on_recording_stop)

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
        
        # UI streaming callbacks (set by WebSocketManager)
        self.on_character_response_chunk = None
        self.on_character_response_full = None


    async def initialize(self):
        """"""


    async def start_new_chat(self):
        """"""

    async def get_active_characters(self, active_characters: List[Character]):
        """Get active characters in conversation"""

        active_characters = await db.get_active_characters()

    
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
    

    async def send_conversation_prompt(self, user_name: str = "Jay"):
        """Build message structure and initiate character response for single and multi-character conversations."""

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
        full_response = ""
        chunk_index = 0
        sentence_index = 0

        async def chunk_generator() -> AsyncGenerator[str, None]:
            """Inner generator that extracts content and fires UI callbacks"""
            nonlocal full_response, chunk_index

            async for chunk in text_stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content

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
            await self.on_character_response_full(full_response, message_id, character.name)

        return full_response

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

        # Callbacks
        self.on_audio_stream_start: Optional[Callback] = None
        self.on_audio_chunk: Optional[Callback] = None
        self.on_audio_stream_stop: Optional[Callback] = None

    async def initialize(self):
        """Initialize Higgs Audio serve engine"""
        # TODO: Initialize HiggsAudioServeEngine here
        # self.serve_engine = HiggsAudioServeEngine(...)
        pass

    async def start(self):
        """Start the TTS processing task"""
        self.is_running = True
        self._task = asyncio.create_task(self._process_sentences())
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

    async def _process_sentences(self):
        """Main TTS processing loop - consumes from sentence_queue"""
        current_message_id = None

        while self.is_running:
            try:
                # Wait for sentence with timeout to allow checking is_running
                tts_sentence: TTSSentence = await asyncio.wait_for(
                    self.queues.sentence_queue.get(),
                    timeout=0.1
                )
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
                    await self.on_audio_stream_start(
                        message_id=tts_sentence.message_id,
                        character_name=tts_sentence.character_name
                    )

            # Handle end-of-message sentinel
            if tts_sentence.is_final:
                if self.on_audio_stream_stop:
                    await self.on_audio_stream_stop(tts_sentence.message_id)
                current_message_id = None
                continue

            # Generate TTS audio for this sentence
            try:
                await self.generate_sentence_audio(tts_sentence)
            except Exception as e:
                logger.error(f"TTS generation failed for sentence: {e}")
                # Skip and continue on error
                continue

    def load_voice_reference(self, voice: str) -> List[Message]:
        """Load voice reference messages for TTS"""
        # TODO: Implement voice reference loading
        # This should return the reference messages for the given voice
        return []

    async def generate_sentence_audio(self, tts_sentence: TTSSentence):
        """Generate audio for a sentence using Higgs Audio"""

        messages = self.load_voice_reference(tts_sentence.voice)

        # Add user message with text to generate
        messages.append(Message(role="user", content=tts_sentence.sentence))

        # Create ChatML sample
        chat_ml_sample = ChatMLSample(messages=messages)

        async for delta in self.serve_engine.generate_delta_stream(
            chat_ml_sample=chat_ml_sample,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            stop_strings=['<|end_of_text|>', '<|eot_id|>'],
            ras_win_len=7,
            ras_win_max_num_repeat=2,
            force_audio_gen=True,
        ):
            # TODO: Process delta and fire on_audio_chunk callback
            # audio_chunk = AudioChunk(...)
            # if self.on_audio_chunk:
            #     await self.on_audio_chunk(audio_chunk)
            pass

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
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

    async def initialize(self):
        """Initialize all services and start background tasks"""

        self.queues = Queues()

        # Initialize STT service with callbacks via constructor injection
        self.transcribe = Transcribe(
            on_realtime_update=self.on_realtime_update,
            on_realtime_stabilized=self.on_realtime_stabilized,
            on_final_transcription=self.on_final_transcription,
        )

        # Initialize LLM service with queues and API key
        self.chat = Chat(
            queues=self.queues,
            api_key=self.openrouter_api_key
        )
        await self.chat.initialize()

        # Wire up Chat UI streaming callbacks
        self.chat.on_character_response_chunk = self.on_character_response_chunk
        self.chat.on_character_response_full = self.on_character_response_full

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


    async def shutdown(self):
        """Shutdown all services gracefully"""
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
                # Clear conversation history
                if self.chat:
                    self.chat.clear_conversation_history()
                logger.info("Conversation history cleared")

            elif message_type == "interrupt":
                # Signal interrupt to stop current generation (LLM and TTS)
                # TODO: Implement interrupt handling
                if self.chat:
                    self.chat.interrupt_event.set()
                if self.tts:
                    # TODO: Add interrupt method to TTS class
                    pass
                logger.info("Interrupt signal sent to LLM and TTS")

            elif message_type == "refresh_active_characters":
                # Refresh active characters from database
                if self.chat:
                    await self.chat.load_active_characters_from_db()
                logger.info("Active characters refreshed from database")

            elif message_type == "ping":
                # Respond to ping with pong for connection health check
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
