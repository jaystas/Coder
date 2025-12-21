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


@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis"""
    text: str
    index: int
    session_id: str
    is_final: bool = False


@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming"""
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    is_final: bool = False

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

class StreamingQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(self):
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
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

        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_realtime_update': on_transcription_update,
            'on_realtime_stabilized': on_transcription_stabilized,
            'on_final_transcription': on_transcription_finished,
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
                    if self.callbacks.on_transcription_finished:
                        self.run_callback(self.callbacks.on_transcription_finished, user_message)

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
        self.is_listening = True
        logger.info("Started listening for audio")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False
        logger.info("Stopped listening for audio")

    def _on_transcription_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self.run_callback(self.callback.get('on_transcription_update'), text)

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
##--          LLM Processor         --##
########################################

class LLMStreamProcessor:
    """Processes LLM stream into sentences for TTS"""

    def __init__(self, queues: StreamingQueues, api_key: str):
        self.queues = queues
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    async def process_prompt(
        self,
        prompt: str,
        session_id: str,
        model: str = "meta-llama/llama-3.1-8b-instruct",
        on_text_chunk: Optional[callable] = None
    ) -> str:
        """
        Stream LLM response, extract sentences, queue for TTS.
        Returns full response text.
        """
        sentence_index = 0
        full_response = ""

        # Create streaming completion
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Keep responses concise and conversational."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        # Generator that extracts text chunks from OpenAI stream
        async def chunk_generator():
            nonlocal full_response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # Fire callback for UI streaming
                        if on_text_chunk:
                            await on_text_chunk(content)
                        yield content

        # Process chunks through stream2sentence
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
                tts_sentence = TTSSentence(
                    text=sentence_text,
                    index=sentence_index,
                    session_id=session_id,
                    is_final=False
                )
                await self.queues.sentence_queue.put(tts_sentence)
                logger.info(f"[LLM] Queued sentence {sentence_index}: {sentence_text[:50]}...")
                sentence_index += 1

        # Signal end of stream
        await self.queues.sentence_queue.put(TTSSentence(
            text="",
            index=sentence_index,
            session_id=session_id,
            is_final=True
        ))
        logger.info(f"[LLM] Stream complete, {sentence_index} sentences queued")

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

class HiggsTTSWorker:
    """TTS worker that synthesizes sentences using Higgs Audio"""

    def __init__(self, queues: StreamingQueues):
        self.queues = queues
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Audio settings
        self.sample_rate = 24000
        self._chunk_size = 10  # Audio tokens per chunk before decoding
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Crossfade settings (equal-power for smooth transitions)
        self._crossfade_duration = 0.04  # 40ms crossfade
        self._crossfade_samples = int(self._crossfade_duration * self.sample_rate)

        # Voice reference
        self.voice_dir = "backend/voices"
        self.voice_name = "lydia"

    async def initialize(self):
        """Initialize Higgs Audio engine"""
        logger.info(f"Initializing Higgs Audio TTS on {self._device}...")

        self.engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=self._device
        )

        logger.info("Higgs Audio TTS initialized")

    async def start(self):
        """Start the TTS worker task"""
        self.is_running = True
        self._task = asyncio.create_task(self.process_sentences())
        logger.info("TTS worker started")

    async def stop(self):
        """Stop the TTS worker"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TTS worker stopped")

    async def process_sentences(self):
        """Main processing loop - consumes from sentence_queue"""
        while self.is_running:
            try:
                sentence = await asyncio.wait_for(
                    self.queues.sentence_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Handle end-of-stream sentinel
            if sentence.is_final:
                await self.queues.audio_queue.put(AudioChunk(
                    audio_bytes=b"",
                    sentence_index=sentence.index,
                    chunk_index=0,
                    session_id=sentence.session_id,
                    is_final=True
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
                        is_final=False
                    )
                    await self.queues.audio_queue.put(audio_chunk)
                    chunk_index += 1
                logger.info(f"[TTS] Sentence {sentence.index} complete, {chunk_index} chunks")
            except Exception as e:
                # Skip on error, move to next sentence
                logger.error(f"[TTS] Error generating audio: {e}")
                continue

    def _load_voice_messages(self) -> list[Message]:
        """Load amelia voice reference for few-shot cloning"""
        audio_path = os.path.join(self.voice_dir, f"{self.voice_name}.wav")
        text_path = os.path.join(self.voice_dir, f"{self.voice_name}.txt")

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        return [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

    def _create_crossfade_curves(self) -> tuple[np.ndarray, np.ndarray]:
        """Create equal-power crossfade curves (sine/cosine) for smooth transitions"""
        t = np.linspace(0, 1, self._crossfade_samples, dtype=np.float32)
        # Equal-power: sin²(x) + cos²(x) = 1, maintains constant loudness
        fade_in = np.sin(t * np.pi / 2)
        fade_out = np.cos(t * np.pi / 2)
        return fade_in, fade_out

    def _apply_crossfade(
        self,
        waveform: np.ndarray,
        prev_tail: Optional[np.ndarray],
        fade_in: np.ndarray,
        fade_out: np.ndarray,
        is_final: bool = False
    ) -> tuple[bytes, Optional[np.ndarray]]:
        """
        Apply equal-power crossfade between chunks.
        Returns (pcm_bytes_to_send, new_tail_to_keep).
        """
        cf_samples = self._crossfade_samples

        if prev_tail is None:
            # First chunk: send everything except tail (save for next crossfade)
            if cf_samples > 0 and waveform.size > cf_samples and not is_final:
                to_send = waveform[:-cf_samples]
                new_tail = waveform[-cf_samples:]
            else:
                to_send = waveform
                new_tail = None
        else:
            # Subsequent chunks: crossfade with previous tail
            if cf_samples > 0 and waveform.size >= cf_samples:
                # Equal-power crossfade: overlap = prev * cos + curr * sin
                overlap = prev_tail * fade_out + waveform[:cf_samples] * fade_in

                if is_final:
                    # Final chunk: send overlap + rest, no new tail
                    rest = waveform[cf_samples:]
                    to_send = np.concatenate([overlap, rest]) if rest.size > 0 else overlap
                    new_tail = None
                elif waveform.size > 2 * cf_samples:
                    # Normal chunk: send overlap + middle, keep new tail
                    middle = waveform[cf_samples:-cf_samples]
                    to_send = np.concatenate([overlap, middle])
                    new_tail = waveform[-cf_samples:]
                else:
                    # Short chunk: just send overlap
                    to_send = overlap
                    new_tail = waveform[-cf_samples:] if waveform.size > cf_samples else None
            else:
                # Chunk too small for crossfade
                to_send = waveform
                new_tail = None

        # Convert to PCM16 bytes
        if to_send.size > 0:
            pcm = np.clip(to_send, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            return pcm16.tobytes(), new_tail
        return b"", new_tail

    async def generate_audio_for_sentence(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming with equal-power crossfade"""

        # Build messages with amelia voice reference
        messages = self._load_voice_messages()
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        # Crossfade state
        fade_in, fade_out = self._create_crossfade_curves()
        prev_tail: Optional[np.ndarray] = None

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
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

                        # Apply crossfade and yield
                        pcm_bytes, prev_tail = self._apply_crossfade(
                            waveform_np, prev_tail, fade_in, fade_out
                        )
                        if pcm_bytes:
                            yield pcm_bytes

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

                # Final chunk crossfade
                pcm_bytes, prev_tail = self._apply_crossfade(
                    waveform_np, prev_tail, fade_in, fade_out, is_final=True
                )
                if pcm_bytes:
                    yield pcm_bytes

            except Exception as e:
                logger.warning(f"Error flushing remaining audio: {e}")

        # Yield any remaining tail
        if prev_tail is not None and prev_tail.size > 0:
            pcm = np.clip(prev_tail, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            yield pcm16.tobytes()

########################################
##--         Audio Streamer         --##
########################################

class AudioStreamer:
    """Streams audio from queue to WebSocket clients"""

    def __init__(self, queues: StreamingQueues):
        self.queues = queues
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.websocket: Optional[WebSocket] = None
        # Event to signal when streaming is complete (is_final processed)
        self.stream_complete: asyncio.Event = asyncio.Event()

    async def start(self, websocket: WebSocket):
        """Start streaming to a WebSocket"""
        self.websocket = websocket
        self.is_running = True
        self.stream_complete.clear()  # Reset for new session
        self._task = asyncio.create_task(self._stream_loop())
        logger.info("Audio streamer started")

    async def stop(self):
        """Stop streaming"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Audio streamer stopped")

    async def wait_for_complete(self, timeout: float = 60.0):
        """Wait for the stream to complete (is_final received and sent)"""
        try:
            await asyncio.wait_for(self.stream_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[Stream] Timeout waiting for stream complete")

    async def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(
                    self.queues.audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if chunk.is_final:
                # Send end-of-stream signal
                await self.websocket.send_json({
                    "type": "audio_complete",
                    "session_id": chunk.session_id
                })
                logger.info(f"[Stream] Audio complete sent")
                # Signal that streaming is complete
                self.stream_complete.set()
                continue

            # Send audio data as binary
            await self.websocket.send_bytes(chunk.audio_bytes)

# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class StreamingPipeline:
    """Orchestrates the complete LLM → TTS → Audio pipeline"""

    def __init__(self, api_key: str):
        self.queues = StreamingQueues()
        self.llm_processor = LLMStreamProcessor(self.queues, api_key)
        self.tts_worker = HiggsTTSWorker(self.queues)
        self.audio_streamer = AudioStreamer(self.queues)
        self._initialized = False

    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        await self.tts_worker.initialize()
        self._initialized = True

    async def start_tts_worker(self):
        """Start the TTS background worker"""
        await self.tts_worker.start()

    async def stop_tts_worker(self):
        """Stop the TTS worker"""
        await self.tts_worker.stop()

    async def process_and_stream(
        self,
        prompt: str,
        websocket: WebSocket,
        session_id: str,
        on_text_chunk: Optional[callable] = None
    ) -> str:
        """
        Process a prompt and stream audio to WebSocket.

        Runs LLM processor and audio streamer concurrently:
        - LLM processor: extracts sentences → sentence_queue
        - TTS worker (background): sentence_queue → audio_queue
        - Audio streamer: audio_queue → WebSocket
        """
        # Start audio streaming to this websocket
        await self.audio_streamer.start(websocket)

        try:
            # Process LLM stream (populates sentence_queue)
            # TTS worker runs in background, consuming sentences
            # Audio streamer runs in background, sending to client
            full_response = await self.llm_processor.process_prompt(
                prompt=prompt,
                session_id=session_id,
                on_text_chunk=on_text_chunk
            )

            # Wait for audio streaming to complete (is_final processed)
            # This ensures all audio is sent before we return
            await self.audio_streamer.wait_for_complete(timeout=60.0)

            return full_response

        finally:
            await self.audio_streamer.stop()


########################################
##--      Conversation Pipeline     --##
########################################

class ConversationPipeline:
    """Orchestrates Complete LLM → TTS → Audio Pipeline"""

    def __init__(self):
        """"""



########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket and Routes Messages"""

    def __init__(self):

        self.transcribe = Transcribe(
            on_realtime_update=self.on_transcription_update,
            on_realtime_stabilized=self.on_transcription_stabilized,
            on_final_transcription=self.on_transcription_finished,
        )

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
            await self.websocket.send_text(json.dumps(data))
    
    async def stream_audio_to_client(self, audio_bytes: bytes):
        """Send binary audio to client (TTS)"""
        if self.websocket:
            await self.websocket.send_bytes(audio_bytes)

    async def on_transcription_update(self, text: str):
        await self.send_text_to_client({"type": "stt_update", "text": text})
    
    async def on_transcription_stabilized(self, text: str):
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})
    
    async def on_transcription_finished(self, user_message: str):
        await self.queues.transcribe_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_finished", "text": user_message})

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