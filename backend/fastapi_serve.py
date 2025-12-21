# Standard library
import os
import re
import json
import uuid
import asyncio
import logging
import threading
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Callable, Optional, Dict, List, AsyncGenerator, Awaitable

# Third-party
import torch
import numpy as np
import uvicorn
from openai import AsyncOpenAI
from supabase import create_client, Client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from backend.RealtimeSTT import AudioToTextRecorder
from backend.stream2sentence import generate_sentences_async
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent

# Database Director - centralized CRUD operations for Supabase
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

########################################
##--          Configuration         --##
########################################

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

########################################
##--          Data Classes          --##
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
    """LLM model configuration"""
    model: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0.0
    top_k: int = 40
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0


########################################
##--        Queue Management        --##
########################################

class StreamingQueues:
    """Manages asyncio queues for the streaming pipeline"""

    def __init__(self):
        # User speech → text (from STT or direct input)
        self.transcribe_queue: asyncio.Queue[str] = asyncio.Queue()
        # LLM sentences → TTS (with character context)
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
        # TTS audio → WebSocket
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

    def clear(self):
        """Clear all queues"""
        while not self.transcribe_queue.empty():
            try:
                self.transcribe_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break


########################################
##--  Speech to Text Transcription  --##
########################################

Callback = Callable[..., Optional[Awaitable[None]]]


class Transcriber:
    """Realtime transcription of user's audio using RealtimeSTT"""

    def __init__(
        self,
        on_transcription_update: Optional[Callback] = None,
        on_transcription_stabilized: Optional[Callback] = None,
        on_transcription_finished: Optional[Callback] = None,
        on_vad_detect_start: Optional[Callback] = None,
        on_vad_detect_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,
    ):
        # Event loop for async callback dispatching
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.is_listening = False

        # Callback registry
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_transcription_update': on_transcription_update,
            'on_transcription_stabilized': on_transcription_stabilized,
            'on_transcription_finished': on_transcription_finished,
            'on_vad_detect_start': on_vad_detect_start,
            'on_vad_detect_stop': on_vad_detect_stop,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
        }

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
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.5,
            spinner=False,
            level=logging.WARNING,
            use_microphone=False
        )

    def _run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run callback from RealtimeSTT background thread"""
        if callback is None or self.loop is None:
            return
        if asyncio.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)
        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def _transcription_loop(self):
        """Background thread for transcription"""
        while self.is_listening:
            try:
                text = self.recorder.text()
                if text and text.strip():
                    callback = self.callbacks.get('on_transcription_finished')
                    if callback:
                        self._run_callback(callback, text)
            except Exception as e:
                logger.error(f"Transcription error: {e}")

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio: {e}")

    def start_listening(self):
        """Start listening for audio input"""
        if self.is_listening:
            return
        self.is_listening = True
        self._thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self._thread.start()
        logger.info("Transcriber started listening")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Transcriber stopped listening")

    # RealtimeSTT callbacks
    def _on_transcription_update(self, text: str) -> None:
        self._run_callback(self.callbacks.get('on_transcription_update'), text)

    def _on_transcription_stabilized(self, text: str) -> None:
        self._run_callback(self.callbacks.get('on_transcription_stabilized'), text)

    def _on_vad_detect_start(self) -> None:
        self._run_callback(self.callbacks.get('on_vad_detect_start'))

    def _on_vad_detect_stop(self) -> None:
        self._run_callback(self.callbacks.get('on_vad_detect_stop'))

    def _on_recording_start(self) -> None:
        self._run_callback(self.callbacks.get('on_recording_start'))

    def _on_recording_stop(self) -> None:
        self._run_callback(self.callbacks.get('on_recording_stop'))


########################################
##--         LLM Processor          --##
########################################

class LLMProcessor:
    """
    Streams LLM responses from OpenRouter and extracts sentences for TTS.
    Queues TTSSentence objects with character context.
    """

    def __init__(self, api_key: str):
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
                stream=True
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
##--          TTS Worker            --##
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
    """
    TTS worker using Higgs Audio. Singleton - initialized once at app startup.
    Consumes from sentence_queue, outputs to audio_queue.
    Caches voice references to avoid reloading per sentence.
    """

    def __init__(self):
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._initialized = False

        # Voice cache: voice_id → loaded voice messages
        self._voice_cache: Dict[str, List[Message]] = {}

        # Audio settings
        self.sample_rate = 24000
        self._chunk_size = 20
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    async def initialize(self):
        """Initialize Higgs Audio engine (call once at app startup)"""
        if self._initialized:
            return

        logger.info(f"[TTS] Initializing Higgs Audio on {self._device}...")
        self.engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=self._device
        )
        self._initialized = True
        logger.info("[TTS] Higgs Audio initialized")

    async def start(
        self,
        sentence_queue: asyncio.Queue[TTSSentence],
        audio_queue: asyncio.Queue[AudioChunk]
    ):
        """Start the TTS worker background task"""
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(
            self._process_loop(sentence_queue, audio_queue)
        )
        logger.info("[TTS] Worker started")

    async def stop(self):
        """Stop the TTS worker"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[TTS] Worker stopped")

    def _load_voice(self, voice_id: str) -> List[Message]:
        """Load voice reference messages with caching"""
        if voice_id in self._voice_cache:
            return self._voice_cache[voice_id]

        # Get voice from database
        voice = db.voices.get_sync(voice_id) if hasattr(db.voices, 'get_sync') else None

        if not voice:
            # Try to load from default voice directory
            voice_dir = "backend/voices"
            audio_path = os.path.join(voice_dir, f"{voice_id}.wav")
            text_path = os.path.join(voice_dir, f"{voice_id}.txt")

            if not os.path.exists(audio_path) or not os.path.exists(text_path):
                logger.warning(f"[TTS] Voice '{voice_id}' not found, using default")
                voice_id = "amelia"
                audio_path = os.path.join(voice_dir, "amelia.wav")
                text_path = os.path.join(voice_dir, "amelia.txt")

            with open(text_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()

            voice_messages = [
                Message(role="user", content=ref_text),
                Message(role="assistant", content=AudioContent(audio_url=audio_path))
            ]
        else:
            # Load from database paths
            with open(voice.text_path, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()

            voice_messages = [
                Message(role="user", content=ref_text),
                Message(role="assistant", content=AudioContent(audio_url=voice.audio_path))
            ]

        self._voice_cache[voice_id] = voice_messages
        logger.info(f"[TTS] Cached voice: {voice_id}")

        return voice_messages

    async def _process_loop(
        self,
        sentence_queue: asyncio.Queue[TTSSentence],
        audio_queue: asyncio.Queue[AudioChunk]
    ):
        """Main processing loop - consumes sentences, produces audio"""
        while self.is_running:
            try:
                sentence = await asyncio.wait_for(
                    sentence_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Pass through sentinels
            if sentence.is_final:
                await audio_queue.put(AudioChunk(
                    audio_bytes=b"",
                    sentence_index=sentence.index,
                    chunk_index=0,
                    session_id=sentence.session_id,
                    character_id=sentence.character_id,
                    is_final=True,
                    is_session_complete=sentence.is_session_complete
                ))
                logger.info(f"[TTS] {sentence.character_name} audio sentinel passed")
                continue

            # Generate audio for this sentence
            logger.info(f"[TTS] Generating: {sentence.character_name} #{sentence.index}")
            chunk_index = 0
            try:
                async for pcm_bytes in self._generate_audio(sentence):
                    await audio_queue.put(AudioChunk(
                        audio_bytes=pcm_bytes,
                        sentence_index=sentence.index,
                        chunk_index=chunk_index,
                        session_id=sentence.session_id,
                        character_id=sentence.character_id,
                        is_final=False,
                        is_session_complete=False
                    ))
                    chunk_index += 1
                logger.info(f"[TTS] {sentence.character_name} #{sentence.index}: {chunk_index} chunks")
            except Exception as e:
                logger.error(f"[TTS] Error generating audio: {e}")
                continue

    async def _generate_audio(self, sentence: TTSSentence) -> AsyncGenerator[bytes, None]:
        """Generate audio for a sentence using character's voice"""
        voice_messages = self._load_voice(sentence.voice_id)

        messages = voice_messages + [
            Message(role="user", content=sentence.text)
        ]

        chat_sample = ChatMLSample(messages=messages)

        audio_tokens: List[torch.Tensor] = []
        seq_len = 0

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

                audio_tokens.append(delta.audio_tokens[:, None])

                # Count non-padding tokens
                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1

                # Decode when chunk size reached
                if seq_len > 0 and seq_len % self._chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    try:
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

                        if isinstance(waveform, torch.Tensor):
                            waveform_np = waveform.detach().cpu().numpy()
                        else:
                            waveform_np = np.asarray(waveform, dtype=np.float32)

                        pcm = np.clip(waveform_np, -1.0, 1.0)
                        pcm16 = (pcm * 32767.0).astype(np.int16)
                        yield pcm16.tobytes()

                    except Exception as e:
                        logger.warning(f"[TTS] Chunk decode error: {e}")
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
                logger.warning(f"[TTS] Flush error: {e}")


########################################
##--        Audio Streamer          --##
########################################

class AudioStreamer:
    """Streams audio chunks from queue to WebSocket client"""

    def __init__(self):
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.websocket: Optional[WebSocket] = None
        self.session_complete: asyncio.Event = asyncio.Event()

    async def start(
        self,
        websocket: WebSocket,
        audio_queue: asyncio.Queue[AudioChunk]
    ):
        """Start streaming to WebSocket"""
        self.websocket = websocket
        self.is_running = True
        self.session_complete.clear()
        self._task = asyncio.create_task(self._stream_loop(audio_queue))
        logger.info("[Stream] Audio streamer started")

    async def stop(self):
        """Stop streaming"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[Stream] Audio streamer stopped")

    async def wait_for_session_complete(self, timeout: float = 120.0):
        """Wait for all audio to finish streaming"""
        try:
            await asyncio.wait_for(self.session_complete.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("[Stream] Timeout waiting for session complete")

    async def _stream_loop(self, audio_queue: asyncio.Queue[AudioChunk]):
        """Main streaming loop"""
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(
                    audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                # Handle character-level completion
                if chunk.is_final:
                    await self.websocket.send_json({
                        "type": "character_audio_complete",
                        "character_id": chunk.character_id,
                        "session_id": chunk.session_id
                    })

                    # Check if session is complete
                    if chunk.is_session_complete:
                        await self.websocket.send_json({
                            "type": "session_audio_complete",
                            "session_id": chunk.session_id
                        })
                        self.session_complete.set()
                        logger.info("[Stream] Session audio complete")
                    continue

                # Send audio bytes
                await self.websocket.send_bytes(chunk.audio_bytes)

            except Exception as e:
                logger.error(f"[Stream] Error sending: {e}")
                break


########################################
##--     Conversation Pipeline      --##
########################################

class ConversationPipeline:
    """
    Central orchestrator for the complete conversation flow.
    Manages: STT → Character Selection → LLM → TTS → Audio Stream
    """

    def __init__(self, api_key: str, tts_worker: TTSWorker):
        # Queues (owned by this pipeline instance)
        self.queues = StreamingQueues()

        # Components
        self.llm = LLMProcessor(api_key)
        self.tts_worker = tts_worker  # Shared singleton
        self.audio_streamer = AudioStreamer()

        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.active_characters: List[Character] = []
        self.model_settings: ModelSettings = ModelSettings()

        # Session
        self.websocket: Optional[WebSocket] = None
        self.session_id: str = ""
        self._conversation_task: Optional[asyncio.Task] = None
        self._is_processing: bool = False

    async def start_session(self, websocket: WebSocket):
        """Start a new conversation session"""
        self.websocket = websocket
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.queues.clear()

        # Start audio streamer for this session
        await self.audio_streamer.start(websocket, self.queues.audio_queue)

        # Start TTS worker with our queues
        await self.tts_worker.start(
            self.queues.sentence_queue,
            self.queues.audio_queue
        )

        # Start main conversation loop
        self._conversation_task = asyncio.create_task(self._conversation_loop())
        logger.info(f"[Pipeline] Session started: {self.session_id}")

    async def stop_session(self):
        """Stop the current session"""
        if self._conversation_task:
            self._conversation_task.cancel()
            try:
                await self._conversation_task
            except asyncio.CancelledError:
                pass

        await self.audio_streamer.stop()
        # Note: TTS worker keeps running (singleton)
        logger.info(f"[Pipeline] Session stopped: {self.session_id}")

    async def _conversation_loop(self):
        """Main loop: consume transcriptions, orchestrate responses"""
        while True:
            try:
                user_message = await asyncio.wait_for(
                    self.queues.transcribe_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if not self._is_processing:
                await self._handle_user_message(user_message)

    async def _handle_user_message(self, user_message: str):
        """
        Process user message through the full multi-character pipeline.

        Flow:
        1. Add user message to history
        2. Determine responding characters
        3. For each character (sequential LLM, concurrent audio):
           - Build messages with character's system prompt
           - Stream LLM response → sentence_queue
           - Once LLM complete, add to history (don't wait for audio)
        4. After last character, wait for all audio to complete
        """
        self._is_processing = True
        self.audio_streamer.session_complete.clear()

        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Notify client
            await self._send_to_client({
                "type": "user_message_received",
                "text": user_message,
                "session_id": self.session_id
            })

            # Determine which characters respond
            responding_characters = self._parse_character_mentions(user_message)

            if not responding_characters:
                logger.warning("[Pipeline] No characters to respond")
                self._is_processing = False
                return

            logger.info(f"[Pipeline] Characters responding: {[c.name for c in responding_characters]}")

            # Process each character's response
            for i, character in enumerate(responding_characters):
                is_last = (i == len(responding_characters) - 1)
                await self._generate_character_response(
                    character=character,
                    is_last_character=is_last
                )

            # Wait for all audio to finish streaming
            await self.audio_streamer.wait_for_session_complete(timeout=120.0)

        except Exception as e:
            logger.error(f"[Pipeline] Error handling message: {e}", exc_info=True)
        finally:
            self._is_processing = False

    async def _generate_character_response(
        self,
        character: Character,
        is_last_character: bool
    ):
        """Generate response for a single character"""

        # Notify client that character is starting
        await self._send_to_client({
            "type": "character_response_start",
            "character_id": character.id,
            "character_name": character.name
        })

        # Build messages for this character
        messages = self._build_messages_for_character(character)

        # Stream LLM response (queues sentences for TTS)
        full_response = await self.llm.stream_character_response(
            messages=messages,
            character=character,
            session_id=self.session_id,
            model_settings=self.model_settings,
            sentence_queue=self.queues.sentence_queue,
            on_text_chunk=lambda chunk: self._send_to_client({
                "type": "llm_chunk",
                "character_id": character.id,
                "character_name": character.name,
                "text": chunk
            }),
            is_last_character=is_last_character
        )

        # Add character's response to history with tags
        if full_response:
            tagged_response = f"<{character.name}>{full_response}</{character.name}>"
            self.conversation_history.append({
                "role": "assistant",
                "content": tagged_response
            })

        # Notify client that LLM stream is complete
        await self._send_to_client({
            "type": "character_llm_complete",
            "character_id": character.id,
            "character_name": character.name,
            "text": full_response
        })

    def _build_messages_for_character(self, character: Character) -> List[Dict[str, str]]:
        """
        Build the message list for OpenRouter API call.

        Structure:
        1. Character's system prompt
        2. Conversation history
        3. Instruction to respond as this character
        """
        messages = []

        # Character's system prompt
        if character.system_prompt:
            messages.append({
                "role": "system",
                "content": character.system_prompt
            })

        # Conversation history
        messages.extend(self.conversation_history)

        # Instruction for this character
        messages.append({
            "role": "system",
            "content": f"Respond as {character.name}. Provide only {character.name}'s reply. Do not respond as anyone else."
        })

        return messages

    def _parse_character_mentions(self, message: str) -> List[Character]:
        """Parse message for character mentions, return in order mentioned"""
        if not self.active_characters:
            return []

        mentioned = []
        seen_ids = set()
        mentions_with_position = []

        for character in self.active_characters:
            for name_part in character.name.lower().split():
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    if character.id not in seen_ids:
                        mentions_with_position.append({
                            'character': character,
                            'position': match.start()
                        })
                        seen_ids.add(character.id)
                    break

        # Sort by position in message
        mentions_with_position.sort(key=lambda x: x['position'])
        mentioned = [m['character'] for m in mentions_with_position]

        # If no mentions, default to all active characters (sorted by name)
        if not mentioned:
            mentioned = sorted(self.active_characters, key=lambda c: c.name)

        return mentioned

    def set_active_characters(self, characters: List[Character]):
        """Set the list of active characters"""
        self.active_characters = characters
        logger.info(f"[Pipeline] Active characters: {[c.name for c in characters]}")

    def set_model_settings(self, settings: ModelSettings):
        """Update model settings"""
        self.model_settings = settings
        logger.info(f"[Pipeline] Model settings updated: {settings.model}")

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("[Pipeline] Conversation history cleared")

    async def _send_to_client(self, data: dict):
        """Send JSON message to WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"[Pipeline] Failed to send: {e}")


########################################
##--       WebSocket Manager        --##
########################################

class WebSocketManager:
    """
    Thin routing layer for WebSocket connections.
    Delegates conversation logic to ConversationPipeline.
    """

    def __init__(self, pipeline: ConversationPipeline):
        self.pipeline = pipeline
        self.websocket: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Transcription handler
        self.transcriber = Transcriber(
            on_transcription_update=self._on_stt_update,
            on_transcription_stabilized=self._on_stt_stabilized,
            on_transcription_finished=self._on_stt_finished,
        )

    async def initialize(self):
        """Initialize on app startup"""
        self.loop = asyncio.get_running_loop()
        self.transcriber.loop = self.loop
        logger.info("[WS] WebSocketManager initialized")

    async def shutdown(self):
        """Cleanup on app shutdown"""
        self.transcriber.stop_listening()
        logger.info("[WS] WebSocketManager shut down")

    async def connect(self, websocket: WebSocket):
        """Handle new WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        await self.pipeline.start_session(websocket)
        logger.info("[WS] Client connected")

    async def disconnect(self):
        """Handle WebSocket disconnect"""
        self.transcriber.stop_listening()
        await self.pipeline.stop_session()
        self.websocket = None
        logger.info("[WS] Client disconnected")

    async def handle_text_message(self, message: str):
        """Route incoming text messages"""
        try:
            data = json.loads(message)
            msg_type = data.get("type", "")
            payload = data.get("data", {})

            if msg_type == "user_message":
                text = payload.get("text", "")
                if text:
                    await self.pipeline.queues.transcribe_queue.put(text)

            elif msg_type == "start_listening":
                self.transcriber.start_listening()

            elif msg_type == "stop_listening":
                self.transcriber.stop_listening()

            elif msg_type == "set_active_characters":
                character_ids = payload.get("character_ids", [])
                characters = await self._load_characters(character_ids)
                self.pipeline.set_active_characters(characters)

            elif msg_type == "model_settings":
                settings = ModelSettings(
                    model=payload.get("model", "meta-llama/llama-3.1-8b-instruct"),
                    temperature=float(payload.get("temperature", 0.7)),
                    top_p=float(payload.get("top_p", 0.9)),
                    min_p=float(payload.get("min_p", 0.0)),
                    top_k=int(payload.get("top_k", 40)),
                    frequency_penalty=float(payload.get("frequency_penalty", 0.0)),
                    presence_penalty=float(payload.get("presence_penalty", 0.0)),
                    repetition_penalty=float(payload.get("repetition_penalty", 1.0))
                )
                self.pipeline.set_model_settings(settings)

            elif msg_type == "clear_history":
                self.pipeline.clear_history()

        except json.JSONDecodeError as e:
            logger.error(f"[WS] Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"[WS] Error handling message: {e}", exc_info=True)

    def handle_audio_message(self, audio_bytes: bytes):
        """Feed audio to transcriber"""
        self.transcriber.feed_audio(audio_bytes)

    async def _on_stt_update(self, text: str):
        """STT realtime update"""
        await self._send_to_client({"type": "stt_update", "text": text})

    async def _on_stt_stabilized(self, text: str):
        """STT stabilized"""
        await self._send_to_client({"type": "stt_stabilized", "text": text})

    async def _on_stt_finished(self, text: str):
        """STT complete - route to pipeline"""
        await self.pipeline.queues.transcribe_queue.put(text)
        await self._send_to_client({"type": "stt_finished", "text": text})

    async def _load_characters(self, character_ids: List[str]) -> List[Character]:
        """Load characters from database"""
        characters = []
        for cid in character_ids:
            try:
                char = db.characters.get(cid)
                if char:
                    characters.append(char)
            except Exception as e:
                logger.error(f"[WS] Failed to load character {cid}: {e}")
        return characters

    async def _send_to_client(self, data: dict):
        """Send JSON to client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"[WS] Failed to send: {e}")


########################################
##--           FastAPI App          --##
########################################

# Singleton TTS worker (initialized once, shared across sessions)
tts_worker = TTSWorker()

# Pipeline and manager
pipeline = ConversationPipeline(api_key=OPENROUTER_API_KEY, tts_worker=tts_worker)
ws_manager = WebSocketManager(pipeline=pipeline)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan: startup and shutdown"""
    logger.info("Starting up services...")

    # Initialize TTS worker (load Higgs model)
    await tts_worker.initialize()

    # Initialize WebSocket manager
    await ws_manager.initialize()

    logger.info("All services initialized!")
    yield

    logger.info("Shutting down services...")
    await ws_manager.shutdown()
    await tts_worker.stop()
    logger.info("All services shut down!")


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
    """Main WebSocket endpoint for conversation"""
    await ws_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                await ws_manager.handle_text_message(message["text"])

            elif "bytes" in message:
                ws_manager.handle_audio_message(message["bytes"])

    except WebSocketDisconnect:
        await ws_manager.disconnect()

    except Exception as e:
        logger.error(f"[WS] WebSocket error: {e}")
        await ws_manager.disconnect()


########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
