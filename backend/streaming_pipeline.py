"""
Concurrent LLM → TTS Streaming Pipeline

Streams text from AsyncOpenAI, extracts sentences via stream2sentence,
generates audio via Higgs TTS, and streams PCM to WebSocket clients.
"""

import os
import asyncio
import logging
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, AsyncGenerator
from openai import AsyncOpenAI
from fastapi import WebSocket

from backend.stream2sentence import generate_sentences_async
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes
# ============================================================================

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
    audio_data: bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    is_final: bool = False


# ============================================================================
# Queue Manager
# ============================================================================

class StreamingQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(self):
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()


# ============================================================================
# LLM Stream Processor
# ============================================================================

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


# ============================================================================
# Higgs TTS Worker
# ============================================================================

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
        self._chunk_size = 20  # Audio tokens per chunk before decoding
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Voice reference
        self.voice_dir = "backend/voices"
        self.voice_name = "amelia"

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
        self._task = asyncio.create_task(self._process_loop())
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

    async def _process_loop(self):
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
                    audio_data=b"",
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
                async for pcm_bytes in self._generate_audio(sentence.text):
                    audio_chunk = AudioChunk(
                        audio_data=pcm_bytes,
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

    async def _generate_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming"""

        # Build messages with amelia voice reference
        messages = self._load_voice_messages()
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
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


# ============================================================================
# Audio Streamer
# ============================================================================

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
            await self.websocket.send_bytes(chunk.audio_data)


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
