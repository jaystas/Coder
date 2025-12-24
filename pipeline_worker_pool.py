"""
Concurrent LLM → TTS Streaming Pipeline

Streams text from AsyncOpenAI, extracts sentences via stream2sentence,
generates audio via Higgs TTS, and streams PCM to WebSocket clients.
"""

import os
import time
import asyncio
import logging
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, AsyncGenerator, Callable
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
# Pipeline Metrics
# ============================================================================

@dataclass
class PipelineMetrics:
    """Timing metrics for a single pipeline run"""
    session_id: str

    # Timestamps (monotonic clock)
    prompt_received_at: float = 0.0
    first_llm_token_at: float = 0.0
    first_sentence_complete_at: float = 0.0
    first_audio_chunk_at: float = 0.0
    first_audio_sent_at: float = 0.0
    stream_complete_at: float = 0.0

    # Counts
    total_sentences: int = 0
    total_audio_chunks: int = 0
    total_audio_bytes: int = 0

    # Computed latencies (ms) - populated on finalize
    ttft_llm_ms: float = 0.0      # Time to first LLM token
    ttfs_ms: float = 0.0          # Time to first sentence
    ttfa_ms: float = 0.0          # Time to first audio chunk generated
    ttfas_ms: float = 0.0         # Time to first audio sent to client
    total_duration_ms: float = 0.0

    def finalize(self):
        """Calculate derived latency metrics"""
        if self.first_llm_token_at and self.prompt_received_at:
            self.ttft_llm_ms = (self.first_llm_token_at - self.prompt_received_at) * 1000
        if self.first_sentence_complete_at and self.prompt_received_at:
            self.ttfs_ms = (self.first_sentence_complete_at - self.prompt_received_at) * 1000
        if self.first_audio_chunk_at and self.prompt_received_at:
            self.ttfa_ms = (self.first_audio_chunk_at - self.prompt_received_at) * 1000
        if self.first_audio_sent_at and self.prompt_received_at:
            self.ttfas_ms = (self.first_audio_sent_at - self.prompt_received_at) * 1000
        if self.stream_complete_at and self.prompt_received_at:
            self.total_duration_ms = (self.stream_complete_at - self.prompt_received_at) * 1000


class MetricsCollector:
    """Collects and reports pipeline metrics across sessions"""

    def __init__(self):
        self._sessions: dict[str, PipelineMetrics] = {}

    def start_session(self, session_id: str) -> PipelineMetrics:
        """Start tracking a new session"""
        metrics = PipelineMetrics(
            session_id=session_id,
            prompt_received_at=time.monotonic()
        )
        self._sessions[session_id] = metrics
        return metrics

    def get_session(self, session_id: str) -> Optional[PipelineMetrics]:
        """Get metrics for a session"""
        return self._sessions.get(session_id)

    def record_first_token(self, session_id: str):
        """Record when first LLM token arrives"""
        if m := self._sessions.get(session_id):
            if m.first_llm_token_at == 0:
                m.first_llm_token_at = time.monotonic()

    def record_first_sentence(self, session_id: str):
        """Record when first complete sentence is ready"""
        if m := self._sessions.get(session_id):
            if m.first_sentence_complete_at == 0:
                m.first_sentence_complete_at = time.monotonic()
            m.total_sentences += 1

    def record_first_audio(self, session_id: str):
        """Record when first audio chunk is generated"""
        if m := self._sessions.get(session_id):
            if m.first_audio_chunk_at == 0:
                m.first_audio_chunk_at = time.monotonic()
            m.total_audio_chunks += 1

    def record_audio_sent(self, session_id: str, num_bytes: int):
        """Record when audio is sent to client"""
        if m := self._sessions.get(session_id):
            if m.first_audio_sent_at == 0:
                m.first_audio_sent_at = time.monotonic()
            m.total_audio_bytes += num_bytes

    def finalize_session(self, session_id: str) -> Optional[PipelineMetrics]:
        """Finalize and log metrics for a session"""
        if m := self._sessions.pop(session_id, None):
            m.stream_complete_at = time.monotonic()
            m.finalize()
            logger.info(
                f"[Metrics] Session {session_id}: "
                f"TTFT={m.ttft_llm_ms:.0f}ms, "
                f"TTFS={m.ttfs_ms:.0f}ms, "
                f"TTFA={m.ttfa_ms:.0f}ms, "
                f"TTFAS={m.ttfas_ms:.0f}ms, "
                f"Total={m.total_duration_ms:.0f}ms, "
                f"Sentences={m.total_sentences}, "
                f"Chunks={m.total_audio_chunks}, "
                f"Bytes={m.total_audio_bytes}"
            )
            return m
        return None


# ============================================================================
# TTS Configuration
# ============================================================================

@dataclass
class TTSConfig:
    """Configuration for TTS synthesis"""
    # Worker pool settings
    num_workers: int = 3

    # Audio generation settings
    chunk_size: int = 10
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95

    # RAS (Repetition Avoidance Sampling)
    ras_win_len: int = 7
    ras_win_max_num_repeat: int = 2

    # Audio processing
    sample_rate: int = 24000
    crossfade_duration: float = 0.04  # 40ms

    # Voice reference
    voice_dir: str = "backend/voices"
    voice_name: str = "lydia"

    # Model paths
    model_name_or_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"


# ============================================================================
# Queue Manager
# ============================================================================

class StreamingQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(
        self,
        sentence_queue_size: int = 10,
        audio_queue_size: int = 50
    ):
        # Bounded queues provide backpressure if TTS can't keep up
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue(maxsize=sentence_queue_size)
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=audio_queue_size)


# ============================================================================
# LLM Stream Processor
# ============================================================================

class LLMStreamProcessor:
    """Processes LLM stream into sentences for TTS"""

    def __init__(
        self,
        queues: StreamingQueues,
        api_key: str,
        metrics: Optional[MetricsCollector] = None
    ):
        self.queues = queues
        self.metrics = metrics
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
                {"role": "system", "content": "You are Erica, a 25 year old actress from Australia. You're a total flirt and love to tease Jay (user). Use varied sentence structure, length, and burstiness. Keeping your replies relatively short, 3-4 sentences at most."},
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
                        # Record first LLM token timing
                        if self.metrics:
                            self.metrics.record_first_token(session_id)
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
                # Record sentence timing
                if self.metrics:
                    self.metrics.record_first_sentence(session_id)
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

class HiggsTTSWorker:
    """
    TTS worker that synthesizes sentences using Higgs Audio.

    Each worker owns its own HiggsAudioServeEngine with dedicated KV caches,
    enabling parallel synthesis across multiple workers.
    """

    def __init__(
        self,
        config: TTSConfig,
        worker_id: int = 0,
        metrics: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.worker_id = worker_id
        self.metrics = metrics
        self.engine: Optional[HiggsAudioServeEngine] = None

        # Derived settings
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._crossfade_samples = int(config.crossfade_duration * config.sample_rate)

        # Cached at initialization (avoid recomputing per-sentence)
        self._cached_voice_messages: Optional[list[Message]] = None
        self._fade_in: Optional[np.ndarray] = None
        self._fade_out: Optional[np.ndarray] = None

    async def initialize(self):
        """Initialize Higgs Audio engine and cache reusable data"""
        logger.info(f"[Worker {self.worker_id}] Initializing Higgs Audio TTS on {self._device}...")

        self.engine = HiggsAudioServeEngine(
            model_name_or_path=self.config.model_name_or_path,
            audio_tokenizer_name_or_path=self.config.audio_tokenizer_path,
            device=self._device
        )

        # Cache voice reference messages (avoid disk I/O per sentence)
        self._cached_voice_messages = self._load_voice_messages()
        logger.info(f"[Worker {self.worker_id}] Cached voice reference: {self.config.voice_name}")

        # Cache crossfade curves (they never change)
        self._fade_in, self._fade_out = self._create_crossfade_curves()
        logger.info(f"[Worker {self.worker_id}] Cached crossfade curves: {self._crossfade_samples} samples")

        logger.info(f"[Worker {self.worker_id}] Higgs Audio TTS initialized")

    async def synthesize_sentence(self, sentence: TTSSentence) -> list[AudioChunk]:
        """
        Synthesize audio for a single sentence.
        Returns list of AudioChunks (used by worker pool for resequencing).
        """
        logger.info(f"[Worker {self.worker_id}] Generating audio for sentence {sentence.index}")
        chunks: list[AudioChunk] = []
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
                chunks.append(audio_chunk)
                # Record first audio chunk timing
                if self.metrics:
                    self.metrics.record_first_audio(sentence.session_id)
                chunk_index += 1

            logger.info(f"[Worker {self.worker_id}] Sentence {sentence.index} complete, {chunk_index} chunks")

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Error generating audio for sentence {sentence.index}: {e}")

        return chunks

    def _load_voice_messages(self) -> list[Message]:
        """Load voice reference for few-shot cloning"""
        audio_path = os.path.join(self.config.voice_dir, f"{self.config.voice_name}.wav")
        text_path = os.path.join(self.config.voice_dir, f"{self.config.voice_name}.txt")

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

    async def _generate_audio(self, text: str) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming with equal-power crossfade"""

        # Build messages with cached voice reference (shallow copy, append new)
        messages = self._cached_voice_messages.copy()
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        # Use cached crossfade curves
        fade_in, fade_out = self._fade_in, self._fade_out
        prev_tail: Optional[np.ndarray] = None

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                force_audio_gen=True,
                ras_win_len=self.config.ras_win_len,
                ras_win_max_num_repeat=self.config.ras_win_max_num_repeat,
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
                if seq_len > 0 and seq_len % self.config.chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    try:
                        # Revert delay pattern and decode
                        vq_code = (
                            revert_delay_pattern(
                                audio_tensor,
                                start_idx=seq_len - self.config.chunk_size + 1
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
        if seq_len > 0 and seq_len % self.config.chunk_size != 0 and audio_tokens:
            audio_tensor = torch.cat(audio_tokens, dim=-1)
            remaining = seq_len % self.config.chunk_size

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


# ============================================================================
# TTS Worker Pool (Parallel Synthesis with Resequencing)
# ============================================================================

class TTSWorkerPool:
    """
    Manages multiple TTS workers for parallel sentence synthesis.

    Workers pull sentences from a shared queue and synthesize in parallel.
    A resequencing buffer ensures audio chunks are output in sentence order.
    """

    def __init__(
        self,
        config: TTSConfig,
        queues: StreamingQueues,
        metrics: Optional[MetricsCollector] = None
    ):
        self.config = config
        self.queues = queues
        self.metrics = metrics
        self.workers: list[HiggsTTSWorker] = []
        self.is_running = False
        self._tasks: list[asyncio.Task] = []

        # Resequencing state
        self._pending_results: dict[int, list[AudioChunk]] = {}
        self._next_output_index: int = 0
        self._resequence_lock = asyncio.Lock()
        self._final_sentence_index: int = -1

    async def initialize(self):
        """Initialize all TTS worker engines"""
        logger.info(f"Initializing TTS worker pool with {self.config.num_workers} workers...")

        for i in range(self.config.num_workers):
            worker = HiggsTTSWorker(
                config=self.config,
                worker_id=i,
                metrics=self.metrics
            )
            await worker.initialize()
            self.workers.append(worker)

        logger.info(f"TTS worker pool initialized: {self.config.num_workers} workers ready")

    async def start(self):
        """Start all worker tasks"""
        self.is_running = True
        self._next_output_index = 0
        self._pending_results.clear()
        self._final_sentence_index = -1

        for worker in self.workers:
            task = asyncio.create_task(self._worker_loop(worker))
            self._tasks.append(task)

        logger.info(f"TTS worker pool started: {len(self._tasks)} workers running")

    async def stop(self):
        """Stop all worker tasks"""
        self.is_running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()
        logger.info("TTS worker pool stopped")

    async def _worker_loop(self, worker: HiggsTTSWorker):
        """
        Worker loop: pull sentences from queue, synthesize, submit for resequencing.
        """
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
                # Record which sentence index is final
                async with self._resequence_lock:
                    self._final_sentence_index = sentence.index
                    # Check if we can flush final now
                    await self._try_flush_final(sentence.session_id)
                continue

            # Synthesize the sentence (this is the expensive parallel work)
            chunks = await worker.synthesize_sentence(sentence)

            # Submit to resequencer
            await self._submit_completed(sentence.index, chunks, sentence.session_id)

    async def _submit_completed(
        self,
        index: int,
        chunks: list[AudioChunk],
        session_id: str
    ):
        """Submit completed sentence chunks for resequencing and output"""
        async with self._resequence_lock:
            self._pending_results[index] = chunks

            # Flush all ready sentences in order
            while self._next_output_index in self._pending_results:
                ready_chunks = self._pending_results.pop(self._next_output_index)
                for chunk in ready_chunks:
                    await self.queues.audio_queue.put(chunk)
                self._next_output_index += 1

            # Check if we can flush final
            await self._try_flush_final(session_id)

    async def _try_flush_final(self, session_id: str):
        """
        Check if all sentences have been output and send final sentinel.
        Must be called with _resequence_lock held.
        """
        if (self._final_sentence_index >= 0 and
            self._next_output_index == self._final_sentence_index and
            len(self._pending_results) == 0):

            # All sentences output, send final sentinel
            await self.queues.audio_queue.put(AudioChunk(
                audio_data=b"",
                sentence_index=self._final_sentence_index,
                chunk_index=0,
                session_id=session_id,
                is_final=True
            ))
            logger.info(f"[Pool] All {self._final_sentence_index} sentences complete, final sentinel sent")


# ============================================================================
# Audio Streamer
# ============================================================================

class AudioStreamer:
    """Streams audio from queue to WebSocket clients"""

    def __init__(
        self,
        queues: StreamingQueues,
        metrics: Optional[MetricsCollector] = None
    ):
        self.queues = queues
        self.metrics = metrics
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.websocket: Optional[WebSocket] = None
        self._current_session_id: Optional[str] = None
        # Event to signal when streaming is complete (is_final processed)
        self.stream_complete: asyncio.Event = asyncio.Event()

    async def start(self, websocket: WebSocket, session_id: Optional[str] = None):
        """Start streaming to a WebSocket"""
        self.websocket = websocket
        self._current_session_id = session_id
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
            # Record audio sent timing
            if self.metrics:
                self.metrics.record_audio_sent(chunk.session_id, len(chunk.audio_data))


# ============================================================================
# Pipeline Orchestrator
# ============================================================================

class StreamingPipeline:
    """Orchestrates the complete LLM → TTS → Audio pipeline"""

    def __init__(
        self,
        api_key: str,
        tts_config: Optional[TTSConfig] = None
    ):
        self.tts_config = tts_config or TTSConfig()
        self.queues = StreamingQueues()
        self.metrics = MetricsCollector()
        self.llm_processor = LLMStreamProcessor(self.queues, api_key, metrics=self.metrics)
        self.tts_pool = TTSWorkerPool(
            config=self.tts_config,
            queues=self.queues,
            metrics=self.metrics
        )
        self.audio_streamer = AudioStreamer(self.queues, metrics=self.metrics)
        self._initialized = False

    async def initialize(self):
        """Initialize all components (TTS worker pool)"""
        if self._initialized:
            return
        await self.tts_pool.initialize()
        self._initialized = True

    async def start_tts_workers(self):
        """Start the TTS worker pool"""
        await self.tts_pool.start()

    async def stop_tts_workers(self):
        """Stop the TTS worker pool"""
        await self.tts_pool.stop()

    async def process_and_stream(
        self,
        prompt: str,
        websocket: WebSocket,
        session_id: str,
        on_text_chunk: Optional[Callable] = None
    ) -> tuple[str, Optional[PipelineMetrics]]:
        """
        Process a prompt and stream audio to WebSocket.

        Runs LLM processor and audio streamer concurrently:
        - LLM processor: extracts sentences → sentence_queue
        - TTS worker pool (parallel): sentence_queue → audio_queue (with resequencing)
        - Audio streamer: audio_queue → WebSocket

        Returns (full_response_text, metrics).
        """
        # Start metrics tracking for this session
        self.metrics.start_session(session_id)

        # Start audio streaming to this websocket
        await self.audio_streamer.start(websocket, session_id=session_id)

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

            # Finalize and log metrics
            session_metrics = self.metrics.finalize_session(session_id)

            return full_response, session_metrics

        finally:
            await self.audio_streamer.stop()
