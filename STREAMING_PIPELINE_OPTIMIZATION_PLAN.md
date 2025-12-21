# Streaming Pipeline Optimization Plan

## Overview

This document outlines the implementation plan for optimizing `backend/streaming_pipeline.py` based on our analysis. The optimizations target latency reduction, throughput improvement, and observability.

---

## Table of Contents

1. [Architecture Changes](#1-architecture-changes)
2. [Implementation Details](#2-implementation-details)
3. [Task Breakdown](#3-task-breakdown)
4. [Risk Assessment](#4-risk-assessment)

---

## 1. Architecture Changes

### Current Architecture

```
LLM Stream ──► [sentence_queue] ──► TTS Worker (1) ──► [audio_queue] ──► WebSocket
                                         │
                                    (sequential)
```

**Problem:** Single TTS worker processes sentences sequentially. While sentence N streams audio, sentence N+1 waits in queue.

### Proposed Architecture

```
                                    ┌─────────────────┐
LLM Stream ──────────────────────►  │  Sentence Queue │
                                    │   (bounded=10)  │
                                    └────────┬────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                        ▼
             ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
             │ TTS Engine  │          │ TTS Engine  │          │ TTS Engine  │
             │   Worker 0  │          │   Worker 1  │          │   Worker 2  │
             │ (dedicated  │          │ (dedicated  │          │ (dedicated  │
             │  KV cache)  │          │  KV cache)  │          │  KV cache)  │
             └──────┬──────┘          └──────┬──────┘          └──────┬──────┘
                    │                        │                        │
                    └────────────────────────┼────────────────────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │   Resequencing      │
                                  │      Buffer         │
                                  │ (maintains order)   │
                                  └──────────┬──────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │   Audio Queue       │
                                  │   (bounded=50)      │
                                  └──────────┬──────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │  WebSocket Streamer │
                                  └─────────────────────┘
```

### Key Insight: Why Multiple Engines?

Looking at `serve_engine.py:239-248`, the `HiggsAudioServeEngine` uses **static KV caches**:

```python
self.kv_caches = {
    length: StaticCache(...)
    for length in sorted(kv_cache_lengths)
}
```

And `generate_delta_stream` calls `self._prepare_kv_caches()` which resets these caches. This means:
- A single engine instance cannot process multiple sentences concurrently
- Each parallel worker needs its own engine instance with dedicated KV caches

**Trade-off:** Multiple engines = more GPU memory, but true parallelism for TTS synthesis.

### Alternative: Prefetch-and-Buffer (Single Engine)

If GPU memory is constrained, we can use a single engine with intelligent prefetching:

```
Sentence Queue ──► Prefetch Scheduler ──► TTS Engine (1) ──► Completion Buffer
                         │                                          │
                    (looks ahead,                            (holds completed
                     prioritizes)                             sentences until
                                                              order is ready)
```

This synthesizes sentences eagerly but sequentially, buffering ahead while audio streams.

**Recommendation:** Start with 2 workers (configurable), measure GPU memory, adjust.

---

## 2. Implementation Details

### 2.1 Quick Wins (Low-Risk, High-Value)

#### 2.1.1 Cache Voice Reference Messages

**Current:** `_load_voice_messages()` reads from disk every sentence (lines 257-268, 336)

**Change:**
```python
class HiggsTTSWorker:
    async def initialize(self):
        self.engine = HiggsAudioServeEngine(...)
        # Cache voice messages at init
        self._cached_voice_messages = self._load_voice_messages()

    async def _generate_audio(self, text: str):
        # Shallow copy cached list, append new message
        messages = self._cached_voice_messages.copy()
        messages.append(Message(role="user", content=text))
```

**Impact:** Eliminates disk I/O per sentence.

#### 2.1.2 Cache Crossfade Curves

**Current:** `_create_crossfade_curves()` called every sentence (line 346)

**Change:**
```python
class HiggsTTSWorker:
    async def initialize(self):
        # Pre-compute crossfade curves (they never change)
        self._fade_in, self._fade_out = self._create_crossfade_curves()
```

**Impact:** Minor CPU savings, cleaner code.

#### 2.1.3 Remove Duplicate `revert_delay_pattern`

**Current:** Function imported (line 21) AND defined locally (lines 148-159)

**Change:** Delete lines 148-159, use the imported version.

#### 2.1.4 Add RAS Parameters

**Current:** `generate_delta_stream` call (line 350-356) missing RAS params

**Change:**
```python
async for delta in self.engine.generate_delta_stream(
    chat_ml_sample=chat_sample,
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    force_audio_gen=True,
    ras_win_len=7,                    # ADD: Repetition avoidance
    ras_win_max_num_repeat=2,         # ADD: Max repeats in window
):
```

**Impact:** Prevents audio token repetition artifacts.

---

### 2.2 Parallel TTS Workers with Resequencing

#### Design

```python
@dataclass
class TTSConfig:
    """Configuration for TTS synthesis"""
    num_workers: int = 2
    chunk_size: int = 10
    crossfade_duration: float = 0.04
    max_tokens: int = 1024
    ras_win_len: int = 7
    ras_win_max_num_repeat: int = 2
    voice_name: str = "lydia"


class TTSWorkerPool:
    """Manages multiple TTS workers with ordered output"""

    def __init__(self, config: TTSConfig, queues: StreamingQueues):
        self.config = config
        self.queues = queues
        self.workers: list[HiggsTTSWorker] = []

        # Resequencing state
        self._pending_results: dict[int, list[AudioChunk]] = {}
        self._next_output_index: int = 0
        self._resequence_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize N independent TTS engine instances"""
        for i in range(self.config.num_workers):
            worker = HiggsTTSWorker(
                worker_id=i,
                config=self.config,
                on_chunk_ready=self._handle_chunk
            )
            await worker.initialize()
            self.workers.append(worker)

    async def start(self):
        """Start all workers consuming from sentence_queue"""
        self._next_output_index = 0
        self._pending_results.clear()

        for worker in self.workers:
            asyncio.create_task(self._worker_loop(worker))

    async def _worker_loop(self, worker: HiggsTTSWorker):
        """Each worker pulls from shared sentence_queue"""
        while self.is_running:
            sentence = await self.queues.sentence_queue.get()

            if sentence.is_final:
                # Put back for other workers to see, then handle
                await self.queues.sentence_queue.put(sentence)
                await self._handle_final(sentence)
                break

            # Generate audio (worker has dedicated engine)
            chunks = await worker.generate_audio(sentence)

            # Submit to resequencer
            await self._submit_completed(sentence.index, chunks)

    async def _submit_completed(self, index: int, chunks: list[AudioChunk]):
        """Resequence completed sentences to maintain order"""
        async with self._resequence_lock:
            self._pending_results[index] = chunks

            # Flush all ready sentences in order
            while self._next_output_index in self._pending_results:
                ready_chunks = self._pending_results.pop(self._next_output_index)
                for chunk in ready_chunks:
                    await self.queues.audio_queue.put(chunk)
                self._next_output_index += 1
```

#### Why This Works

1. **Parallel Synthesis:** Workers 0, 1, 2 can synthesize sentences 0, 1, 2 simultaneously
2. **Ordered Output:** Resequencing buffer holds sentence 2's audio until sentences 0, 1 complete
3. **Backpressure:** Bounded queues prevent runaway memory growth
4. **Independent Engines:** Each worker owns a `HiggsAudioServeEngine` with dedicated KV caches

#### Memory Consideration

Each `HiggsAudioServeEngine` loads:
- Model weights (shared via PyTorch memory if same GPU)
- KV caches (per-engine, ~100-500MB each depending on cache lengths)
- Audio tokenizer (can potentially be shared)

**Mitigation:** Make `num_workers` configurable, default to 2.

---

### 2.3 Metrics & Instrumentation

#### Metrics Data Structure

```python
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

    # Computed latencies (populated on finalize)
    ttft_llm_ms: float = 0.0          # Time to first LLM token
    ttfs_ms: float = 0.0               # Time to first sentence
    ttfa_ms: float = 0.0               # Time to first audio chunk generated
    ttfas_ms: float = 0.0              # Time to first audio sent to client
    total_duration_ms: float = 0.0

    def finalize(self):
        """Calculate derived metrics"""
        if self.first_llm_token_at:
            self.ttft_llm_ms = (self.first_llm_token_at - self.prompt_received_at) * 1000
        if self.first_sentence_complete_at:
            self.ttfs_ms = (self.first_sentence_complete_at - self.prompt_received_at) * 1000
        if self.first_audio_chunk_at:
            self.ttfa_ms = (self.first_audio_chunk_at - self.prompt_received_at) * 1000
        if self.first_audio_sent_at:
            self.ttfas_ms = (self.first_audio_sent_at - self.prompt_received_at) * 1000
        if self.stream_complete_at:
            self.total_duration_ms = (self.stream_complete_at - self.prompt_received_at) * 1000


class MetricsCollector:
    """Collects and reports pipeline metrics"""

    def __init__(self):
        self._sessions: dict[str, PipelineMetrics] = {}

    def start_session(self, session_id: str) -> PipelineMetrics:
        metrics = PipelineMetrics(
            session_id=session_id,
            prompt_received_at=time.monotonic()
        )
        self._sessions[session_id] = metrics
        return metrics

    def record_first_token(self, session_id: str):
        if m := self._sessions.get(session_id):
            if m.first_llm_token_at == 0:
                m.first_llm_token_at = time.monotonic()

    # ... similar methods for other events

    def finalize_session(self, session_id: str) -> PipelineMetrics:
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
                f"Chunks={m.total_audio_chunks}"
            )
            return m
        return None
```

#### Integration Points

```python
# In LLMStreamProcessor.process_prompt():
async for chunk in stream:
    if content:
        self.metrics.record_first_token(session_id)  # First token timing
        ...

async for sentence in generate_sentences_async(...):
    self.metrics.record_first_sentence(session_id)  # First sentence timing
    ...

# In HiggsTTSWorker._generate_audio():
if pcm_bytes:
    self.metrics.record_first_audio(session_id)  # First audio chunk timing
    yield pcm_bytes

# In AudioStreamer._stream_loop():
await self.websocket.send_bytes(chunk.audio_data)
self.metrics.record_audio_sent(session_id, len(chunk.audio_data))
```

---

### 2.4 Prefetching Clarification

You asked: *"Aren't we beginning synthesis on sentence N+1 while streaming audio for sentence N?"*

**Current Reality:** Not quite. Here's the actual flow:

```
Time ──────────────────────────────────────────────────────────────────────►

LLM:        [S0 tokens...]──►[S1 tokens...]──►[S2 tokens...]
                  │                │                │
                  ▼                ▼                ▼
Sentence Q:     [S0]            [S1]            [S2]
                  │                │                │
                  ▼                │                │
TTS Worker:   [Synth S0]─────────►[Synth S1]─────►[Synth S2]  ← SEQUENTIAL!
                  │                    │               │
                  ▼                    ▼               ▼
Audio Q:      [A0 chunks]        [A1 chunks]     [A2 chunks]
                  │                    │               │
                  ▼                    ▼               ▼
WebSocket:    [Stream A0]────────►[Stream A1]───►[Stream A2]
```

The TTS worker processes sentences **one at a time**. While S0 audio streams to the client, S1 sits in the queue waiting for the TTS worker to finish S0.

**With Parallel Workers:**

```
Time ──────────────────────────────────────────────────────────────────────►

LLM:          [S0]──►[S1]──►[S2]──►[S3]
                │      │      │      │
Sentence Q:   [S0]   [S1]   [S2]   [S3]
                │      │      │      │
                ▼      ▼      ▼      ▼
TTS Workers:  W0:S0  W1:S1  W0:S2  W1:S3   ← PARALLEL!
                │      │      │      │
                └──────┼──────┴──────┘
                       ▼
Resequence:    [wait] [S0 done] → output S0
               [wait] [S1 done] → output S1
                ...

WebSocket:    [Stream A0]──►[Stream A1]──►[Stream A2]
```

**Benefit:** Sentence S1 synthesis starts as soon as it's ready, not after S0 completes.

---

### 2.5 Bounded Queues for Backpressure

```python
class StreamingQueues:
    def __init__(self, sentence_queue_size: int = 10, audio_queue_size: int = 50):
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue(maxsize=sentence_queue_size)
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=audio_queue_size)
```

**Why:**
- Prevents unbounded memory growth if TTS stalls
- Creates natural backpressure to LLM consumer
- `maxsize=10` for sentences: LLM usually faster than TTS, this buffers ahead
- `maxsize=50` for audio: ~5 seconds of audio buffer at typical chunk rates

---

## 3. Task Breakdown

### Phase 1: Quick Wins (Low Risk)
1. [ ] Cache voice reference messages at initialization
2. [ ] Cache crossfade curves at initialization
3. [ ] Remove duplicate `revert_delay_pattern` function
4. [ ] Add RAS parameters to `generate_delta_stream` call
5. [ ] Add bounded queue sizes

### Phase 2: Metrics & Observability
6. [ ] Create `PipelineMetrics` dataclass
7. [ ] Create `MetricsCollector` class
8. [ ] Instrument LLM processor with timing hooks
9. [ ] Instrument TTS worker with timing hooks
10. [ ] Instrument audio streamer with timing hooks
11. [ ] Add metrics logging on session complete

### Phase 3: Parallel TTS Workers
12. [ ] Create `TTSConfig` dataclass
13. [ ] Refactor `HiggsTTSWorker` to accept worker_id and config
14. [ ] Create `TTSWorkerPool` class with resequencing logic
15. [ ] Update `StreamingPipeline` to use worker pool
16. [ ] Add graceful shutdown for worker pool
17. [ ] Test with num_workers=1, 2, 3 and measure GPU memory

### Phase 4: Testing & Validation
18. [ ] Unit tests for resequencing buffer
19. [ ] Integration test: verify audio order matches sentence order
20. [ ] Load test: measure latency improvements
21. [ ] Memory profiling under sustained load

---

## 4. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Multiple TTS engines | High GPU memory | Make `num_workers` configurable, default=2, document memory requirements |
| Resequencing buffer | Ordering bugs | Extensive unit tests, logging, verify with sentence indices |
| Bounded queues | Potential deadlock | Ensure producers handle full queues gracefully (timeout + retry) |
| Metrics collection | Performance overhead | Use monotonic clock, minimize allocations in hot path |

---

## 5. Open Questions

1. **GPU Memory Budget:** How much GPU memory is available for multiple engines? This determines max `num_workers`.

2. **Voice Reference Caching:** Should we cache the *tokenized* voice audio (from `audio_tokenizer.encode()`) rather than just the file paths? This happens in `_prepare_inputs` and is expensive.

3. **Engine Sharing:** Can we share the model weights and audio tokenizer across workers, only duplicating KV caches? Would require refactoring `HiggsAudioServeEngine`.

4. **Sentence Fragmentation Tuning:** Current settings:
   - `minimum_first_fragment_length=10`
   - `minimum_sentence_length=15`

   Lower values = faster first audio but more TTS calls. Should we tune these based on metrics?

---

## Next Steps

Please review this plan and let me know:
1. Which phase to start with (I recommend Phase 1 → Phase 2 → Phase 3)
2. Your GPU memory constraints for worker count decisions
3. Any concerns about the parallel worker architecture
4. Whether you want the single-engine prefetch approach instead of multi-engine

Once approved, I'll implement the changes incrementally with commits after each phase.
