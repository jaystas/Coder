# Concurrent LLM -> TTS Streaming Pipeline Plan

## Executive Summary

This document outlines the design and implementation plan for a low-latency concurrent pipeline that streams text from an LLM, extracts sentences, generates audio via Higgs Audio TTS, and streams audio bytes to the client in real-time.

**Key Goal**: Minimize time-to-first-audio (TTFA) by running LLM text generation and TTS audio generation concurrently.

---

## 1. Current System Analysis

### 1.1 Available Components

| Component | File | Status | Description |
|-----------|------|--------|-------------|
| `HiggsAudioServeEngine` | `serve_engine.py` | Complete | TTS engine with streaming support |
| `StreamingVoiceGenerator` | `serve_engine.py` | Complete | Rolling context for voice consistency |
| `AsyncHiggsAudioStreamer` | `serve_engine.py` | Complete | Async streamer for audio tokens |
| `generate_sentences_from_stream` | `sentence_stream_pipeline.py` | Complete | Sentence extraction from text stream |
| `produce_sentences_to_queue` | `sentence_stream_pipeline.py` | Complete | Queue-based sentence producer |
| `STTPipeline` | `coder_server.py` | Complete | Speech-to-text service |
| `LLMPipeline` | `coder_server.py` | Partial | LLM text generation (needs sentence streaming integration) |
| `TTSPipeline` | `coder_server.py` | **Stub** | Needs full implementation |

### 1.2 Key Technical Specifications

- **Higgs Audio Tokenizer**: 50 tokens/second (tps), 16kHz sampling rate
- **Samples per token**: 320 samples (16000 / 50)
- **Audio codebooks**: 8 (num_codebooks in model config)
- **Minimum streaming chunk**: ~64 tokens = 1.28 seconds of audio
- **Voice context buffer**: Recommended 5 chunks for consistency

---

## 2. Architecture Design

### 2.1 Pipeline Overview (Fan-Out Pattern)

```
                                    ┌─────────────────────┐
                                    │   Audio Token       │
                                    │   Accumulator       │
                                    │   (per sentence)    │
                                    └──────────┬──────────┘
                                               │
                                               ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐    ┌─────────────┐
│   LLM       │───▶│  Sentence   │───▶│   TTS Generator     │───▶│   Audio     │
│   Stream    │    │  Producer   │    │   (StreamingVoice   │    │   Decoder   │
│             │    │             │    │    Generator)       │    │             │
└─────────────┘    └──────┬──────┘    └─────────────────────┘    └──────┬──────┘
                          │                                              │
                          │                                              │
                   sentence_queue                                 audio_bytes_queue
                   (SentenceItem)                                 (AudioChunk)
                          │                                              │
                          │                                              ▼
                          │                                    ┌─────────────────────┐
                          └───────────────────────────────────▶│   WebSocket         │
                                   (text for UI display)       │   Client            │
                                                               └─────────────────────┘
```

### 2.2 Data Flow

1. **LLM Stream** → AsyncOpenAI yields text deltas
2. **Sentence Producer** → `stream2sentence` extracts complete sentences
3. **TTS Generator** → `StreamingVoiceGenerator.generate_streaming()` produces audio tokens
4. **Audio Decoder** → Converts tokens to PCM16 waveform chunks
5. **WebSocket** → Streams binary audio + JSON text to client

### 2.3 Queue Types

```python
# Sentence queue items (from sentence_stream_pipeline.py)
SentenceQueueItem = SentenceItem | StreamComplete | StreamError

# Audio queue item (new)
@dataclass
class AudioChunkItem:
    audio_bytes: bytes          # PCM16 @ 16kHz audio data
    sentence_index: int         # Which sentence this belongs to
    chunk_index: int            # Chunk within sentence
    message_id: str             # Parent message ID
    character_id: str           # Character speaking
    is_sentence_final: bool     # Last chunk for this sentence?
    is_message_final: bool      # Last chunk for entire message?

# Sentinel types
@dataclass
class AudioStreamComplete:
    total_sentences: int
    total_audio_chunks: int

@dataclass
class AudioStreamError:
    exception: Exception
    sentence_index: int
```

---

## 3. Implementation Plan

### 3.1 Phase 1: TTSPipeline Core Implementation

**File**: `coder_server.py` - Expand `TTSPipeline` class

```python
class TTSPipeline:
    """
    TTS Pipeline using Higgs Audio with streaming voice consistency.

    Responsibilities:
    1. Initialize and manage HiggsAudioServeEngine
    2. Maintain per-character StreamingVoiceGenerator instances
    3. Consume sentences from sentence_queue
    4. Generate audio tokens and decode to PCM chunks
    5. Push audio chunks to audio_queue for streaming
    """

    def __init__(
        self,
        queues: Queues,
        model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
        tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
        device: str = "cuda",
        audio_chunk_size: int = 64,  # tokens per streaming chunk
        voice_context_buffer_size: int = 5,
    ):
        self.queues = queues
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.audio_chunk_size = audio_chunk_size
        self.voice_context_buffer_size = voice_context_buffer_size

        # Initialized later
        self.serve_engine: Optional[HiggsAudioServeEngine] = None

        # Per-character voice generators for multi-character support
        self.voice_generators: Dict[str, StreamingVoiceGenerator] = {}

        # Interrupt handling
        self.interrupt_event = asyncio.Event()

    async def initialize(self):
        """Initialize the Higgs Audio serve engine."""
        self.serve_engine = HiggsAudioServeEngine(
            model_name_or_path=self.model_path,
            audio_tokenizer_name_or_path=self.tokenizer_path,
            device=self.device,
        )

    def get_or_create_voice_generator(
        self,
        character: Character,
        voice: Voice,
    ) -> StreamingVoiceGenerator:
        """Get or create a voice generator for a character."""
        # ... implementation details in Phase 1
```

**Key Methods to Implement**:

1. `initialize()` - Load HiggsAudioServeEngine
2. `get_or_create_voice_generator(character, voice)` - Per-character voice context
3. `generate_audio_for_sentence(sentence, character, voice)` - Single sentence TTS
4. `decode_audio_tokens_streaming(tokens)` - Chunked audio token decoding
5. `speech_loop()` - Main consumer loop
6. `shutdown()` - Cleanup resources

### 3.2 Phase 2: Concurrent Pipeline Integration

**File**: `coder_server.py` - Modify `LLMPipeline` and `WebSocketManager`

#### 3.2.1 LLMPipeline Modifications

```python
async def character_response_stream_with_tts(
    self,
    character: Character,
    voice: Voice,
    text_stream: AsyncIterator,
) -> str:
    """
    Generate character response with concurrent TTS.

    This method:
    1. Streams text chunks to UI (response_queue)
    2. Extracts sentences via stream2sentence
    3. Queues sentences for TTS processing (sentence_queue)
    """
    message_id = f"msg-{character.id}-{int(time.time() * 1000)}"

    # Create text delta extractor
    async def text_delta_generator():
        async for chunk in text_stream:
            content = chunk.choices[0].delta.content
            if content:
                self.response_text += content
                # Send to UI immediately
                await self.queues.response_queue.put(TextChunk(...))
                yield content

    # Run sentence extraction concurrently
    sentence_count = await produce_sentences_to_queue(
        text_stream=text_delta_generator(),
        sentence_queue=self.queues.sentence_queue,
        quick_yield=True,
        min_first_fragment_length=10,
    )

    return self.response_text
```

#### 3.2.2 WebSocketManager Task Coordination

```python
async def start_service_tasks(self):
    """Start all concurrent service tasks."""
    self.service_tasks = [
        asyncio.create_task(self.stt_pipeline.run_transcription_loop()),
        asyncio.create_task(self.llm_pipeline.conversation_loop()),
        asyncio.create_task(self.tts_pipeline.speech_loop()),  # TTS consumer
        asyncio.create_task(self.stream_text_to_client()),     # Text UI
        asyncio.create_task(self.stream_audio_to_client()),    # Audio streaming
    ]
```

### 3.3 Phase 3: Real-Time Audio Streaming

**Key Challenge**: Decode audio tokens in chunks for minimal latency

```python
async def decode_audio_tokens_streaming(
    self,
    audio_token_generator: AsyncIterator[torch.Tensor],
    chunk_size: int = 64,
) -> AsyncIterator[bytes]:
    """
    Decode audio tokens to PCM bytes in streaming chunks.

    Args:
        audio_token_generator: Yields individual audio tokens
        chunk_size: Number of tokens to accumulate before decoding

    Yields:
        PCM16 audio bytes (16kHz, mono)
    """
    buffer = []

    async for token in audio_token_generator:
        buffer.append(token)

        if len(buffer) >= chunk_size:
            # Decode chunk
            audio_chunk = torch.stack(buffer, dim=1)
            vq_code = revert_delay_pattern(audio_chunk).clip(
                0, self.serve_engine.audio_codebook_size - 1
            )[:, 1:-1]

            waveform = self.serve_engine.audio_tokenizer.decode(
                vq_code.unsqueeze(0)
            )[0, 0]

            # Convert to PCM16 bytes
            pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()
            yield pcm_bytes

            buffer = []  # Clear buffer (could keep overlap for continuity)

    # Process remaining tokens
    if buffer:
        audio_chunk = torch.stack(buffer, dim=1)
        vq_code = revert_delay_pattern(audio_chunk).clip(
            0, self.serve_engine.audio_codebook_size - 1
        )[:, 1:-1]
        waveform = self.serve_engine.audio_tokenizer.decode(
            vq_code.unsqueeze(0)
        )[0, 0]
        pcm_bytes = (waveform * 32767).astype(np.int16).tobytes()
        yield pcm_bytes
```

### 3.4 Phase 4: Voice Configuration & Character Mapping

**File**: `coder_server.py` - Extend `Voice` and character handling

```python
@dataclass
class Voice:
    voice: str                  # Voice identifier
    method: str                 # "description" | "clone" | "reference"
    speaker_desc: str           # e.g., "feminine;warm;moderate pitch"
    scene_prompt: str           # e.g., "Audio recorded in quiet room"
    audio_path: str = ""        # Reference audio for cloning
    temperature: float = 0.7    # TTS sampling temperature
    top_p: float = 0.95         # TTS nucleus sampling

def get_voice_for_character(character: Character) -> Voice:
    """
    Map character to voice configuration.

    Options:
    1. Character.voice contains a JSON voice config
    2. Character.voice is a preset name (lookup table)
    3. Default voice based on character name
    """
    # Implementation...
```

---

## 4. Detailed Implementation Checklist

### 4.1 TTSPipeline Implementation

- [ ] Initialize HiggsAudioServeEngine in `initialize()`
- [ ] Implement `get_or_create_voice_generator()` with caching
- [ ] Implement `speech_loop()` that consumes from `sentence_queue`
- [ ] Implement `generate_audio_for_sentence()` using `generate_streaming_with_context`
- [ ] Implement `decode_audio_tokens_streaming()` for chunked decoding
- [ ] Implement interrupt handling (`interrupt_event`)
- [ ] Implement `shutdown()` for cleanup

### 4.2 LLMPipeline Modifications

- [ ] Modify `character_response_stream()` to also produce sentences
- [ ] Integrate `produce_sentences_to_queue()` from sentence_stream_pipeline.py
- [ ] Ensure text chunks still go to `response_queue` for UI
- [ ] Add sentence metadata (character_id, message_id) to queue items

### 4.3 WebSocketManager Updates

- [ ] Add `stream_audio_to_client()` task
- [ ] Ensure proper task coordination/cancellation
- [ ] Handle audio chunk binary streaming format

### 4.4 Voice Configuration

- [ ] Extend `Voice` dataclass with TTS parameters
- [ ] Implement character-to-voice mapping
- [ ] Support voice presets and custom configurations

### 4.5 Testing & Optimization

- [ ] Unit test TTSPipeline with mock sentences
- [ ] Integration test full pipeline (STT → LLM → TTS)
- [ ] Measure and optimize TTFA (time to first audio)
- [ ] Test interrupt handling mid-generation
- [ ] Test multi-character voice consistency

---

## 5. Questions for Discussion

### Q1: Audio Chunk Size
**Current proposal**: 64 tokens = ~1.28 seconds of audio

**Trade-offs**:
- Smaller chunks = lower latency, but more decode overhead and potential audio artifacts
- Larger chunks = smoother audio, but higher latency

**Question**: What's the acceptable latency for your use case? Should we target <500ms TTFA?

### Q2: Voice Context Scope
**Current proposal**: Per-character voice generators with rolling buffer

**Options**:
1. **Session-scoped**: Voice context persists for entire conversation
2. **Message-scoped**: Reset voice context per message
3. **Hybrid**: Keep reference but limit rolling buffer

**Question**: Should voice context persist across multiple user turns, or reset each response?

### Q3: Multi-Character TTS
**Current system**: Characters can be mentioned and respond in sequence

**Question**: For multi-character responses, should we:
1. Generate all characters' audio in sequence (simpler, maintains context)
2. Generate characters' audio in parallel (faster, but context isolation)

### Q4: Audio Format
**Current proposal**: PCM16 @ 16kHz mono

**Options**:
- PCM16 @ 16kHz (raw, highest quality, largest bandwidth)
- PCM16 @ 24kHz (if model supports upsampling)
- Opus encoding (compressed, lower bandwidth, requires client decoder)

**Question**: What audio format does the frontend client expect?

### Q5: Sentence Boundaries
**Current**: Using `stream2sentence` with `quick_yield=True`

**Concern**: Quick yield might produce short fragments that result in choppy TTS

**Question**: Should we tune `min_sentence_length` / `min_first_fragment_length` for smoother audio?

---

## 6. Latency Analysis

### 6.1 Current Estimated Latencies

| Stage | Latency | Notes |
|-------|---------|-------|
| LLM first token | ~200-500ms | Depends on model/provider |
| Sentence accumulation | ~500-2000ms | First complete sentence |
| TTS first token | ~100-200ms | Model inference startup |
| Audio token accumulation | ~1280ms | 64 tokens @ 50 tps |
| Audio decode | ~50-100ms | GPU accelerated |
| WebSocket transfer | ~10-50ms | Network dependent |
| **Total TTFA** | **~2.1-4.1s** | Without optimization |

### 6.2 Optimization Strategies

1. **Reduce sentence accumulation time**: Lower `min_first_fragment_length` to 8-10 chars
2. **Reduce audio chunk size**: 32 tokens = ~640ms (but more artifacts)
3. **Overlap LLM/TTS**: Start TTS on first sentence while LLM continues
4. **Pre-warm TTS**: Keep KV caches warm between sentences

---

## 7. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU OOM with concurrent TTS | High | Sequential TTS, batch size limits |
| Voice inconsistency between sentences | Medium | Larger context buffer, lower temperature |
| Audio artifacts from chunked decoding | Medium | Overlap/crossfade between chunks |
| Client buffer underrun | Medium | Adaptive chunk size, prebuffering |
| Interrupt handling race conditions | Low | Proper event synchronization |

---

## 8. Next Steps

1. **Review this plan** - Discuss questions and design decisions
2. **Implement Phase 1** - TTSPipeline core
3. **Implement Phase 2** - Pipeline integration
4. **Implement Phase 3** - Streaming audio
5. **Testing & iteration** - Measure latencies, optimize

---

## Appendix A: Code Structure After Implementation

```
backend/
├── coder_server.py           # Main FastAPI server (modified)
│   ├── STTPipeline           # Speech-to-text (existing)
│   ├── LLMPipeline           # LLM generation (modified)
│   ├── TTSPipeline           # TTS generation (new implementation)
│   ├── WebSocketManager      # Coordination (modified)
│   └── Queues                # Queue management (extended)
├── sentence_stream_pipeline.py    # Sentence extraction (existing)
└── boson_multimodal/
    └── serve/
        └── serve_engine.py   # Higgs Audio engine (existing)
```

## Appendix B: Example Usage Flow

```python
# 1. User speaks → STT transcribes
user_message = "Hey Luna, what do you think about AI?"

# 2. LLM generates response (streaming)
# 3. Sentences extracted concurrently
# 4. TTS generates audio per sentence
# 5. Audio streamed to client as PCM chunks

# Timeline:
# T+0ms:     User message received
# T+200ms:   LLM first token
# T+800ms:   First sentence complete ("I think AI is fascinating!")
# T+900ms:   TTS starts on first sentence
# T+2100ms:  First audio chunk ready (64 tokens)
# T+2150ms:  Audio streaming to client begins
```
