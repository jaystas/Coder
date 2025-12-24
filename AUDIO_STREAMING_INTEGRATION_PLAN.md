# Audio Streaming Pipeline Integration Plan

## Overview

This document analyzes the current state of `backend/fastapi_serve.py` and provides a roadmap to complete the streaming/concurrent text-to-audio pipeline with direct browser playback.

**Target Architecture**: User Input → STT → LLM Stream → Sentence Extraction → TTS Synthesis → WebSocket Audio Stream → Browser Playback

---

## Data Flow Analysis

### Current Pipeline Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INTENDED DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐        │
│  │   Browser    │───▶│  WebSocketManager │───▶│  transcribe_queue  │        │
│  │  (Audio In)  │    │   (Transcribe)    │    │                    │        │
│  └──────────────┘    └───────────────────┘    └─────────┬──────────┘        │
│                                                          │                   │
│                                                          ▼                   │
│                                              ┌────────────────────┐          │
│                                              │   ChatFunctions    │          │
│                                              │ (process_user_msg) │          │
│                                              └─────────┬──────────┘          │
│                                                        │                     │
│                                                        ▼                     │
│                                              ┌────────────────────┐          │
│                                              │   LLMTextStream    │          │
│                                              │ (OpenRouter API)   │          │
│                                              └─────────┬──────────┘          │
│                                                        │                     │
│                                                        ▼                     │
│  ┌──────────────┐    ┌───────────────────┐    ┌────────────────────┐        │
│  │   Browser    │◀───│  WebSocketManager │◀───│  sentence_queue    │        │
│  │ (Audio Out)  │    │ (stream_audio)    │    │                    │        │
│  └──────────────┘    └───────────────────┘    └─────────┬──────────┘        │
│         ▲                                               │                    │
│         │                                               ▼                    │
│         │                                    ┌────────────────────┐          │
│         │                                    │     TTSWorker      │          │
│         │                                    │   (Higgs Audio)    │          │
│         │                                    └─────────┬──────────┘          │
│         │                                              │                     │
│         │    ┌───────────────────┐                     │                     │
│         └────│   audio_queue     │◀────────────────────┘                     │
│              │    (REMOVED)      │                                           │
│              └───────────────────┘                                           │
│                                                                              │
│  NEW: Stream directly from TTSWorker → WebSocket (skip audio_queue)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Critical Issues Identified

### 1. Disconnected Queue Instances (BLOCKER)

**Location**: Lines 772, 823, 993-994

**Problem**: Multiple independent `PipeQueues` instances exist that don't share data:

```python
# Line 772 - ConversationPipeline creates its own queues
class ConversationPipeline:
    def __init__(self, api_key: str):
        self.queues = PipeQueues()  # Instance A

# Line 823 - WebSocketManager creates separate queues
class WebSocketManager:
    def __init__(self):
        self.queues = PipeQueues()  # Instance B (DIFFERENT!)

# Lines 993-994 - Both are instantiated separately
convo_pipe = ConversationPipeline()  # Uses Instance A
ws_manager = WebSocketManager()       # Uses Instance B
```

**Impact**: Messages placed in `ws_manager.queues.transcribe_queue` will never be consumed by `convo_pipe` because they operate on different queue instances.

**Fix**: Use a single shared `PipeQueues` instance across all components.

---

### 2. Constructor Signature Mismatches (BLOCKER)

**Location**: Lines 771, 842, 993

**Problem A**: `ConversationPipeline` requires `api_key` but none is provided:
```python
# Line 771 - Requires api_key
def __init__(self, api_key: str):

# Line 993 - Called without api_key
convo_pipe = ConversationPipeline()  # TypeError!
```

**Problem B**: `ChatFunctions` constructor mismatch:
```python
# Lines 284-289 - Takes no parameters
class ChatFunctions:
    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []

# Line 842 - Called with parameters it doesn't accept
self.chat = ChatFunctions(queues=self.queues, api_key=self.openrouter_api_key)  # TypeError!
```

**Problem C**: `openrouter_api_key` is not defined:
```python
# Line 842 - References undefined attribute
api_key=self.openrouter_api_key  # AttributeError!
```

---

### 3. Method Name Mismatches (BLOCKER)

**Location**: Lines 388, 605, 640, 792

**Problem A**: Singular vs plural method name:
```python
# Line 792 - Calls singular (doesn't exist)
await self.chat.process_user_message(...)

# Line 388 - Actual method is plural
async def process_user_messages(...)
```

**Problem B**: Missing `voice` parameter in TTS call:
```python
# Line 605 - Calls with only text
async for pcm_bytes in self.generate_audio_for_sentence(sentence.text):

# Line 640 - Method requires both text AND voice
async def generate_audio_for_sentence(self, text: str, voice: str) -> AsyncGenerator[bytes, None]:
```

---

### 4. AudioChunk Field Mismatches (BLOCKER)

**Location**: Lines 82-90 vs 973-979

**Problem**: `on_audio_chunk` references fields that don't exist in `AudioChunk`:

```python
# Lines 82-90 - Actual AudioChunk definition
@dataclass
class AudioChunk:
    audio_bytes: bytes       # ← Actual field
    sentence_index: int      # ← Actual field
    chunk_index: int         # ← Actual field
    session_id: str          # ← Actual field
    character_id: str        # ← Actual field
    is_final: bool
    is_session_complete: bool

# Lines 973-979 - References non-existent fields
await self.send_text_to_client({
    "message_id": audio_chunk.message_id,       # ← MISSING
    "chunk_id": audio_chunk.chunk_id,           # ← MISSING (is chunk_index)
    "character_name": audio_chunk.character_name, # ← MISSING (only character_id exists)
    "index": audio_chunk.index,                 # ← MISSING (is sentence_index)
})
await self.stream_audio_to_client(audio_chunk.audio_data)  # ← WRONG (is audio_bytes)
```

---

### 5. Empty/Stubbed Methods (INCOMPLETE)

**Location**: Lines 799-807

**Problem**: Core pipeline methods have no implementation:

```python
# Lines 799-803 - Empty conversate method
async def conversate(self, user_message: str, character: Character, websocket: WebSocket):
    """Main conversation flow: user message → LLM → TTS → audio stream"""
    # This is where calls will go (creates one easy to read top down flow).
    # Do not put whole functions here, just calls.
    pass  # EMPTY!

# Lines 805-807 - Empty audio player
async def audio_player(self):
    """Streams PCM Audio to Browser for Real-time Playback"""
    pass  # EMPTY!
```

---

### 6. TTS Worker Never Started (BLOCKER)

**Location**: Lines 548-573

**Problem**: `TTSWorker` has `initialize()` and `start()` methods, but neither is called:

```python
# Lines 558-561 - start() method exists
async def start(self):
    self.is_running = True
    self._task = asyncio.create_task(self.process_sentences())

# NOWHERE in the codebase is this called!
```

**Impact**: The sentence queue will fill up but audio will never be generated.

---

### 7. No Audio Queue Consumer (BLOCKER)

**Location**: Lines 116, 805-807

**Problem**: Audio chunks are placed into `audio_queue` but nothing consumes them:

```python
# Line 116 - Queue exists
self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

# Lines 586-616 - Chunks are put into queue
await self.queues.audio_queue.put(audio_chunk)

# Lines 805-807 - Consumer is empty
async def audio_player(self):
    """Streams PCM Audio to Browser for Real-time Playback"""
    pass  # NOTHING READS FROM audio_queue!
```

---

### 8. Duplicate Function Definition

**Location**: Lines 39, 517-528

**Problem**: `revert_delay_pattern` is both imported and redefined:

```python
# Line 39 - Imported
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# Lines 517-528 - Redefined locally (shadows import)
def revert_delay_pattern(data: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
```

**Fix**: Remove the local definition (lines 517-528) and use the imported version.

---

### 9. Missing Environment Variable Loading

**Location**: Lines 420-425, 771

**Problem**: OpenRouter API key needs to be loaded from environment:

```python
# Line 420-425 - LLMTextStream requires api_key
class LLMTextStream:
    def __init__(self, queues: PipeQueues, api_key: str):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key  # Must be provided
        )
```

**Fix**: Add `OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")` and validate it exists.

---

### 10. process_user_messages Has Unused Parameter

**Location**: Line 388

**Problem**: `queues` parameter is passed but never used:

```python
async def process_user_messages(
    self,
    queues: PipeQueues,  # ← NEVER USED in method body
    user_message: str,
    user_name: str,
    ...
)
```

---

## New Architecture: Direct Browser Streaming

Per the requirement to stream TTS audio **directly to browser** (bypassing `audio_queue`), here's the updated design:

### Simplified Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIRECT STREAMING ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Message                                                                │
│       │                                                                      │
│       ▼                                                                      │
│  ┌────────────────┐                                                          │
│  │ ChatFunctions  │ ─── LLM Stream ───▶ generate_sentences_async()          │
│  └────────────────┘                              │                           │
│                                                  │                           │
│                                     ┌────────────┴────────────┐              │
│                                     │     For each sentence   │              │
│                                     └────────────┬────────────┘              │
│                                                  │                           │
│                                                  ▼                           │
│                                     ┌─────────────────────────┐              │
│                                     │  TTS generate_audio()   │              │
│                                     │   (yields PCM chunks)   │              │
│                                     └────────────┬────────────┘              │
│                                                  │                           │
│                                                  ▼                           │
│                                     ┌─────────────────────────┐              │
│                                     │  WebSocket.send_bytes() │ ───▶ Browser │
│                                     │    (direct stream)      │              │
│                                     └─────────────────────────┘              │
│                                                                              │
│  Key Change: No audio_queue intermediate step                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Remove `audio_queue`** - Stream PCM directly to WebSocket as it's generated
2. **Keep `sentence_queue`** - Still needed to decouple LLM streaming from TTS generation
3. **Single Shared Queues** - All components reference the same `PipeQueues` instance
4. **Async TTS Integration** - TTS worker yields chunks that are immediately sent to browser

---

## Implementation Tasks

### Phase 1: Fix Critical Blockers

#### Task 1.1: Unify Queue Management
- Create single `PipeQueues` instance at module level
- Pass shared instance to all components
- Remove duplicate instantiations

#### Task 1.2: Fix Constructor Signatures
- Add `OPENROUTER_API_KEY` environment variable loading
- Update `ChatFunctions.__init__()` to accept required params OR remove params from caller
- Fix `ConversationPipeline` instantiation with api_key

#### Task 1.3: Fix Method Name Mismatches
- Rename `process_user_message` → `process_user_messages` call
- Add `voice` parameter to TTS generation call chain

#### Task 1.4: Align AudioChunk Fields
- Update `on_audio_chunk` to use correct field names
- OR update `AudioChunk` dataclass to include needed fields

### Phase 2: Implement Direct Streaming

#### Task 2.1: Modify TTSWorker for Direct Streaming
```python
async def synthesize_and_stream(
    self,
    sentence: TTSSentence,
    websocket: WebSocket
) -> None:
    """Generate audio and stream directly to WebSocket"""
    async for pcm_chunk in self.generate_audio_for_sentence(
        text=sentence.text,
        voice=sentence.voice_id
    ):
        # Send metadata
        await websocket.send_json({
            "type": "audio_chunk",
            "sentence_index": sentence.index,
            "character_id": sentence.character_id,
        })
        # Send binary audio
        await websocket.send_bytes(pcm_chunk)
```

#### Task 2.2: Implement Audio Streaming Orchestrator
```python
async def stream_audio_to_browser(
    self,
    websocket: WebSocket,
    session_id: str
) -> None:
    """Consume sentence_queue, synthesize, stream to browser"""
    while True:
        sentence = await self.queues.sentence_queue.get()

        if sentence.is_session_complete:
            await websocket.send_json({"type": "audio_complete"})
            break

        await self.tts_worker.synthesize_and_stream(sentence, websocket)
```

#### Task 2.3: Wire Up Pipeline in WebSocket Handler
```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)

    # Start background audio streaming task
    audio_task = asyncio.create_task(
        convo_pipe.stream_audio_to_browser(websocket, session_id)
    )

    try:
        while True:
            message = await websocket.receive()
            # Handle messages...
    finally:
        audio_task.cancel()
```

### Phase 3: Initialize Components Properly

#### Task 3.1: Update Lifespan Handler
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    # Initialize shared queues
    shared_queues = PipeQueues()

    # Initialize conversation pipeline with shared queues
    await convo_pipe.initialize(shared_queues, api_key)

    # Initialize TTS engine (loads model)
    await convo_pipe.tts_worker.initialize()

    # Initialize WebSocket manager with shared queues
    await ws_manager.initialize(shared_queues)

    yield

    await ws_manager.shutdown()
```

### Phase 4: Remove Unused Code

#### Task 4.1: Clean Up
- Remove duplicate `revert_delay_pattern` function (lines 517-528)
- Remove `audio_queue` from `PipeQueues` if not needed
- Remove empty `audio_player` method
- Remove unused `queues` parameter from `process_user_messages`

---

## WebSocket Protocol Specification

### Browser → Server Messages

```json
// User text message (typed)
{"type": "user_message", "data": {"text": "Hello, how are you?"}}

// Start voice recording
{"type": "start_listening"}

// Stop voice recording
{"type": "stop_listening"}

// Model settings
{"type": "model_settings", "data": {"model": "...", "temperature": 0.7}}
```

### Server → Browser Messages

```json
// Text streaming start
{"type": "text_stream_start", "message_id": "...", "character_name": "..."}

// Text chunk (for UI display)
{"type": "text_chunk", "text": "...", "message_id": "..."}

// Text streaming complete
{"type": "text_stream_stop", "message_id": "..."}

// Audio streaming start
{"type": "audio_stream_start", "message_id": "...", "character_name": "..."}

// Audio chunk metadata (followed by binary frame)
{"type": "audio_chunk_meta", "sentence_index": 0, "chunk_index": 0}

// [Binary Frame: PCM16 audio bytes]

// Audio streaming complete
{"type": "audio_complete", "session_id": "..."}

// STT updates
{"type": "stt_update", "text": "..."}
{"type": "stt_finished", "text": "..."}
```

### Audio Format
- **Format**: Raw PCM
- **Sample Rate**: 24000 Hz (Higgs Audio native)
- **Bit Depth**: 16-bit signed integer (PCM16)
- **Channels**: Mono
- **Endianness**: Little-endian (native)

---

## Reference: Working Implementation

The file `backend/streaming_pipeline.py` contains a working reference implementation with:
- `StreamingQueues` - Clean queue management
- `LLMStreamProcessor` - OpenRouter streaming with sentence extraction
- `HiggsTTSWorker` - Higgs Audio synthesis
- `AudioStreamer` - WebSocket streaming
- `StreamingPipeline` - Orchestration

Consider adapting patterns from this file when fixing `fastapi_serve.py`.

---

## Testing Checklist

- [ ] Server starts without errors
- [ ] WebSocket connection establishes
- [ ] Text message triggers LLM response
- [ ] Sentences stream to TTS worker
- [ ] Audio chunks stream to browser
- [ ] Browser plays audio in real-time
- [ ] STT transcription works
- [ ] Multi-character responses work correctly
- [ ] Graceful error handling
- [ ] Clean disconnection handling

---

## Summary of Required Changes

| Priority | Issue | Location | Effort |
|----------|-------|----------|--------|
| P0 | Disconnected queue instances | Lines 772, 823 | Low |
| P0 | Missing api_key in constructor | Line 993 | Low |
| P0 | ChatFunctions constructor mismatch | Line 842 | Low |
| P0 | Method name mismatch | Line 792 | Low |
| P0 | Missing voice param in TTS | Line 605 | Low |
| P0 | AudioChunk field mismatches | Lines 973-979 | Low |
| P0 | TTS worker never started | N/A | Low |
| P1 | Empty audio_player method | Lines 805-807 | Medium |
| P1 | Implement direct streaming | New code | Medium |
| P2 | Remove duplicate function | Lines 517-528 | Low |
| P2 | Remove unused parameter | Line 388 | Low |

**Estimated Total Effort**: 2-4 hours to reach working state
