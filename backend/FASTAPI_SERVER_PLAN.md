# FastAPI Server Review & Planning Document

## Executive Summary

The server implements a voice chat pipeline: **STT (Transcribe) → LLM (ChatLLM) → TTS (Speech)**

After thorough review, I've identified several categories of issues:
1. Missing/incomplete `__init__` methods
2. Undefined instance attributes referenced throughout
3. Mismatched callback parameter names
4. Method name typos
5. Missing lifecycle methods (initialize, connect, disconnect, shutdown)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      WebSocketManager                            │
│  - Owns all pipeline components                                  │
│  - Routes messages between client and pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │PipeQueues│    │Transcribe│    │ ChatLLM  │    │  Speech  │  │
│  │          │    │  (STT)   │    │          │    │  (TTS)   │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       │    transcribe_queue    sentence_queue    audio_queue    │
│       └───────────────┴───────────────┴───────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Class-by-Class Analysis

### 1. `PipeQueues` (Lines 91-98) ✅ OK

**Current State:** Properly implemented

```python
class PipeQueues:
    def __init__(self):
        self.transcribe_queue = asyncio.Queue()  # user messages
        self.sentence_queue = asyncio.Queue()    # TTSSentence objects
        self.audio_queue = asyncio.Queue()       # AudioChunk objects
```

**Status:** No changes needed.

---

### 2. `Transcribe` (Lines 106-238) ⚠️ ISSUES FOUND

**Current `__init__`:**
```python
def __init__(self,
    on_transcription_update: Optional[Callback] = None,
    on_transcription_stabilized: Optional[Callback] = None,
    on_transcription_finished: Optional[Callback] = None,
    # ... more callbacks
):
    self.callbacks: Dict[str, Optional[Callback]] = {
        'on_realtime_update': on_transcription_update,  # ← key mismatch
        # ...
    }
    self.is_listening = False
    self.recorder = AudioToTextRecorder(...)
```

**Issues Found:**
| Line | Issue | Description |
|------|-------|-------------|
| 121-131 | Key mismatch | Callback dict keys don't match parameter names |
| 177 | Missing attribute | `self.loop` is never set but referenced |
| 167 | Wrong syntax | `self.callbacks.on_transcription_finished` should use `.get()` |
| 206 | Typo | `self.callback` should be `self.callbacks` |

**Recommended `__init__`:**
```python
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
    # Store callbacks with CONSISTENT key names
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
    self.loop: Optional[asyncio.AbstractEventLoop] = None  # ← ADD THIS

    self.recorder = AudioToTextRecorder(...)
```

**Additional Fix Needed - `set_event_loop` method:**
```python
def set_event_loop(self, loop: asyncio.AbstractEventLoop):
    """Set the asyncio event loop for callback execution"""
    self.loop = loop
```

**Bug Fixes Required:**
- Line 167: Change `self.callbacks.on_transcription_finished` → `self.callbacks.get('on_transcription_finished')`
- Line 206: Change `self.callback.get(...)` → `self.callbacks.get(...)`

---

### 3. `ChatLLM` (Lines 245-463) ⚠️ ISSUES FOUND

**Current `__init__`:**
```python
def __init__(self, queues: PipeQueues, api_key: str):
    self.conversation_history: List[Dict] = []
    self.conversation_id: Optional[str] = None
    self.queues = queues
    self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    self.model_settings: Optional[ModelSettings] = None
    self.active_characters: List[Character] = []
```

**Status:** The `__init__` is actually fine!

**Issues are in methods, not `__init__`:**

| Line | Issue | Fix |
|------|-------|-----|
| 350 | Wrong method name | `self.character_instruction_message` → `self.create_character_instruction_message` |
| 366 | Undefined `self.chatllm` | Should be `self.stream_character_response` |
| 369-372 | Undefined variables | `session_id`, `sentence_queue`, `on_text_chunk` need to be passed as parameters |
| 376 | Wrong method name | `self.wrap_character_tags` → `self.wrap_with_character_tags` |

**The `process_user_messages` method signature should be:**
```python
async def process_user_messages(
    self,
    user_message: str,
    user_name: str,
    session_id: str,                          # ADD
    sentence_queue: asyncio.Queue[TTSSentence],  # ADD
    on_text_chunk: Optional[Callable[[str], Awaitable[None]]] = None  # ADD
):
```

---

### 4. `Speech` (Lines 482-693) ⚠️ ISSUES FOUND

**Current `__init__`:**
```python
def __init__(self, queues: PipeQueues):
    self.queues = queues
    self.engine: Optional[HiggsAudioServeEngine] = None
    self.is_running = False
    self._task: Optional[asyncio.Task] = None
    self.sample_rate = 24000
    self._chunk_size = 14
    self._device = "cuda" if torch.cuda.is_available() else "cpu"
    self.voice_dir = "backend/voices"
    self.voice_name = "lydia"
```

**Status:** The `__init__` is fine!

**Issue in method call:**
| Line | Issue | Fix |
|------|-------|-----|
| 557 | Missing argument | `self.generate_audio_for_sentence(sentence.text)` missing `voice` parameter |

**Fix:**
```python
# Line 557 should be:
async for pcm_bytes in self.generate_audio_for_sentence(sentence.text, sentence.voice_id):
```

---

### 5. `WebSocketManager` (Lines 699-795) ❌ MAJOR ISSUES

**Current `__init__`:**
```python
def __init__(self):
    self.transcribe = Transcribe(
        on_realtime_update=self.on_transcription_update,      # Wrong param name
        on_realtime_stabilized=self.on_transcription_stabilized,  # Wrong param name
        on_final_transcription=self.on_transcription_finished,    # Wrong param name
    )
```

**Missing Attributes (referenced but never defined):**
- `self.websocket` (lines 759, 764)
- `self.queues` (lines 755, 774, 785, 793)
- `self.chat` (lines 741, 790)
- `self.llm_stream` (line 792)
- `self.consumer_task` (line 779)
- `self.speech` (for TTS)

**Missing Methods (called but not defined):**
- `initialize()` (line 806)
- `connect(websocket)` (line 829)
- `disconnect()` (lines 842, 846)
- `shutdown()` (line 810)

**Recommended Complete `__init__`:**
```python
def __init__(self):
    # Pipeline queues - shared between all components
    self.queues = PipeQueues()

    # WebSocket connection (set on connect)
    self.websocket: Optional[WebSocket] = None

    # Pipeline components (initialized in initialize())
    self.transcribe: Optional[Transcribe] = None
    self.chat: Optional[ChatLLM] = None
    self.speech: Optional[Speech] = None

    # Background tasks
    self.consumer_task: Optional[asyncio.Task] = None
    self.audio_streamer_task: Optional[asyncio.Task] = None
```

**Required Methods to Add:**

```python
async def initialize(self):
    """Initialize all pipeline components"""
    api_key = os.getenv("OPENROUTER_API_KEY", "")

    # Initialize transcription (STT)
    self.transcribe = Transcribe(
        on_transcription_update=self.on_transcription_update,
        on_transcription_stabilized=self.on_transcription_stabilized,
        on_transcription_finished=self.on_transcription_finished,
    )
    self.transcribe.set_event_loop(asyncio.get_event_loop())

    # Initialize chat (LLM)
    self.chat = ChatLLM(queues=self.queues, api_key=api_key)
    self.chat.active_characters = await self.chat.get_active_characters()

    # Initialize speech (TTS)
    self.speech = Speech(queues=self.queues)
    await self.speech.initialize()

async def connect(self, websocket: WebSocket):
    """Accept WebSocket connection and start pipeline"""
    await websocket.accept()
    self.websocket = websocket

    # Start TTS worker
    await self.speech.start()

    # Start pipeline consumer
    await self.start_pipeline()

    # Start audio streaming to client
    self.audio_streamer_task = asyncio.create_task(self.stream_audio_to_client_loop())

async def disconnect(self):
    """Clean up on disconnect"""
    if self.transcribe:
        self.transcribe.stop_listening()

    if self.speech:
        await self.speech.stop()

    if self.consumer_task:
        self.consumer_task.cancel()

    if self.audio_streamer_task:
        self.audio_streamer_task.cancel()

    self.websocket = None

async def shutdown(self):
    """Shutdown all services"""
    await self.disconnect()

async def stream_audio_to_client_loop(self):
    """Background task: stream audio chunks to WebSocket client"""
    while True:
        try:
            chunk: AudioChunk = await self.queues.audio_queue.get()

            if chunk.audio_bytes:
                await self.websocket.send_bytes(chunk.audio_bytes)

            if chunk.is_session_complete:
                await self.send_text_to_client({"type": "audio_complete"})

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
```

---

## Dataclass Review

### `TTSSentence` ✅ OK
```python
@dataclass
class TTSSentence:
    text: str
    index: int
    session_id: str
    character_id: str
    character_name: str
    voice_id: str
    is_final: bool = False
    is_session_complete: bool = False
```
**Status:** Complete and correct. Used to pass sentences from LLM to TTS.

### `AudioChunk` ✅ OK
```python
@dataclass
class AudioChunk:
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    character_id: str
    is_final: bool = False
    is_session_complete: bool = False
```
**Status:** Complete and correct. Used to pass audio from TTS to WebSocket.

### `ModelSettings` ✅ OK
```python
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
```
**Status:** Complete and correct. Used for LLM configuration.

---

## Summary of Required Changes

### High Priority (Server won't run without these)

| # | Class | Change | Lines |
|---|-------|--------|-------|
| 1 | `WebSocketManager` | Add complete `__init__` with all required attributes | 702-708 |
| 2 | `WebSocketManager` | Add `initialize()` method | NEW |
| 3 | `WebSocketManager` | Add `connect(websocket)` method | NEW |
| 4 | `WebSocketManager` | Add `disconnect()` method | NEW |
| 5 | `WebSocketManager` | Add `shutdown()` method | NEW |
| 6 | `WebSocketManager` | Add `stream_audio_to_client_loop()` method | NEW |
| 7 | `Transcribe` | Add `self.loop` attribute | 133 |
| 8 | `Transcribe` | Add `set_event_loop()` method | NEW |

### Medium Priority (Runtime errors)

| # | Class | Change | Lines |
|---|-------|--------|-------|
| 9 | `Transcribe` | Fix callback dict key names | 121-131 |
| 10 | `Transcribe` | Fix `self.callback` → `self.callbacks` typo | 206 |
| 11 | `Transcribe` | Fix `self.callbacks.on_...` → `self.callbacks.get('on_...')` | 167 |
| 12 | `ChatLLM` | Fix method name `character_instruction_message` | 350 |
| 13 | `ChatLLM` | Fix `self.chatllm` → `self` | 366 |
| 14 | `ChatLLM` | Fix method name `wrap_character_tags` | 376 |
| 15 | `ChatLLM` | Add parameters to `process_user_messages` | 354 |
| 16 | `Speech` | Pass `voice` to `generate_audio_for_sentence` | 557 |

### Low Priority (Improvements)

| # | Class | Change | Lines |
|---|-------|--------|-------|
| 17 | `WebSocketManager` | Fix callback parameter names in Transcribe instantiation | 704-708 |

---

## Discussion Points

Before implementing, let's discuss:

1. **User Identity:** The `process_user_messages` method takes `user_name` - where should this come from? Should it be stored in `WebSocketManager`?

2. **LLM Text Streaming:** There's a reference to `self.llm_stream` (line 792) and `on_text_chunk`. Do you want to stream LLM text tokens to the frontend in real-time?

3. **Multiple Connections:** Currently the server uses a single global `ws_manager`. Do you want to support multiple simultaneous WebSocket connections (multiple users)? If so, we'd need a connection ID system.

4. **API Key:** Where should the OpenRouter API key come from? Currently I've assumed an environment variable `OPENROUTER_API_KEY`.

5. **Character Loading:** Should characters be loaded once at startup, or refreshed per-message?

---

## Ready for Implementation

Once you've reviewed this document and we've discussed the points above, I can implement all the changes. Would you like to proceed with the implementation, or do you have questions about any of the findings?
