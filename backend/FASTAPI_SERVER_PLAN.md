# FastAPI Server Review & Planning Document (v3 - Simplified)

## Executive Summary

The server implements a voice chat pipeline: **STT (Transcribe) → LLM (ChatLLM) → TTS (Speech)**

**Requirements:**
- Single user: "Jay"
- Single WebSocket connection
- Stream LLM text tokens to frontend in real-time
- Characters loaded at startup, refreshed when changed
- **Zero database calls during response flow** (latency critical!)
- `message_id` generated locally via `uuid.uuid4()` for audio tracking

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      WebSocketManager                            │
│  - Owns all pipeline components                                  │
│  - Routes messages between client and pipeline                   │
│  - Single user: "Jay"                                            │
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

## Simplified `message_id` Approach

**No database calls during response!** Just generate a UUID locally:

```python
message_id = str(uuid.uuid4())  # Instant, no latency
```

The `message_id` is purely for:
1. Tracking which audio chunks belong to which character response
2. Letting frontend know when audio for a response is complete

Database persistence (if needed later) can happen asynchronously after the response is complete.

---

## Updated Dataclasses

### `TTSSentence` - SIMPLIFIED
```python
@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis"""
    text: str
    index: int                    # Sentence index within this message
    message_id: str               # Local UUID for tracking
    character_id: str
    character_name: str
    voice_id: str
    is_final: bool = False        # True = last sentence for this message
```

**Removed:** `session_id`, `conversation_id`, `is_session_complete`

### `AudioChunk` - SIMPLIFIED
```python
@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming"""
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    message_id: str               # Local UUID for tracking
    character_id: str
    is_final: bool = False        # True = last chunk for this message
```

**Removed:** `session_id`, `conversation_id`, `is_session_complete`

### `ModelSettings` - NO CHANGES
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

---

## Class-by-Class Changes

### 1. `PipeQueues` (Lines 91-98) ✅ NO CHANGES NEEDED

```python
class PipeQueues:
    def __init__(self):
        self.transcribe_queue = asyncio.Queue()  # user messages (str)
        self.sentence_queue = asyncio.Queue()    # TTSSentence objects
        self.audio_queue = asyncio.Queue()       # AudioChunk objects
```

---

### 2. `Transcribe` (Lines 106-238) - FIXES REQUIRED

**Issues:**
| Line | Issue | Fix |
|------|-------|-----|
| 121-131 | Callback dict keys mismatch | Use consistent key names |
| 133 | Missing `self.loop` | Add event loop attribute |
| 167 | Wrong syntax | `self.callbacks.on_...` → `self.callbacks.get('...')` |
| 206 | Typo | `self.callback` → `self.callbacks` |

**Updated `__init__`:**
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
    self.loop: Optional[asyncio.AbstractEventLoop] = None  # ADD THIS

    self.recorder = AudioToTextRecorder(...)  # existing code
```

**Add method:**
```python
def set_event_loop(self, loop: asyncio.AbstractEventLoop):
    """Set the asyncio event loop for callback execution"""
    self.loop = loop
```

---

### 3. `ChatLLM` (Lines 245-463) - FIXES REQUIRED

**`__init__` - Minor addition:**
```python
def __init__(self, queues: PipeQueues, api_key: str):
    self.conversation_history: List[Dict] = []
    self.queues = queues
    self.client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    self.model_settings: Optional[ModelSettings] = None
    self.active_characters: List[Character] = []
    # REMOVED: self.conversation_id (not needed for now)
    # REMOVED: self.db (no database calls during response)
```

**Method fixes:**
| Line | Issue | Fix |
|------|-------|-----|
| 350 | Wrong method name | `character_instruction_message` → `create_character_instruction_message` |
| 354 | Missing parameters | Add `sentence_queue`, `on_text_chunk` parameters |
| 366 | Wrong self reference | `self.chatllm.stream_...` → `self.stream_...` |
| 376 | Wrong method name | `wrap_character_tags` → `wrap_with_character_tags` |

**Updated `process_user_messages` - NO DATABASE CALLS:**
```python
async def process_user_messages(
    self,
    user_message: str,
    sentence_queue: asyncio.Queue[TTSSentence],
    on_text_chunk: Optional[Callable[[str, str], Awaitable[None]]] = None,
):
    """Process user message and generate character responses"""

    # Add to local conversation history (no database call)
    self.conversation_history.append({
        "role": "user",
        "name": "Jay",
        "content": user_message
    })

    # Determine which characters respond
    responding_characters = self.parse_character_mentions(
        message=user_message,
        active_characters=self.active_characters
    )

    for i, character in enumerate(responding_characters):
        is_last = (i == len(responding_characters) - 1)

        # Generate message_id locally - INSTANT, no latency!
        message_id = str(uuid.uuid4())

        messages = self.build_messages_for_character(character)

        full_response = await self.stream_character_response(
            messages=messages,
            character=character,
            message_id=message_id,
            model_settings=self.get_model_settings(),
            sentence_queue=sentence_queue,
            on_text_chunk=on_text_chunk,
        )

        response_wrapped = self.wrap_with_character_tags(full_response, character.name)
        self.conversation_history.append({
            "role": "assistant",
            "name": character.name,
            "content": response_wrapped
        })
```

**Updated `stream_character_response`:**
```python
async def stream_character_response(
    self,
    messages: List[Dict[str, str]],
    character: Character,
    message_id: str,
    model_settings: ModelSettings,
    sentence_queue: asyncio.Queue[TTSSentence],
    on_text_chunk: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> str:
    """Stream LLM response, extract sentences, queue for TTS."""

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
            stream=True,
            extra_body={
                "top_k": model_settings.top_k,
                "min_p": model_settings.min_p,
                "repetition_penalty": model_settings.repetition_penalty,
            }
        )

        async def chunk_generator() -> AsyncGenerator[str, None]:
            nonlocal full_response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        if on_text_chunk:
                            await on_text_chunk(content, character.id)
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
                    message_id=message_id,
                    character_id=character.id,
                    character_name=character.name,
                    voice_id=character.voice,
                    is_final=False,
                ))
                sentence_index += 1

    except Exception as e:
        logger.error(f"[LLM] Error streaming for {character.name}: {e}")

    # Final sentinel
    await sentence_queue.put(TTSSentence(
        text="",
        index=sentence_index,
        message_id=message_id,
        character_id=character.id,
        character_name=character.name,
        voice_id=character.voice,
        is_final=True,
    ))

    return full_response
```

---

### 4. `Speech` (Lines 482-693) - MINOR FIXES

**`__init__` is fine.**

**Fix in `process_sentences` (line 557):**
```python
# Current (missing voice parameter):
async for pcm_bytes in self.generate_audio_for_sentence(sentence.text):

# Fixed:
async for pcm_bytes in self.generate_audio_for_sentence(sentence.text, sentence.voice_id):
```

**Update AudioChunk creation to use simplified dataclass:**
```python
audio_chunk = AudioChunk(
    audio_bytes=pcm_bytes,
    sentence_index=sentence.index,
    chunk_index=chunk_index,
    message_id=sentence.message_id,
    character_id=sentence.character_id,
    is_final=False,
)
```

**Update sentinel passthrough:**
```python
if sentence.is_final:
    await self.queues.audio_queue.put(AudioChunk(
        audio_bytes=b"",
        sentence_index=sentence.index,
        chunk_index=0,
        message_id=sentence.message_id,
        character_id=sentence.character_id,
        is_final=True,
    ))
```

---

### 5. `WebSocketManager` (Lines 699-795) - MAJOR REWRITE

**Complete `__init__`:**
```python
def __init__(self):
    # Pipeline queues - shared between all components
    self.queues = PipeQueues()

    # WebSocket connection
    self.websocket: Optional[WebSocket] = None

    # Pipeline components (initialized in initialize())
    self.transcribe: Optional[Transcribe] = None
    self.chat: Optional[ChatLLM] = None
    self.speech: Optional[Speech] = None

    # Background tasks
    self.consumer_task: Optional[asyncio.Task] = None
    self.audio_streamer_task: Optional[asyncio.Task] = None

    # User info
    self.user_name = "Jay"
```

**Add `initialize()` method:**
```python
async def initialize(self):
    """Initialize all pipeline components at startup"""
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

    logger.info(f"Initialized with {len(self.chat.active_characters)} active characters")
```

**Add `connect()` method:**
```python
async def connect(self, websocket: WebSocket):
    """Accept WebSocket connection and start pipeline"""
    await websocket.accept()
    self.websocket = websocket

    # Start TTS worker
    await self.speech.start()

    # Start pipeline consumer
    await self.start_pipeline()

    # Start audio streaming to client
    self.audio_streamer_task = asyncio.create_task(self.stream_audio_loop())

    logger.info("WebSocket connected, pipeline started")
```

**Add `disconnect()` method:**
```python
async def disconnect(self):
    """Clean up on disconnect"""
    if self.transcribe:
        self.transcribe.stop_listening()

    if self.speech:
        await self.speech.stop()

    if self.consumer_task:
        self.consumer_task.cancel()
        try:
            await self.consumer_task
        except asyncio.CancelledError:
            pass

    if self.audio_streamer_task:
        self.audio_streamer_task.cancel()
        try:
            await self.audio_streamer_task
        except asyncio.CancelledError:
            pass

    self.websocket = None
    logger.info("WebSocket disconnected, pipeline stopped")
```

**Add `shutdown()` method:**
```python
async def shutdown(self):
    """Shutdown all services"""
    await self.disconnect()
    logger.info("All services shut down")
```

**Add `stream_audio_loop()` method:**
```python
async def stream_audio_loop(self):
    """Background task: stream audio chunks to WebSocket client"""
    while True:
        try:
            chunk: AudioChunk = await self.queues.audio_queue.get()

            if chunk.audio_bytes:
                await self.websocket.send_bytes(chunk.audio_bytes)

            if chunk.is_final:
                await self.send_text_to_client({
                    "type": "audio_complete",
                    "message_id": chunk.message_id,
                    "character_id": chunk.character_id
                })

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error streaming audio: {e}")
```

**Add `refresh_active_characters()` method:**
```python
async def refresh_active_characters(self):
    """Refresh active characters from database (call when characters change)"""
    if self.chat:
        self.chat.active_characters = await self.chat.get_active_characters()
        logger.info(f"Refreshed to {len(self.chat.active_characters)} active characters")
```

**Update `get_user_messages()`:**
```python
async def get_user_messages(self):
    """Background task: get user message from transcribe queue and process."""
    while True:
        try:
            user_message: str = await self.queues.transcribe_queue.get()

            if user_message and user_message.strip():
                await self.chat.process_user_messages(
                    user_message=user_message,
                    sentence_queue=self.queues.sentence_queue,
                    on_text_chunk=self.on_llm_text_chunk,
                )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
```

**Add `on_llm_text_chunk()` for real-time text streaming:**
```python
async def on_llm_text_chunk(self, text_chunk: str, character_id: str):
    """Stream LLM text chunks to frontend in real-time"""
    await self.send_text_to_client({
        "type": "llm_chunk",
        "text": text_chunk,
        "character_id": character_id
    })
```

**Update `handle_text_message()` to handle character changes:**
```python
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

        elif message_type == "refresh_characters":
            await self.refresh_active_characters()

        elif message_type == "clear_history":
            if self.chat:
                self.chat.clear_conversation_history()
            await self.send_text_to_client({"type": "history_cleared"})

    except Exception as e:
        logger.error(f"Error handling message: {e}", exc_info=True)
```

---

## Complete End-to-End Flow (No Database Calls!)

### 1. User Speaks (STT Path)
```
Browser microphone → WebSocket binary
    → ws_manager.handle_audio_message()
    → transcribe.feed_audio()
    → RealtimeSTT processes audio
    → on_transcription_finished callback
    → queues.transcribe_queue.put(user_message)
```

### 2. User Types (Direct Path)
```
Browser text input → WebSocket JSON {"type": "user_message", "data": {"text": "..."}}
    → ws_manager.handle_text_message()
    → ws_manager.handle_user_message()
    → queues.transcribe_queue.put(user_message)
```

### 3. Message Processing (LLM) - ZERO DB CALLS
```
get_user_messages() loop:
    → queues.transcribe_queue.get()
    → chat.process_user_messages()
        → conversation_history.append(user_message)  # Local only
        → For each character:
            → message_id = uuid.uuid4()  # INSTANT!
            → stream_character_response()
                → OpenRouter streaming API
                → For each text chunk:
                    → on_text_chunk() → WebSocket {"type": "llm_chunk"}
                → For each sentence:
                    → sentence_queue.put(TTSSentence with message_id)
                → sentence_queue.put(final sentinel)
```

### 4. Text-to-Speech (TTS)
```
speech.process_sentences() loop:
    → sentence_queue.get()
    → If is_final: pass through sentinel to audio_queue
    → Else: generate_audio_for_sentence()
        → Higgs Audio streaming
        → For each PCM chunk:
            → audio_queue.put(AudioChunk with message_id)
```

### 5. Audio Streaming to Client
```
stream_audio_loop():
    → audio_queue.get()
    → websocket.send_bytes(audio)
    → If is_final:
        → websocket.send_text({"type": "audio_complete", "message_id": ...})
```

---

## Summary of Changes

| Component | Change |
|-----------|--------|
| `TTSSentence` | Simplified: removed `session_id`, `conversation_id`, `is_session_complete` |
| `AudioChunk` | Simplified: removed `session_id`, `conversation_id`, `is_session_complete` |
| `Transcribe` | Fix callback keys, add `self.loop`, fix typos |
| `ChatLLM` | Remove DB calls, generate `message_id` locally, fix method names |
| `Speech` | Pass `voice_id` to generator, update dataclass usage |
| `WebSocketManager` | Add all missing methods and attributes |

---

## Files to Modify

1. **backend/fastapi_server.py** - All the changes above

---

## Ready for Implementation

All changes are latency-friendly with zero database calls during the response flow. Ready to implement when you approve!
