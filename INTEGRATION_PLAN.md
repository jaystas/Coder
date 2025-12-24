# FastAPI Server & Frontend Integration Plan

## Executive Summary

After thorough analysis of both backend (`backend/fastapi_server.py`) and frontend JavaScript files, I've identified critical message type mismatches and variable naming inconsistencies preventing proper integration. **The backend workflow logic is complete and functional**, but the frontend is unable to receive and display LLM responses due to these mismatches.

---

## Critical Issues Identified

### 1. **MESSAGE TYPE MISMATCH (CRITICAL - Prevents LLM Responses from Displaying)**

**Issue**: Frontend and backend use different message type identifiers.

| Component | Message Type | Data Structure |
|-----------|-------------|----------------|
| **Backend sends** | `llm_chunk` | `{text, character_id}` |
| **Frontend expects** | `text_chunk` | `{text, character_name, message_id, is_final}` |

**Impact**: User messages reach OpenRouter successfully, but responses are never displayed in the chat UI because the frontend doesn't recognize `llm_chunk` messages.

**Location**:
- Backend: `fastapi_server.py:874-880` (sends `llm_chunk`)
- Frontend: `chat.js:83-86` (expects `text_chunk`)

---

### 2. **STT MESSAGE TYPE MISMATCH**

**Issue**: STT completion events use different type identifiers.

| Component | Message Type |
|-----------|-------------|
| **Backend sends** | `stt_finished` |
| **Frontend expects** | `stt_final` |

**Impact**: Transcribed text won't finalize properly in the chat UI (though it still goes to transcribe_queue).

**Location**:
- Backend: `fastapi_server.py:872` (sends `stt_finished`)
- Frontend: `chat.js:79` (expects `stt_final`)

---

### 3. **VARIABLE NAME MISMATCHES**

**Issue**: Multiple variable naming inconsistencies between frontend and backend.

#### Text vs User Message
- **Backend**: Uses `user_message` consistently (correct)
- **Frontend**: Sometimes uses `text` in message payloads
- **Location**: `websocket.js:354` sends `{text}`, backend expects this and converts correctly

#### Character Identification
- **Backend**: Sends `character_id` (UUID string)
- **Frontend**: Expects `character_name` (human-readable name)
- **Missing**: `message_id` not sent with llm_chunk
- **Missing**: `is_final` flag not sent with llm_chunk

**Impact**: Even if message type is fixed, character attribution won't work correctly.

---

### 4. **MISSING RESPONSE METADATA**

The backend doesn't send complete metadata with LLM chunks:

**Backend sends**:
```python
{
    "type": "llm_chunk",
    "text": text_chunk,
    "character_id": character_id
}
```

**Frontend needs**:
```javascript
{
    "type": "text_chunk",
    "data": {
        "text": text_chunk,
        "character_name": character_name,  // ❌ Missing
        "message_id": message_id,          // ❌ Missing
        "is_final": is_final               // ❌ Missing
    }
}
```

**Location**: `fastapi_server.py:874-880`

---

### 5. **AUDIO COMPLETION NOTIFICATION**

**Issue**: Backend sends `audio_complete` messages but frontend doesn't handle them.

**Backend sends**: `fastapi_server.py:919-923`
```python
{
    "type": "audio_complete",
    "message_id": message_id,
    "character_id": character_id
}
```

**Frontend**: No handler in `chat.js` for this message type.

**Impact**: No way to know when audio generation is complete for a message.

---

## Backend Workflow Analysis

### ✅ Current Backend Flow (COMPLETE)

The backend workflow **IS complete** and processes the entire pipeline correctly:

```
1. WebSocket receives message
   ↓
2. handle_text_message() parses JSON
   ↓
3. handle_user_message() → transcribe_queue.put(user_message)
   ↓
4. get_user_messages() background task picks up message
   ↓
5. chat.process_user_messages()
   ├─ Adds to conversation_history
   ├─ Determines responding characters
   └─ For each character:
      ↓
6. stream_character_response()
   ├─ Calls OpenRouter API ✅
   ├─ Streams chunks via generate_sentences_async ✅
   ├─ Sends llm_chunk to frontend ✅ (but wrong format)
   └─ Queues sentences to sentence_queue ✅
      ↓
7. TTS Worker (Speech.process_sentences)
   ├─ Generates audio chunks ✅
   └─ Queues to audio_queue ✅
      ↓
8. stream_audio_loop()
   └─ Streams audio bytes to WebSocket ✅
```

**Verdict**: Backend logic is sound. Issue is purely in the message format sent to frontend.

---

## Detailed Component Analysis

### Backend Components

#### ✅ WebSocket Message Handler (`fastapi_server.py:804-848`)
- **Status**: Working correctly
- **Handles**: `user_message`, `start_listening`, `stop_listening`, `model_settings`, `refresh_characters`, `clear_history`
- **Issue**: None - properly routes all message types

#### ⚠️ LLM Text Chunk Sender (`fastapi_server.py:874-880`)
- **Status**: Sends messages but wrong format
- **Current**:
  ```python
  await self.send_text_to_client({
      "type": "llm_chunk",
      "text": text_chunk,
      "character_id": character_id
  })
  ```
- **Should be**:
  ```python
  await self.send_text_to_client({
      "type": "text_chunk",
      "data": {
          "text": text_chunk,
          "character_name": character.name,
          "message_id": message_id,
          "is_final": False
      }
  })
  ```

#### ⚠️ STT Finished Handler (`fastapi_server.py:870-872`)
- **Status**: Wrong message type
- **Current**: `{"type": "stt_finished", "text": user_message}`
- **Should be**: `{"type": "stt_final", "text": user_message}`

#### ✅ Transcribe Queue Processing (`fastapi_server.py:892-907`)
- **Status**: Working correctly
- **Receives**: Messages from transcribe_queue
- **Calls**: `chat.process_user_messages()` correctly

#### ✅ LLM Processing (`fastapi_server.py:357-398`)
- **Status**: Working correctly
- **Adds message to history**: Line 366-370 ✅
- **Parses character mentions**: Line 373-376 ✅
- **Streams from OpenRouter**: Line 383-397 ✅

#### ⚠️ Character Response Streaming (`fastapi_server.py:399-480`)
- **Status**: Works but missing final chunk notification
- **Sends sentence chunks**: Line 453-461 ✅
- **Sends final sentinel**: Line 469-477 ✅
- **Missing**: No way to notify frontend when streaming is complete for a character

---

### Frontend Components

#### ✅ WebSocket Connection (`websocket.js`)
- **Status**: Working correctly
- **Handles**: Message routing, reconnection, heartbeat
- **Issue**: None

#### ⚠️ Chat Message Handler (`chat.js:69-87`)
- **Status**: Expects wrong message types
- **Expects**: `text_chunk` (backend sends `llm_chunk`)
- **Expects**: `stt_final` (backend sends `stt_finished`)
- **Missing**: `audio_complete` handler

#### ✅ Message Sending (`chat.js:297-302`)
- **Status**: Working correctly
- **Sends**: User messages via WebSocket
- **Adds to UI**: Correctly displays user messages

#### ⚠️ Response Chunk Handler (`chat.js:93-127`)
- **Status**: Never called due to wrong message type
- **Expects**: `data: {text, character_name, message_id, is_final}`
- **Backend provides**: `{text, character_id}` (missing data wrapper)

#### ✅ Editor Send (`editor.js:438-461`)
- **Status**: Working correctly
- **Gets text**: From editor
- **Calls**: `chat.sendMessage(content)`
- **Clears editor**: After sending

---

## Resolution Plan

### Phase 1: Critical Message Type Fixes

#### 1.1 Fix LLM Chunk Message Format (HIGHEST PRIORITY)

**File**: `backend/fastapi_server.py`

**Location**: `on_llm_text_chunk` method (Line 874-880)

**Change**:
```python
# BEFORE
async def on_llm_text_chunk(self, text_chunk: str, character_id: str):
    """Stream LLM text chunks to frontend in real-time"""
    await self.send_text_to_client({
        "type": "llm_chunk",
        "text": text_chunk,
        "character_id": character_id
    })

# AFTER
async def on_llm_text_chunk(self, text_chunk: str, character_id: str, character_name: str, message_id: str):
    """Stream LLM text chunks to frontend in real-time"""
    await self.send_text_to_client({
        "type": "text_chunk",
        "data": {
            "text": text_chunk,
            "character_name": character_name,
            "message_id": message_id,
            "is_final": False
        }
    })
```

**Required**: Update method signature and all callers to pass `character_name` and `message_id`

---

#### 1.2 Update LLM Streaming to Pass Additional Parameters

**File**: `backend/fastapi_server.py`

**Location**: `stream_character_response` method (Line 399-480)

**Change at Line 439-440**:
```python
# BEFORE
if on_text_chunk:
    await on_text_chunk(content, character.id)

# AFTER
if on_text_chunk:
    await on_text_chunk(content, character.id, character.name, message_id)
```

**Change at Line 902** (when calling process_user_messages):
```python
# Signature is already correct, just ensure on_text_chunk is passed
await self.chat.process_user_messages(
    user_message=user_message,
    sentence_queue=self.queues.sentence_queue,
    on_text_chunk=self.on_llm_text_chunk,  # ✅ Already passed
)
```

---

#### 1.3 Send Final Chunk Notification

**File**: `backend/fastapi_server.py`

**Location**: `stream_character_response` method (Line 468-478)

**Add after final sentence sentinel**:
```python
# After line 478
# Send final chunk notification to frontend
if on_text_chunk:
    await on_text_chunk("", character.id, character.name, message_id, is_final=True)
```

**Update on_llm_text_chunk signature**:
```python
async def on_llm_text_chunk(
    self,
    text_chunk: str,
    character_id: str,
    character_name: str,
    message_id: str,
    is_final: bool = False
):
    """Stream LLM text chunks to frontend in real-time"""
    await self.send_text_to_client({
        "type": "text_chunk",
        "data": {
            "text": text_chunk,
            "character_name": character_name,
            "message_id": message_id,
            "is_final": is_final
        }
    })
```

---

#### 1.4 Fix STT Message Type

**File**: `backend/fastapi_server.py`

**Location**: `on_transcription_finished` method (Line 870-872)

**Change**:
```python
# BEFORE
async def on_transcription_finished(self, user_message: str):
    await self.queues.transcribe_queue.put(user_message)
    await self.send_text_to_client({"type": "stt_finished", "text": user_message})

# AFTER
async def on_transcription_finished(self, user_message: str):
    await self.queues.transcribe_queue.put(user_message)
    await self.send_text_to_client({"type": "stt_final", "text": user_message})
```

---

### Phase 2: Frontend Enhancements

#### 2.1 Add Audio Complete Handler

**File**: `frontend/chat.js`

**Location**: `handleServerMessage` function (Line 69-87)

**Add case**:
```javascript
function handleServerMessage(message) {
  switch (message.type) {
    case 'stt_update':
      updateSTTPreview(message.text, false)
      break

    case 'stt_stabilized':
      updateSTTPreview(message.text, true)
      break

    case 'stt_final':
      finalizeUserMessage(message.text)
      break

    case 'text_chunk':
      handleResponseChunk(message.data)
      break

    // NEW: Handle audio completion
    case 'audio_complete':
      handleAudioComplete(message)
      break
  }
}

// NEW: Audio complete handler
function handleAudioComplete(data) {
  const { message_id, character_id } = data
  console.log(`[Chat] Audio complete for message ${message_id}`)
  // Optional: Add visual indicator that audio is ready
}
```

---

### Phase 3: Variable Naming Cleanup

#### 3.1 Standardize on `user_message`

**Files**: All frontend files already use correct variable names when calling backend.

**Verification needed**: Ensure all places where user input is captured use `user_message` consistently.

**No changes required** - Current implementation is correct.

---

### Phase 4: Testing & Validation

#### 4.1 Test Manual Text Input

**Test Steps**:
1. Type text in editor
2. Click Send button
3. **Verify**: Message appears in chat UI
4. **Verify**: Console shows message sent to server
5. **Verify**: LLM response streams into chat UI
6. **Verify**: Audio plays after response completes

**Expected Console Output**:
```
[Editor] Sending message: Hello
[Chat] User message added
[WS] Sent: {"type":"user_message","data":{"text":"Hello"}}
[Chat] Received text_chunk: {"data":{"text":"Hello there!",...}}
[TTS] Queued audio chunk
[TTS] Playing audio
```

---

#### 4.2 Test STT Audio Input (Once Fixed)

**Test Steps**:
1. Click microphone button
2. Grant microphone permissions
3. Speak a message
4. **Verify**: STT preview shows real-time transcription
5. **Verify**: Stabilized text appears
6. **Verify**: Final transcription added to chat
7. **Verify**: LLM response appears
8. **Verify**: Audio plays

---

#### 4.3 Test Multi-Character Responses

**Test Steps**:
1. Send message mentioning multiple characters (e.g., "Alice and Bob, hello")
2. **Verify**: Each character's response appears with correct name
3. **Verify**: Responses appear in order
4. **Verify**: Audio plays for each character sequentially

---

## Implementation Checklist

### Backend Changes (fastapi_server.py)

- [ ] **Line 874-880**: Update `on_llm_text_chunk` signature to accept `character_name`, `message_id`, `is_final`
- [ ] **Line 874-880**: Change message type from `llm_chunk` to `text_chunk`
- [ ] **Line 874-880**: Wrap data in `data` object with all required fields
- [ ] **Line 439-440**: Pass `character.name` and `message_id` to `on_text_chunk` callback
- [ ] **Line 468-478**: Send final chunk notification with `is_final=True`
- [ ] **Line 872**: Change message type from `stt_finished` to `stt_final`
- [ ] **Test**: Manual text input reaches OpenRouter and response displays
- [ ] **Test**: Multiple characters respond correctly
- [ ] **Test**: Audio generation completes for all responses

### Frontend Changes (chat.js)

- [ ] **Line 83-86**: Verify `text_chunk` handler works (no changes needed)
- [ ] **Line 79-81**: Verify `stt_final` handler works (no changes needed)
- [ ] **Add**: `audio_complete` message handler
- [ ] **Test**: LLM responses display correctly with character names
- [ ] **Test**: Final chunk closes message properly
- [ ] **Test**: Audio complete notification received

---

## Questions to Resolve

Before implementation, please clarify:

### 1. Character Name Lookup
**Question**: The backend sends `character_id` but needs to send `character_name`. The `character` object is available in `stream_character_response`, but not in the calling context of `on_llm_text_chunk`.

**Options**:
- A) Pass `character.name` as parameter (requires updating call site)
- B) Store character mapping in WebSocketManager and look up by ID
- C) Change frontend to accept `character_id` and do lookup in characterCache

**Recommendation**: Option A (pass as parameter) - simplest and most explicit.

---

### 2. Message ID Tracking
**Question**: `message_id` is generated in `stream_character_response` but needs to be passed through the callback chain.

**Current**: `on_text_chunk` callback signature is `(text_chunk: str, character_id: str)`

**Needed**: `on_text_chunk` callback signature should be `(text_chunk: str, character_id: str, character_name: str, message_id: str, is_final: bool = False)`

**Confirm**: Is this acceptable?

---

### 3. STT Not Functional
**Question**: You mentioned "STT audio capture is currently not functional."

**Analysis**:
- Frontend has complete STT implementation (stt-audio.js, stt-processor.js)
- Backend has Transcribe class with RealtimeSTT integration
- WebSocket routes audio bytes correctly

**Potential Issues**:
- Browser permissions not granted?
- AudioWorklet processor not loading?
- Backend `use_microphone=False` on line 154?

**Recommendation**:
1. Check browser console for microphone permission errors
2. Verify AudioWorklet loads: Check for `[STT] Audio capture initialized` in console
3. Backend line 154: Should `use_microphone` be `True` or does it use browser audio stream?

---

### 4. Audio Playback Coordination
**Question**: Should TTS audio automatically pause when user starts speaking (STT active)?

**Current**: `stt-audio.js` sets `isTTSPlaying` to pause recording during TTS, and `tts-audio.js` notifies STT when playing.

**This appears correct** - just confirm this is desired behavior.

---

## Summary

### Root Cause
The integration issue is **NOT** a workflow problem - the backend successfully processes messages through the entire pipeline (transcribe → LLM → TTS → audio streaming). The issue is purely **message format mismatches** between frontend and backend.

### Primary Fix Required
Change 6 lines in `backend/fastapi_server.py`:
1. Rename `llm_chunk` → `text_chunk`
2. Add `data` wrapper with `character_name`, `message_id`, `is_final`
3. Rename `stt_finished` → `stt_final`
4. Update method signatures to pass additional parameters

### Impact
Once these changes are made:
- ✅ Manual text input will display LLM responses
- ✅ Character names will appear correctly
- ✅ Message completion will be detected
- ✅ STT final transcriptions will appear
- ✅ Audio will play correctly (already working)

### Estimated Effort
- Backend changes: ~30 minutes
- Frontend changes: ~15 minutes
- Testing: ~30 minutes
- **Total**: ~1.5 hours

---

## Next Steps

1. **Review this plan** - Confirm analysis is correct
2. **Answer questions** - Clarify any ambiguities in "Questions to Resolve" section
3. **Prioritize fixes** - Phase 1 is critical, Phase 2-4 are enhancements
4. **Implement changes** - Start with Phase 1.1-1.4 (backend message format)
5. **Test incrementally** - Verify each fix before moving to next
6. **Debug STT separately** - Once message flow works, focus on audio capture issues

Let me know if you'd like me to implement any of these changes or if you have questions about the analysis!
