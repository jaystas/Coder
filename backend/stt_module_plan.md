# Real-Time Speech-to-Text Module Implementation Plan

## Overview

This plan describes the implementation of a real-time STT module using RealtimeSTT for automatic audio recording and transcription with voice activity detection (VAD). The module will integrate with the existing WebSocket infrastructure to stream live transcription updates to the frontend chat interface.

---

## 1. Architecture Overview

### System Flow
```
Browser Mic Button Press
        ↓
Frontend: Start Audio Capture (PCM16 @ 16kHz)
        ↓ WebSocket (binary audio chunks)
Backend: STTModule.feed_audio(chunk)
        ↓
AudioToTextRecorder (VAD processing)
        ↓
┌─────────────────────────────────────────────────────┐
│              State Machine Flow                      │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐  │
│  │ LISTENING│───→│ RECORDING │───→│ TRANSCRIBING │  │
│  └──────────┘    └───────────┘    └──────────────┘  │
│       ↑               │                   │         │
│       │    (silence)  │    (VAD stop)     │         │
│       └───────────────┴───────────────────┘         │
└─────────────────────────────────────────────────────┘
        ↓
Callbacks → WebSocket → Frontend UI
```

### Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        STTModule                                 │
├─────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────┐  ┌─────────────────────────────────────┐│
│ │ AudioToTextRecorder │  │         Callback Handlers           ││
│ │  (use_microphone=   │  │  • on_realtime_transcription_update ││
│ │       False)        │  │  • on_realtime_transcription_stable ││
│ │                     │  │  • on_recording_start               ││
│ │  • VAD (WebRTC +    │  │  • on_recording_stop                ││
│ │    Silero)          │  │  • on_vad_start / on_vad_stop       ││
│ │  • Real-time        │  │  • on_transcription_finished        ││
│ │    transcription    │  │                                     ││
│ │  • Final            │  └─────────────────────────────────────┘│
│ │    transcription    │                                         │
│ └─────────────────────┘                                         │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                  TranscriptionLoop                           │ │
│ │  • Runs in background thread                                 │ │
│ │  • Calls recorder.text(on_transcription_finished)            │ │
│ │  • Manages session lifecycle                                 │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │                  WebSocket Bridge                            │ │
│ │  • Streams transcription updates to client                   │ │
│ │  • Receives audio chunks from client                         │ │
│ │  • Session management                                        │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Implementation

### 2.1 File Structure

```
backend/
├── stt_module.py              # Main STT module (NEW)
├── streaming_server.py        # Add STT WebSocket handlers (MODIFY)
└── RealtimeSTT/
    └── audio_recorder.py      # Existing RealtimeSTT
```

### 2.2 STTModule Class Design

```python
# backend/stt_module.py

from dataclasses import dataclass
from typing import Callable, Optional
from enum import Enum
import asyncio
import threading
import logging

logger = logging.getLogger(__name__)


class STTState(Enum):
    """States for the STT state machine"""
    IDLE = "idle"                    # Not listening, waiting for activation
    LISTENING = "listening"          # Listening for voice activity
    RECORDING = "recording"          # Voice detected, recording audio
    TRANSCRIBING = "transcribing"    # Processing final transcription


@dataclass
class TranscriptionResult:
    """Result object for transcription callbacks"""
    text: str
    session_id: str
    is_final: bool
    is_stabilized: bool = False


@dataclass
class STTCallbacks:
    """Callback container for STT events"""
    on_realtime_update: Optional[Callable[[str, str], None]] = None      # (text, session_id)
    on_realtime_stabilized: Optional[Callable[[str, str], None]] = None  # (text, session_id)
    on_transcription_finished: Optional[Callable[[str, str], None]] = None  # (text, session_id)
    on_state_change: Optional[Callable[[STTState, str], None]] = None    # (state, session_id)
    on_vad_start: Optional[Callable[[str], None]] = None                 # (session_id)
    on_vad_stop: Optional[Callable[[str], None]] = None                  # (session_id)
    on_recording_start: Optional[Callable[[str], None]] = None           # (session_id)
    on_recording_stop: Optional[Callable[[str], None]] = None            # (session_id)


class STTModule:
    """
    Real-time Speech-to-Text module using RealtimeSTT.

    Handles:
    - Audio input from WebSocket (PCM16 @ 16kHz)
    - Voice Activity Detection (VAD) for auto start/stop
    - Real-time transcription streaming
    - Final transcription on silence detection
    """

    def __init__(
        self,
        callbacks: STTCallbacks,
        # Transcription model settings
        model: str = "medium.en",
        language: str = "en",
        device: str = "cuda",
        compute_type: str = "float16",
        # Real-time transcription settings
        enable_realtime_transcription: bool = True,
        realtime_model_type: str = "tiny.en",
        realtime_processing_pause: float = 0.1,
        # VAD settings
        silero_sensitivity: float = 0.4,
        webrtc_sensitivity: int = 3,
        post_speech_silence_duration: float = 0.6,
        min_length_of_recording: float = 0.5,
        pre_recording_buffer_duration: float = 1.0,
    ):
        """Initialize STT module with configuration."""
        self.callbacks = callbacks
        self.state = STTState.IDLE
        self.current_session_id: Optional[str] = None
        self.is_active = False
        self._recorder = None
        self._transcription_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Store config for recorder initialization
        self._config = {
            "model": model,
            "language": language,
            "device": device,
            "compute_type": compute_type,
            "enable_realtime_transcription": enable_realtime_transcription,
            "realtime_model_type": realtime_model_type,
            "realtime_processing_pause": realtime_processing_pause,
            "silero_sensitivity": silero_sensitivity,
            "webrtc_sensitivity": webrtc_sensitivity,
            "post_speech_silence_duration": post_speech_silence_duration,
            "min_length_of_recording": min_length_of_recording,
            "pre_recording_buffer_duration": pre_recording_buffer_duration,
        }

    def initialize(self) -> None:
        """Initialize the AudioToTextRecorder with callbacks."""
        from RealtimeSTT import AudioToTextRecorder

        self._recorder = AudioToTextRecorder(
            # Model configuration
            model=self._config["model"],
            language=self._config["language"],
            device=self._config["device"],
            compute_type=self._config["compute_type"],

            # Disable microphone - we'll feed audio via WebSocket
            use_microphone=False,

            # Real-time transcription
            enable_realtime_transcription=self._config["enable_realtime_transcription"],
            realtime_model_type=self._config["realtime_model_type"],
            realtime_processing_pause=self._config["realtime_processing_pause"],
            on_realtime_transcription_update=self._on_realtime_update,
            on_realtime_transcription_stabilized=self._on_realtime_stabilized,

            # VAD configuration
            silero_sensitivity=self._config["silero_sensitivity"],
            webrtc_sensitivity=self._config["webrtc_sensitivity"],
            post_speech_silence_duration=self._config["post_speech_silence_duration"],
            min_length_of_recording=self._config["min_length_of_recording"],
            pre_recording_buffer_duration=self._config["pre_recording_buffer_duration"],

            # State callbacks
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,

            # Additional settings
            spinner=False,
            no_log_file=True,
        )
        logger.info("STTModule initialized with AudioToTextRecorder")

    def start_listening(self, session_id: str) -> None:
        """
        Start the transcription loop for a session.
        Called when user presses the microphone button.
        """
        if self.is_active:
            logger.warning(f"STT already active for session {self.current_session_id}")
            return

        self.current_session_id = session_id
        self.is_active = True
        self._stop_event.clear()
        self._set_state(STTState.LISTENING)

        # Start transcription loop in background thread
        self._transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True
        )
        self._transcription_thread.start()
        logger.info(f"Started listening for session {session_id}")

    def stop_listening(self) -> None:
        """
        Stop the transcription loop.
        Called when user releases the microphone button or ends session.
        """
        if not self.is_active:
            return

        self._stop_event.set()
        self.is_active = False

        # Interrupt any ongoing transcription
        if self._recorder:
            self._recorder.abort()

        self._set_state(STTState.IDLE)
        logger.info(f"Stopped listening for session {self.current_session_id}")
        self.current_session_id = None

    def feed_audio(self, audio_chunk: bytes, sample_rate: int = 16000) -> None:
        """
        Feed audio chunk from WebSocket to the recorder.

        Args:
            audio_chunk: Raw PCM16 audio bytes
            sample_rate: Sample rate of the audio (should be 16000)
        """
        if not self.is_active or not self._recorder:
            return

        self._recorder.feed_audio(audio_chunk, original_sample_rate=sample_rate)

    def shutdown(self) -> None:
        """Clean shutdown of the STT module."""
        self.stop_listening()
        if self._recorder:
            self._recorder.shutdown()
            self._recorder = None
        logger.info("STTModule shut down")

    # ─────────────────────────────────────────────────────────────────
    # Private Methods
    # ─────────────────────────────────────────────────────────────────

    def _transcription_loop(self) -> None:
        """
        Main transcription loop running in background thread.
        Continuously processes audio and yields transcriptions.
        """
        logger.info("Transcription loop started")

        while not self._stop_event.is_set() and self.is_active:
            try:
                # This blocks until transcription is complete (silence detected)
                # The recorder handles VAD internally
                text = self._recorder.text(
                    on_transcription_finished=self._on_transcription_finished
                )

                # If text() returns directly (no callback), handle it here
                if text and not self._stop_event.is_set():
                    self._on_transcription_finished(text)

            except Exception as e:
                logger.error(f"Error in transcription loop: {e}", exc_info=True)
                if self._stop_event.is_set():
                    break

        logger.info("Transcription loop ended")

    def _set_state(self, new_state: STTState) -> None:
        """Update state and notify via callback."""
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state
        logger.debug(f"STT state: {old_state.value} → {new_state.value}")

        if self.callbacks.on_state_change and self.current_session_id:
            self.callbacks.on_state_change(new_state, self.current_session_id)

    # ─────────────────────────────────────────────────────────────────
    # Callback Handlers (called by AudioToTextRecorder)
    # ─────────────────────────────────────────────────────────────────

    def _on_realtime_update(self, text: str) -> None:
        """Called when real-time transcription updates (may be unstable)."""
        if self.callbacks.on_realtime_update and self.current_session_id:
            self.callbacks.on_realtime_update(text, self.current_session_id)

    def _on_realtime_stabilized(self, text: str) -> None:
        """Called when real-time transcription stabilizes (more accurate)."""
        if self.callbacks.on_realtime_stabilized and self.current_session_id:
            self.callbacks.on_realtime_stabilized(text, self.current_session_id)

    def _on_transcription_finished(self, text: str) -> None:
        """Called when final transcription is complete (after silence)."""
        self._set_state(STTState.LISTENING)  # Return to listening

        if self.callbacks.on_transcription_finished and self.current_session_id:
            self.callbacks.on_transcription_finished(text, self.current_session_id)

    def _on_vad_start(self) -> None:
        """Called when voice activity is detected."""
        if self.callbacks.on_vad_start and self.current_session_id:
            self.callbacks.on_vad_start(self.current_session_id)

    def _on_vad_stop(self) -> None:
        """Called when voice activity stops."""
        if self.callbacks.on_vad_stop and self.current_session_id:
            self.callbacks.on_vad_stop(self.current_session_id)

    def _on_recording_start(self) -> None:
        """Called when recording starts (VAD triggered)."""
        self._set_state(STTState.RECORDING)

        if self.callbacks.on_recording_start and self.current_session_id:
            self.callbacks.on_recording_start(self.current_session_id)

    def _on_recording_stop(self) -> None:
        """Called when recording stops (silence detected)."""
        self._set_state(STTState.TRANSCRIBING)

        if self.callbacks.on_recording_stop and self.current_session_id:
            self.callbacks.on_recording_stop(self.current_session_id)
```

---

## 3. WebSocket Integration

### 3.1 Message Protocol

#### Client → Server Messages

```json
// Start listening (mic button pressed)
{
    "type": "stt_start",
    "session_id": "uuid-v4"
}

// Stop listening (mic button released / toggle off)
{
    "type": "stt_stop",
    "session_id": "uuid-v4"
}

// Binary audio data (sent continuously while recording)
// Format: Raw PCM16 bytes at 16kHz mono
// (Binary WebSocket message, not JSON)
```

#### Server → Client Messages

```json
// Listening started acknowledgment
{
    "type": "stt_listening",
    "session_id": "uuid-v4"
}

// Voice activity detected (start of utterance)
{
    "type": "stt_vad_start",
    "session_id": "uuid-v4"
}

// Voice activity stopped (end of utterance)
{
    "type": "stt_vad_stop",
    "session_id": "uuid-v4"
}

// Recording started
{
    "type": "stt_recording_start",
    "session_id": "uuid-v4"
}

// Recording stopped
{
    "type": "stt_recording_stop",
    "session_id": "uuid-v4"
}

// Real-time transcription update (unstable, frequent)
{
    "type": "stt_update",
    "text": "Hello how are",
    "session_id": "uuid-v4"
}

// Stabilized transcription (more accurate, less frequent)
{
    "type": "stt_stabilized",
    "text": "Hello, how are you",
    "session_id": "uuid-v4"
}

// Final transcription (complete utterance)
{
    "type": "stt_final",
    "text": "Hello, how are you doing today?",
    "session_id": "uuid-v4"
}

// State change notification
{
    "type": "stt_state",
    "state": "listening" | "recording" | "transcribing" | "idle",
    "session_id": "uuid-v4"
}

// Error message
{
    "type": "stt_error",
    "error": "Error description",
    "session_id": "uuid-v4"
}
```

### 3.2 WebSocket Handler Integration

Add to `streaming_server.py`:

```python
# Additional imports
from stt_module import STTModule, STTCallbacks, STTState

# Create global STT module instance
stt_module: Optional[STTModule] = None


def create_stt_callbacks(websocket: WebSocket) -> STTCallbacks:
    """Create callbacks that send messages via WebSocket."""

    async def send_json_safe(data: dict):
        """Send JSON message, handling connection errors."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send STT message: {e}")

    def sync_send(data: dict):
        """Synchronous wrapper for async send (for use in callbacks)."""
        asyncio.run_coroutine_threadsafe(
            send_json_safe(data),
            asyncio.get_event_loop()
        )

    return STTCallbacks(
        on_realtime_update=lambda text, sid: sync_send({
            "type": "stt_update",
            "text": text,
            "session_id": sid
        }),
        on_realtime_stabilized=lambda text, sid: sync_send({
            "type": "stt_stabilized",
            "text": text,
            "session_id": sid
        }),
        on_transcription_finished=lambda text, sid: sync_send({
            "type": "stt_final",
            "text": text,
            "session_id": sid
        }),
        on_state_change=lambda state, sid: sync_send({
            "type": "stt_state",
            "state": state.value,
            "session_id": sid
        }),
        on_vad_start=lambda sid: sync_send({
            "type": "stt_vad_start",
            "session_id": sid
        }),
        on_vad_stop=lambda sid: sync_send({
            "type": "stt_vad_stop",
            "session_id": sid
        }),
        on_recording_start=lambda sid: sync_send({
            "type": "stt_recording_start",
            "session_id": sid
        }),
        on_recording_stop=lambda sid: sync_send({
            "type": "stt_recording_stop",
            "session_id": sid
        }),
    )


@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())

    # Initialize STT with WebSocket callbacks
    global stt_module
    if stt_module is None:
        callbacks = create_stt_callbacks(websocket)
        stt_module = STTModule(callbacks=callbacks)
        stt_module.initialize()

    try:
        while True:
            message = await websocket.receive()

            # Handle binary audio data
            if "bytes" in message:
                audio_chunk = message["bytes"]
                stt_module.feed_audio(audio_chunk, sample_rate=16000)
                continue

            # Handle JSON messages
            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "stt_start":
                    stt_module.start_listening(data.get("session_id", session_id))
                    await websocket.send_json({
                        "type": "stt_listening",
                        "session_id": session_id
                    })

                elif msg_type == "stt_stop":
                    stt_module.stop_listening()

                elif msg_type == "prompt":
                    # Handle text prompt (existing logic)
                    # ...
                    pass

                elif msg_type == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
        if stt_module:
            stt_module.stop_listening()

    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
```

---

## 4. Frontend Integration

### 4.1 Modifications to `stt-audio.js`

Update the existing STT audio capture to send to WebSocket:

```javascript
// stt-audio.js additions

class STTAudioCapture {
    constructor(websocket) {
        this.websocket = websocket;
        this.audioContext = null;
        this.mediaStream = null;
        this.workletNode = null;
        this.isCapturing = false;
        this.sessionId = null;
    }

    async start(sessionId) {
        if (this.isCapturing) return;

        this.sessionId = sessionId;

        // Create audio context at 16kHz for STT
        this.audioContext = new AudioContext({ sampleRate: 16000 });

        // Get microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                sampleRate: 16000,
                channelCount: 1
            }
        });

        // Load AudioWorklet processor
        await this.audioContext.audioWorklet.addModule('stt-processor.js');

        // Create source and processor nodes
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        this.workletNode = new AudioWorkletNode(this.audioContext, 'stt-processor');

        // Handle audio data from processor
        this.workletNode.port.onmessage = (event) => {
            if (event.data.type === 'audio') {
                // Convert Float32Array to Int16 PCM
                const float32Data = event.data.audio;
                const int16Data = this._float32ToInt16(float32Data);

                // Send binary audio via WebSocket
                this.websocket.sendAudio(int16Data.buffer);
            }
        };

        // Connect nodes
        source.connect(this.workletNode);

        // Tell server to start listening
        this.websocket.sendText({
            type: 'stt_start',
            session_id: sessionId
        });

        this.isCapturing = true;
    }

    stop() {
        if (!this.isCapturing) return;

        // Tell server to stop
        this.websocket.sendText({
            type: 'stt_stop',
            session_id: this.sessionId
        });

        // Cleanup
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        this.isCapturing = false;
        this.sessionId = null;
    }

    _float32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            const s = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        return int16Array;
    }
}
```

### 4.2 Modifications to `websocket.js`

Add STT message handling:

```javascript
// websocket.js additions

class WebSocketClient {
    // ... existing code ...

    // Add STT event listeners
    onSTTUpdate(callback) {
        this.sttUpdateListeners.push(callback);
    }

    onSTTStabilized(callback) {
        this.sttStabilizedListeners.push(callback);
    }

    onSTTFinal(callback) {
        this.sttFinalListeners.push(callback);
    }

    onSTTStateChange(callback) {
        this.sttStateListeners.push(callback);
    }

    _handleMessage(data) {
        // ... existing message handling ...

        switch (data.type) {
            case 'stt_update':
                this.sttUpdateListeners.forEach(cb => cb(data.text, data.session_id));
                break;
            case 'stt_stabilized':
                this.sttStabilizedListeners.forEach(cb => cb(data.text, data.session_id));
                break;
            case 'stt_final':
                this.sttFinalListeners.forEach(cb => cb(data.text, data.session_id));
                break;
            case 'stt_state':
                this.sttStateListeners.forEach(cb => cb(data.state, data.session_id));
                break;
            case 'stt_vad_start':
                this.emit('vad_start', data.session_id);
                break;
            case 'stt_vad_stop':
                this.emit('vad_stop', data.session_id);
                break;
            // ... other cases ...
        }
    }
}
```

### 4.3 Modifications to `chat.js`

Update UI to display real-time transcription:

```javascript
// chat.js additions

class ChatUI {
    constructor(websocket, sttCapture) {
        this.websocket = websocket;
        this.sttCapture = sttCapture;
        this.currentSTTPreview = null;
        this.isMicActive = false;

        // Listen for STT events
        websocket.onSTTUpdate((text, sessionId) => {
            this.updateSTTPreview(text, false);
        });

        websocket.onSTTStabilized((text, sessionId) => {
            this.updateSTTPreview(text, true);
        });

        websocket.onSTTFinal((text, sessionId) => {
            this.finalizeSTTPreview(text);
        });

        websocket.onSTTStateChange((state, sessionId) => {
            this.updateMicState(state);
        });
    }

    // Toggle microphone button
    toggleMic() {
        if (this.isMicActive) {
            this.sttCapture.stop();
            this.isMicActive = false;
            this.updateMicButton(false);
        } else {
            const sessionId = this.generateSessionId();
            this.sttCapture.start(sessionId);
            this.isMicActive = true;
            this.updateMicButton(true);
            this.createSTTPreview();
        }
    }

    createSTTPreview() {
        // Create a preview message bubble for real-time transcription
        this.currentSTTPreview = document.createElement('div');
        this.currentSTTPreview.className = 'stt-preview message user-message';
        this.currentSTTPreview.innerHTML = `
            <div class="stt-indicator">
                <span class="pulse-dot"></span>
                <span class="stt-text">Listening...</span>
            </div>
        `;
        this.messagesContainer.appendChild(this.currentSTTPreview);
        this.scrollToBottom();
    }

    updateSTTPreview(text, isStabilized) {
        if (!this.currentSTTPreview) return;

        const textElement = this.currentSTTPreview.querySelector('.stt-text');
        textElement.textContent = text || 'Listening...';
        textElement.classList.toggle('stabilized', isStabilized);
    }

    finalizeSTTPreview(text) {
        if (!this.currentSTTPreview) return;

        // Convert preview to final message
        this.currentSTTPreview.className = 'message user-message';
        this.currentSTTPreview.innerHTML = `<p>${text}</p>`;
        this.currentSTTPreview = null;

        // Automatically send as user message
        this.sendUserMessage(text);
    }

    updateMicState(state) {
        const micButton = document.getElementById('mic-button');
        const stateIndicator = document.getElementById('stt-state');

        micButton.dataset.state = state;
        stateIndicator.textContent = state;

        // Update visual indicators
        switch (state) {
            case 'listening':
                micButton.classList.add('listening');
                micButton.classList.remove('recording');
                break;
            case 'recording':
                micButton.classList.remove('listening');
                micButton.classList.add('recording');
                break;
            case 'transcribing':
                micButton.classList.add('processing');
                break;
            default:
                micButton.classList.remove('listening', 'recording', 'processing');
        }
    }
}
```

---

## 5. Configuration Options

### 5.1 Backend Configuration

```python
# config.py or environment variables

STT_CONFIG = {
    # Main transcription model
    "model": "medium.en",              # Options: tiny, base, small, medium, large-v2
    "language": "en",                   # Language code
    "device": "cuda",                   # cuda or cpu
    "compute_type": "float16",          # float16, int8_float16, int8

    # Real-time transcription
    "enable_realtime_transcription": True,
    "realtime_model_type": "tiny.en",   # Smaller model for real-time
    "realtime_processing_pause": 0.1,    # Seconds between updates

    # Voice Activity Detection
    "silero_sensitivity": 0.4,          # 0.0 to 1.0 (higher = more sensitive)
    "webrtc_sensitivity": 3,            # 0 to 3 (lower = more sensitive)
    "post_speech_silence_duration": 0.6, # Seconds of silence to end recording
    "min_length_of_recording": 0.5,     # Minimum recording length
    "pre_recording_buffer_duration": 1.0, # Buffer before VAD trigger
}
```

### 5.2 Frontend Configuration

```javascript
// config.js

const STT_CONFIG = {
    sampleRate: 16000,
    channels: 1,
    bufferSize: 4096,
    audioConstraints: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
    },
    ui: {
        showRealtimePreview: true,
        showStabilizedOnly: false,  // If true, only show stabilized text
        autoSendOnFinal: true,      // Automatically send message when done
    }
};
```

---

## 6. State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STT State Machine                                │
└─────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │       IDLE       │
                    │  (Mic disabled)  │
                    └────────┬─────────┘
                             │
              User clicks    │
              mic button     │
                             ▼
                    ┌──────────────────┐
            ┌──────▶│    LISTENING     │◀─────────────────────┐
            │       │ (Waiting for     │                      │
            │       │  voice activity) │                      │
            │       └────────┬─────────┘                      │
            │                │                                │
            │    VAD start   │ (WebRTC + Silero detect voice) │
            │   callback     │                                │
            │                ▼                                │
            │       ┌──────────────────┐                      │
            │       │    RECORDING     │                      │
            │       │ (Voice detected, │                      │
            │       │  capturing audio)│                      │
            │       └────────┬─────────┘                      │
            │                │                                │
            │                │ During recording:              │
            │                │ • on_realtime_update (unstable)│
            │                │ • on_realtime_stabilized       │
            │                │                                │
            │    VAD stop    │ (Silence detected)             │
            │   callback     │                                │
            │                ▼                                │
            │       ┌──────────────────┐                      │
            │       │  TRANSCRIBING    │                      │
            │       │ (Final whisper   │                      │
            │       │  transcription)  │                      │
            │       └────────┬─────────┘                      │
            │                │                                │
            │                │ on_transcription_finished      │
            │                │ (Final text sent to UI)        │
            │                │                                │
            └────────────────┴────────────────────────────────┘
                             │
              User clicks    │
              mic button     │
              (stop)         │
                             ▼
                    ┌──────────────────┐
                    │       IDLE       │
                    └──────────────────┘
```

---

## 7. Error Handling

### 7.1 Backend Error Handling

```python
class STTModule:
    # ... existing code ...

    def _transcription_loop(self) -> None:
        """Transcription loop with error handling."""
        while not self._stop_event.is_set() and self.is_active:
            try:
                text = self._recorder.text(
                    on_transcription_finished=self._on_transcription_finished
                )
                if text and not self._stop_event.is_set():
                    self._on_transcription_finished(text)

            except KeyboardInterrupt:
                logger.info("Transcription interrupted by user")
                break

            except Exception as e:
                logger.error(f"Transcription error: {e}", exc_info=True)

                # Send error to client
                if self.callbacks.on_error and self.current_session_id:
                    self.callbacks.on_error(str(e), self.current_session_id)

                # Brief pause before retry
                if not self._stop_event.wait(0.5):
                    continue
                break
```

### 7.2 WebSocket Error Messages

```json
{
    "type": "stt_error",
    "error": "Microphone access denied",
    "code": "MIC_ACCESS_DENIED",
    "session_id": "uuid"
}

{
    "type": "stt_error",
    "error": "Transcription model failed to load",
    "code": "MODEL_LOAD_ERROR",
    "session_id": "uuid"
}

{
    "type": "stt_error",
    "error": "Audio processing timeout",
    "code": "PROCESSING_TIMEOUT",
    "session_id": "uuid"
}
```

---

## 8. Performance Considerations

### 8.1 Audio Buffering

- **Browser → Server**: Audio chunks sent every ~50-100ms
- **Buffer size**: 4096 samples at 16kHz ≈ 256ms per chunk
- **WebSocket binary messages**: Efficient for raw PCM data

### 8.2 Model Selection

| Use Case | Main Model | Realtime Model | Notes |
|----------|------------|----------------|-------|
| Fast/Low latency | tiny.en | tiny.en | ~50ms latency |
| Balanced | small.en | tiny.en | Good accuracy/speed |
| High accuracy | medium.en | small.en | More GPU usage |
| Maximum accuracy | large-v2 | base.en | Highest GPU usage |

### 8.3 GPU Memory Usage

```
tiny.en:    ~1GB VRAM
base.en:    ~1GB VRAM
small.en:   ~2GB VRAM
medium.en:  ~5GB VRAM
large-v2:   ~10GB VRAM
```

When using separate models for real-time + final: add memory requirements together.

---

## 9. Testing Strategy

### 9.1 Unit Tests

```python
# tests/test_stt_module.py

import pytest
from stt_module import STTModule, STTCallbacks, STTState

def test_state_transitions():
    """Test STT state machine transitions."""
    states = []
    callbacks = STTCallbacks(
        on_state_change=lambda state, sid: states.append(state)
    )
    stt = STTModule(callbacks)

    # Start listening
    stt._set_state(STTState.LISTENING)
    assert states[-1] == STTState.LISTENING

    # VAD triggered
    stt._set_state(STTState.RECORDING)
    assert states[-1] == STTState.RECORDING

    # Recording stopped
    stt._set_state(STTState.TRANSCRIBING)
    assert states[-1] == STTState.TRANSCRIBING

def test_audio_feeding():
    """Test audio chunk feeding."""
    # Mock recorder
    # ...
```

### 9.2 Integration Tests

```python
# tests/test_stt_websocket.py

import pytest
from fastapi.testclient import TestClient
from streaming_server import app

def test_stt_websocket_flow():
    """Test full STT WebSocket flow."""
    client = TestClient(app)

    with client.websocket_connect("/ws/stream") as websocket:
        # Start STT
        websocket.send_json({"type": "stt_start", "session_id": "test"})
        response = websocket.receive_json()
        assert response["type"] == "stt_listening"

        # Send audio chunk
        audio_chunk = bytes(1024)  # Silence
        websocket.send_bytes(audio_chunk)

        # Stop STT
        websocket.send_json({"type": "stt_stop", "session_id": "test"})
```

---

## 10. Implementation Checklist

### Phase 1: Core Module
- [ ] Create `stt_module.py` with STTModule class
- [ ] Implement STTCallbacks dataclass
- [ ] Implement STTState enum
- [ ] Set up AudioToTextRecorder integration
- [ ] Implement transcription loop
- [ ] Add callback handlers

### Phase 2: WebSocket Integration
- [ ] Add STT WebSocket endpoints to `streaming_server.py`
- [ ] Implement message protocol (stt_start, stt_stop, etc.)
- [ ] Handle binary audio messages
- [ ] Implement callback → WebSocket bridge

### Phase 3: Frontend Integration
- [ ] Update `stt-audio.js` for WebSocket streaming
- [ ] Update `websocket.js` with STT message handlers
- [ ] Update `chat.js` with real-time transcription UI
- [ ] Add mic button toggle functionality
- [ ] Add state indicators (listening/recording/transcribing)

### Phase 4: Testing & Refinement
- [ ] Unit tests for STTModule
- [ ] Integration tests for WebSocket flow
- [ ] End-to-end testing with real audio
- [ ] Performance tuning (model selection, buffer sizes)
- [ ] Error handling edge cases

---

## 11. Dependencies

### Backend

```
# requirements.txt additions
RealtimeSTT>=0.1.15
faster-whisper>=0.9.0
webrtcvad>=2.0.10
torch>=2.0.0
numpy>=1.24.0
```

### Frontend

No additional dependencies - uses native Web Audio API.

---

## 12. Example Usage

### Backend Initialization

```python
from stt_module import STTModule, STTCallbacks

# Create callbacks
callbacks = STTCallbacks(
    on_realtime_update=lambda text, sid: print(f"[LIVE] {text}"),
    on_realtime_stabilized=lambda text, sid: print(f"[STABLE] {text}"),
    on_transcription_finished=lambda text, sid: print(f"[FINAL] {text}"),
    on_state_change=lambda state, sid: print(f"State: {state.value}"),
)

# Initialize module
stt = STTModule(
    callbacks=callbacks,
    model="medium.en",
    enable_realtime_transcription=True,
)
stt.initialize()

# Start listening (triggered by WebSocket message)
stt.start_listening(session_id="user-123")

# Feed audio chunks (from WebSocket binary messages)
stt.feed_audio(audio_chunk, sample_rate=16000)

# Stop listening (triggered by WebSocket message)
stt.stop_listening()

# Cleanup
stt.shutdown()
```

### Frontend Usage

```javascript
// Initialize
const ws = new WebSocketClient('ws://localhost:8000/ws/stream');
const sttCapture = new STTAudioCapture(ws);
const chat = new ChatUI(ws, sttCapture);

// Handle mic button click
document.getElementById('mic-button').onclick = () => {
    chat.toggleMic();
};

// STT events are handled automatically by ChatUI
```
