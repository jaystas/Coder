"""
STT Service Module for Low-Latency Voice Chat

This module provides an async-compatible Speech-to-Text service that wraps
RealtimeSTT's AudioToTextRecorder for use in a WebSocket-based voice chat
application. Audio is received from a browser client (PCM16 @ 16kHz) and
transcribed in real-time with support for interrupts during TTS playback.

Key Features:
- Real-time transcription with update/stabilized/final callbacks
- Voice Activity Detection (VAD) for interrupt handling
- Async event queues for integration with asyncio-based pipelines
- Clean lifecycle management for multi-turn conversations
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import AsyncIterator, Callable, Optional

import numpy as np

from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)


class TranscriptionEventType(Enum):
    """Types of transcription events emitted by the STT service."""

    # Real-time updates (may change as more audio is processed)
    REALTIME_UPDATE = auto()

    # Stabilized text (less likely to change, good for display)
    REALTIME_STABILIZED = auto()

    # Final transcription (complete utterance after silence detected)
    FINAL = auto()

    # Voice activity detected (user started speaking)
    VAD_START = auto()

    # Voice activity ended (user stopped speaking)
    VAD_STOP = auto()

    # Recording started
    RECORDING_START = auto()

    # Recording stopped
    RECORDING_STOP = auto()

    # Error occurred during transcription
    ERROR = auto()


@dataclass
class TranscriptionEvent:
    """An event emitted by the STT service."""

    event_type: TranscriptionEventType
    text: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class STTService:
    """
    Async-compatible Speech-to-Text service for voice chat applications.

    This service wraps RealtimeSTT's AudioToTextRecorder and provides:
    - Async event queues for transcription results
    - Thread-safe audio feeding from WebSocket handlers
    - VAD-based interrupt detection for barge-in during TTS
    - Clean lifecycle management for multi-turn conversations

    Usage:
        async with STTService() as stt:
            # Feed audio from WebSocket
            stt.feed_audio(audio_chunk)

            # Process transcription events
            async for event in stt.events():
                if event.event_type == TranscriptionEventType.FINAL:
                    # Send to LLM
                    pass
    """

    def __init__(
        self,
        # Model configuration
        model: str = "large-v3-turbo",
        realtime_model: str = "base",
        language: str = "en",
        device: str = "cuda",
        compute_type: str = "float16",

        # Real-time transcription settings
        enable_realtime_transcription: bool = True,
        realtime_processing_pause: float = 0.1,
        init_realtime_after_seconds: float = 0.15,

        # VAD settings
        silero_sensitivity: float = 0.4,
        webrtc_sensitivity: int = 3,
        post_speech_silence_duration: float = 0.5,
        min_length_of_recording: float = 0.3,
        pre_recording_buffer_duration: float = 0.5,

        # Event queue settings
        event_queue_maxsize: int = 100,

        # Additional recorder kwargs
        **recorder_kwargs
    ):
        """
        Initialize the STT service.

        Args:
            model: Main transcription model (used for final transcription)
            realtime_model: Model for real-time transcription (faster, less accurate)
            language: Language code for transcription
            device: Device to run models on ("cuda" or "cpu")
            compute_type: Compute type for models
            enable_realtime_transcription: Enable real-time transcription updates
            realtime_processing_pause: Pause between real-time transcriptions
            init_realtime_after_seconds: Delay before starting realtime transcription
            silero_sensitivity: Silero VAD sensitivity (0-1, higher = more sensitive)
            webrtc_sensitivity: WebRTC VAD sensitivity (0-3, higher = less sensitive)
            post_speech_silence_duration: Silence duration to end recording
            min_length_of_recording: Minimum recording length
            pre_recording_buffer_duration: Audio buffer before VAD triggers
            event_queue_maxsize: Maximum size of the event queue
            **recorder_kwargs: Additional kwargs passed to AudioToTextRecorder
        """
        self._model = model
        self._realtime_model = realtime_model
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._enable_realtime = enable_realtime_transcription
        self._realtime_pause = realtime_processing_pause
        self._realtime_init_delay = init_realtime_after_seconds
        self._silero_sensitivity = silero_sensitivity
        self._webrtc_sensitivity = webrtc_sensitivity
        self._post_speech_silence = post_speech_silence_duration
        self._min_recording_length = min_length_of_recording
        self._pre_recording_buffer = pre_recording_buffer_duration
        self._recorder_kwargs = recorder_kwargs

        # Event queue for async consumers
        self._event_queue: asyncio.Queue[TranscriptionEvent] = asyncio.Queue(
            maxsize=event_queue_maxsize
        )

        # Asyncio event loop reference (set during start)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # The underlying recorder
        self._recorder: Optional[AudioToTextRecorder] = None

        # State tracking
        self._is_running = False
        self._is_listening = False
        self._is_recording = False

        # Interrupt detection state
        self._interrupt_detected = False
        self._tts_playing = False

        # Lock for thread-safe state access
        self._state_lock = threading.Lock()

        # Transcription task
        self._transcription_task: Optional[asyncio.Task] = None

    def _emit_event(self, event: TranscriptionEvent) -> None:
        """
        Emit an event to the async queue (thread-safe).

        Called from AudioToTextRecorder callbacks which run in separate threads.
        """
        if self._loop is None:
            return

        try:
            self._loop.call_soon_threadsafe(
                self._event_queue.put_nowait, event
            )
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event: %s", event.event_type)

    # -------------------------------------------------------------------------
    # Callback handlers (called by AudioToTextRecorder from background threads)
    # -------------------------------------------------------------------------

    def _on_realtime_update(self, text: str) -> None:
        """Handle real-time transcription updates."""
        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.REALTIME_UPDATE,
            text=text
        ))

    def _on_realtime_stabilized(self, text: str) -> None:
        """Handle stabilized real-time transcription."""
        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.REALTIME_STABILIZED,
            text=text
        ))

    def _on_vad_start(self) -> None:
        """Handle voice activity detection start."""
        with self._state_lock:
            # Check for interrupt during TTS playback
            if self._tts_playing:
                self._interrupt_detected = True
                logger.info("Interrupt detected: User started speaking during TTS")

        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.VAD_START,
            metadata={"interrupt": self._interrupt_detected}
        ))

    def _on_vad_stop(self) -> None:
        """Handle voice activity detection stop."""
        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.VAD_STOP
        ))

    def _on_recording_start(self) -> None:
        """Handle recording start."""
        with self._state_lock:
            self._is_recording = True

        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.RECORDING_START
        ))

    def _on_recording_stop(self) -> None:
        """Handle recording stop."""
        with self._state_lock:
            self._is_recording = False

        self._emit_event(TranscriptionEvent(
            event_type=TranscriptionEventType.RECORDING_STOP
        ))

    # -------------------------------------------------------------------------
    # Recorder initialization and lifecycle
    # -------------------------------------------------------------------------

    def _create_recorder(self) -> AudioToTextRecorder:
        """Create and configure the AudioToTextRecorder instance."""
        return AudioToTextRecorder(
            # Model configuration
            model=self._model,
            realtime_model_type=self._realtime_model,
            language=self._language,
            device=self._device,
            compute_type=self._compute_type,

            # Disable microphone - we feed audio via WebSocket
            use_microphone=False,

            # Real-time transcription
            enable_realtime_transcription=self._enable_realtime,
            realtime_processing_pause=self._realtime_pause,
            init_realtime_after_seconds=self._realtime_init_delay,
            on_realtime_transcription_update=self._on_realtime_update,
            on_realtime_transcription_stabilized=self._on_realtime_stabilized,

            # VAD configuration
            silero_sensitivity=self._silero_sensitivity,
            webrtc_sensitivity=self._webrtc_sensitivity,
            post_speech_silence_duration=self._post_speech_silence,
            min_length_of_recording=self._min_recording_length,
            pre_recording_buffer_duration=self._pre_recording_buffer,

            # VAD callbacks
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,

            # Recording callbacks
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,

            # Disable spinner (server-side)
            spinner=False,

            # Additional settings
            no_log_file=True,

            # Pass through any additional kwargs
            **self._recorder_kwargs
        )

    async def start(self) -> None:
        """
        Start the STT service.

        Initializes the AudioToTextRecorder and starts the transcription loop.
        """
        if self._is_running:
            logger.warning("STT service already running")
            return

        self._loop = asyncio.get_running_loop()

        # Create recorder in thread pool to avoid blocking
        logger.info("Initializing STT service with model: %s", self._model)
        self._recorder = await self._loop.run_in_executor(
            None, self._create_recorder
        )

        self._is_running = True
        logger.info("STT service started successfully")

    async def stop(self) -> None:
        """
        Stop the STT service.

        Shuts down the recorder and cleans up resources.
        """
        if not self._is_running:
            return

        self._is_running = False
        self._is_listening = False

        if self._transcription_task and not self._transcription_task.done():
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        if self._recorder:
            await self._loop.run_in_executor(None, self._recorder.shutdown)
            self._recorder = None

        logger.info("STT service stopped")

    async def __aenter__(self) -> "STTService":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()

    # -------------------------------------------------------------------------
    # Audio feeding and listening control
    # -------------------------------------------------------------------------

    def feed_audio(self, chunk: bytes, sample_rate: int = 16000) -> None:
        """
        Feed an audio chunk to the STT service.

        This method is thread-safe and can be called from WebSocket handlers.
        Audio should be PCM16 format.

        Args:
            chunk: Raw audio bytes (PCM16 format)
            sample_rate: Sample rate of the audio (default 16kHz)
        """
        if not self._is_running or not self._recorder:
            return

        # Convert bytes to numpy array if needed
        if isinstance(chunk, bytes):
            audio_data = np.frombuffer(chunk, dtype=np.int16)
        else:
            audio_data = chunk

        self._recorder.feed_audio(audio_data, original_sample_rate=sample_rate)

    def start_listening(self) -> None:
        """
        Start listening for voice activity.

        Puts the recorder in listening mode, waiting for VAD to trigger recording.
        Call this when the user activates the microphone.
        """
        if not self._is_running or not self._recorder:
            logger.warning("Cannot start listening: STT service not running")
            return

        with self._state_lock:
            self._is_listening = True
            self._interrupt_detected = False

        self._recorder.listen()
        logger.debug("STT listening started")

    def stop_listening(self) -> None:
        """
        Stop listening for voice activity.

        Stops the current recording if any and clears the audio buffer.
        """
        if not self._is_running or not self._recorder:
            return

        with self._state_lock:
            self._is_listening = False

        if self._is_recording:
            self._recorder.stop()

        self._recorder.clear_audio_queue()
        logger.debug("STT listening stopped")

    def abort_recording(self) -> None:
        """
        Abort the current recording without transcribing.

        Useful when the user cancels or an interrupt occurs.
        """
        if not self._is_running or not self._recorder:
            return

        self._recorder.abort()
        logger.debug("Recording aborted")

    # -------------------------------------------------------------------------
    # Interrupt handling for barge-in
    # -------------------------------------------------------------------------

    def set_tts_playing(self, playing: bool) -> None:
        """
        Set whether TTS audio is currently playing.

        When TTS is playing and VAD detects voice activity, an interrupt
        is flagged. This allows the system to stop TTS and process the
        user's interruption.

        Args:
            playing: True if TTS is playing, False otherwise
        """
        with self._state_lock:
            self._tts_playing = playing
            if not playing:
                self._interrupt_detected = False

    def is_interrupt_detected(self) -> bool:
        """
        Check if an interrupt was detected.

        Returns True if the user started speaking while TTS was playing.
        """
        with self._state_lock:
            return self._interrupt_detected

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        with self._state_lock:
            self._interrupt_detected = False

    # -------------------------------------------------------------------------
    # Transcription methods
    # -------------------------------------------------------------------------

    async def transcribe_turn(self) -> Optional[str]:
        """
        Wait for and transcribe a complete speech turn.

        This method blocks until the user finishes speaking (detected by
        voice activity ending) and returns the final transcription.

        Returns:
            The final transcription text, or None if aborted/error
        """
        if not self._is_running or not self._recorder:
            return None

        # Start listening if not already
        if not self._is_listening:
            self.start_listening()

        try:
            # Run the blocking text() call in a thread pool
            transcription = await self._loop.run_in_executor(
                None, self._recorder.text
            )

            if transcription:
                # Emit final transcription event
                self._emit_event(TranscriptionEvent(
                    event_type=TranscriptionEventType.FINAL,
                    text=transcription
                ))

            return transcription

        except Exception as e:
            logger.error("Transcription error: %s", e)
            self._emit_event(TranscriptionEvent(
                event_type=TranscriptionEventType.ERROR,
                text=str(e)
            ))
            return None

    async def events(self) -> AsyncIterator[TranscriptionEvent]:
        """
        Async iterator for transcription events.

        Yields events as they occur (real-time updates, VAD changes, etc.)

        Usage:
            async for event in stt.events():
                match event.event_type:
                    case TranscriptionEventType.REALTIME_UPDATE:
                        # Update UI with partial transcription
                        pass
                    case TranscriptionEventType.VAD_START:
                        # User started speaking
                        pass
        """
        while self._is_running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=0.1
                )
                yield event
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    async def get_next_event(self, timeout: Optional[float] = None) -> Optional[TranscriptionEvent]:
        """
        Get the next transcription event.

        Args:
            timeout: Maximum time to wait (None = wait forever)

        Returns:
            The next event, or None if timeout occurs
        """
        try:
            if timeout is not None:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=timeout
                )
            else:
                event = await self._event_queue.get()
            return event
        except asyncio.TimeoutError:
            return None

    # -------------------------------------------------------------------------
    # State queries
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Check if the service is running."""
        return self._is_running

    @property
    def is_listening(self) -> bool:
        """Check if the service is listening for voice activity."""
        with self._state_lock:
            return self._is_listening

    @property
    def is_recording(self) -> bool:
        """Check if voice is being recorded."""
        with self._state_lock:
            return self._is_recording

    @property
    def state(self) -> str:
        """Get the current state of the recorder."""
        if self._recorder:
            return self._recorder.state
        return "inactive"


class STTTranscriptionLoop:
    """
    Manages continuous transcription for multi-turn conversations.

    This class provides a higher-level interface for running continuous
    speech-to-text in a voice chat context, handling turn-taking and
    interrupt detection.

    Usage:
        stt = STTService()
        loop = STTTranscriptionLoop(stt)

        async with stt:
            async for transcription in loop.run():
                # Process each complete user utterance
                response = await llm.generate(transcription)
                await tts.speak(response)
    """

    def __init__(
        self,
        stt_service: STTService,
        on_realtime_update: Optional[Callable[[str], None]] = None,
        on_vad_start: Optional[Callable[[], None]] = None,
        on_vad_stop: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize the transcription loop.

        Args:
            stt_service: The STT service instance
            on_realtime_update: Callback for real-time transcription updates
            on_vad_start: Callback when voice activity starts
            on_vad_stop: Callback when voice activity stops
        """
        self._stt = stt_service
        self._on_realtime_update = on_realtime_update
        self._on_vad_start = on_vad_start
        self._on_vad_stop = on_vad_stop
        self._running = False

    async def run(self) -> AsyncIterator[str]:
        """
        Run the continuous transcription loop.

        Yields final transcriptions as they become available.
        Handles VAD events and real-time updates via callbacks.
        """
        self._running = True

        # Start background task to process events
        event_task = asyncio.create_task(self._process_events())

        try:
            while self._running and self._stt.is_running:
                # Wait for a complete transcription turn
                transcription = await self._stt.transcribe_turn()

                if transcription:
                    yield transcription

        finally:
            self._running = False
            event_task.cancel()
            try:
                await event_task
            except asyncio.CancelledError:
                pass

    async def _process_events(self) -> None:
        """Process events and dispatch to callbacks."""
        try:
            async for event in self._stt.events():
                if not self._running:
                    break

                match event.event_type:
                    case TranscriptionEventType.REALTIME_UPDATE:
                        if self._on_realtime_update:
                            self._on_realtime_update(event.text)

                    case TranscriptionEventType.REALTIME_STABILIZED:
                        # Could use stabilized text for more reliable updates
                        if self._on_realtime_update:
                            self._on_realtime_update(event.text)

                    case TranscriptionEventType.VAD_START:
                        if self._on_vad_start:
                            self._on_vad_start()

                    case TranscriptionEventType.VAD_STOP:
                        if self._on_vad_stop:
                            self._on_vad_stop()

        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Stop the transcription loop."""
        self._running = False
