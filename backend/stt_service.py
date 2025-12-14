"""
Low-latency Speech-to-Text service for WebSocket-based voice chat.

This module provides a streaming STT pipeline using RealtimeSTT with:
- Remote audio feeding via WebSocket (PCM16 @ 16kHz)
- Real-time transcription with update, stabilized, and final callbacks
- Voice Activity Detection (VAD) for automatic recording start/stop
- Interrupt detection capability for TTS playback interruption

Architecture:
    Browser Audio (PCM16@16kHz) → WebSocket → feed_audio() → RealtimeSTT
                                                                ↓
    STT Callbacks ← transcription_loop() ← AudioToTextRecorder

Usage:
    # Initialize the service
    stt = STTService()
    stt.initialize()

    # Set callbacks for transcription events
    stt.set_callbacks(STTCallbacks(
        on_realtime_update=lambda text: print(f"Update: {text}"),
        on_realtime_stabilized=lambda text: print(f"Stabilized: {text}"),
        on_final_transcription=lambda text: print(f"Final: {text}"),
        on_vad_start=lambda: print("Voice detected!"),
    ))

    # Set the event loop for async callback scheduling
    stt.set_event_loop(asyncio.get_event_loop())

    # Start listening when user clicks microphone button
    stt.start_listening()

    # Feed audio chunks from WebSocket
    stt.feed_audio(audio_bytes)

    # Check for interrupts during TTS playback
    if stt.is_voice_detected():
        # User is speaking - interrupt TTS
        tts.stop()

    # Stop listening
    stt.stop_listening()

    # Cleanup
    stt.shutdown()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Union

import numpy as np

from backend.RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Types
# -----------------------------------------------------------------------------


class STTState(Enum):
    """States for the STT service lifecycle."""

    IDLE = auto()          # Not listening, waiting for start_listening()
    LISTENING = auto()     # Waiting for voice activity (VAD)
    RECORDING = auto()     # Voice detected, recording in progress
    TRANSCRIBING = auto()  # Recording complete, transcription in progress
    INTERRUPTED = auto()   # Interrupted by user/system
    ERROR = auto()         # Error state


@dataclass
class STTCallbacks:
    """
    Callback functions for STT events.

    All callbacks are optional. When set, they will be invoked on the
    appropriate events. Callbacks can be either sync or async functions.

    Attributes:
        on_realtime_update: Called frequently with partial transcription.
            Useful for showing live "typing" text.
        on_realtime_stabilized: Called when transcription stabilizes.
            More accurate but slightly delayed vs update.
        on_final_transcription: Called with the final, complete transcription
            after recording stops and full transcription completes.
        on_vad_start: Called when voice activity is first detected.
            Useful for interrupt detection during TTS playback.
        on_vad_stop: Called when voice activity stops.
        on_recording_start: Called when audio recording begins.
        on_recording_stop: Called when audio recording ends.
        on_state_change: Called when STT state changes.
    """

    on_realtime_update: Optional[Callable[[str], Any]] = None
    on_realtime_stabilized: Optional[Callable[[str], Any]] = None
    on_final_transcription: Optional[Callable[[str], Any]] = None
    on_vad_start: Optional[Callable[[], Any]] = None
    on_vad_stop: Optional[Callable[[], Any]] = None
    on_recording_start: Optional[Callable[[], Any]] = None
    on_recording_stop: Optional[Callable[[], Any]] = None
    on_state_change: Optional[Callable[[STTState, STTState], Any]] = None


@dataclass
class STTConfig:
    """
    Configuration for the STT service.

    Attributes:
        model: Whisper model for final transcription ("tiny", "base", "small", etc.)
        realtime_model: Model for realtime transcription (can be smaller for speed)
        language: Language code (empty for auto-detection)
        sample_rate: Expected input audio sample rate (should be 16000)
        silero_sensitivity: VAD sensitivity (0.0-1.0, higher = more sensitive)
        webrtc_sensitivity: WebRTC VAD aggressiveness (0-3, higher = less sensitive)
        post_speech_silence_duration: Seconds of silence before ending recording
        min_recording_length: Minimum recording duration in seconds
        realtime_processing_pause: Pause between realtime transcription updates
    """

    model: str = "small.en"
    realtime_model: str = "small.en"
    language: str = "en"
    sample_rate: int = 16000
    silero_sensitivity: float = 0.4
    webrtc_sensitivity: int = 3
    post_speech_silence_duration: float = 0.6
    min_recording_length: float = 0.5
    realtime_processing_pause: float = 0.1
    device: str = "cuda"
    compute_type: str = "float16"


# -----------------------------------------------------------------------------
# STT Service
# -----------------------------------------------------------------------------


class STTService:
    """
    Speech-to-Text service using RealtimeSTT for WebSocket-based voice chat.

    This service handles:
    - Audio feeding from WebSocket (PCM16 @ 16kHz)
    - Real-time transcription with streaming callbacks
    - Voice Activity Detection (VAD) for automatic recording
    - State management for the listening lifecycle
    - Async/sync callback bridging for FastAPI compatibility

    Thread Safety:
        - feed_audio() is thread-safe and can be called from any thread
        - Callbacks are scheduled on the event loop if set
        - State changes are protected by locks
    """

    def __init__(self, config: Optional[STTConfig] = None):
        """
        Initialize the STT service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or STTConfig()

        # Core components
        self._recorder: Optional[AudioToTextRecorder] = None
        self._transcription_thread: Optional[threading.Thread] = None

        # State management
        self._state = STTState.IDLE
        self._state_lock = threading.Lock()
        self._is_listening = threading.Event()
        self._should_stop = threading.Event()

        # VAD state for interrupt detection
        self._vad_active = threading.Event()
        self._last_vad_start_time: float = 0
        self._last_vad_stop_time: float = 0

        # Callbacks
        self._callbacks: STTCallbacks = STTCallbacks()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialization flag
        self._initialized = False

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Initialize the STT recorder and start the transcription thread.

        This should be called once during application startup.

        Raises:
            RuntimeError: If initialization fails.
        """
        if self._initialized:
            logger.warning("STT service already initialized")
            return

        logger.info("Initializing STT service...")

        try:
            self._recorder = AudioToTextRecorder(
                # Model configuration
                model=self.config.model,
                realtime_model_type=self.config.realtime_model,
                language=self.config.language,
                device=self.config.device,
                compute_type=self.config.compute_type,

                # Audio configuration
                use_microphone=False,  # We feed audio via WebSocket
                sample_rate=self.config.sample_rate,

                # Realtime transcription
                enable_realtime_transcription=True,
                realtime_processing_pause=self.config.realtime_processing_pause,
                on_realtime_transcription_update=self._on_realtime_update,
                on_realtime_transcription_stabilized=self._on_realtime_stabilized,

                # VAD configuration
                silero_sensitivity=self.config.silero_sensitivity,
                webrtc_sensitivity=self.config.webrtc_sensitivity,
                post_speech_silence_duration=self.config.post_speech_silence_duration,
                min_length_of_recording=self.config.min_recording_length,

                # VAD callbacks
                on_vad_detect_start=self._on_vad_detect_start,
                on_vad_detect_stop=self._on_vad_detect_stop,

                # Recording callbacks
                on_recording_start=self._on_recording_start,
                on_recording_stop=self._on_recording_stop,

                # Other settings
                spinner=False,
                level=logging.WARNING,
                no_log_file=True,
            )

            # Start the transcription loop thread
            self._transcription_thread = threading.Thread(
                target=self._transcription_loop,
                name="STT-Transcription",
                daemon=True,
            )
            self._transcription_thread.start()

            self._initialized = True
            logger.info("STT service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize STT service: {e}")
            raise RuntimeError(f"STT initialization failed: {e}") from e

    def shutdown(self) -> None:
        """
        Shutdown the STT service and cleanup resources.

        This should be called during application shutdown.
        """
        logger.info("Shutting down STT service...")

        # Signal threads to stop
        self._should_stop.set()
        self._is_listening.set()  # Unblock the transcription loop

        # Wait for transcription thread
        if self._transcription_thread and self._transcription_thread.is_alive():
            self._transcription_thread.join(timeout=5.0)
            if self._transcription_thread.is_alive():
                logger.warning("Transcription thread did not stop in time")

        # Shutdown the recorder
        if self._recorder:
            try:
                self._recorder.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down recorder: {e}")

        self._initialized = False
        self._set_state(STTState.IDLE)
        logger.info("STT service shut down")

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_callbacks(self, callbacks: STTCallbacks) -> None:
        """
        Set callback functions for STT events.

        Args:
            callbacks: STTCallbacks instance with desired callbacks set.
        """
        self._callbacks = callbacks

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Set the event loop for scheduling async callbacks.

        This should be called from the main async context (FastAPI).

        Args:
            loop: The asyncio event loop to use for callbacks.
        """
        self._event_loop = loop

    # -------------------------------------------------------------------------
    # Listening Control
    # -------------------------------------------------------------------------

    def start_listening(self) -> None:
        """
        Start listening for voice activity.

        This puts the service in LISTENING state, waiting for voice
        activity to begin recording. Call this when the user clicks
        the microphone button.
        """
        if not self._initialized:
            logger.error("Cannot start listening: STT service not initialized")
            return

        if self._is_listening.is_set():
            logger.debug("Already listening")
            return

        logger.info("Starting STT listening")
        self._set_state(STTState.LISTENING)
        self._is_listening.set()

        # Put recorder in listening mode (waits for VAD to trigger recording)
        if self._recorder:
            self._recorder.listen()

    def stop_listening(self) -> None:
        """
        Stop listening and abort any in-progress recording.

        Call this when the user clicks the microphone button again
        to stop, or when the conversation ends.
        """
        if not self._is_listening.is_set():
            logger.debug("Not currently listening")
            return

        logger.info("Stopping STT listening")

        # Abort any in-progress recording
        if self._recorder:
            self._recorder.abort()

        self._is_listening.clear()
        self._vad_active.clear()
        self._set_state(STTState.IDLE)

    def abort(self) -> None:
        """
        Abort the current recording/transcription.

        Use this to cancel an in-progress recording without
        producing a transcription.
        """
        if self._recorder:
            self._recorder.abort()
        self._set_state(STTState.INTERRUPTED)

    # -------------------------------------------------------------------------
    # Audio Feeding
    # -------------------------------------------------------------------------

    def feed_audio(self, audio_data: bytes, sample_rate: int = 16000) -> None:
        """
        Feed raw audio data into the STT pipeline.

        This method is thread-safe and should be called for each
        audio chunk received from the WebSocket.

        Args:
            audio_data: Raw PCM16 audio bytes (little-endian, mono)
            sample_rate: Sample rate of the audio (default 16000)

        Note:
            Audio is buffered internally and processed when enough
            data is accumulated. The recorder handles resampling if
            the sample rate differs from 16kHz.
        """
        if not self._initialized or not self._recorder:
            return

        if not self._is_listening.is_set():
            return

        try:
            self._recorder.feed_audio(audio_data, original_sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Error feeding audio: {e}")

    def feed_audio_numpy(
        self,
        audio_array: np.ndarray,
        sample_rate: int = 16000
    ) -> None:
        """
        Feed audio as a numpy array.

        Args:
            audio_array: Audio samples as numpy array (int16 or float32)
            sample_rate: Sample rate of the audio
        """
        if not self._initialized or not self._recorder:
            return

        if not self._is_listening.is_set():
            return

        try:
            self._recorder.feed_audio(audio_array, original_sample_rate=sample_rate)
        except Exception as e:
            logger.error(f"Error feeding audio array: {e}")

    # -------------------------------------------------------------------------
    # State and VAD Queries
    # -------------------------------------------------------------------------

    @property
    def state(self) -> STTState:
        """Get the current STT state."""
        with self._state_lock:
            return self._state

    @property
    def is_listening(self) -> bool:
        """Check if the service is currently listening."""
        return self._is_listening.is_set()

    @property
    def is_recording(self) -> bool:
        """Check if audio is currently being recorded."""
        return self._state == STTState.RECORDING

    def is_voice_detected(self) -> bool:
        """
        Check if voice activity is currently detected.

        This is useful for interrupt detection: if the user starts
        speaking during TTS playback, you can use this to detect it
        and stop the TTS.

        Returns:
            True if voice activity is currently detected.
        """
        return self._vad_active.is_set()

    def get_vad_timing(self) -> tuple[float, float]:
        """
        Get the timing of the last VAD events.

        Returns:
            Tuple of (last_vad_start_time, last_vad_stop_time)
        """
        return self._last_vad_start_time, self._last_vad_stop_time

    # -------------------------------------------------------------------------
    # Transcription Loop
    # -------------------------------------------------------------------------

    def _transcription_loop(self) -> None:
        """
        Main transcription loop running in a background thread.

        This loop waits for listening to be enabled, then calls
        recorder.text() to get transcriptions. The recorder's text()
        method blocks until a complete utterance is captured and
        transcribed.
        """
        logger.info("STT transcription loop started")

        while not self._should_stop.is_set():
            # Wait until listening is enabled
            self._is_listening.wait(timeout=0.1)

            if self._should_stop.is_set():
                break

            if not self._is_listening.is_set():
                continue

            try:
                # Get transcription (blocks until complete utterance)
                self._set_state(STTState.LISTENING)
                text = self._recorder.text()

                # Check if we were interrupted or stopped
                if self._should_stop.is_set():
                    break

                if not self._is_listening.is_set():
                    continue

                # Process the transcription
                if text and text.strip():
                    logger.info(f"Final transcription: {text}")
                    self._on_final_transcription(text.strip())
                else:
                    logger.debug("Empty transcription received")

            except Exception as e:
                if not self._should_stop.is_set():
                    logger.error(f"Error in transcription loop: {e}")
                    self._set_state(STTState.ERROR)

        logger.info("STT transcription loop stopped")

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def _set_state(self, new_state: STTState) -> None:
        """Set the STT state with proper locking and callback."""
        with self._state_lock:
            old_state = self._state
            if old_state == new_state:
                return
            self._state = new_state

        logger.debug(f"STT state: {old_state.name} → {new_state.name}")

        if self._callbacks.on_state_change:
            self._schedule_callback(
                self._callbacks.on_state_change,
                old_state,
                new_state
            )

    # -------------------------------------------------------------------------
    # Internal Callbacks (called from recorder thread)
    # -------------------------------------------------------------------------

    def _on_realtime_update(self, text: str) -> None:
        """Called when realtime transcription updates."""
        if self._callbacks.on_realtime_update:
            self._schedule_callback(self._callbacks.on_realtime_update, text)

    def _on_realtime_stabilized(self, text: str) -> None:
        """Called when realtime transcription stabilizes."""
        if self._callbacks.on_realtime_stabilized:
            self._schedule_callback(self._callbacks.on_realtime_stabilized, text)

    def _on_final_transcription(self, text: str) -> None:
        """Called when final transcription is ready."""
        self._set_state(STTState.LISTENING)
        if self._callbacks.on_final_transcription:
            self._schedule_callback(self._callbacks.on_final_transcription, text)

    def _on_vad_detect_start(self) -> None:
        """Called when VAD starts detecting (voice activity begins)."""
        self._vad_active.set()
        self._last_vad_start_time = time.time()
        logger.debug("VAD: Voice activity detected")

        if self._callbacks.on_vad_start:
            self._schedule_callback(self._callbacks.on_vad_start)

    def _on_vad_detect_stop(self) -> None:
        """Called when VAD stops detecting (voice activity ends)."""
        self._vad_active.clear()
        self._last_vad_stop_time = time.time()
        logger.debug("VAD: Voice activity stopped")

        if self._callbacks.on_vad_stop:
            self._schedule_callback(self._callbacks.on_vad_stop)

    def _on_recording_start(self) -> None:
        """Called when audio recording starts."""
        self._set_state(STTState.RECORDING)
        logger.debug("Recording started")

        if self._callbacks.on_recording_start:
            self._schedule_callback(self._callbacks.on_recording_start)

    def _on_recording_stop(self) -> None:
        """Called when audio recording stops."""
        self._set_state(STTState.TRANSCRIBING)
        logger.debug("Recording stopped")

        if self._callbacks.on_recording_stop:
            self._schedule_callback(self._callbacks.on_recording_stop)

    # -------------------------------------------------------------------------
    # Callback Scheduling (thread-safe async bridging)
    # -------------------------------------------------------------------------

    def _schedule_callback(self, callback: Callable, *args: Any) -> None:
        """
        Schedule a callback to run on the event loop.

        This bridges sync callbacks from the recorder thread to
        async callbacks in the FastAPI event loop.
        """
        if self._event_loop is None:
            # No event loop set, try to run callback directly
            try:
                if asyncio.iscoroutinefunction(callback):
                    # Can't run async without event loop
                    logger.warning("Cannot run async callback without event loop")
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
            return

        try:
            if asyncio.iscoroutinefunction(callback):
                # Schedule async callback
                self._event_loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(callback(*args))
                )
            else:
                # Schedule sync callback
                self._event_loop.call_soon_threadsafe(callback, *args)
        except Exception as e:
            logger.error(f"Error scheduling callback: {e}")


# -----------------------------------------------------------------------------
# Interrupt Monitor (for TTS Interruption)
# -----------------------------------------------------------------------------


class InterruptMonitor:
    """
    Monitor for detecting user speech during TTS playback.

    This class provides a clean interface for detecting when the user
    starts speaking while TTS audio is playing, enabling "barge-in"
    or interrupt functionality.

    Usage:
        # Create monitor linked to STT service
        monitor = InterruptMonitor(stt_service)

        # Start monitoring when TTS begins
        monitor.start()

        # Check periodically or use callback
        if monitor.is_interrupted:
            tts.stop()

        # Stop monitoring when TTS ends
        monitor.stop()

    Alternative with callback:
        monitor = InterruptMonitor(stt_service)
        monitor.on_interrupt = lambda: tts.stop()
        monitor.start()
    """

    def __init__(
        self,
        stt_service: STTService,
        min_voice_duration_ms: int = 100,
    ):
        """
        Initialize the interrupt monitor.

        Args:
            stt_service: The STT service to monitor for voice activity.
            min_voice_duration_ms: Minimum voice duration (ms) to trigger
                interrupt. Helps filter out noise/false positives.
        """
        self._stt = stt_service
        self._min_duration = min_voice_duration_ms / 1000.0
        self._monitoring = False
        self._interrupted = threading.Event()
        self._start_time: float = 0
        self._monitor_thread: Optional[threading.Thread] = None

        # Callback for interrupt events
        self.on_interrupt: Optional[Callable[[], Any]] = None

    def start(self) -> None:
        """
        Start monitoring for voice activity interrupts.

        Call this when TTS playback begins.
        """
        if self._monitoring:
            return

        self._monitoring = True
        self._interrupted.clear()
        self._start_time = time.time()

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="InterruptMonitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def stop(self) -> None:
        """
        Stop monitoring for voice activity.

        Call this when TTS playback ends.
        """
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt was detected."""
        return self._interrupted.is_set()

    def reset(self) -> None:
        """Reset the interrupt flag for reuse."""
        self._interrupted.clear()

    def _monitor_loop(self) -> None:
        """Background loop checking for voice activity."""
        voice_start_time: Optional[float] = None

        while self._monitoring:
            if self._stt.is_voice_detected():
                if voice_start_time is None:
                    voice_start_time = time.time()
                elif time.time() - voice_start_time >= self._min_duration:
                    # Voice detected for long enough - trigger interrupt
                    self._interrupted.set()
                    logger.info("Interrupt detected: User started speaking")

                    if self.on_interrupt:
                        try:
                            self.on_interrupt()
                        except Exception as e:
                            logger.error(f"Error in interrupt callback: {e}")

                    self._monitoring = False
                    break
            else:
                voice_start_time = None

            time.sleep(0.01)  # 10ms polling


class AsyncInterruptMonitor:
    """
    Async version of InterruptMonitor for use with asyncio.

    Usage:
        monitor = AsyncInterruptMonitor(stt_service)

        async def tts_playback():
            monitor.start()
            try:
                async for chunk in tts_stream:
                    if monitor.is_interrupted:
                        break
                    await play(chunk)
            finally:
                monitor.stop()
    """

    def __init__(
        self,
        stt_service: STTService,
        min_voice_duration_ms: int = 100,
    ):
        self._stt = stt_service
        self._min_duration = min_voice_duration_ms / 1000.0
        self._monitoring = False
        self._interrupted = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        self.on_interrupt: Optional[Callable[[], Any]] = None

    def start(self) -> None:
        """Start monitoring (creates async task)."""
        if self._monitoring:
            return

        self._monitoring = True
        self._interrupted.clear()

        try:
            loop = asyncio.get_running_loop()
            self._monitor_task = loop.create_task(self._monitor_loop())
        except RuntimeError:
            logger.warning("No running event loop for async interrupt monitor")

    def stop(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

    @property
    def is_interrupted(self) -> bool:
        """Check if an interrupt was detected."""
        return self._interrupted.is_set()

    def reset(self) -> None:
        """Reset the interrupt flag."""
        self._interrupted.clear()

    async def wait_for_interrupt(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for an interrupt to occur.

        Args:
            timeout: Maximum seconds to wait (None = forever)

        Returns:
            True if interrupted, False if timeout
        """
        try:
            await asyncio.wait_for(self._interrupted.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def _monitor_loop(self) -> None:
        """Async monitoring loop."""
        voice_start_time: Optional[float] = None

        while self._monitoring:
            if self._stt.is_voice_detected():
                if voice_start_time is None:
                    voice_start_time = time.time()
                elif time.time() - voice_start_time >= self._min_duration:
                    self._interrupted.set()
                    logger.info("Interrupt detected: User started speaking")

                    if self.on_interrupt:
                        try:
                            if asyncio.iscoroutinefunction(self.on_interrupt):
                                await self.on_interrupt()
                            else:
                                self.on_interrupt()
                        except Exception as e:
                            logger.error(f"Error in interrupt callback: {e}")

                    self._monitoring = False
                    break
            else:
                voice_start_time = None

            await asyncio.sleep(0.01)


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------


def create_stt_service(
    model: str = "small.en",
    language: str = "en",
    device: str = "cuda",
    **kwargs: Any,
) -> STTService:
    """
    Create and initialize an STT service with common settings.

    Args:
        model: Whisper model to use
        language: Language code
        device: "cuda" or "cpu"
        **kwargs: Additional STTConfig parameters

    Returns:
        Initialized STTService ready for use
    """
    config = STTConfig(
        model=model,
        realtime_model=model,
        language=language,
        device=device,
        **kwargs,
    )
    service = STTService(config)
    service.initialize()
    return service
