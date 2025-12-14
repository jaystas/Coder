"""
STT Service Module for Low-Latency Voice Chat

This module provides an async-compatible Speech-to-Text service that wraps
RealtimeSTT's AudioToTextRecorder for use in a WebSocket-based voice chat
application. Audio is received from a browser client (PCM16 @ 16kHz) and
transcribed in real-time with support for interrupts during TTS playback.

Uses RealtimeSTT's native callback system, bridging to asyncio via
run_coroutine_threadsafe for FastAPI/WebSocket compatibility.
"""

import asyncio
import logging
import threading
from typing import Awaitable, Callable, Optional, Union

import numpy as np

from RealtimeSTT import AudioToTextRecorder

logger = logging.getLogger(__name__)

# Type alias for callbacks that can be sync or async
Callback = Callable[..., Optional[Awaitable[None]]]


class STTService:
    """
    Async-compatible Speech-to-Text service for voice chat applications.

    Wraps RealtimeSTT's AudioToTextRecorder and bridges its threaded callbacks
    to asyncio for use with FastAPI WebSocket handlers.

    Usage:
        stt = STTService(
            on_realtime_update=lambda text: broadcast({"type": "partial", "text": text}),
            on_final_transcription=lambda text: llm_queue.put(text),
            on_vad_start=handle_interrupt,
        )

        await stt.start()
        stt.start_listening()  # User clicked mic

        # Audio flows in from WebSocket handler:
        async def handle_audio(data: bytes):
            stt.feed_audio(data)
    """

    def __init__(
        self,
        # Async callbacks - bridged from RealtimeSTT's threads
        on_realtime_update: Optional[Callback] = None,
        on_realtime_stabilized: Optional[Callback] = None,
        on_final_transcription: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,

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

        # Additional recorder kwargs
        **recorder_kwargs
    ):
        """
        Initialize the STT service.

        Args:
            on_realtime_update: Called with partial transcription text (may change)
            on_realtime_stabilized: Called with stabilized text (more reliable)
            on_final_transcription: Called with final transcription after speech ends
            on_vad_start: Called when voice activity detected
            on_vad_stop: Called when voice activity ends
            on_recording_start: Called when recording begins
            on_recording_stop: Called when recording ends

            model: Main transcription model (for final transcription)
            realtime_model: Model for real-time transcription (faster)
            language: Language code for transcription
            device: Device to run models on ("cuda" or "cpu")
            compute_type: Compute type for models

            enable_realtime_transcription: Enable real-time updates
            realtime_processing_pause: Pause between real-time transcriptions
            init_realtime_after_seconds: Delay before starting realtime

            silero_sensitivity: Silero VAD sensitivity (0-1)
            webrtc_sensitivity: WebRTC VAD sensitivity (0-3)
            post_speech_silence_duration: Silence duration to end recording
            min_length_of_recording: Minimum recording length
            pre_recording_buffer_duration: Audio buffer before VAD triggers

            **recorder_kwargs: Additional kwargs for AudioToTextRecorder
        """
        # Store user callbacks
        self._on_realtime_update = on_realtime_update
        self._on_realtime_stabilized = on_realtime_stabilized
        self._on_final_transcription = on_final_transcription
        self._on_vad_start = on_vad_start
        self._on_vad_stop = on_vad_stop
        self._on_recording_start = on_recording_start
        self._on_recording_stop = on_recording_stop

        # Store config
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

        # Runtime state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._recorder: Optional[AudioToTextRecorder] = None
        self._is_running = False
        self._is_listening = False

        # Interrupt detection
        self._tts_playing = False
        self._interrupt_detected = False
        self._state_lock = threading.Lock()

        # Transcription loop task
        self._transcription_task: Optional[asyncio.Task] = None

    # -------------------------------------------------------------------------
    # Callback bridge: RealtimeSTT threads -> asyncio
    # -------------------------------------------------------------------------

    def _invoke_callback(self, callback: Optional[Callback], *args) -> None:
        """
        Invoke a user callback from a RealtimeSTT background thread.

        Handles both sync and async callbacks, bridging to the event loop
        when necessary.
        """
        if callback is None or self._loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            # Async callback - schedule on event loop
            asyncio.run_coroutine_threadsafe(callback(*args), self._loop)
        else:
            # Sync callback - run thread-safe on event loop
            self._loop.call_soon_threadsafe(callback, *args)

    def _handle_realtime_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self._invoke_callback(self._on_realtime_update, text)

    def _handle_realtime_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self._invoke_callback(self._on_realtime_stabilized, text)

    def _handle_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        with self._state_lock:
            if self._tts_playing:
                self._interrupt_detected = True
                logger.info("Interrupt detected: user speaking during TTS")

        self._invoke_callback(self._on_vad_start)

    def _handle_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self._invoke_callback(self._on_vad_stop)

    def _handle_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self._invoke_callback(self._on_recording_start)

    def _handle_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self._invoke_callback(self._on_recording_stop)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def _create_recorder(self) -> AudioToTextRecorder:
        """Create and configure the AudioToTextRecorder."""
        return AudioToTextRecorder(
            # Model config
            model=self._model,
            realtime_model_type=self._realtime_model,
            language=self._language,
            device=self._device,
            compute_type=self._compute_type,

            # No microphone - audio fed via WebSocket
            use_microphone=False,

            # Real-time transcription with our bridged callbacks
            enable_realtime_transcription=self._enable_realtime,
            realtime_processing_pause=self._realtime_pause,
            init_realtime_after_seconds=self._realtime_init_delay,
            on_realtime_transcription_update=self._handle_realtime_update,
            on_realtime_transcription_stabilized=self._handle_realtime_stabilized,

            # VAD config with our bridged callbacks
            silero_sensitivity=self._silero_sensitivity,
            webrtc_sensitivity=self._webrtc_sensitivity,
            post_speech_silence_duration=self._post_speech_silence,
            min_length_of_recording=self._min_recording_length,
            pre_recording_buffer_duration=self._pre_recording_buffer,
            on_vad_start=self._handle_vad_start,
            on_vad_stop=self._handle_vad_stop,

            # Recording callbacks
            on_recording_start=self._handle_recording_start,
            on_recording_stop=self._handle_recording_stop,

            # Server-side settings
            spinner=False,
            no_log_file=True,

            **self._recorder_kwargs
        )

    async def start(self) -> None:
        """Start the STT service and initialize models."""
        if self._is_running:
            logger.warning("STT service already running")
            return

        self._loop = asyncio.get_running_loop()

        logger.info("Initializing STT with model: %s", self._model)
        self._recorder = await self._loop.run_in_executor(
            None, self._create_recorder
        )

        self._is_running = True
        logger.info("STT service started")

    async def stop(self) -> None:
        """Stop the STT service and clean up."""
        if not self._is_running:
            return

        self._is_running = False
        self._is_listening = False

        # Cancel transcription loop if running
        if self._transcription_task and not self._transcription_task.done():
            self._transcription_task.cancel()
            try:
                await self._transcription_task
            except asyncio.CancelledError:
                pass

        # Shutdown recorder
        if self._recorder:
            await self._loop.run_in_executor(None, self._recorder.shutdown)
            self._recorder = None

        logger.info("STT service stopped")

    async def __aenter__(self) -> "STTService":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()

    # -------------------------------------------------------------------------
    # Audio input
    # -------------------------------------------------------------------------

    def feed_audio(self, chunk: bytes, sample_rate: int = 16000) -> None:
        """
        Feed audio data from WebSocket.

        Args:
            chunk: PCM16 audio bytes
            sample_rate: Sample rate (default 16kHz)
        """
        if not self._is_running or not self._recorder:
            return

        if isinstance(chunk, bytes):
            audio_data = np.frombuffer(chunk, dtype=np.int16)
        else:
            audio_data = chunk

        self._recorder.feed_audio(audio_data, original_sample_rate=sample_rate)

    # -------------------------------------------------------------------------
    # Listening control
    # -------------------------------------------------------------------------

    def start_listening(self) -> None:
        """
        Start listening for voice activity.

        Call when user activates microphone. The recorder will wait for
        VAD to trigger, then start recording automatically.
        """
        if not self._is_running or not self._recorder:
            logger.warning("Cannot start listening: STT not running")
            return

        with self._state_lock:
            self._is_listening = True
            self._interrupt_detected = False

        self._recorder.listen()
        logger.debug("Listening started")

    def stop_listening(self) -> None:
        """Stop listening and clear audio buffer."""
        if not self._is_running or not self._recorder:
            return

        with self._state_lock:
            self._is_listening = False

        self._recorder.clear_audio_queue()
        logger.debug("Listening stopped")

    def abort(self) -> None:
        """Abort current recording without transcribing."""
        if self._recorder:
            self._recorder.abort()
            logger.debug("Recording aborted")

    # -------------------------------------------------------------------------
    # Interrupt handling
    # -------------------------------------------------------------------------

    def set_tts_playing(self, playing: bool) -> None:
        """
        Set TTS playback state for interrupt detection.

        When TTS is playing and VAD detects speech, an interrupt is flagged.
        """
        with self._state_lock:
            self._tts_playing = playing
            if not playing:
                self._interrupt_detected = False

    def is_interrupt_detected(self) -> bool:
        """Check if user interrupted during TTS playback."""
        with self._state_lock:
            return self._interrupt_detected

    def clear_interrupt(self) -> None:
        """Clear the interrupt flag."""
        with self._state_lock:
            self._interrupt_detected = False

    # -------------------------------------------------------------------------
    # Transcription
    # -------------------------------------------------------------------------

    async def get_transcription(self) -> Optional[str]:
        """
        Wait for complete speech turn and return transcription.

        Blocks until user finishes speaking (silence detected).
        Returns the final transcription text.
        """
        if not self._is_running or not self._recorder:
            return None

        if not self._is_listening:
            self.start_listening()

        try:
            # text() blocks until speech complete
            text = await self._loop.run_in_executor(None, self._recorder.text)

            if text and self._on_final_transcription:
                self._invoke_callback(self._on_final_transcription, text)

            return text

        except Exception as e:
            logger.error("Transcription error: %s", e)
            return None

    async def run_transcription_loop(self) -> None:
        """
        Run continuous transcription loop.

        Continuously listens and transcribes, calling on_final_transcription
        for each complete utterance. Run this as a background task.
        """
        logger.info("Starting transcription loop")

        while self._is_running:
            try:
                if not self._is_listening:
                    self.start_listening()

                text = await self.get_transcription()

                if text:
                    logger.debug("Transcription: %s", text[:50] + "..." if len(text) > 50 else text)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Transcription loop error: %s", e)
                await asyncio.sleep(0.1)

        logger.info("Transcription loop ended")

    def start_transcription_loop(self) -> asyncio.Task:
        """
        Start the transcription loop as a background task.

        Returns the task so it can be cancelled if needed.
        """
        if self._transcription_task and not self._transcription_task.done():
            logger.warning("Transcription loop already running")
            return self._transcription_task

        self._transcription_task = asyncio.create_task(self.run_transcription_loop())
        return self._transcription_task

    def stop_transcription_loop(self) -> None:
        """Stop the transcription loop."""
        if self._transcription_task:
            self._transcription_task.cancel()

    # -------------------------------------------------------------------------
    # State
    # -------------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_listening(self) -> bool:
        with self._state_lock:
            return self._is_listening

    @property
    def state(self) -> str:
        """Current recorder state: inactive, listening, recording, transcribing."""
        if self._recorder:
            return self._recorder.state
        return "inactive"
