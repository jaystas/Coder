"""
Headless Audio Player Module
----------------------------
Provides a PyAudio-free audio player implementation that processes audio chunks
without actual audio device playback. Perfect for:

- Server-side TTS generation on remote GPU servers
- Streaming audio to browser clients via WebSocket
- Processing audio for file output or other purposes
- Testing and development without audio hardware

The headless player:
- Reads audio chunks from the engine queue
- Processes timing information for word-level callbacks
- Triggers on_audio_chunk callbacks for each processed chunk
- Simulates playback timing for accurate word timing events
- Supports pause/resume/stop operations
"""

from typing import Callable, Optional
import threading
import logging
import queue
import time
import io

import numpy as np

from .audio_formats import AudioFormat
from .audio_player_base import AudioPlayerBase, AudioConfig


class HeadlessBufferManager:
    """
    Manages an audio buffer without PyAudio dependency.
    Tracks sample counts and provides buffered duration calculations.
    """

    def __init__(self, audio_buffer: queue.Queue, timings: queue.Queue, config: AudioConfig):
        """
        Args:
            audio_buffer: Queue containing audio chunks
            timings: Queue containing timing information
            config: Audio configuration
        """
        self.audio_buffer = audio_buffer
        self.timings = timings
        self.config = config
        self.total_samples = 0
        self._lock = threading.Lock()

    def add_to_buffer(self, audio_data: bytes) -> None:
        """Add audio data to the buffer."""
        self.audio_buffer.put(audio_data)
        bytes_per_frame = self.config.get_bytes_per_frame()
        with self._lock:
            self.total_samples += len(audio_data) // bytes_per_frame

    def get_from_buffer(self, timeout: float = 0.05) -> tuple:
        """
        Retrieve audio data from the buffer.

        Args:
            timeout: Time to wait before returning if buffer is empty

        Returns:
            Tuple of (success: bool, chunk: bytes or None)
        """
        try:
            chunk = self.audio_buffer.get(timeout=timeout)
            bytes_per_frame = self.config.get_bytes_per_frame()
            with self._lock:
                if chunk:
                    self.total_samples -= len(chunk) // bytes_per_frame
            return True, chunk
        except queue.Empty:
            return False, None

    def get_buffered_seconds(self) -> float:
        """Calculate duration of buffered audio in seconds."""
        rate = self.config.rate if self.config.rate > 0 else 16000
        with self._lock:
            return self.total_samples / rate

    def clear_buffer(self) -> None:
        """Clear all data from the buffer."""
        while not self.audio_buffer.empty():
            try:
                self.audio_buffer.get_nowait()
            except queue.Empty:
                break
        while not self.timings.empty():
            try:
                self.timings.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self.total_samples = 0


class HeadlessPlayer(AudioPlayerBase):
    """
    A headless audio player that processes audio without PyAudio.

    This player:
    - Pulls audio chunks from the engine queue
    - Triggers callbacks (on_audio_chunk, on_word_spoken, etc.)
    - Simulates timing for accurate word-level events
    - Does NOT play audio to any device

    Use cases:
    - Remote TTS server streaming to browser clients
    - File-only output
    - Audio processing pipelines
    """

    def __init__(
        self,
        audio_buffer: queue.Queue,
        timings: queue.Queue,
        config: AudioConfig,
        on_playback_start: Optional[Callable] = None,
        on_playback_stop: Optional[Callable] = None,
        on_audio_chunk: Optional[Callable[[bytes], None]] = None,
        on_word_spoken: Optional[Callable] = None,
        simulate_realtime: bool = False,
    ):
        """
        Initialize the headless player.

        Args:
            audio_buffer: Queue containing audio chunks from the engine
            timings: Queue containing word timing information
            config: Audio configuration
            on_playback_start: Callback when playback starts
            on_playback_stop: Callback when playback stops
            on_audio_chunk: Callback for each processed audio chunk
            on_word_spoken: Callback for word timing events
            simulate_realtime: If True, adds delays to simulate real-time playback
        """
        super().__init__(
            audio_buffer, timings, config,
            on_playback_start, on_playback_stop,
            on_audio_chunk, on_word_spoken
        )

        self.buffer_manager = HeadlessBufferManager(audio_buffer, timings, config)
        self.timings_queue = timings
        self.timings_list = []

        self.playback_active = False
        self.immediate_stop = threading.Event()
        self.pause_event = threading.Event()
        self.playback_thread = None
        self.first_chunk_processed = False
        self.simulate_realtime = simulate_realtime

    def _get_sub_chunk_size(self) -> int:
        """Calculate the sub-chunk size for processing."""
        if self.config.playout_chunk_size > 0:
            return self.config.playout_chunk_size

        if self.config.frames_per_buffer == AudioFormat.FRAMES_PER_BUFFER_UNSPECIFIED:
            return 512  # Default small chunk size
        else:
            return self.config.frames_per_buffer * self.config.get_bytes_per_frame()

    def _process_chunk(self, chunk: bytes) -> None:
        """
        Process a single audio chunk.

        Args:
            chunk: Raw audio data bytes
        """
        # Handle MPEG format specially (would need external decoder in real use)
        if self.config.is_mpeg():
            self._process_mpeg_chunk(chunk)
            return

        self._process_pcm_chunk(chunk)

    def _process_mpeg_chunk(self, chunk: bytes) -> None:
        """
        Process an MPEG audio chunk.

        For headless mode, we just pass through the raw MPEG data.
        The receiving client (browser) can decode it.
        """
        if not self.first_chunk_processed and self.on_playback_start:
            self.on_playback_start()
            self.first_chunk_processed = True

        if self.on_audio_chunk:
            self.on_audio_chunk(chunk)

        # Handle pause
        while self.pause_event.is_set() and not self.immediate_stop.is_set():
            time.sleep(0.01)

    def _process_pcm_chunk(self, chunk: bytes) -> None:
        """
        Process a PCM audio chunk.

        Splits into sub-chunks, handles timing, and triggers callbacks.
        """
        bytes_per_frame = self.config.get_bytes_per_frame()
        rate = self.config.rate if self.config.rate > 0 else 24000
        sub_chunk_size = self._get_sub_chunk_size()

        # Process timing information from queue
        while True:
            try:
                timing = self.timings_queue.get_nowait()
                self.timings_list.append(timing)
            except queue.Empty:
                break

        # Split into sub-chunks for granular processing
        for i in range(0, len(chunk), sub_chunk_size):
            if self.immediate_stop.is_set():
                break

            sub_chunk = chunk[i:i + sub_chunk_size]

            # Trigger playback start on first chunk
            if not self.first_chunk_processed and self.on_playback_start:
                self.on_playback_start()
                self.first_chunk_processed = True

            # Update playback time
            chunk_duration = len(sub_chunk) / (rate * bytes_per_frame)
            self.seconds_played += chunk_duration

            # Check for word timing events
            for timing in list(self.timings_list):
                if timing.start_time <= self.seconds_played:
                    if self.on_word_spoken:
                        self.on_word_spoken(timing)
                    self.timings_list.remove(timing)

            # Trigger audio chunk callback
            if self.on_audio_chunk:
                self.on_audio_chunk(sub_chunk)

            # Simulate real-time playback if requested
            if self.simulate_realtime and not self.muted:
                time.sleep(chunk_duration)

            # Handle pause
            while self.pause_event.is_set() and not self.immediate_stop.is_set():
                time.sleep(0.01)

    def _process_buffer(self) -> None:
        """
        Main processing loop - reads chunks from buffer and processes them.
        """
        while self.playback_active or not self.buffer_manager.audio_buffer.empty():
            success, chunk = self.buffer_manager.get_from_buffer()

            if chunk:
                self._process_chunk(chunk)

            if self.immediate_stop.is_set():
                logging.info("Immediate stop requested, aborting playback")
                break

        # Playback complete
        if self.on_playback_stop:
            self.on_playback_stop()

    def start(self) -> None:
        """Start the audio processing loop."""
        self.first_chunk_processed = False
        self.playback_active = True
        self.seconds_played = 0.0
        self.timings_list = []

        if not self.playback_thread or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(target=self._process_buffer)
            self.playback_thread.start()

    def stop(self, immediate: bool = False) -> None:
        """
        Stop audio processing.

        Args:
            immediate: If True, stop immediately without draining buffer
        """
        if not self.playback_thread:
            logging.warning("No playback thread found, cannot stop")
            return

        if immediate:
            self.immediate_stop.set()
            while self.playback_active:
                time.sleep(0.001)
            self.immediate_stop.clear()
            return

        self.playback_active = False

        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join()

        self.buffer_manager.clear_buffer()
        self.playback_thread = None
        self.immediate_stop.clear()

    def pause(self) -> None:
        """Pause audio processing."""
        self.pause_event.set()

    def resume(self) -> None:
        """Resume audio processing."""
        self.pause_event.clear()

    def get_buffered_seconds(self) -> float:
        """Get the duration of buffered audio in seconds."""
        return self.buffer_manager.get_buffered_seconds()

    def is_playing(self) -> bool:
        """Check if processing is currently active."""
        return self.playback_active and (
            self.playback_thread is not None and
            self.playback_thread.is_alive()
        )
