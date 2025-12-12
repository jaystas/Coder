"""
Audio Player Base Module
------------------------
Provides an abstract base class for audio players, allowing different playback
backends to be used interchangeably.

This enables:
- PyAudio-based playback for local speaker output
- Headless playback for server-side processing (no audio device needed)
- WebSocket streaming for browser-based playback
- Custom playback implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional
import queue
import logging

from .audio_formats import AudioFormat


@dataclass
class AudioConfig:
    """
    Configuration for audio streams.

    Attributes:
        format: Audio format (e.g., AudioFormat.INT16)
        channels: Number of audio channels (1=mono, 2=stereo)
        rate: Sample rate in Hz (e.g., 24000, 44100)
        muted: If True, audio playback is muted
        frames_per_buffer: Frames per buffer (0 = unspecified)
        playout_chunk_size: Size of chunks for playback in bytes (-1 = auto)
    """
    format: int = AudioFormat.INT16
    channels: int = 1
    rate: int = 16000
    muted: bool = False
    frames_per_buffer: int = AudioFormat.FRAMES_PER_BUFFER_UNSPECIFIED
    playout_chunk_size: int = -1

    def get_bytes_per_sample(self) -> int:
        """Returns bytes per sample for the configured format."""
        return AudioFormat.get_bytes_per_sample(self.format)

    def get_bytes_per_frame(self) -> int:
        """Returns bytes per frame (sample * channels)."""
        return self.get_bytes_per_sample() * self.channels

    def is_mpeg(self) -> bool:
        """Returns True if this is an MPEG/MP3 stream configuration."""
        return AudioFormat.is_mpeg_format(self.format, self.channels, self.rate)


class AudioPlayerBase(ABC):
    """
    Abstract base class for audio players.

    Defines the interface that all audio player implementations must follow.
    This allows TextToAudioStream to work with different playback backends.
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
    ):
        """
        Initialize the audio player.

        Args:
            audio_buffer: Queue containing audio chunks to play
            timings: Queue containing word timing information
            config: Audio configuration
            on_playback_start: Callback when playback starts
            on_playback_stop: Callback when playback stops
            on_audio_chunk: Callback for each audio chunk (after processing)
            on_word_spoken: Callback for word timing events
        """
        self.audio_buffer = audio_buffer
        self.timings = timings
        self.config = config
        self.on_playback_start = on_playback_start
        self.on_playback_stop = on_playback_stop
        self.on_audio_chunk = on_audio_chunk
        self.on_word_spoken = on_word_spoken
        self.muted = config.muted
        self.seconds_played = 0.0

    @abstractmethod
    def start(self) -> None:
        """Start audio playback."""
        pass

    @abstractmethod
    def stop(self, immediate: bool = False) -> None:
        """
        Stop audio playback.

        Args:
            immediate: If True, stop immediately without draining buffer
        """
        pass

    @abstractmethod
    def pause(self) -> None:
        """Pause audio playback."""
        pass

    @abstractmethod
    def resume(self) -> None:
        """Resume paused audio playback."""
        pass

    def mute(self, muted: bool = True) -> None:
        """
        Mute or unmute audio playback.

        Args:
            muted: True to mute, False to unmute
        """
        self.muted = muted

    @abstractmethod
    def get_buffered_seconds(self) -> float:
        """
        Get the duration of buffered audio in seconds.

        Returns:
            Duration of buffered audio in seconds
        """
        pass

    @abstractmethod
    def is_playing(self) -> bool:
        """
        Check if playback is currently active.

        Returns:
            True if playing, False otherwise
        """
        pass


def create_player(
    backend: str,
    audio_buffer: queue.Queue,
    timings: queue.Queue,
    config: AudioConfig,
    **kwargs
) -> AudioPlayerBase:
    """
    Factory function to create an audio player with the specified backend.

    Args:
        backend: Player backend type ('pyaudio', 'headless', 'websocket')
        audio_buffer: Queue containing audio chunks
        timings: Queue containing word timing info
        config: Audio configuration
        **kwargs: Additional arguments passed to the player constructor

    Returns:
        An AudioPlayerBase implementation

    Raises:
        ValueError: If the backend is not recognized
        ImportError: If the backend dependencies are not available
    """
    backend = backend.lower()

    if backend == "headless":
        from .headless_player import HeadlessPlayer
        return HeadlessPlayer(audio_buffer, timings, config, **kwargs)

    elif backend == "pyaudio":
        from .stream_player import StreamPlayer, AudioConfiguration
        # Convert AudioConfig to legacy AudioConfiguration
        legacy_config = AudioConfiguration(
            format=config.format,
            channels=config.channels,
            rate=config.rate,
            muted=config.muted,
            frames_per_buffer=config.frames_per_buffer,
            playout_chunk_size=config.playout_chunk_size,
        )
        return StreamPlayer(audio_buffer, timings, legacy_config, **kwargs)

    else:
        raise ValueError(
            f"Unknown player backend: {backend}. "
            f"Available backends: 'headless', 'pyaudio'"
        )
