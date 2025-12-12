"""
Audio Format Constants Module
-----------------------------
Provides PyAudio-compatible format constants without requiring PyAudio installation.
This allows RealtimeTTS to run in headless/server environments without audio device access.

These constants mirror PyAudio's format definitions for compatibility with engines
that need to specify audio formats.
"""


class AudioFormat:
    """
    Audio format constants that mirror PyAudio's format definitions.

    These allow engines and stream components to specify audio formats
    without importing pyaudio directly.
    """

    # Standard PCM formats (values match PyAudio's internal constants)
    FLOAT32 = 0x01       # 32-bit floating point (-1.0 to 1.0)
    INT32 = 0x02         # 32-bit signed integer
    INT24 = 0x04         # 24-bit signed integer (packed)
    INT16 = 0x08         # 16-bit signed integer (most common)
    INT8 = 0x10          # 8-bit signed integer
    UINT8 = 0x20         # 8-bit unsigned integer

    # Special format for MPEG/MP3 streaming (uses external player like mpv)
    CUSTOM = 0x00010000  # Custom format indicator (e.g., MPEG streams)

    # Buffer size constant
    FRAMES_PER_BUFFER_UNSPECIFIED = 0

    # Mapping from format to bytes per sample
    FORMAT_BYTES = {
        FLOAT32: 4,
        INT32: 4,
        INT24: 3,
        INT16: 2,
        INT8: 1,
        UINT8: 1,
        CUSTOM: 4,  # Default for custom formats
    }

    # Mapping from format to numpy dtype
    FORMAT_DTYPE = {
        FLOAT32: 'float32',
        INT32: 'int32',
        INT16: 'int16',
        INT8: 'int8',
        UINT8: 'uint8',
    }

    @classmethod
    def get_bytes_per_sample(cls, format_id: int) -> int:
        """
        Returns the number of bytes per sample for a given format.

        Args:
            format_id: One of the AudioFormat constants

        Returns:
            Number of bytes per sample
        """
        return cls.FORMAT_BYTES.get(format_id, 2)  # Default to 2 (INT16)

    @classmethod
    def get_numpy_dtype(cls, format_id: int) -> str:
        """
        Returns the numpy dtype string for a given format.

        Args:
            format_id: One of the AudioFormat constants

        Returns:
            Numpy dtype string (e.g., 'int16', 'float32')
        """
        return cls.FORMAT_DTYPE.get(format_id, 'int16')

    @classmethod
    def is_mpeg_format(cls, format_id: int, channels: int = 1, rate: int = 0) -> bool:
        """
        Checks if the format represents an MPEG/MP3 stream.

        The MPEG format is identified by CUSTOM format with channels=-1 and rate=-1.

        Args:
            format_id: The audio format constant
            channels: Number of channels (-1 for MPEG)
            rate: Sample rate (-1 for MPEG)

        Returns:
            True if this represents an MPEG stream
        """
        return format_id == cls.CUSTOM and channels == -1 and rate == -1


# Convenience aliases for backwards compatibility
paInt16 = AudioFormat.INT16
paInt24 = AudioFormat.INT24
paInt32 = AudioFormat.INT32
paFloat32 = AudioFormat.FLOAT32
paInt8 = AudioFormat.INT8
paUInt8 = AudioFormat.UINT8
paCustomFormat = AudioFormat.CUSTOM
paFramesPerBufferUnspecified = AudioFormat.FRAMES_PER_BUFFER_UNSPECIFIED
