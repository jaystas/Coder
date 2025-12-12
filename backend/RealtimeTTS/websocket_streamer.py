"""
WebSocket Audio Streamer Module
-------------------------------
Provides real-time audio streaming from RealtimeTTS to browser clients via WebSocket.

This module enables the distributed architecture:
- Remote GPU Server: Runs TTS engines, generates audio using HeadlessPlayer
- Browser Client: Receives audio via WebSocket, plays via Web Audio API

Usage:
    from RealtimeTTS import TextToAudioStream, KokoroEngine
    from RealtimeTTS.websocket_streamer import WebSocketStreamer

    # Create streamer (starts WebSocket server)
    streamer = WebSocketStreamer(host="0.0.0.0", port=8765)

    # Create TTS stream in headless mode
    engine = KokoroEngine()
    stream = TextToAudioStream(engine, player_backend="headless")

    # Stream audio to connected clients
    stream.play(
        on_audio_chunk=streamer.send_chunk,
        muted=True
    )

Browser Client Example (JavaScript):
    const ws = new WebSocket('ws://server:8765');
    const audioContext = new AudioContext({ sampleRate: 24000 });

    ws.onmessage = async (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'audio') {
            const audioData = base64ToFloat32(data.audio);
            const buffer = audioContext.createBuffer(1, audioData.length, data.sampleRate);
            buffer.getChannelData(0).set(audioData);
            // Queue for playback...
        }
    };
"""

import asyncio
import base64
import json
import logging
import threading
from typing import Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field
import numpy as np

from .audio_formats import AudioFormat


@dataclass
class AudioStreamConfig:
    """Configuration for audio stream metadata sent to clients."""
    format: int = AudioFormat.INT16
    channels: int = 1
    sample_rate: int = 24000

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "format": self.format,
            "channels": self.channels,
            "sampleRate": self.sample_rate,
            "bytesPerSample": AudioFormat.get_bytes_per_sample(self.format),
        }


class WebSocketStreamer:
    """
    Streams audio chunks to browser clients via WebSocket.

    This class provides:
    - WebSocket server for real-time audio streaming
    - Automatic client connection management
    - Audio format conversion for Web Audio API compatibility
    - Event callbacks for connection events
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        config: Optional[AudioStreamConfig] = None,
        on_client_connect: Optional[Callable[[str], None]] = None,
        on_client_disconnect: Optional[Callable[[str], None]] = None,
        convert_to_float32: bool = True,
    ):
        """
        Initialize the WebSocket streamer.

        Args:
            host: Host address to bind the WebSocket server
            port: Port to listen on
            config: Audio stream configuration (format, channels, rate)
            on_client_connect: Callback when a client connects
            on_client_disconnect: Callback when a client disconnects
            convert_to_float32: If True, convert INT16 audio to Float32 for Web Audio API
        """
        self.host = host
        self.port = port
        self.config = config or AudioStreamConfig()
        self.on_client_connect = on_client_connect
        self.on_client_disconnect = on_client_disconnect
        self.convert_to_float32 = convert_to_float32

        self._clients: Set = set()
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._server_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start the WebSocket server in a background thread."""
        if self._running:
            logging.warning("WebSocket server already running")
            return

        self._running = True
        self._server_thread = threading.Thread(target=self._run_server, daemon=True)
        self._server_thread.start()
        logging.info(f"WebSocket streamer starting on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        if self._loop and self._server:
            self._loop.call_soon_threadsafe(self._shutdown_server)

    def _shutdown_server(self):
        """Shutdown the server (called from the event loop thread)."""
        if self._server:
            self._server.close()

    def _run_server(self) -> None:
        """Run the WebSocket server (in background thread)."""
        try:
            import websockets
            import websockets.server
        except ImportError:
            raise ImportError(
                "websockets package required for WebSocket streaming. "
                "Install with: pip install websockets"
            )

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def handler(websocket, path=None):
            """Handle WebSocket connections."""
            client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"

            with self._lock:
                self._clients.add(websocket)

            logging.info(f"Client connected: {client_id}")
            if self.on_client_connect:
                self.on_client_connect(client_id)

            # Send stream configuration to client
            config_msg = json.dumps({
                "type": "config",
                "config": self.config.to_dict()
            })
            await websocket.send(config_msg)

            try:
                # Keep connection alive, handle any incoming messages
                async for message in websocket:
                    # Handle client messages if needed (e.g., control commands)
                    try:
                        data = json.loads(message)
                        await self._handle_client_message(websocket, data)
                    except json.JSONDecodeError:
                        pass
            except Exception as e:
                logging.debug(f"Client {client_id} disconnected: {e}")
            finally:
                with self._lock:
                    self._clients.discard(websocket)

                logging.info(f"Client disconnected: {client_id}")
                if self.on_client_disconnect:
                    self.on_client_disconnect(client_id)

        async def main():
            try:
                self._server = await websockets.serve(handler, self.host, self.port)
                logging.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
                await self._server.wait_closed()
            except Exception as e:
                logging.error(f"WebSocket server error: {e}")

        try:
            self._loop.run_until_complete(main())
        except Exception as e:
            logging.error(f"Error running WebSocket server: {e}")
        finally:
            self._loop.close()

    async def _handle_client_message(self, websocket, data: dict) -> None:
        """Handle incoming messages from clients."""
        msg_type = data.get("type")

        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong"}))

    def send_chunk(self, chunk: bytes) -> None:
        """
        Send an audio chunk to all connected clients.

        This method is designed to be used as the on_audio_chunk callback
        for TextToAudioStream.

        Args:
            chunk: Raw audio bytes (INT16 or FLOAT32 depending on engine)
        """
        if not self._clients:
            return

        # Convert audio format if needed
        if self.convert_to_float32 and self.config.format == AudioFormat.INT16:
            # Convert INT16 to Float32 for Web Audio API
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            audio_bytes = audio_float.tobytes()
            send_format = "float32"
        else:
            audio_bytes = chunk
            send_format = "int16" if self.config.format == AudioFormat.INT16 else "float32"

        # Encode as base64 for JSON transport
        audio_b64 = base64.b64encode(audio_bytes).decode('ascii')

        message = json.dumps({
            "type": "audio",
            "audio": audio_b64,
            "format": send_format,
            "sampleRate": self.config.sample_rate,
            "channels": self.config.channels,
        })

        # Send to all connected clients
        self._broadcast(message)

    def _broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients."""
        if not self._loop or not self._clients:
            return

        with self._lock:
            clients = list(self._clients)

        async def send_to_all():
            for client in clients:
                try:
                    await client.send(message)
                except Exception as e:
                    logging.debug(f"Error sending to client: {e}")

        try:
            asyncio.run_coroutine_threadsafe(send_to_all(), self._loop)
        except Exception as e:
            logging.debug(f"Error broadcasting: {e}")

    def send_event(self, event_type: str, data: dict = None) -> None:
        """
        Send an event to all connected clients.

        Args:
            event_type: Type of event (e.g., 'start', 'stop', 'word')
            data: Additional event data
        """
        message = json.dumps({
            "type": "event",
            "event": event_type,
            "data": data or {}
        })
        self._broadcast(message)

    def update_config(self, config: AudioStreamConfig) -> None:
        """
        Update the audio configuration and notify clients.

        Args:
            config: New audio stream configuration
        """
        self.config = config
        config_msg = json.dumps({
            "type": "config",
            "config": self.config.to_dict()
        })
        self._broadcast(config_msg)

    @property
    def client_count(self) -> int:
        """Return the number of connected clients."""
        with self._lock:
            return len(self._clients)

    @property
    def is_running(self) -> bool:
        """Return True if the server is running."""
        return self._running


# Convenience function for creating a streaming callback
def create_websocket_callback(
    host: str = "0.0.0.0",
    port: int = 8765,
    sample_rate: int = 24000,
    channels: int = 1,
) -> tuple:
    """
    Create a WebSocket streamer and return it with its send_chunk callback.

    This is a convenience function for quick setup.

    Args:
        host: WebSocket server host
        port: WebSocket server port
        sample_rate: Audio sample rate
        channels: Number of audio channels

    Returns:
        Tuple of (WebSocketStreamer, send_chunk_callback)

    Usage:
        streamer, on_chunk = create_websocket_callback(port=8765)
        streamer.start()

        stream = TextToAudioStream(engine, player_backend="headless")
        stream.play(on_audio_chunk=on_chunk)
    """
    config = AudioStreamConfig(
        format=AudioFormat.INT16,
        channels=channels,
        sample_rate=sample_rate,
    )
    streamer = WebSocketStreamer(host=host, port=port, config=config)
    return streamer, streamer.send_chunk
