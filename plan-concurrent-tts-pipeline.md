# Concurrent LLM → TTS Streaming Pipeline Implementation Plan

## Overview

This plan describes a concurrent pipeline that streams text from an AsyncOpenAI client, extracts sentences in real-time using `stream2sentence`, and generates TTS audio using Higgs Audio `generate_delta_stream` - all while playing audio sequentially to the browser.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           CONCURRENT PIPELINE FLOW                            │
└──────────────────────────────────────────────────────────────────────────────┘

    AsyncOpenAI Stream          stream2sentence           Sentence Queue
    ┌─────────────┐            ┌─────────────┐           ┌──────────────┐
    │  "Hello, "  │──────────▶│  Accumulate │           │              │
    │  "world! "  │           │   & Detect  │──────────▶│  "Hello, "   │
    │  "How are"  │           │  Boundaries │           │  "world!"    │
    │  " you?"    │           └─────────────┘           │  "How are"   │
    └─────────────┘                                     │  "you?"      │
                                                        └──────┬───────┘
                                                               │
                                          ┌────────────────────┘
                                          ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         TTS WORKER (asyncio.Task)                        │
    │  ┌─────────────────────────────────────────────────────────────────┐    │
    │  │  await sentence_queue.get()                                      │    │
    │  │  async for delta in engine.generate_delta_stream(sentence):      │    │
    │  │      # Accumulate audio tokens, decode chunks                    │    │
    │  │      await audio_queue.put(pcm_chunk)                            │    │
    │  └─────────────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │     Audio Queue       │
                              │  (PCM16 bytes)        │
                              │  ┌─────┐ ┌─────┐      │
                              │  │chunk│ │chunk│ ...  │
                              │  └─────┘ └─────┘      │
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │   WebSocket Client    │
                              │   (Browser)           │
                              │   - Receives PCM16    │
                              │   - Queues in order   │
                              │   - Plays via Web     │
                              │     Audio API         │
                              └───────────────────────┘
```

---

## Architecture Components

### 1. Data Classes

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis"""
    text: str
    index: int
    session_id: str
    is_final: bool = False  # Sentinel for end-of-stream

@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming"""
    audio_data: bytes      # PCM16 bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    is_final: bool = False  # Last chunk of session
```

### 2. Queue Manager

```python
class StreamingQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(self):
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()
```

---

## Component Details

### Component 1: LLM Stream Text Extractor

**File**: `backend/streaming_pipeline.py`

**Purpose**: Consumes AsyncOpenAI stream, uses stream2sentence to detect sentence boundaries, puts sentences into queue.

```python
from openai import AsyncOpenAI
from backend.stream2sentence import generate_sentences_async

class LLMStreamProcessor:
    """Processes LLM stream into sentences for TTS"""

    def __init__(self, queues: StreamingQueues, api_key: str):
        self.queues = queues
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    async def process_prompt(
        self,
        prompt: str,
        session_id: str,
        model: str = "meta-llama/llama-3.1-8b-instruct"
    ) -> str:
        """
        Stream LLM response, extract sentences, queue for TTS.
        Returns full response text.
        """
        sentence_index = 0
        full_response = ""

        # Create streaming completion
        stream = await self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )

        # Generator that extracts text chunks from OpenAI stream
        async def chunk_generator():
            nonlocal full_response
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield content

        # Process chunks through stream2sentence
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
                tts_sentence = TTSSentence(
                    text=sentence_text,
                    index=sentence_index,
                    session_id=session_id,
                    is_final=False
                )
                await self.queues.sentence_queue.put(tts_sentence)
                sentence_index += 1

        # Signal end of stream
        await self.queues.sentence_queue.put(TTSSentence(
            text="",
            index=sentence_index,
            session_id=session_id,
            is_final=True
        ))

        return full_response
```

**Key Points**:
- Uses `generate_sentences_async` with `quick_yield_single_sentence_fragment=True` for low latency
- Wraps the OpenAI stream in an async generator that yields content strings
- Puts `TTSSentence` objects into `sentence_queue`
- Sends final sentinel with `is_final=True` to signal stream completion

---

### Component 2: Higgs TTS Worker

**File**: `backend/streaming_pipeline.py`

**Purpose**: Consumes sentences from queue, generates audio via Higgs `generate_delta_stream`, puts PCM chunks into audio queue.

```python
import torch
import numpy as np
from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

class HiggsTTSWorker:
    """TTS worker that synthesizes sentences using Higgs Audio"""

    def __init__(self, queues: StreamingQueues):
        self.queues = queues
        self.engine: Optional[HiggsAudioServeEngine] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None

        # Audio settings
        self.sample_rate = 24000
        self._chunk_size = 20  # Audio tokens per chunk before decoding
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        # Voice reference directory
        self.voice_dir = "voices"
        self.default_voice = "default"  # Name of default voice files

    async def initialize(self):
        """Initialize Higgs Audio engine"""
        self.engine = HiggsAudioServeEngine(
            model_name_or_path="bosonai/higgs-audio-v2-generation-3B-base",
            audio_tokenizer_name_or_path="bosonai/higgs-audio-v2-tokenizer",
            device=self._device
        )

    async def start(self):
        """Start the TTS worker task"""
        self.is_running = True
        self._task = asyncio.create_task(self._process_loop())

    async def stop(self):
        """Stop the TTS worker"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _process_loop(self):
        """Main processing loop - consumes from sentence_queue"""
        while self.is_running:
            try:
                # Get sentence with timeout
                sentence = await asyncio.wait_for(
                    self.queues.sentence_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            # Handle end-of-stream sentinel
            if sentence.is_final:
                await self.queues.audio_queue.put(AudioChunk(
                    audio_data=b"",
                    sentence_index=sentence.index,
                    chunk_index=0,
                    session_id=sentence.session_id,
                    is_final=True
                ))
                continue

            # Generate audio for this sentence
            chunk_index = 0
            async for pcm_bytes in self._generate_audio(sentence.text):
                audio_chunk = AudioChunk(
                    audio_data=pcm_bytes,
                    sentence_index=sentence.index,
                    chunk_index=chunk_index,
                    session_id=sentence.session_id,
                    is_final=False
                )
                await self.queues.audio_queue.put(audio_chunk)
                chunk_index += 1

    def _load_voice_messages(self, voice_name: str) -> list[Message]:
        """Load voice reference for few-shot cloning"""
        import os

        audio_path = os.path.join(self.voice_dir, f"{voice_name}.wav")
        text_path = os.path.join(self.voice_dir, f"{voice_name}.txt")

        if not os.path.exists(audio_path) or not os.path.exists(text_path):
            # Return empty list if no voice reference
            return []

        with open(text_path, 'r', encoding='utf-8') as f:
            ref_text = f.read().strip()

        return [
            Message(role="user", content=ref_text),
            Message(role="assistant", content=AudioContent(audio_url=audio_path))
        ]

    async def _generate_audio(
        self,
        text: str,
        voice: str = "default"
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for text using Higgs streaming"""

        # Build messages with optional voice reference
        messages = self._load_voice_messages(voice)
        messages.append(Message(role="user", content=text))

        chat_sample = ChatMLSample(messages=messages)

        # Initialize streaming state
        audio_tokens: list[torch.Tensor] = []
        seq_len = 0

        with torch.inference_mode():
            async for delta in self.engine.generate_delta_stream(
                chat_ml_sample=chat_sample,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.95,
                force_audio_gen=True,
            ):
                if delta.audio_tokens is None:
                    continue

                # Check for end token
                if torch.all(delta.audio_tokens == 1025):
                    break

                # Accumulate tokens
                audio_tokens.append(delta.audio_tokens[:, None])

                # Count non-padding tokens
                if torch.all(delta.audio_tokens != 1024):
                    seq_len += 1

                # Decode when chunk size reached
                if seq_len > 0 and seq_len % self._chunk_size == 0:
                    audio_tensor = torch.cat(audio_tokens, dim=-1)

                    # Revert delay pattern and decode
                    vq_code = (
                        revert_delay_pattern(
                            audio_tensor,
                            start_idx=seq_len - self._chunk_size + 1
                        )
                        .clip(0, 1023)
                        .to(self._device)
                    )

                    waveform = self.engine.audio_tokenizer.decode(
                        vq_code.unsqueeze(0)
                    )[0, 0]

                    # Convert to numpy
                    if isinstance(waveform, torch.Tensor):
                        waveform_np = waveform.detach().cpu().numpy()
                    else:
                        waveform_np = np.asarray(waveform, dtype=np.float32)

                    # Convert to PCM16 bytes
                    pcm = np.clip(waveform_np, -1.0, 1.0)
                    pcm16 = (pcm * 32767.0).astype(np.int16)
                    yield pcm16.tobytes()

        # Flush remaining tokens
        if seq_len > 0 and seq_len % self._chunk_size != 0 and audio_tokens:
            audio_tensor = torch.cat(audio_tokens, dim=-1)
            remaining = seq_len % self._chunk_size

            vq_code = (
                revert_delay_pattern(
                    audio_tensor,
                    start_idx=seq_len - remaining + 1
                )
                .clip(0, 1023)
                .to(self._device)
            )

            waveform = self.engine.audio_tokenizer.decode(
                vq_code.unsqueeze(0)
            )[0, 0]

            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
            else:
                waveform_np = np.asarray(waveform, dtype=np.float32)

            pcm = np.clip(waveform_np, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            yield pcm16.tobytes()
```

**Key Points**:
- Uses `generate_delta_stream` for streaming token generation
- Accumulates audio tokens and decodes in chunks of `_chunk_size` tokens
- Uses `revert_delay_pattern` for proper audio token decoding
- Yields PCM16 bytes ready for WebSocket streaming
- Supports voice cloning via reference audio (few-shot approach)

---

### Component 3: Audio Queue Streamer

**File**: `backend/streaming_pipeline.py`

**Purpose**: Consumes audio chunks from queue, streams to WebSocket clients.

```python
class AudioStreamer:
    """Streams audio from queue to WebSocket clients"""

    def __init__(self, queues: StreamingQueues):
        self.queues = queues
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self.websocket: Optional[WebSocket] = None

    async def start(self, websocket: WebSocket):
        """Start streaming to a WebSocket"""
        self.websocket = websocket
        self.is_running = True
        self._task = asyncio.create_task(self._stream_loop())

    async def stop(self):
        """Stop streaming"""
        self.is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _stream_loop(self):
        """Main streaming loop"""
        while self.is_running:
            try:
                chunk = await asyncio.wait_for(
                    self.queues.audio_queue.get(),
                    timeout=0.1
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            if chunk.is_final:
                # Send end-of-stream signal
                await self.websocket.send_json({
                    "type": "audio_complete",
                    "session_id": chunk.session_id
                })
                continue

            # Send audio data as binary
            await self.websocket.send_bytes(chunk.audio_data)
```

---

### Component 4: Pipeline Orchestrator

**File**: `backend/streaming_pipeline.py`

**Purpose**: Coordinates all components, manages lifecycle.

```python
class StreamingPipeline:
    """Orchestrates the complete LLM → TTS → Audio pipeline"""

    def __init__(self, api_key: str):
        self.queues = StreamingQueues()
        self.llm_processor = LLMStreamProcessor(self.queues, api_key)
        self.tts_worker = HiggsTTSWorker(self.queues)
        self.audio_streamer = AudioStreamer(self.queues)
        self._initialized = False

    async def initialize(self):
        """Initialize all components"""
        if self._initialized:
            return
        await self.tts_worker.initialize()
        self._initialized = True

    async def start_tts_worker(self):
        """Start the TTS background worker"""
        await self.tts_worker.start()

    async def stop_tts_worker(self):
        """Stop the TTS worker"""
        await self.tts_worker.stop()

    async def process_and_stream(
        self,
        prompt: str,
        websocket: WebSocket,
        session_id: str
    ):
        """
        Process a prompt and stream audio to WebSocket.

        This runs the LLM processor and audio streamer concurrently:
        - LLM processor: extracts sentences → sentence_queue
        - TTS worker (background): sentence_queue → audio_queue
        - Audio streamer: audio_queue → WebSocket
        """
        # Start audio streaming to this websocket
        await self.audio_streamer.start(websocket)

        try:
            # Process LLM stream (this populates sentence_queue)
            # TTS worker runs in background, consuming sentences
            # Audio streamer runs in background, sending to client
            full_response = await self.llm_processor.process_prompt(
                prompt=prompt,
                session_id=session_id
            )

            # Wait for audio queue to drain
            while not self.queues.audio_queue.empty():
                await asyncio.sleep(0.1)

            return full_response

        finally:
            await self.audio_streamer.stop()
```

---

### Component 5: FastAPI Server

**File**: `backend/streaming_server.py`

```python
import os
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.streaming_pipeline import StreamingPipeline

# Global pipeline instance
pipeline: Optional[StreamingPipeline] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup pipeline"""
    global pipeline

    api_key = os.getenv("OPENROUTER_API_KEY", "your-api-key")
    pipeline = StreamingPipeline(api_key=api_key)

    print("Initializing Higgs Audio TTS...")
    await pipeline.initialize()
    await pipeline.start_tts_worker()
    print("Pipeline ready!")

    yield

    print("Shutting down...")
    await pipeline.stop_tts_worker()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS audio"""
    await websocket.accept()

    try:
        while True:
            # Wait for user prompt
            data = await websocket.receive_json()

            if data.get("type") == "prompt":
                prompt = data.get("text", "")
                session_id = data.get("session_id", "default")

                # Notify client that streaming is starting
                await websocket.send_json({
                    "type": "stream_start",
                    "session_id": session_id
                })

                # Process and stream audio
                full_response = await pipeline.process_and_stream(
                    prompt=prompt,
                    websocket=websocket,
                    session_id=session_id
                )

                # Send the full text response
                await websocket.send_json({
                    "type": "text_response",
                    "text": full_response,
                    "session_id": session_id
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

# Serve static files (index.html)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Component 6: Browser Interface (index.html)

**File**: `frontend/stream_test.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Concurrent TTS Stream Test</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }
        h1 { color: #00d9ff; }
        .container {
            background: #16213e;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 12px;
            border: 2px solid #0f3460;
            border-radius: 8px;
            background: #1a1a2e;
            color: #eee;
            font-size: 16px;
            resize: vertical;
        }
        textarea:focus {
            outline: none;
            border-color: #00d9ff;
        }
        button {
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 10px;
        }
        button:hover { background: #00b8d9; }
        button:disabled {
            background: #555;
            cursor: not-allowed;
        }
        #status {
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .connected { background: #1e5128; }
        .disconnected { background: #5c1a1a; }
        .streaming { background: #4a3f00; }
        #response {
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            min-height: 100px;
            white-space: pre-wrap;
            font-family: monospace;
        }
        #audioVisualizer {
            width: 100%;
            height: 60px;
            background: #0f3460;
            border-radius: 8px;
        }
        .stats {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 14px;
            color: #888;
        }
    </style>
</head>
<body>
    <h1>Concurrent LLM → TTS Pipeline Test</h1>

    <div class="container">
        <h3>Prompt</h3>
        <textarea id="prompt" placeholder="Enter your prompt here...">Tell me a short story about a robot learning to paint.</textarea>
        <button id="sendBtn" onclick="sendPrompt()">Send & Stream</button>
        <div id="status" class="disconnected">Disconnected</div>
    </div>

    <div class="container">
        <h3>Audio Stream</h3>
        <canvas id="audioVisualizer"></canvas>
        <div class="stats">
            <span>Chunks received: <span id="chunkCount">0</span></span>
            <span>Buffer: <span id="bufferSize">0</span>ms</span>
            <span>Latency: <span id="latency">-</span>ms</span>
        </div>
    </div>

    <div class="container">
        <h3>Response Text</h3>
        <div id="response"></div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let audioQueue = [];
        let isPlaying = false;
        let chunkCount = 0;
        let startTime = 0;
        let firstChunkTime = 0;

        const SAMPLE_RATE = 24000;

        // Initialize WebSocket
        function connect() {
            const wsUrl = `ws://${window.location.host}/ws/stream`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                document.getElementById('status').className = 'connected';
                document.getElementById('sendBtn').disabled = false;
            };

            ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                document.getElementById('status').className = 'disconnected';
                document.getElementById('sendBtn').disabled = true;
                setTimeout(connect, 2000);
            };

            ws.onerror = (e) => {
                console.error('WebSocket error:', e);
            };

            ws.onmessage = async (event) => {
                if (event.data instanceof Blob) {
                    // Binary audio data
                    const arrayBuffer = await event.data.arrayBuffer();
                    handleAudioChunk(arrayBuffer);
                } else {
                    // JSON message
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                }
            };
        }

        function handleMessage(data) {
            switch (data.type) {
                case 'stream_start':
                    document.getElementById('status').textContent = 'Streaming...';
                    document.getElementById('status').className = 'streaming';
                    document.getElementById('response').textContent = '';
                    chunkCount = 0;
                    startTime = Date.now();
                    firstChunkTime = 0;
                    break;

                case 'text_response':
                    document.getElementById('response').textContent = data.text;
                    break;

                case 'audio_complete':
                    document.getElementById('status').textContent = 'Complete';
                    document.getElementById('status').className = 'connected';
                    break;
            }
        }

        function handleAudioChunk(arrayBuffer) {
            if (!audioContext) {
                audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
            }

            // Track first chunk latency
            if (chunkCount === 0 && startTime > 0) {
                firstChunkTime = Date.now() - startTime;
                document.getElementById('latency').textContent = firstChunkTime;
            }

            chunkCount++;
            document.getElementById('chunkCount').textContent = chunkCount;

            // Convert PCM16 to Float32
            const int16Array = new Int16Array(arrayBuffer);
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }

            // Create audio buffer
            const audioBuffer = audioContext.createBuffer(1, float32Array.length, SAMPLE_RATE);
            audioBuffer.getChannelData(0).set(float32Array);

            // Queue for playback
            audioQueue.push(audioBuffer);
            updateBufferStats();

            // Start playback if not already playing
            if (!isPlaying) {
                playNextChunk();
            }

            // Visualize
            visualize(float32Array);
        }

        function playNextChunk() {
            if (audioQueue.length === 0) {
                isPlaying = false;
                return;
            }

            isPlaying = true;
            const buffer = audioQueue.shift();
            updateBufferStats();

            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.onended = playNextChunk;
            source.start();
        }

        function updateBufferStats() {
            let totalSamples = audioQueue.reduce((sum, buf) => sum + buf.length, 0);
            let bufferMs = Math.round((totalSamples / SAMPLE_RATE) * 1000);
            document.getElementById('bufferSize').textContent = bufferMs;
        }

        function visualize(samples) {
            const canvas = document.getElementById('audioVisualizer');
            const ctx = canvas.getContext('2d');
            const width = canvas.width = canvas.offsetWidth;
            const height = canvas.height;

            ctx.fillStyle = '#0f3460';
            ctx.fillRect(0, 0, width, height);

            ctx.beginPath();
            ctx.strokeStyle = '#00d9ff';
            ctx.lineWidth = 2;

            const step = Math.ceil(samples.length / width);
            for (let i = 0; i < width; i++) {
                const idx = i * step;
                const val = samples[idx] || 0;
                const y = (1 - val) * height / 2;
                if (i === 0) ctx.moveTo(i, y);
                else ctx.lineTo(i, y);
            }
            ctx.stroke();
        }

        function sendPrompt() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt || !ws || ws.readyState !== WebSocket.OPEN) return;

            // Reset state
            audioQueue = [];
            isPlaying = false;

            ws.send(JSON.stringify({
                type: 'prompt',
                text: prompt,
                session_id: Date.now().toString()
            }));
        }

        // Connect on page load
        connect();
    </script>
</body>
</html>
```

---

## Concurrency Model

The pipeline achieves concurrency through three independent asyncio tasks:

```
┌───────────────────────────────────────────────────────────────────┐
│                        MAIN ASYNC LOOP                             │
├───────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Task 1: LLM Processor              Task 2: TTS Worker            │
│   ┌─────────────────────┐            ┌─────────────────────┐       │
│   │ await stream.create │            │ while is_running:   │       │
│   │ async for chunk:    │            │   sentence = await  │       │
│   │   yield to s2s      │            │     queue.get()     │       │
│   │ async for sentence: │────────────│   async for delta:  │       │
│   │   queue.put(sent)   │            │     yield pcm       │       │
│   └─────────────────────┘            │   queue.put(audio)  │       │
│          │                           └──────────┬──────────┘       │
│          │ sentence_queue                       │ audio_queue      │
│          ▼                                      ▼                  │
│   ┌─────────────────────────────────────────────────────────┐      │
│   │                   Task 3: Audio Streamer                │      │
│   │   while is_running:                                     │      │
│   │       chunk = await audio_queue.get()                   │      │
│   │       await websocket.send_bytes(chunk.audio_data)      │      │
│   └─────────────────────────────────────────────────────────┘      │
│                              │                                     │
│                              ▼                                     │
│                        WebSocket Client                            │
└───────────────────────────────────────────────────────────────────┘
```

**Key Concurrency Benefits**:
1. **LLM Processor** doesn't wait for TTS - just puts sentences in queue
2. **TTS Worker** runs independently, processing sentences as they arrive
3. **Audio Streamer** sends audio immediately when available
4. All three tasks run concurrently using asyncio event loop

---

## File Structure

```
backend/
├── streaming_pipeline.py    # All pipeline components
├── streaming_server.py      # FastAPI server
├── stream2sentence.py       # (existing) Sentence extraction
├── boson_multimodal/
│   └── serve/
│       └── serve_engine.py  # (existing) Higgs TTS engine
└── voices/
    ├── default.wav          # Default voice reference audio
    └── default.txt          # Default voice reference text

frontend/
└── stream_test.html         # Browser test interface
```

---

## Key Implementation Notes

### 1. stream2sentence Configuration

For lowest latency TTS, use these settings:
```python
generate_sentences_async(
    generator,
    minimum_first_fragment_length=10,  # Yield first fragment quickly
    minimum_sentence_length=15,         # Balance between latency and naturalness
    quick_yield_single_sentence_fragment=True,  # Fast first fragment
    sentence_fragment_delimiters=".?!;:,\n…)]}。-",  # Punctuation triggers
)
```

### 2. Higgs Audio Streaming

Key aspects of `generate_delta_stream`:
- Returns `HiggsAudioStreamerDelta` with `audio_tokens` tensor
- Audio tokens use delay pattern - must call `revert_delay_pattern()`
- Decode in chunks (e.g., 20 tokens) for smooth streaming
- Token 1024 = padding, Token 1025 = end-of-audio

### 3. Audio Format

- Higgs outputs at 24000 Hz sample rate
- Convert to PCM16 (int16) for efficient WebSocket transfer
- Browser converts back to Float32 for Web Audio API

### 4. Voice Cloning

For voice cloning, provide reference audio in messages:
```python
messages = [
    Message(role="user", content="Reference transcript text"),
    Message(role="assistant", content=AudioContent(audio_url="path/to/ref.wav")),
    Message(role="user", content="Text to synthesize"),
]
```

---

## Questions Before Implementation

1. **Voice Selection**: Should the API support multiple voices, or use a single default voice for this demo?

2. **Crossfade**: The existing server.py has crossfade logic for smooth audio chunk transitions. Should we include this, or keep the demo simpler?

3. **Error Handling**: How should we handle LLM or TTS errors mid-stream? Options:
   - Send error message to client
   - Retry with backoff
   - Skip failed sentence

4. **Buffer Strategy**: The current plan streams immediately. Should we buffer a minimum amount before starting playback for smoother experience?

5. **Interrupt Support**: Should the pipeline support interruption (user speaks/types during playback)?
