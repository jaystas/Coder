"""
Concurrent Streaming TTS Server

FastAPI server for the LLM → TTS streaming pipeline.
Run with: python -m backend.streaming_server
"""

import os
import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from backend.streaming_pipeline import StreamingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline: Optional[StreamingPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup pipeline"""
    global pipeline

    api_key = os.getenv(
        "OPENROUTER_API_KEY",
        "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"
    )
    pipeline = StreamingPipeline(api_key=api_key)

    logger.info("=" * 60)
    logger.info("Initializing Concurrent TTS Streaming Pipeline...")
    logger.info("=" * 60)

    await pipeline.initialize()
    await pipeline.start_tts_worker()

    logger.info("Pipeline ready! TTS worker running.")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down pipeline...")
    await pipeline.stop_tts_worker()
    logger.info("Pipeline shutdown complete.")


app = FastAPI(
    title="Concurrent TTS Streaming Server",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "pipeline_ready": pipeline._initialized if pipeline else False}


@app.websocket("/ws/stream")
async def stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS audio"""
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            # Wait for user prompt
            data = await websocket.receive_json()

            if data.get("type") == "prompt":
                prompt = data.get("text", "")
                session_id = data.get("session_id", "default")

                logger.info(f"Received prompt: {prompt[:50]}...")

                # Notify client that streaming is starting
                await websocket.send_json({
                    "type": "stream_start",
                    "session_id": session_id
                })

                # Callback to stream text chunks to client
                async def on_text_chunk(chunk: str):
                    await websocket.send_json({
                        "type": "text_chunk",
                        "text": chunk
                    })

                # Process and stream audio
                full_response = await pipeline.process_and_stream(
                    prompt=prompt,
                    websocket=websocket,
                    session_id=session_id,
                    on_text_chunk=on_text_chunk
                )

                # Send the full text response
                await websocket.send_json({
                    "type": "text_complete",
                    "text": full_response,
                    "session_id": session_id
                })

                logger.info(f"Stream complete for session {session_id}")

            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")


# Serve static files (frontend)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "backend.streaming_server:app",
        host="0.0.0.0",
        port=8001,
        reload=False
    )
