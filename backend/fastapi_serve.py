# Standard library
import os
import re
import json
import uuid
import asyncio
import logging
import threading
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Callable, Optional, Dict, List, AsyncGenerator, Awaitable

# Third-party
import uvicorn
from supabase import create_client, Client
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Local imports
from backend.RealtimeSTT import AudioToTextRecorder
from backend.stream2sentence import generate_sentences_async

# Database Director - centralized CRUD operations for Supabase
from backend.database_director import (
    db,
    Character,
    CharacterCreate,
    CharacterUpdate,
    Voice,
    VoiceCreate,
    VoiceUpdate,
    Conversation,
    ConversationCreate,
    ConversationUpdate,
    Message as ConversationMessage,
    MessageCreate,
)

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jslevsbvapopncjehhva.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpzbGV2c2J2YXBvcG5jamVoaHZhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgwNTQwOTMsImV4cCI6MjA3MzYzMDA5M30.DotbJM3IrvdVzwfScxOtsSpxq0xsj7XxI3DvdiqDSrE")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

logging.basicConfig(filename="filelogger.log", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class TTSSentence:
    """Sentence queued for TTS synthesis"""
    text: str
    index: int
    session_id: str
    is_final: bool = False


@dataclass
class AudioChunk:
    """PCM audio chunk ready for streaming"""
    audio_bytes: bytes
    sentence_index: int
    chunk_index: int
    session_id: str
    is_final: bool = False

@dataclass
class ModelSettings:
    model: str
    temperature: float 
    top_p: float
    min_p: float
    top_k: int
    frequency_penalty: float
    presence_penalty: float
    repetition_penalty: float

########################################
##--        Queue Management        --##
########################################

class StreamingQueues:
    """Manages asyncio queues for the pipeline"""

    def __init__(self):
        # Queue for transcribed user messages (STT → Chat)
        self.transcribe_queue: asyncio.Queue[str] = asyncio.Queue()
        # Queue for sentences to be synthesized (Chat → TTS)
        self.sentence_queue: asyncio.Queue[TTSSentence] = asyncio.Queue()
        # Queue for audio chunks to stream (TTS → WebSocket)
        self.audio_queue: asyncio.Queue[AudioChunk] = asyncio.Queue()

########################################
##--  Speech to Text Transcription  --##
########################################

Callback = Callable[..., Optional[Awaitable[None]]]

class Transcribe:
    """Realtime transcription of user's audio prompt"""

    def __init__(self,
        on_transcription_update: Optional[Callback] = None,
        on_transcription_stabilized: Optional[Callback] = None,
        on_transcription_finished: Optional[Callback] = None,
        on_vad_detect_start: Optional[Callback] = None,
        on_vad_detect_stop: Optional[Callback] = None,
        on_vad_start: Optional[Callback] = None,
        on_vad_stop: Optional[Callback] = None,
        on_recording_start: Optional[Callback] = None,
        on_recording_stop: Optional[Callback] = None,
    ):
        # Event loop for async callback dispatching (set by WebSocketManager)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Transcription thread
        self._thread: Optional[threading.Thread] = None

        # Callback registry - keys match the internal callback method names
        self.callbacks: Dict[str, Optional[Callback]] = {
            'on_transcription_update': on_transcription_update,
            'on_transcription_stabilized': on_transcription_stabilized,
            'on_transcription_finished': on_transcription_finished,
            'on_vad_detect_start': on_vad_detect_start,
            'on_vad_detect_stop': on_vad_detect_stop,
            'on_vad_start': on_vad_start,
            'on_vad_stop': on_vad_stop,
            'on_recording_start': on_recording_start,
            'on_recording_stop': on_recording_stop,
        }

        self.is_listening = False

        self.recorder = AudioToTextRecorder(
            model="small.en",
            language="en",
            enable_realtime_transcription=True,
            realtime_processing_pause=0.1,
            realtime_model_type="small.en",
            on_realtime_transcription_update=self._on_transcription_update,
            on_realtime_transcription_stabilized=self._on_transcription_stabilized,
            on_recording_start=self._on_recording_start,
            on_recording_stop=self._on_recording_stop,
            on_vad_detect_start=self._on_vad_detect_start,
            on_vad_detect_stop=self._on_vad_detect_stop,
            on_vad_start=self._on_vad_start,
            on_vad_stop=self._on_vad_stop,
            silero_sensitivity=0.4,
            webrtc_sensitivity=3,
            post_speech_silence_duration=0.7,
            min_length_of_recording=0.5,
            spinner=False,
            level=logging.WARNING,
            use_microphone=False
        )

    def transcriber(self):
        """Transcribes in real-time from browser audio feed"""

        while self.is_listening:
            try:
                user_message = self.recorder.text()

                if user_message and user_message.strip():
                    callback = self.callbacks.get('on_transcription_finished')
                    if callback:
                        self.run_callback(callback, user_message)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")

        
    def run_callback(self, callback: Optional[Callback], *args) -> None:
        """Run a user callback from a RealtimeSTT background thread."""

        if callback is None or self.loop is None:
            return

        if asyncio.iscoroutinefunction(callback):
            asyncio.run_coroutine_threadsafe(callback(*args), self.loop)

        else:
            self.loop.call_soon_threadsafe(callback, *args)

    def feed_audio(self, audio_bytes: bytes):
        """Feed raw PCM audio bytes (16kHz, 16-bit, mono)"""
        if self.recorder:
            try:
                self.recorder.feed_audio(audio_bytes, original_sample_rate=16000)
            except Exception as e:
                logger.error(f"Failed to feed audio to recorder: {e}")

    def start_listening(self):
        """Start listening for audio input"""
        if self.is_listening:
            return  # Already listening

        self.is_listening = True

        # Start transcription in background thread
        self._thread = threading.Thread(target=self.transcriber, daemon=True)
        self._thread.start()
        logger.info("Started listening for audio")

    def stop_listening(self):
        """Stop listening for audio input"""
        self.is_listening = False

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
            self._thread = None

        logger.info("Stopped listening for audio")

    def _on_transcription_update(self, text: str) -> None:
        """RealtimeSTT callback: real-time transcription update."""
        self.run_callback(self.callbacks.get('on_transcription_update'), text)

    def _on_transcription_stabilized(self, text: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_stabilized'), text)

    def _on_transcription_finished(self, user_message: str) -> None:
        """RealtimeSTT callback: stabilized transcription."""
        self.run_callback(self.callbacks.get('on_transcription_finished'), user_message)

    def _on_vad_detect_start(self) -> None:
        """RealtimeSTT callback: started listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_start'))

    def _on_vad_detect_stop(self) -> None:
        """RealtimeSTT callback: stopped listening for voice activity."""
        self.run_callback(self.callbacks.get('on_vad_detect_stop'))

    def _on_vad_start(self) -> None:
        """RealtimeSTT callback: voice activity started."""
        self.run_callback(self.callbacks.get('on_vad_start'))

    def _on_vad_stop(self) -> None:
        """RealtimeSTT callback: voice activity stopped."""
        self.run_callback(self.callbacks.get('on_vad_stop'))

    def _on_recording_start(self) -> None:
        """RealtimeSTT callback: recording started."""
        self.run_callback(self.callbacks.get('on_recording_start'))

    def _on_recording_stop(self) -> None:
        """RealtimeSTT callback: recording stopped."""
        self.run_callback(self.callbacks.get('on_recording_stop'))


########################################
##--       LLM Chat Functions       --##
########################################

class ChatFunctions:
    """Manages LLM chat sessions and conversation history"""

    def __init__(self):
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_id: Optional[str] = None
        self.model_settings: Optional[ModelSettings] = None
        self.active_characters: List[Character] = []

    async def start_new_chat(self):
        """Start a new chat session"""
        self.conversation_history = []
        self.conversation_id = str(uuid.uuid4())

    def set_model_settings(self, model_settings: ModelSettings):
        """Set model settings for LLM requests"""

        self.model_settings = model_settings

    def clear_conversation_history(self):
        """Clear the conversation history"""

        self.conversation_history = []

    def wrap_with_character_tags(self, text: str, character_name: str) -> str:
        """Wrap response text with character name XML tags for conversation history."""

        return f"<{character_name}>{text}</{character_name}>"

    def create_character_instruction_message(self, character: Character) -> Dict[str, str]:
        """Create character instruction message for group chat with character tags."""

        return {
            'role': 'system',
            'content': f'Based on the conversation history above provide the next reply as {character.name}. Your response should include only {character.name}\'s reply. Do not respond for/as anyone else.'
        }

    def parse_character_mentions(self, message: str, active_characters: List[Character]) -> List[Character]:
        """Parse a message for character mentions in order of appearance"""

        mentioned_characters = []
        processed_characters = set()

        name_mentions = []

        for character in active_characters:
            name_parts = character.name.lower().split()

            for name_part in name_parts:
                pattern = r'\b' + re.escape(name_part) + r'\b'
                for match in re.finditer(pattern, message, re.IGNORECASE):
                    name_mentions.append({
                        'character': character,
                        'position': match.start(),
                        'name_part': name_part
                    })

        name_mentions.sort(key=lambda x: x['position'])

        for mention in name_mentions:
            if mention['character'].id not in processed_characters:
                mentioned_characters.append(mention['character'])
                processed_characters.add(mention['character'].id)

        if not mentioned_characters:
            mentioned_characters = sorted(active_characters, key=lambda c: c.name)

        return mentioned_characters
    
    def get_model_settings(self) -> ModelSettings:
        """Get current model settings for the LLM request"""
        if self.model_settings is None:
            # Return default settings if not set
            return ModelSettings(
                model="meta-llama/llama-3.1-8b-instruct",
                temperature=0.7,
                top_p=0.9,
                min_p=0.0,
                top_k=40,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                repetition_penalty=1.0
            )
        return self.model_settings

    async def prepare_conversation_structure(self, messages: str, character: Character, user_name: str = "Jay", model_settings: ModelSettings = None, user_message=str):
        """Include Conversation History, Model Settings etc. with User Message"""

        self.conversation_history.append({"role": "user", "name": user_name, "content": user_message})

        mentioned_characters = self.parse_character_mentions(message=user_message, active_characters=self.active_characters)


    async def send_message_to_character(self, user_message: str, character: Character):
        """Send user message to a character and get streaming response"""
        # TODO: Integrate with streaming pipeline
        pass

    async def character_response_stream(self, messages: List[Dict[str, str]], character: Character) -> AsyncGenerator[str, None]:
        """Character response text stream from OpenRouter API"""
        # TODO: Integration point with streaming pipeline
        yield ""




########################################
##--      Conversation Pipeline     --##
########################################

class ConversationPipeline:
    """Orchestrates Complete LLM → TTS → Audio Pipeline"""

    def __init__(self):
        self.chat = ChatFunctions()
        self.queues = StreamingQueues()

    async def conversate(self, user_message: str, character: Character, websocket: WebSocket):
        """Main conversation flow: user message → LLM → TTS → audio stream"""
        # TODO: Integrate with streaming pipeline
        pass




########################################
##--        WebSocket Manager       --##
########################################

class WebSocketManager:
    """Manages WebSocket connections and routes messages"""

    def __init__(self):
        # WebSocket connection
        self.websocket: Optional[WebSocket] = None

        # Event loop reference for async callback dispatching
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Shared queues for pipeline
        self.queues = StreamingQueues()

        # Chat functions for LLM interaction
        self.chat = ChatFunctions()

        # Transcription handler with callbacks
        self.transcribe = Transcribe(
            on_transcription_update=self.on_transcription_update,
            on_transcription_stabilized=self.on_transcription_stabilized,
            on_transcription_finished=self.on_transcription_finished,
        )

    async def initialize(self):
        """Initialize resources on app startup"""
        self.loop = asyncio.get_running_loop()
        # Pass event loop to transcribe for async callback dispatching
        self.transcribe.loop = self.loop
        logger.info("WebSocketManager initialized")

    async def shutdown(self):
        """Clean up resources on app shutdown"""
        if self.transcribe:
            self.transcribe.stop_listening()
        logger.info("WebSocketManager shut down")

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.websocket = websocket
        logger.info("WebSocket client connected")

    async def disconnect(self):
        """Clean up on WebSocket disconnect"""
        if self.transcribe:
            self.transcribe.stop_listening()
        self.websocket = None
        logger.info("WebSocket client disconnected")

    async def handle_text_message(self, message: str):
        """Handle incoming text messages from WebSocket client"""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            payload = data.get("data", {})

            if message_type == "user_message":
                user_message = payload.get("text", "")
                await self.handle_user_message(user_message)

            elif message_type == "start_listening":
                if self.transcribe:
                    self.transcribe.start_listening()

            elif message_type == "stop_listening":
                if self.transcribe:
                    self.transcribe.stop_listening()

            elif message_type == "model_settings":
                settings_data = payload
                model_settings = ModelSettings(
                    model=settings_data.get("model", "meta-llama/llama-3.1-8b-instruct"),
                    temperature=float(settings_data.get("temperature", 0.7)),
                    top_p=float(settings_data.get("top_p", 0.9)),
                    min_p=float(settings_data.get("min_p", 0.0)),
                    top_k=int(settings_data.get("top_k", 40)),
                    frequency_penalty=float(settings_data.get("frequency_penalty", 0.0)),
                    presence_penalty=float(settings_data.get("presence_penalty", 0.0)),
                    repetition_penalty=float(settings_data.get("repetition_penalty", 1.0))
                )
                if self.chat:
                    self.chat.set_model_settings(model_settings)
                logger.info(f"Model settings updated: {model_settings.model}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)

    async def handle_audio_message(self, audio_bytes: bytes):
        """Feed audio for transcription"""
        if self.transcribe:
            self.transcribe.feed_audio(audio_bytes)

    async def handle_user_message(self, user_message: str):
        """Process manually sent user message"""
        await self.queues.transcribe_queue.put(user_message)

    async def send_text_to_client(self, data: dict):
        """Send JSON message to client"""
        if self.websocket:
            try:
                await self.websocket.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Failed to send text to client: {e}")

    async def stream_audio_to_client(self, audio_bytes: bytes):
        """Send binary audio to client (TTS)"""
        if self.websocket:
            try:
                await self.websocket.send_bytes(audio_bytes)
            except Exception as e:
                logger.error(f"Failed to send audio to client: {e}")

    async def on_transcription_update(self, text: str):
        """Callback: realtime transcription update"""
        await self.send_text_to_client({"type": "stt_update", "text": text})

    async def on_transcription_stabilized(self, text: str):
        """Callback: stabilized transcription"""
        await self.send_text_to_client({"type": "stt_stabilized", "text": text})

    async def on_transcription_finished(self, user_message: str):
        """Callback: final transcription complete"""
        await self.queues.transcribe_queue.put(user_message)
        await self.send_text_to_client({"type": "stt_finished", "text": user_message})

########################################
##--           FastAPI App          --##
########################################

convo_pipe = ConversationPipeline()
ws_manager = WebSocketManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up services...")
    await ws_manager.initialize()
    print("All services initialized!")
    yield
    print("Shutting down services...")
    await ws_manager.shutdown()
    print("All services shut down!")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

########################################
##--       WebSocket Endpoint       --##
########################################

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                await ws_manager.handle_text_message(message["text"])
            
            elif "bytes" in message:
                await ws_manager.handle_audio_message(message["bytes"])
    
    except WebSocketDisconnect:
        await ws_manager.disconnect()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await ws_manager.disconnect()

########################################
##--           Run Server           --##
########################################

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
