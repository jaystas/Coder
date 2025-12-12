"""
ZonosEngine: RealtimeTTS Engine for Zonos TTS

This engine integrates the Zonos TTS model with RealtimeTTS for real-time
text-to-speech synthesis using streaming audio generation.
"""

import torch
import torchaudio
import numpy as np
from typing import Union, List
from RealtimeTTS.engines.base_engine import BaseEngine
from RealtimeTTS.audio_formats import AudioFormat

try:
    from zonos import Zonos
except ImportError:
    print("Zonos library not found. Please install the Zonos TTS package.")
    raise


class ZonosEngine(BaseEngine):
    """
    RealtimeTTS engine for Zonos TTS model.
    
    Provides real-time text-to-speech synthesis using the Zonos transformer model
    with streaming audio generation capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "Zyphra/Zonos-v0.1-transformer",
        reference_audio_path: str = "/workspace/tts/voices/emily.wav",
        language: str = "en-us",
        chunk_schedule: List[int] = None,
        chunk_overlap: int = 1,
        device: str = None
    ):
        """
        Initialize the ZonosEngine.
        
        Args:
            model_name (str): Zonos model identifier for loading from pretrained
            reference_audio_path (str): Path to reference speaker audio file
            language (str): Language code for synthesis
            chunk_schedule (List[int]): Audio chunk schedule for streaming
            chunk_overlap (int): Overlap between audio chunks
            device (str): Device to run model on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        
        self.engine_name = "zonos"
        self.can_consume_generators = False  # Process sentence by sentence
        
        # Configuration
        self.model_name = model_name
        self.reference_audio_path = reference_audio_path
        self.language = language
        self.chunk_schedule = chunk_schedule or [128]
        self.chunk_overlap = chunk_overlap
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Audio configuration - Zonos typically outputs at 22050 Hz
        self.sample_rate = 22050
        self.channels = 1
        self.format = AudioFormat.FLOAT32
        
        # Initialize model and speaker embedding
        self._load_model()
        self._load_speaker_embedding()
        
    def _load_model(self):
        """Load the Zonos model."""
        print(f"Loading Zonos model: {self.model_name}")
        try:
            self.model = Zonos.from_pretrained(self.model_name, device=self.device)
            self.model.requires_grad_(False).eval()
            print("Zonos model loaded successfully")
        except Exception as e:
            print(f"Error loading Zonos model: {e}")
            raise
            
    def _load_speaker_embedding(self):
        """Load and create speaker embedding from reference audio."""
        print(f"Loading reference audio: {self.reference_audio_path}")
        try:
            wav, sr = torchaudio.load(self.reference_audio_path)
            self.speaker_embedding = self.model.make_speaker_embedding(wav, sr)
            print("Speaker embedding created successfully")
        except Exception as e:
            print(f"Error loading reference audio: {e}")
            raise
            
    def get_stream_info(self):
        """
        Returns the audio stream configuration information.

        Returns:
            tuple: (format, channels, sample_rate)
        """
        return self.format, self.channels, self.sample_rate
        
    def synthesize(self, text: str) -> bool:
        """
        Synthesizes text to audio stream using Zonos TTS.
        
        Args:
            text (str): Text to synthesize
            
        Returns:
            bool: True if synthesis successful, False otherwise
        """
        if not text or not text.strip():
            return True
            
        try:
            # Call synthesis start callback if available
            if self.on_playback_start:
                self.on_playback_start()
                
            # Create generator for Zonos streaming
            def condition_generator():
                yield {
                    "text": text.strip(),
                    "speaker": self.speaker_embedding,
                    "language": self.language,
                }
            
            # Generate streaming audio using Zonos
            stream_generator = self.model.stream(
                cond_dicts_generator=condition_generator(),
                chunk_schedule=self.chunk_schedule,
                chunk_overlap=self.chunk_overlap,
            )
            
            # Process each audio chunk
            for i, audio_chunk in enumerate(stream_generator):
                # Convert tensor to numpy array
                np_chunk = audio_chunk.cpu().numpy()
                
                # Convert to bytes (raw PCM data)
                audio_bytes = np_chunk.astype(np.float32).tobytes()
                
                # Put audio chunk in queue for playback
                self.queue.put(audio_bytes)
                
                # Call audio chunk callback if available
                if self.on_audio_chunk:
                    self.on_audio_chunk(audio_bytes)
            
            return True
            
        except Exception as e:
            print(f"Error in Zonos synthesis: {e}")
            return False
            
    def get_voices(self):
        """
        Retrieves available voices.
        
        For Zonos, voices are determined by reference audio files.
        This returns the current speaker embedding info.
        
        Returns:
            list: List containing current voice information
        """
        return [{
            "name": "zonos_speaker",
            "id": "zonos_speaker",
            "reference_audio": self.reference_audio_path,
            "language": self.language
        }]
        
    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice by loading a new reference audio file.
        
        Args:
            voice (Union[str, object]): Path to reference audio file or voice object
        """
        if isinstance(voice, str):
            # Assume it's a path to reference audio
            self.reference_audio_path = voice
            self._load_speaker_embedding()
        elif isinstance(voice, dict) and "reference_audio" in voice:
            # Voice object with reference_audio field
            self.reference_audio_path = voice["reference_audio"]
            self._load_speaker_embedding()
        else:
            print(f"Unsupported voice format: {type(voice)}")
            
    def set_voice_parameters(self, **voice_parameters):
        """
        Sets voice parameters for synthesis.
        
        Args:
            **voice_parameters: Voice parameters such as:
                - language: Language code
                - chunk_schedule: Audio chunk schedule
                - chunk_overlap: Overlap between chunks
        """
        if "language" in voice_parameters:
            self.language = voice_parameters["language"]
            
        if "chunk_schedule" in voice_parameters:
            self.chunk_schedule = voice_parameters["chunk_schedule"]
            
        if "chunk_overlap" in voice_parameters:
            self.chunk_overlap = voice_parameters["chunk_overlap"]
            
        if "reference_audio" in voice_parameters:
            self.reference_audio_path = voice_parameters["reference_audio"]
            self._load_speaker_embedding()
            
    def shutdown(self):
        """
        Shuts down the engine and releases resources.
        """
        if hasattr(self, 'model'):
            # Clear model from memory
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print("ZonosEngine shutdown complete")