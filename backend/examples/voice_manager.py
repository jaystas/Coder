"""
Voice Manager Module
Handles all voice-related database operations with Supabase
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from supabase import Client
from fastapi import HTTPException
import logging
import json
import re
import threading

logger = logging.getLogger(__name__)


class Voice(BaseModel):
    voice: str  # Serves as both PK and display name
    method: str
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""
    audio_tokens: Optional[str] = None  # Serialized audio tokens (JSON)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class VoiceCreate(BaseModel):
    voice: str
    method: str
    audio_path: str = ""
    text_path: str = ""
    speaker_desc: str = ""
    scene_prompt: str = ""


class VoiceUpdate(BaseModel):
    voice: Optional[str] = None
    method: Optional[str] = None
    audio_path: Optional[str] = None
    text_path: Optional[str] = None
    speaker_desc: Optional[str] = None
    scene_prompt: Optional[str] = None
    audio_tokens: Optional[str] = None


class VoiceManager:
    """Voice management service using Supabase with caching"""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.table_name = "voices"
        self.voice_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.Lock()
    
    def _serialize_audio_tokens(self, tokens: Optional[list]) -> str:
        """Serialize audio tokens to JSON string"""
        if tokens is None:
            return "[]"
        return json.dumps(tokens)
    
    def _deserialize_audio_tokens(self, tokens_str: Optional[str]) -> list:
        """Deserialize audio tokens from JSON string"""
        if not tokens_str:
            return []
        try:
            return json.loads(tokens_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to deserialize audio tokens: {tokens_str}")
            return []
    
    async def get_all_voices(self) -> List[Voice]:
        """Get all voices from database"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .execute()

            voices = []
            for row in response.data:
                voice_data = {
                    "voice": row["voice"],
                    "method": row["method"],
                    "audio_path": row.get("audio_path", ""),
                    "text_path": row.get("text_path", ""),
                    "speaker_desc": row.get("speaker_desc", ""),
                    "scene_prompt": row.get("scene_prompt", ""),
                    "audio_tokens": row.get("audio_tokens"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at")
                }
                voices.append(Voice(**voice_data))

            logger.info(f"Retrieved {len(voices)} voices from database")
            return voices
            
        except Exception as e:
            logger.error(f"Error getting all voices: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def get_voice(self, voice: str) -> Voice:
        """Get a specific voice by name"""
        # Check cache first
        with self.cache_lock:
            if voice in self.voice_cache:
                logger.debug(f"Retrieved voice {voice} from cache")
                return self.voice_cache[voice]["config"]

        # Not in cache, fetch from database
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("voice", voice)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")

            row = response.data[0]
            voice_data = {
                "voice": row["voice"],
                "method": row["method"],
                "audio_path": row.get("audio_path", ""),
                "text_path": row.get("text_path", ""),
                "speaker_desc": row.get("speaker_desc", ""),
                "scene_prompt": row.get("scene_prompt", ""),
                "audio_tokens": row.get("audio_tokens"),
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at")
            }

            voice = Voice(**voice_data)

            # Add to cache
            with self.cache_lock:
                self.voice_cache[voice] = {
                    "config": voice,
                    "audio_tokens": self._deserialize_audio_tokens(voice.audio_tokens)
                }

            logger.info(f"Retrieved voice {voice} from database and cached")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting voice {voice}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def create_voice(self, voice_data: VoiceCreate) -> Voice:
        """Create a new voice"""
        try:
            db_data = {
                "voice": voice_data.voice,
                "method": voice_data.method,
                "audio_path": voice_data.audio_path,
                "text_path": voice_data.text_path,
                "speaker_desc": voice_data.speaker_desc,
                "scene_prompt": voice_data.scene_prompt,
                "audio_tokens": "[]"  # Initialize with empty tokens
            }

            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create voice")

            voice = await self.get_voice(voice_data.voice)

            # Add to cache
            with self.cache_lock:
                self.voice_cache[voice_data.voice] = {
                    "config": voice,
                    "audio_tokens": []
                }

            logger.info(f"Created voice: {voice.voice}")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating voice: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def update_voice(self, voice: str, voice_data: VoiceUpdate) -> Voice:
        """Update an existing voice"""
        try:
            update_data = {}
            # Note: Cannot update voice name as it's the PK
            if voice_data.method is not None:
                update_data["method"] = voice_data.method
            if voice_data.audio_path is not None:
                update_data["audio_path"] = voice_data.audio_path
            if voice_data.text_path is not None:
                update_data["text_path"] = voice_data.text_path
            if voice_data.speaker_desc is not None:
                update_data["speaker_desc"] = voice_data.speaker_desc
            if voice_data.scene_prompt is not None:
                update_data["scene_prompt"] = voice_data.scene_prompt
            if voice_data.audio_tokens is not None:
                update_data["audio_tokens"] = voice_data.audio_tokens

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = self.supabase.table(self.table_name)\
                .update(update_data)\
                .eq("voice", voice)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Voice not found")

            voice = await self.get_voice(voice)

            # Update cache
            with self.cache_lock:
                if voice in self.voice_cache:
                    self.voice_cache[voice]["config"] = voice
                    if voice_data.audio_tokens is not None:
                        self.voice_cache[voice]["audio_tokens"] = self._deserialize_audio_tokens(voice.audio_tokens)

            logger.info(f"Updated voice: {voice}")
            return voice
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating voice {voice}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def delete_voice(self, voice: str) -> bool:
        """Delete a voice"""
        try:
            await self.get_voice(voice)

            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq("voice", voice)\
                .execute()

            # Remove from cache
            with self.cache_lock:
                if voice in self.voice_cache:
                    del self.voice_cache[voice]

            logger.info(f"Deleted voice: {voice}")
            return True
            
        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting voice {voice}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    def get_cached_audio_tokens(self, voice: str) -> Optional[list]:
        """Get audio tokens from cache if available"""
        with self.cache_lock:
            if voice in self.voice_cache:
                return self.voice_cache[voice]["audio_tokens"]
        return None

    def update_cached_audio_tokens(self, voice: str, audio_tokens: list):
        """Update audio tokens in cache"""
        with self.cache_lock:
            if voice in self.voice_cache:
                self.voice_cache[voice]["audio_tokens"] = audio_tokens
    
    def clear_cache(self):
        """Clear the entire voice cache"""
        with self.cache_lock:
            self.voice_cache.clear()
        logger.info("Voice cache cleared")
