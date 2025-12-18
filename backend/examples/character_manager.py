"""
Character Manager Module
Handles all character-related database operations with Supabase
"""

from typing import List, Optional
from pydantic import BaseModel
from supabase import Client
from fastapi import HTTPException
import logging
import re

logger = logging.getLogger(__name__)


class Character(BaseModel):
    id: str
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class CharacterCreate(BaseModel):
    name: str
    voice: str = ""
    system_prompt: str = ""
    image_url: str = ""
    images: List[str] = []
    is_active: bool = False


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    voice: Optional[str] = None
    system_prompt: Optional[str] = None
    image_url: Optional[str] = None
    images: Optional[List[str]] = None
    is_active: Optional[bool] = None


class CharacterManager:
    """Character management service using Supabase"""
    
    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.table_name = "characters"
    
    def _generate_id(self, name: str) -> str:
        """
        Generate a sequential ID from the character name.
        Format: lowercase-name-with-dashes-###
        Example: "Olivia Barns" -> "olivia-barns-001", "olivia-barns-002", etc.
        """
        # Convert to lowercase and replace spaces with dashes
        base_id = name.lower().strip()
        base_id = re.sub(r'[^a-z0-9\s-]', '', base_id)  # Remove special chars
        base_id = re.sub(r'\s+', '-', base_id)  # Replace spaces with dashes
        base_id = re.sub(r'-+', '-', base_id)  # Remove multiple consecutive dashes
        base_id = base_id.strip('-')  # Remove leading/trailing dashes

        try:
            # Query existing IDs with the same base pattern
            response = self.supabase.table(self.table_name)\
                .select("id")\
                .like("id", f"{base_id}-%")\
                .execute()

            # Find the highest number in existing IDs
            highest_num = 0
            pattern = re.compile(rf"^{re.escape(base_id)}-(\d{{3}})$")

            for row in response.data:
                match = pattern.match(row["id"])
                if match:
                    num = int(match.group(1))
                    highest_num = max(highest_num, num)

            # Generate next sequential number
            next_num = highest_num + 1
            character_id = f"{base_id}-{next_num:03d}"

            logger.info(f"Generated character id: {character_id}")
            return character_id

        except Exception as e:
            logger.error(f"Error generating character id: {e}")
            # Fallback to 001 if there's an error
            return f"{base_id}-001"
    
    async def get_all_characters(self) -> List[Character]:
        """Get all characters"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice": row["voice"] or "",
                    "system_prompt": row["system_prompt"] or "",
                    "image_url": row["image_url"] or "",
                    "images": row["images"] or [],
                    "is_active": row["is_active"] or False,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                characters.append(Character(**character_data))

            logger.info(f"Retrieved {len(characters)} characters")
            return characters

        except Exception as e:
            logger.error(f"Error getting characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def get_active_characters(self) -> List[Character]:
        """Get all active characters"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("is_active", True)\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice": row["voice"] or "",
                    "system_prompt": row["system_prompt"] or "",
                    "image_url": row["image_url"] or "",
                    "images": row["images"] or [],
                    "is_active": row["is_active"] or False,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                characters.append(Character(**character_data))

            logger.info(f"Retrieved {len(characters)} active characters")
            return characters

        except Exception as e:
            logger.error(f"Error getting active characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def get_character(self, character_id: str) -> Character:
        """Get a specific character by ID"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("id", character_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Character not found")

            row = response.data[0]
            character_data = {
                "id": row["id"],
                "name": row["name"],
                "voice": row["voice"] or "",
                "system_prompt": row["system_prompt"] or "",
                "image_url": row["image_url"] or "",
                "images": row["images"] or [],
                "is_active": row["is_active"] or False,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"]
            }

            return Character(**character_data)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def create_character(self, character_data: CharacterCreate) -> Character:
        """Create a new character"""
        try:
            character_id = self._generate_id(character_data.name)

            db_data = {
                "id": character_id,
                "name": character_data.name,
                "voice": character_data.voice,
                "system_prompt": character_data.system_prompt,
                "image_url": character_data.image_url,
                "images": character_data.images,
                "is_active": character_data.is_active
            }

            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create character")

            return await self.get_character(character_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating character: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def update_character(self, character_id: str, character_data: CharacterUpdate) -> Character:
        """Update an existing character"""
        try:
            update_data = {}
            if character_data.name is not None:
                update_data["name"] = character_data.name
            if character_data.voice is not None:
                update_data["voice"] = character_data.voice
            if character_data.system_prompt is not None:
                update_data["system_prompt"] = character_data.system_prompt
            if character_data.image_url is not None:
                update_data["image_url"] = character_data.image_url
            if character_data.images is not None:
                update_data["images"] = character_data.images
            if character_data.is_active is not None:
                update_data["is_active"] = character_data.is_active

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = self.supabase.table(self.table_name)\
                .update(update_data)\
                .eq("id", character_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Character not found")

            return await self.get_character(character_id)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def set_character_active(self, character_id: str, is_active: bool) -> Character:
        """Set character active status"""
        return await self.update_character(character_id, CharacterUpdate(is_active=is_active))
    
    async def delete_character(self, character_id: str) -> bool:
        """Delete a character"""
        try:
            await self.get_character(character_id)

            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq("id", character_id)\
                .execute()

            logger.info(f"Deleted character: {character_id}")
            return True
            
        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting character {character_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    
    async def search_characters(self, query: str) -> List[Character]:
        """Search characters by name"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .ilike("name", f"%{query}%")\
                .execute()

            characters = []
            for row in response.data:
                character_data = {
                    "id": row["id"],
                    "name": row["name"],
                    "voice": row["voice"] or "",
                    "system_prompt": row["system_prompt"] or "",
                    "image_url": row["image_url"] or "",
                    "images": row["images"] or [],
                    "is_active": row["is_active"] or False,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"]
                }
                characters.append(Character(**character_data))

            logger.info(f"Found {len(characters)} characters matching '{query}'")
            return characters
            
        except Exception as e:
            logger.error(f"Error searching characters: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
