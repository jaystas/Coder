"""
Conversation Manager Module
Handles all conversation-related database operations with Supabase
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from supabase import Client
from fastapi import HTTPException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Conversation(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    active_characters: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ConversationCreate(BaseModel):
    title: Optional[str] = None
    active_characters: List[str] = []


class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    active_characters: Optional[List[str]] = None


class ConversationManager:
    """Conversation management service using Supabase"""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.table_name = "conversations"

    def _generate_title(self, first_message: Optional[str] = None) -> str:
        """
        Generate a conversation title.
        If first_message is provided, use the first 50 chars.
        Otherwise, generate based on timestamp.
        """
        if first_message and first_message.strip():
            # Take first 50 characters and add ellipsis if needed
            title = first_message.strip()[:50]
            if len(first_message.strip()) > 50:
                title += "..."
            return title

        # Generate timestamp-based title
        from datetime import datetime
        return f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    async def create_conversation(
        self,
        conversation_data: ConversationCreate,
        auto_generate_title: bool = True
    ) -> Conversation:
        """Create a new conversation"""
        try:
            title = conversation_data.title
            if auto_generate_title and not title:
                title = self._generate_title()

            db_data = {
                "title": title,
                "active_characters": conversation_data.active_characters or []
            }

            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create conversation")

            row = response.data[0]
            conversation = Conversation(
                conversation_id=str(row["conversation_id"]),
                title=row.get("title"),
                active_characters=row.get("active_characters", []),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Created conversation {conversation.conversation_id} with title: {conversation.title}")
            return conversation

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_conversation(self, conversation_id: str) -> Conversation:
        """Get a specific conversation by ID"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Conversation not found")

            row = response.data[0]
            conversation = Conversation(
                conversation_id=str(row["conversation_id"]),
                title=row.get("title"),
                active_characters=row.get("active_characters", []),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Retrieved conversation {conversation_id}")
            return conversation

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_all_conversations(
        self,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Conversation]:
        """Get all conversations ordered by most recent first"""
        try:
            query = self.supabase.table(self.table_name)\
                .select("*")\
                .order("updated_at", desc=True)

            if limit is not None:
                query = query.limit(limit)

            if offset > 0:
                query = query.range(offset, offset + (limit or 1000) - 1)

            response = query.execute()

            conversations = []
            for row in response.data:
                conversation = Conversation(
                    conversation_id=str(row["conversation_id"]),
                    title=row.get("title"),
                    active_characters=row.get("active_characters", []),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                conversations.append(conversation)

            logger.info(f"Retrieved {len(conversations)} conversations")
            return conversations

        except Exception as e:
            logger.error(f"Error getting all conversations: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_conversation(
        self,
        conversation_id: str,
        conversation_data: ConversationUpdate
    ) -> Conversation:
        """Update an existing conversation"""
        try:
            update_data = {}
            if conversation_data.title is not None:
                update_data["title"] = conversation_data.title
            if conversation_data.active_characters is not None:
                update_data["active_characters"] = conversation_data.active_characters

            if not update_data:
                raise HTTPException(status_code=400, detail="No fields to update")

            response = self.supabase.table(self.table_name)\
                .update(update_data)\
                .eq("conversation_id", conversation_id)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=404, detail="Conversation not found")

            logger.info(f"Updated conversation {conversation_id}")
            return await self.get_conversation(conversation_id)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def update_title(self, conversation_id: str, title: str) -> Conversation:
        """Update just the title of a conversation"""
        return await self.update_conversation(
            conversation_id,
            ConversationUpdate(title=title)
        )

    async def update_active_characters(
        self,
        conversation_id: str,
        active_characters: List[str]
    ) -> Conversation:
        """Update the active characters in a conversation"""
        return await self.update_conversation(
            conversation_id,
            ConversationUpdate(active_characters=active_characters)
        )

    async def add_character(self, conversation_id: str, character_id: str) -> Conversation:
        """Add a character to the active_characters list"""
        try:
            conversation = await self.get_conversation(conversation_id)

            if character_id not in conversation.active_characters:
                active_characters = conversation.active_characters + [character_id]
                return await self.update_active_characters(conversation_id, active_characters)

            return conversation

        except Exception as e:
            logger.error(f"Error adding character {character_id} to conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def remove_character(self, conversation_id: str, character_id: str) -> Conversation:
        """Remove a character from the active_characters list"""
        try:
            conversation = await self.get_conversation(conversation_id)

            if character_id in conversation.active_characters:
                active_characters = [c for c in conversation.active_characters if c != character_id]
                return await self.update_active_characters(conversation_id, active_characters)

            return conversation

        except Exception as e:
            logger.error(f"Error removing character {character_id} from conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation (messages will be cascade deleted)"""
        try:
            # Verify conversation exists
            await self.get_conversation(conversation_id)

            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq("conversation_id", conversation_id)\
                .execute()

            logger.info(f"Deleted conversation {conversation_id}")
            return True

        except HTTPException as e:
            if e.status_code == 404:
                raise
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def auto_update_title_from_first_message(
        self,
        conversation_id: str,
        first_message: str
    ) -> Conversation:
        """
        Auto-generate and update title from the first message if title is not set.
        This is useful to call after the first user message is saved.
        """
        try:
            conversation = await self.get_conversation(conversation_id)

            # Only update if title is empty or auto-generated (contains timestamp)
            if not conversation.title or "Conversation" in conversation.title:
                new_title = self._generate_title(first_message)
                return await self.update_title(conversation_id, new_title)

            return conversation

        except Exception as e:
            logger.error(f"Error auto-updating title for conversation {conversation_id}: {e}")
            # Don't raise - this is a nice-to-have feature
            return conversation
