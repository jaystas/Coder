"""
Message Manager Module
Handles all message-related database operations with Supabase
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from supabase import Client
from fastapi import HTTPException
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class Message(BaseModel):
    message_id: str
    conversation_id: str
    role: str  # "user", "assistant", "system"
    name: Optional[str] = None
    content: str
    character_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class MessageCreate(BaseModel):
    conversation_id: str
    role: str
    content: str
    name: Optional[str] = None
    character_id: Optional[str] = None


class MessageManager:
    """Message management service using Supabase"""

    def __init__(self, supabase_client: Client):
        self.supabase = supabase_client
        self.table_name = "messages"

    async def create_message(self, message_data: MessageCreate) -> Message:
        """Create a single message"""
        try:
            db_data = {
                "conversation_id": message_data.conversation_id,
                "role": message_data.role,
                "content": message_data.content,
                "name": message_data.name,
                "character_id": message_data.character_id
            }

            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create message")

            row = response.data[0]
            message = Message(
                message_id=str(row["message_id"]),
                conversation_id=str(row["conversation_id"]),
                role=row["role"],
                name=row.get("name"),
                content=row["content"],
                character_id=row.get("character_id"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at")
            )

            logger.info(f"Created message {message.message_id} in conversation {message.conversation_id}")
            return message

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def create_messages_batch(self, messages: List[MessageCreate]) -> List[Message]:
        """Create multiple messages in a single batch operation"""
        try:
            db_data = [
                {
                    "conversation_id": msg.conversation_id,
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "character_id": msg.character_id
                }
                for msg in messages
            ]

            response = self.supabase.table(self.table_name)\
                .insert(db_data)\
                .execute()

            if not response.data:
                raise HTTPException(status_code=500, detail="Failed to create messages")

            created_messages = []
            for row in response.data:
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                created_messages.append(message)

            logger.info(f"Created {len(created_messages)} messages in batch")
            return created_messages

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating messages batch: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_messages(
        self,
        conversation_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a conversation with optional pagination"""
        try:
            query = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=False)

            if limit is not None:
                query = query.limit(limit)

            if offset > 0:
                query = query.range(offset, offset + (limit or 1000) - 1)

            response = query.execute()

            messages = []
            for row in response.data:
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                messages.append(message)

            logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_recent_messages(self, conversation_id: str, n: int = 10) -> List[Message]:
        """Get the last N messages from a conversation"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=True)\
                .limit(n)\
                .execute()

            # Reverse to get chronological order
            messages = []
            for row in reversed(response.data):
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                messages.append(message)

            logger.info(f"Retrieved last {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting recent messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        async def get_last_message(self, conversation_id: str, n: int = 1) -> Message:
        """Get the last message from a conversation"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("*")\
                .eq("conversation_id", conversation_id)\
                .order("created_at", desc=True)\
                .limit(n)\
                .execute()

            # Reverse to get chronological order
            messages = []
            for row in reversed(response.data):
                message = Message(
                    message_id=str(row["message_id"]),
                    conversation_id=str(row["conversation_id"]),
                    role=row["role"],
                    name=row.get("name"),
                    content=row["content"],
                    character_id=row.get("character_id"),
                    created_at=row.get("created_at"),
                    updated_at=row.get("updated_at")
                )
                messages.append(message)

            logger.info(f"Retrieved last {len(messages)} messages for conversation {conversation_id}")
            return messages

        except Exception as e:
            logger.error(f"Error getting recent messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def get_message_count(self, conversation_id: str) -> int:
        """Get the total number of messages in a conversation"""
        try:
            response = self.supabase.table(self.table_name)\
                .select("message_id", count="exact")\
                .eq("conversation_id", conversation_id)\
                .execute()

            count = response.count if hasattr(response, 'count') else len(response.data)
            logger.info(f"Conversation {conversation_id} has {count} messages")
            return count

        except Exception as e:
            logger.error(f"Error getting message count for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    async def delete_messages(self, conversation_id: str) -> bool:
        """Delete all messages for a conversation"""
        try:
            response = self.supabase.table(self.table_name)\
                .delete()\
                .eq("conversation_id", conversation_id)\
                .execute()

            logger.info(f"Deleted messages for conversation {conversation_id}")
            return True

        except Exception as e:
            logger.error(f"Error deleting messages for conversation {conversation_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
