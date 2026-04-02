"""
Complete RAG System for Turtle History Management.
"""

import json
import contextvars
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our RAG components
from rag.embedder.embedding_model import get_embedding_model
from rag.chunking.json_chunking import get_chunker
from rag.storage.vector_storage import get_vector_storage
from core.paths import RAG_DATA_DIR, ensure_dirs

from core.env import load_env
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

# Load environment variables
load_env()


class TurtleRAGSystem:
    """Complete RAG system for Turtle conversation history"""
    
    def __init__(self, storage_dir: Optional[str] = None):
        """Initialize the complete RAG system"""
        ensure_dirs()
        self.storage_dir = Path(storage_dir) if storage_dir else RAG_DATA_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Temp JSON file for current session
        self.temp_session_file = self.storage_dir / "current_session.json"
        
        # Initialize components
        self.embedder = get_embedding_model()
        self.chunker = get_chunker()
        self.vector_store = get_vector_storage()
        self._vector_lock = threading.Lock()
        
        # Current session tracking
        self.current_session_id = None
        self.session_conversations = []

    def _clear_temp_session_file(self, expected_session_id: str | None = None) -> None:
        if not self.temp_session_file.exists():
            return
        if expected_session_id is None:
            self.temp_session_file.unlink(missing_ok=True)
            return
        try:
            session_data = json.loads(self.temp_session_file.read_text(encoding="utf-8"))
            if session_data.get("session_id") == expected_session_id:
                self.temp_session_file.unlink(missing_ok=True)
        except Exception:
            self.temp_session_file.unlink(missing_ok=True)

    def _index_session_conversations(
        self,
        *,
        session_id: str,
        conversations: list[dict[str, Any]],
        creation_time: str | None = None,
    ) -> bool:
        if not session_id or not conversations:
            return True

        session_data = {
            session_id: {
                "creation_time": creation_time or datetime.now().isoformat(),
                "conversations": conversations,
            }
        }
        chunks = self.chunker.chunk_session_conversations(session_data)
        if not chunks:
            return True

        chunk_contents = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_for_storage(chunk_contents)
        with self._vector_lock:
            self.vector_store.add_chunks(chunks, embeddings)
        return True

    @staticmethod
    def _fallback_chunk_turn_records(
        *,
        session_id: str,
        turn_records: list[dict[str, Any]],
        creation_time: str,
        max_chunk_tokens: int = 200,
        overlap_tokens: int = 80,
    ) -> list[dict[str, Any]]:
        if not turn_records:
            return []

        lines: list[str] = []
        for record in turn_records:
            kind = str(record.get("kind", "unknown")).strip().lower()
            content = str(record.get("content", "")).strip()
            if not content:
                continue
            tool_name = str(record.get("tool_name", "")).strip()
            timestamp = str(record.get("timestamp", "")).strip()
            prefix = f"[{kind}]"
            if tool_name:
                prefix = f"{prefix}[{tool_name}]"
            if timestamp:
                prefix = f"{prefix}[{timestamp}]"
            lines.append(f"{prefix} {content}")

        if not lines:
            return []

        chunks: list[dict[str, Any]] = []
        buffer: list[str] = []
        buffer_tokens = 0
        chunk_index = 0

        for line in lines:
            tokens = max(1, len(line) // 4)
            if buffer and (buffer_tokens + tokens > max_chunk_tokens):
                chunk_text = "\n".join(buffer).strip()
                if chunk_text:
                    chunks.append(
                        {
                            "chunk_id": f"{session_id}_fallback_chunk_{chunk_index:03d}",
                            "session_id": session_id,
                            "creation_time": creation_time,
                            "chunk_index": chunk_index,
                            "content": chunk_text,
                            "char_count": len(chunk_text),
                            "estimated_tokens": max(1, len(chunk_text) // 4),
                        }
                    )
                    chunk_index += 1

                overlap: list[str] = []
                overlap_count = 0
                for existing in reversed(buffer):
                    existing_tokens = max(1, len(existing) // 4)
                    if overlap and overlap_count + existing_tokens > overlap_tokens:
                        break
                    overlap.insert(0, existing)
                    overlap_count += existing_tokens
                buffer = overlap
                buffer_tokens = overlap_count

            buffer.append(line)
            buffer_tokens += tokens

        if buffer:
            chunk_text = "\n".join(buffer).strip()
            if chunk_text:
                chunks.append(
                    {
                        "chunk_id": f"{session_id}_fallback_chunk_{chunk_index:03d}",
                        "session_id": session_id,
                        "creation_time": creation_time,
                        "chunk_index": chunk_index,
                        "content": chunk_text,
                        "char_count": len(chunk_text),
                        "estimated_tokens": max(1, len(chunk_text) // 4),
                    }
                )
        return chunks

    def _index_turn_records(
        self,
        *,
        session_id: str,
        turn_records: list[dict[str, Any]],
        creation_time: str | None = None,
    ) -> bool:
        if not session_id or not turn_records:
            return True

        creation_timestamp = creation_time or datetime.now().isoformat()
        try:
            chunk_turn_records = getattr(self.chunker, "chunk_turn_records", None)
            if not callable(chunk_turn_records):
                raise AttributeError("chunk_turn_records is not implemented on active chunker")
            chunks = chunk_turn_records(
                session_id=session_id,
                turn_records=turn_records,
                creation_time=creation_timestamp,
            )
        except Exception as e:
            print(f"LOG: Turn-record chunking unavailable, using fallback chunker for {session_id}: {e}")
            chunks = self._fallback_chunk_turn_records(
                session_id=session_id,
                turn_records=turn_records,
                creation_time=creation_timestamp,
            )

        if not chunks:
            return False

        chunk_contents = [chunk["content"] for chunk in chunks]
        embeddings = self.embedder.embed_for_storage(chunk_contents)
        with self._vector_lock:
            self.vector_store.add_chunks(chunks, embeddings)
        return True

    @staticmethod
    def _extract_conversations_from_messages(message_history: list[ModelMessage]) -> list[dict[str, Any]]:
        conversations: list[dict[str, Any]] = []
        pending_user: str | None = None

        for message in message_history:
            if isinstance(message, ModelRequest):
                user_parts = [
                    part.content
                    for part in message.parts
                    if isinstance(part, UserPromptPart) and isinstance(part.content, str) and part.content.strip()
                ]
                if user_parts:
                    pending_user = "\n".join(user_parts).strip()
            elif isinstance(message, ModelResponse):
                response_text = (message.text or "").strip()
                if pending_user and response_text:
                    conversations.append(
                        {
                            "user_query": pending_user,
                            "turtle_response": response_text,
                            "timestamp": message.timestamp.isoformat(),
                        }
                    )
                    pending_user = None

        return conversations

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()

    def _extract_turn_records_from_messages(self, message_history: list[ModelMessage]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []

        for message in message_history:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart):
                        content = self._stringify_value(part.content)
                        if content:
                            records.append(
                                {
                                    "kind": "user",
                                    "content": content,
                                    "timestamp": part.timestamp.isoformat(),
                                }
                            )
                    elif isinstance(part, ToolReturnPart):
                        content = self._stringify_value(part.content)
                        if content:
                            records.append(
                                {
                                    "kind": "tool_return",
                                    "tool_name": part.tool_name,
                                    "content": content,
                                    "timestamp": part.timestamp.isoformat(),
                                }
                            )
                    elif isinstance(part, RetryPromptPart):
                        content = part.model_response().strip()
                        if content:
                            records.append(
                                {
                                    "kind": "retry",
                                    "tool_name": part.tool_name,
                                    "content": content,
                                    "timestamp": part.timestamp.isoformat(),
                                }
                            )
            elif isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, TextPart):
                        content = part.content.strip()
                        if content:
                            records.append(
                                {
                                    "kind": "assistant",
                                    "content": content,
                                    "timestamp": message.timestamp.isoformat(),
                                }
                            )
                    elif isinstance(part, ToolCallPart):
                        try:
                            args_text = json.dumps(part.args_as_dict(), ensure_ascii=False)
                        except Exception:
                            args_text = str(part.args).strip()
                        if args_text:
                            records.append(
                                {
                                    "kind": "tool_call",
                                    "tool_name": part.tool_name,
                                    "content": args_text,
                                    "timestamp": message.timestamp.isoformat(),
                                }
                            )

        return records

    async def finalize_archived_session(
        self,
        *,
        session_id: str,
        archive_path: Path,
    ) -> bool:
        try:
            messages_path = archive_path / "messages.json"
            if not messages_path.exists():
                return False

            from pydantic_ai import ModelMessagesTypeAdapter

            message_history = ModelMessagesTypeAdapter.validate_json(messages_path.read_bytes())
            turn_records = self._extract_turn_records_from_messages(message_history)
            if not turn_records:
                return False

            manifest_path = archive_path / "session.json"
            creation_time: str | None = None
            if manifest_path.exists():
                try:
                    creation_time = json.loads(manifest_path.read_text(encoding="utf-8")).get("created_at")
                except Exception:
                    creation_time = None

            indexed = self._index_turn_records(
                session_id=session_id,
                turn_records=turn_records,
                creation_time=creation_time,
            )
            if not indexed:
                return False
            self._clear_temp_session_file(expected_session_id=session_id)
            return True
        except Exception as e:
            print(f"LOG: Failed to finalize archived session {session_id}: {e}")
            return False
    
    async def start_session(self, session_id: str | None = None) -> str:
        """Start a new conversation session"""
        if self.temp_session_file.exists():
            try:
                session_data = json.loads(self.temp_session_file.read_text(encoding="utf-8"))
                existing_session_id = session_data.get("session_id")
                if existing_session_id and (session_id is None or session_id == existing_session_id):
                    self.current_session_id = existing_session_id
                    self.session_conversations = session_data.get("conversations", [])
                    return self.current_session_id
            except Exception:
                pass

        # Process any existing in-memory session first
        if self.current_session_id and self.current_session_id != session_id:
            await self.end_session()
        
        # Create new session
        self.current_session_id = session_id or f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_conversations = []
        
        # Initialize temp JSON
        session_data = {
            "session_id": self.current_session_id,
            "creation_time": datetime.now().isoformat(),
            "conversations": []
        }
        
        with open(self.temp_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        return self.current_session_id
    
    def add_conversation(self, user_query: str, turtle_response: str):
        """Add a conversation pair to the current session"""
        if not self.current_session_id:
            # Create session synchronously if needed
            self.current_session_id = f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.session_conversations = []
        
        conversation = {
            "user_query": user_query,
            "turtle_response": turtle_response,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to memory
        self.session_conversations.append(conversation)
        
        # Update temp JSON file
        if self.temp_session_file.exists():
            with open(self.temp_session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
        else:
            session_data = {
                "session_id": self.current_session_id,
                "creation_time": datetime.now().isoformat(),
                "conversations": []
            }
        
        # Ensure session metadata matches current session
        session_data["session_id"] = self.current_session_id
        session_data.setdefault("creation_time", datetime.now().isoformat())
        
        session_data["conversations"] = self.session_conversations
        
        with open(self.temp_session_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
    
    async def query_history(self, user_query: str) -> str:
        """
        Main RAG query interface:
        1. Generate embeddings for query
        2. Search vector database for similar chunks
        3. Format and return relevant chunks
        """
        try:
            # Generate embeddings for search query
            query_embedding = self.embedder.embed_for_query(user_query)
            
            # Search vector database for top 5 similar chunks
            with self._vector_lock:
                similar_chunks = self.vector_store.search_similar(
                    query_embedding,
                    top_k=5,
                    threshold=0.3
                )
            
            if not similar_chunks:
                return "cannot find in history"
            
            # Return raw chunks as JSON string for main agent to process
            import json
            chunks_data = []
            for chunk in similar_chunks:
                chunk_info = {
                    "content": chunk.get('content', ''),
                    "similarity_score": chunk.get('similarity_score', 0.0),
                    "session_id": chunk.get('session_id', ''),
                    "timestamp": chunk.get('timestamp', '')
                }
                chunks_data.append(chunk_info)
            
            result = json.dumps(chunks_data, indent=2, ensure_ascii=False)
            return result
            
        except Exception as e:
            return "cannot find in history"
    
    async def end_session(self):
        """End current session and process conversations to vector database"""
        if not self.current_session_id or not self.session_conversations:
            return True
        
        try:
            self._index_session_conversations(
                session_id=self.current_session_id,
                conversations=self.session_conversations,
                creation_time=datetime.now().isoformat(),
            )
            
            # Clean up
            self.current_session_id = None
            self.session_conversations = []
            
            # Remove temp file
            self.temp_session_file.unlink(missing_ok=True)
            return True
                
        except Exception as e:
            print(f"LOG: RAG end_session failed: {e}")
            return False
    
    async def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "current_session_id": self.current_session_id,
            "conversations_count": len(self.session_conversations),
            "vector_store_stats": self.vector_store.get_storage_stats(),
            "temp_file_exists": self.temp_session_file.exists()
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get complete system statistics"""
        vector_stats = self.vector_store.get_storage_stats()
        
        return {
            "current_session": self.current_session_id,
            "current_conversations": len(self.session_conversations),
            "total_vectors": vector_stats.get("total_vectors", 0),
            "total_sessions": vector_stats.get("total_sessions", 0),
            "storage_size_mb": vector_stats.get("storage_size_mb", 0),
            "embedding_model": "embed-english-v3.0",
            "vector_db_type": "FAISS IndexFlatIP"
        }


# Context-local RAG instance to avoid cross-session mutable global sharing.
_turtle_rag_ctx: contextvars.ContextVar[TurtleRAGSystem | None] = contextvars.ContextVar(
    "turtle_rag",
    default=None,
)

def get_rag_system() -> TurtleRAGSystem:
    """Get context-local RAG system instance (legacy convenience API)."""
    rag = _turtle_rag_ctx.get()
    if rag is None:
        rag = TurtleRAGSystem()
        _turtle_rag_ctx.set(rag)
    return rag

# Convenience functions for easy integration
async def start_rag_session() -> str:
    """Start new RAG session"""
    rag = get_rag_system()
    return await rag.start_session()

def add_to_rag(user_query: str, turtle_response: str):
    """Add conversation to RAG"""
    rag = get_rag_system()
    rag.add_conversation(user_query, turtle_response)

async def search_rag(query: str) -> str:
    """Search RAG for history"""
    rag = get_rag_system()
    return await rag.query_history(query)

async def end_rag_session():
    """End RAG session"""
    rag = get_rag_system()
    await rag.end_session()

def get_rag_stats() -> Dict[str, Any]:
    """Get RAG statistics"""
    rag = get_rag_system()
    return rag.get_system_stats()

