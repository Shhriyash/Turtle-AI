"""
Complete RAG System for Turtle History Management

This module integrates all RAG components into a unified system:
- Temp JSON storage for active sessions
- Conversation chunking and embedding
- Vector storage and semantic search  
- Gemini LLM for query processing and intent extraction
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our RAG components
from embedding_model import get_embedding_model
from json_chunking import get_chunker
from vector_storage import get_vector_storage

# Pydantic AI for LLM integration
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TurtleRAGSystem:
    """Complete RAG system for Turtle conversation history"""
    
    def __init__(self, storage_dir: str = "rag_data"):
        """Initialize the complete RAG system"""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Temp JSON file for current session
        self.temp_session_file = self.storage_dir / "current_session.json"
        
        # Initialize components
        self.embedder = get_embedding_model()
        self.chunker = get_chunker()
        self.vector_store = get_vector_storage()
        
        # Current session tracking
        self.current_session_id = None
        self.session_conversations = []
    
    async def start_session(self) -> str:
        """Start a new conversation session"""
        # Process any existing session first
        if self.current_session_id:
            await self.end_session()
        
        # Create new session
        self.current_session_id = f"turtle_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            return
        
        try:
            # Create session data structure for chunking
            session_data = {
                self.current_session_id: {
                    "creation_time": datetime.now().isoformat(),
                    "conversations": self.session_conversations
                }
            }
            
            # Chunk the conversations
            chunks = self.chunker.chunk_session_conversations(session_data)
            
            if chunks:
                # Extract content for embedding
                chunk_contents = [chunk["content"] for chunk in chunks]
                
                # Generate embeddings
                embeddings = self.embedder.embed_for_storage(chunk_contents)
                
                # Add to vector database
                self.vector_store.add_chunks(chunks, embeddings)
            
            # Clean up
            self.current_session_id = None
            self.session_conversations = []
            
            # Remove temp file
            if self.temp_session_file.exists():
                self.temp_session_file.unlink()
                
        except Exception as e:
            pass
    
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


# Global RAG system instance
turtle_rag = None

def get_rag_system() -> TurtleRAGSystem:
    """Get global RAG system instance"""
    global turtle_rag
    if turtle_rag is None:
        turtle_rag = TurtleRAGSystem()
    return turtle_rag

# Convenience functions for easy integration
def start_rag_session() -> str:
    """Start new RAG session"""
    rag = get_rag_system()
    return rag.start_session()

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

