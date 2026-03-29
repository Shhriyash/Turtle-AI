"""
FAISS Vector Storage for RAG System

This module handles vector storage and retrieval using FAISS (Facebook AI Similarity Search).
Provides efficient similarity search for conversation chunks.
"""

import os
import json
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime
from core.paths import RAG_VECTOR_DIR, ensure_dirs


class VectorStorage:
    """FAISS-based vector storage for conversation chunks"""
    
    def __init__(self, storage_dir: Optional[str] = None, embedding_dimension: int = 1024):
        """
        Initialize FAISS vector storage
        
        Args:
            storage_dir: Directory to store FAISS index and metadata
            embedding_dimension: Dimension of embeddings (1024 for Cohere embed-english-v3.0)
        """
        ensure_dirs()
        self.storage_dir = Path(storage_dir) if storage_dir else RAG_VECTOR_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_dimension = embedding_dimension
        self.faiss_index = None
        self.chunk_metadata = []
        
        # File paths
        self.index_path = self.storage_dir / "faiss_index.bin"
        self.metadata_path = self.storage_dir / "chunk_metadata.json"
        
        # Initialize or load FAISS index
        self._initialize_index()
        self._load_metadata()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_path.exists():
            try:
                # Load existing index
                self.faiss_index = faiss.read_index(str(self.index_path))
            except Exception as e:
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatIP (Inner Product) for cosine similarity
        # Normalize embeddings before adding for true cosine similarity
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
    
    def _load_metadata(self):
        """Load chunk metadata from file"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_metadata = json.load(f)
            except Exception as e:
                self.chunk_metadata = []
        else:
            self.chunk_metadata = []
    
    def _save_index(self):
        """Save FAISS index to disk"""
        if self.faiss_index:
            faiss.write_index(self.faiss_index, str(self.index_path))
    
    def _save_metadata(self):
        """Save chunk metadata to file"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunk_metadata, f, indent=2, ensure_ascii=False)
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize embeddings for cosine similarity with IndexFlatIP
        
        Args:
            embeddings: Raw embeddings array
            
        Returns:
            L2-normalized embeddings
        """
        # L2 normalize each embedding vector
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms
        return normalized.astype(np.float32)
    
    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Add chunks and their embeddings to vector storage
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: Corresponding embeddings array
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(f"Chunks count ({len(chunks)}) must match embeddings count ({embeddings.shape[0]})")
        
        if embeddings.shape[1] != self.embedding_dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) must match expected ({self.embedding_dimension})")
        
        existing_chunk_ids = {
            metadata.get("chunk_id")
            for metadata in self.chunk_metadata
            if metadata.get("chunk_id") and not metadata.get("deleted", False)
        }
        filtered_pairs = [
            (chunk, embeddings[i])
            for i, chunk in enumerate(chunks)
            if chunk.get("chunk_id") not in existing_chunk_ids
        ]
        if not filtered_pairs:
            return

        filtered_chunks = [chunk for chunk, _ in filtered_pairs]
        filtered_embeddings = np.stack([embedding for _, embedding in filtered_pairs]).astype(np.float32)

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self._normalize_embeddings(filtered_embeddings)
        
        # Add to FAISS index
        self.faiss_index.add(normalized_embeddings)
        
        # Add metadata with vector indices
        start_idx = len(self.chunk_metadata)
        for i, chunk in enumerate(filtered_chunks):
            metadata = {
                "vector_index": start_idx + i,
                "chunk_id": chunk.get("chunk_id", f"chunk_{start_idx + i}"),
                "session_id": chunk.get("session_id", "unknown"),
                "creation_time": chunk.get("creation_time", datetime.now().isoformat()),
                "content": chunk.get("content", ""),
                "chunk_index": chunk.get("chunk_index", i),
                "char_count": chunk.get("char_count", len(chunk.get("content", ""))),
                "estimated_tokens": chunk.get("estimated_tokens", 0),
                "added_timestamp": datetime.now().isoformat(),
                "kind": chunk.get("kind", "turn_chunk"),
                "timestamp_start": chunk.get("timestamp_start"),
                "timestamp_end": chunk.get("timestamp_end"),
                "turn_range": chunk.get("turn_range"),
                "tags": chunk.get("tags", []),
            }
            self.chunk_metadata.append(metadata)
        
        # Save to disk
        self._save_index()
        self._save_metadata()
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using cosine similarity
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar chunks with metadata and scores
        """
        if self.faiss_index.ntotal == 0:
            return []
        
        # Ensure query embedding is 2D and normalized
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_embedding = self._normalize_embeddings(query_embedding)
        
        # Search in FAISS
        k = min(top_k, self.faiss_index.ntotal)
        scores, indices = self.faiss_index.search(query_embedding, k)
        
        # Filter by threshold and prepare results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= threshold and idx < len(self.chunk_metadata):
                metadata = self.chunk_metadata[idx].copy()
                if metadata.get("deleted", False):
                    continue
                metadata["similarity_score"] = float(score)
                results.append(metadata)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk metadata by chunk ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Chunk metadata if found, None otherwise
        """
        for metadata in self.chunk_metadata:
            if metadata.get("chunk_id") == chunk_id:
                return metadata
        return None
    
    def get_chunks_by_session(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks from a specific session
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of chunks from the session
        """
        session_chunks = []
        for metadata in self.chunk_metadata:
            if metadata.get("session_id") == session_id:
                session_chunks.append(metadata)
        return session_chunks
    
    def delete_session(self, session_id: str) -> int:
        """
        Delete all chunks from a specific session
        Note: This marks chunks as deleted but doesn't remove from FAISS index
        
        Args:
            session_id: Session identifier
            
        Returns:
            Number of chunks marked as deleted
        """
        deleted_count = 0
        for metadata in self.chunk_metadata:
            if metadata.get("session_id") == session_id:
                metadata["deleted"] = True
                metadata["deleted_timestamp"] = datetime.now().isoformat()
                deleted_count += 1
        
        if deleted_count > 0:
            self._save_metadata()
        
        return deleted_count
    
    def cleanup_deleted_chunks(self):
        """
        Cleanup deleted chunks safely.

        NOTE: FAISS index entries cannot be removed without rebuilding the index.
        To avoid corrupting index-to-metadata alignment, this method is a no-op
        unless a rebuild strategy is implemented.
        """
        deleted_count = sum(1 for m in self.chunk_metadata if m.get("deleted", False))
        if deleted_count > 0:
            print("LOG: cleanup_deleted_chunks skipped to preserve FAISS index alignment.")
        return 0
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get vector storage statistics
        
        Returns:
            Dictionary with storage statistics
        """
        active_chunks = [m for m in self.chunk_metadata if not m.get("deleted", False)]
        deleted_chunks = [m for m in self.chunk_metadata if m.get("deleted", False)]
        
        sessions = set(m.get("session_id") for m in active_chunks)
        
        return {
            "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
            "active_chunks": len(active_chunks),
            "deleted_chunks": len(deleted_chunks),
            "total_sessions": len(sessions),
            "embedding_dimension": self.embedding_dimension,
            "index_type": type(self.faiss_index).__name__ if self.faiss_index else "None",
            "storage_size_mb": self._get_storage_size()
        }
    
    def _get_storage_size(self) -> float:
        """Get storage size in MB"""
        total_size = 0
        for file_path in [self.index_path, self.metadata_path]:
            if file_path.exists():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # Convert to MB


# Global vector storage instance
vector_storage = None

def get_vector_storage() -> VectorStorage:
    """Get global vector storage instance"""
    global vector_storage
    if vector_storage is None:
        vector_storage = VectorStorage()
    return vector_storage

def add_to_vector_db(chunks: List[Dict[str, Any]], embeddings: np.ndarray):
    """Convenience function to add chunks to vector storage"""
    storage = get_vector_storage()
    storage.add_chunks(chunks, embeddings)

def search_vector_db(query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
    """Convenience function to search vector storage"""
    storage = get_vector_storage()
    return storage.search_similar(query_embedding, top_k, threshold)

