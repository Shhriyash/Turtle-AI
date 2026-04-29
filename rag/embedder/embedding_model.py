"""
Cohere Embedding Model for RAG System

This module handles text embedding generation using Cohere's API.
Follows the official Cohere documentation: https://docs.cohere.com/reference/embed
"""

import os
import numpy as np
from typing import List, Union
import cohere
from core.env import load_env

# Load environment variables
load_env()


class CohereEmbedding:
    """Cohere embedding model wrapper for generating text embeddings"""
    
    def __init__(self, api_key: str = None, model: str = "embed-english-v3.0"):
        """
        Initialize Cohere embedding model
        
        Args:
            api_key: Cohere API key (if None, loads from COHERE_API_KEY env var)
            model: Cohere embedding model to use
        """
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not found. Set COHERE_API_KEY environment variable.")
        
        self.model = model
        self.client = cohere.Client(api_key=self.api_key)
    
    def embed_texts(self, texts: Union[str, List[str]], input_type: str = "search_document") -> np.ndarray:
        """
        Generate embeddings for text(s) using Cohere API
        
        Args:
            texts: Single text string or list of text strings to embed
            input_type: Type of input for embedding optimization
                       - "search_document": For documents to be searched over
                       - "search_query": For search queries
                       - "classification": For text classification
                       - "clustering": For text clustering
        
        Returns:
            numpy array of embeddings with shape (n_texts, embedding_dim)
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Call Cohere embed API
            response = self.client.embed(
                texts=texts,
                model=self.model,
                input_type=input_type,
                embedding_types=["float"]  # Specify float embeddings
            )
            
            # Convert to numpy array - handle new response structure
            if hasattr(response.embeddings, 'float'):
                embeddings = np.array(response.embeddings.float, dtype=np.float32)
            else:
                # Fallback for different response structure
                embeddings = np.array(response.embeddings, dtype=np.float32)
            
            return embeddings
            
        except Exception as e:
            raise
    
    def embed_for_storage(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings optimized for document storage/indexing
        
        Args:
            texts: Text(s) to embed for storage
            
        Returns:
            numpy array of embeddings
        """
        return self.embed_texts(texts, input_type="search_document")
    
    def embed_for_query(self, query: str) -> np.ndarray:
        """
        Generate embedding optimized for search queries
        
        Args:
            query: Search query text
            
        Returns:
            numpy array of embedding
        """
        return self.embed_texts(query, input_type="search_query")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension for the current model
        
        Returns:
            Embedding dimension (usually 1024 for embed-english-v3.0)
        """
        # Test with a simple text to get dimension
        test_embedding = self.embed_texts("test")
        return test_embedding.shape[1]
    
    def batch_embed(self, texts: List[str], batch_size: int = 96) -> np.ndarray:
        """
        Generate embeddings in batches to handle API rate limits
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (Cohere limit is 96)
            
        Returns:
            numpy array of all embeddings
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embed_for_storage(batch)
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])


# Global embedding instance for easy import
cohere_embedder = None

def get_embedding_model() -> CohereEmbedding:
    """Get global Cohere embedding model instance"""
    global cohere_embedder
    if cohere_embedder is None:
        cohere_embedder = CohereEmbedding()
    return cohere_embedder

def embed_texts(texts: Union[str, List[str]]) -> np.ndarray:
    """Convenience function to embed texts"""
    embedder = get_embedding_model()
    return embedder.embed_for_storage(texts)

def embed_query(query: str) -> np.ndarray:
    """Convenience function to embed search query"""
    embedder = get_embedding_model()
    return embedder.embed_for_query(query)


