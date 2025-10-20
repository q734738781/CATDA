"""
Embedding model configuration and utilities.
Provides functions for text embedding and vector operations with caching.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import hashlib
from openai import OpenAI
import logging
import lmdb  # Added import

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """LMDB-based cache for embeddings"""

    def __init__(self, db_path: Optional[Path] = None, map_size: int = 1 << 30): # Default 1GB map size
        """Initialize embedding cache using LMDB

        Args:
            db_path: Path to the LMDB database file. Defaults to './embedding_cache.lmdb'
            map_size: Maximum size the database can grow to.
        """
        self.db_path = db_path or Path('./embedding_cache.lmdb')
        # Ensure the parent directory exists if db_path is specified deeply
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(str(self.db_path), map_size=map_size, subdir=False,
                             readonly=False, lock=True, readahead=False)

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from LMDB cache"""
        cache_key = self._get_cache_key(text).encode()
        try:
            with self.env.begin() as txn:
                raw = txn.get(cache_key)
            if raw is not None:
                # Assuming embeddings are stored as float32
                embedding = np.frombuffer(raw, dtype=np.float32).tolist()
                return embedding
        except lmdb.Error as e:
            logger.warning(f"LMDB get failed for key {cache_key.decode()}: {e}")
        except Exception as e:
             logger.warning(f"Failed to decode embedding from cache for key {cache_key.decode()}: {e}")
        return None

    def set(self, text: str, embedding: List[float]):
        """Save embedding to LMDB cache"""
        cache_key = self._get_cache_key(text).encode()
        try:
            vec = np.array(embedding, dtype=np.float32)
            with self.env.begin(write=True) as txn:
                txn.put(cache_key, vec.tobytes(), overwrite=True) # Overwrite if exists
        except lmdb.Error as e:
            logger.warning(f"LMDB set failed for key {cache_key.decode()}: {e}")
        except Exception as e:
            logger.warning(f"Failed to encode embedding for cache key {cache_key.decode()}: {e}")

    def __del__(self):
        """Close the LMDB environment when the object is destroyed."""
        if hasattr(self, 'env') and self.env:
            self.env.close()


class EntityEmbedding:
    """Entity embedding manager with caching"""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", db_path: Optional[Path] = None):
        """Initialize entity embedding

        Args:
            api_key: OpenAI API key. Defaults to OPENAI_API_KEY environment variable
            model: OpenAI embedding model name
            db_path: Path to the LMDB database file for caching.
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.cache = EmbeddingCache(db_path=db_path) # Pass db_path if provided

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text, using cache if available"""
        # Check cache first
        # Trim whitespace which can affect cache keys and API results
        processed_text = text.strip()
        if not processed_text:
             logger.warning("Attempted to get embedding for empty or whitespace-only string.")
             # Return a zero vector or raise an error, depending on desired behavior
             # For now, let's assume the model handles it or we return zeros of expected dim
             # We need the dimension, which isn't stored directly here.
             # Let's return an empty list and let the caller handle it, or potentially raise.
             return [] # Or raise ValueError("Cannot embed empty string")


        cached = self.cache.get(processed_text)
        if cached is not None:
            logger.debug(f"Cache hit for text snippet: '{processed_text[:50]}...'")
            return cached
        
        logger.debug(f"Cache miss for text snippet: '{processed_text[:50]}...'. Calling API.")
        # Get new embedding
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=processed_text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to get embedding from OpenAI API for text '{processed_text[:50]}...': {e}")
            # Decide error handling: re-raise, return default, etc.
            # For now, re-raising to make the failure explicit.
            raise

        # Cache the result
        self.cache.set(processed_text, embedding)
        return embedding

# Example Usage (Optional - can be removed or kept for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # Ensure OPENAI_API_KEY is set in your environment or pass it directly
    # api_key = "your_openai_api_key"
    # embedder = EntityEmbedding(api_key=api_key)
    embedder = EntityEmbedding() # Assumes env var is set

    texts = ["This is the first test sentence.",
             "This is the second test sentence.",
             "This is the first test sentence."] # Test cache hit

    for text in texts:
        print(f"\nGetting embedding for: '{text}'")
        try:
            emb = embedder.get_embedding(text)
            print(f"Embedding dimension: {len(emb)}")
            # print(f"Embedding preview: {emb[:5]}...") # Uncomment to see embedding values
        except Exception as e:
            print(f"Error getting embedding: {e}")

    # Clean up the cache file after example run (optional)
    # lmdb_file = Path('./embedding_cache.lmdb')
    # if lmdb_file.exists():
    #    os.remove(lmdb_file)
    #    print(f"\nRemoved cache file: {lmdb_file}")