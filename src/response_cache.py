"""Response cache for common queries to reduce latency and API costs."""

import asyncio
import logging
import hashlib
import pickle
import os
from typing import Optional, Dict, Tuple
from pathlib import Path
import time

from src.models import AudioChunk


logger = logging.getLogger(__name__)


class ResponseCache:
    """Cache for LLM responses and TTS audio to reduce latency and costs."""
    
    def __init__(self, cache_dir: str = "./data/response_cache", max_age_seconds: int = 86400):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_age_seconds: Maximum age of cached responses (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.max_age_seconds = max_age_seconds
        
        # In-memory cache for fast access
        self._text_cache: Dict[str, Tuple[str, float]] = {}  # query_hash -> (response, timestamp)
        self._audio_cache: Dict[str, Tuple[list, float]] = {}  # response_hash -> (audio_chunks, timestamp)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache from disk
        self._load_cache()
        
        logger.info(f"ResponseCache initialized: cache_dir={cache_dir}, max_age={max_age_seconds}s")
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            text_cache_path = self.cache_dir / "text_cache.pkl"
            audio_cache_path = self.cache_dir / "audio_cache.pkl"
            
            if text_cache_path.exists():
                with open(text_cache_path, 'rb') as f:
                    self._text_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._text_cache)} text responses from cache")
            
            if audio_cache_path.exists():
                with open(audio_cache_path, 'rb') as f:
                    self._audio_cache = pickle.load(f)
                logger.info(f"Loaded {len(self._audio_cache)} audio responses from cache")
            
            # Clean expired entries
            self._clean_expired()
            
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            self._text_cache = {}
            self._audio_cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            text_cache_path = self.cache_dir / "text_cache.pkl"
            audio_cache_path = self.cache_dir / "audio_cache.pkl"
            
            with open(text_cache_path, 'wb') as f:
                pickle.dump(self._text_cache, f)
            
            with open(audio_cache_path, 'wb') as f:
                pickle.dump(self._audio_cache, f)
            
            logger.debug("Cache saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _clean_expired(self) -> None:
        """Remove expired cache entries."""
        current_time = time.time()
        
        # Clean text cache
        expired_text = [
            key for key, (_, timestamp) in self._text_cache.items()
            if current_time - timestamp > self.max_age_seconds
        ]
        for key in expired_text:
            del self._text_cache[key]
        
        # Clean audio cache
        expired_audio = [
            key for key, (_, timestamp) in self._audio_cache.items()
            if current_time - timestamp > self.max_age_seconds
        ]
        for key in expired_audio:
            del self._audio_cache[key]
        
        if expired_text or expired_audio:
            logger.info(f"Cleaned {len(expired_text)} text and {len(expired_audio)} audio cache entries")
    
    def get_text_response(self, query: str) -> Optional[str]:
        """
        Get cached text response for query.
        
        Args:
            query: User query
            
        Returns:
            Cached response text or None if not found
        """
        query_hash = self._hash_text(query)
        
        if query_hash in self._text_cache:
            response, timestamp = self._text_cache[query_hash]
            
            # Check if expired
            if time.time() - timestamp > self.max_age_seconds:
                del self._text_cache[query_hash]
                logger.debug(f"Text cache expired for query: {query[:50]}")
                return None
            
            logger.info(f"Text cache HIT for query: {query[:50]}")
            return response
        
        logger.debug(f"Text cache MISS for query: {query[:50]}")
        return None
    
    def set_text_response(self, query: str, response: str) -> None:
        """
        Cache text response for query.
        
        Args:
            query: User query
            response: LLM response
        """
        query_hash = self._hash_text(query)
        self._text_cache[query_hash] = (response, time.time())
        
        logger.debug(f"Cached text response for query: {query[:50]}")
        
        # Save to disk asynchronously
        asyncio.create_task(asyncio.to_thread(self._save_cache))
    
    def get_audio_response(self, response_text: str) -> Optional[list]:
        """
        Get cached audio chunks for response text.
        
        Args:
            response_text: Response text
            
        Returns:
            List of AudioChunk objects or None if not found
        """
        response_hash = self._hash_text(response_text)
        
        if response_hash in self._audio_cache:
            audio_chunks, timestamp = self._audio_cache[response_hash]
            
            # Check if expired
            if time.time() - timestamp > self.max_age_seconds:
                del self._audio_cache[response_hash]
                logger.debug(f"Audio cache expired for response: {response_text[:50]}")
                return None
            
            logger.info(f"Audio cache HIT for response: {response_text[:50]}")
            return audio_chunks
        
        logger.debug(f"Audio cache MISS for response: {response_text[:50]}")
        return None
    
    def set_audio_response(self, response_text: str, audio_chunks: list) -> None:
        """
        Cache audio chunks for response text.
        
        Args:
            response_text: Response text
            audio_chunks: List of AudioChunk objects
        """
        response_hash = self._hash_text(response_text)
        self._audio_cache[response_hash] = (audio_chunks, time.time())
        
        logger.debug(f"Cached audio response for text: {response_text[:50]}")
        
        # Save to disk asynchronously
        asyncio.create_task(asyncio.to_thread(self._save_cache))
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._text_cache.clear()
        self._audio_cache.clear()
        
        # Remove cache files
        try:
            text_cache_path = self.cache_dir / "text_cache.pkl"
            audio_cache_path = self.cache_dir / "audio_cache.pkl"
            
            if text_cache_path.exists():
                text_cache_path.unlink()
            if audio_cache_path.exists():
                audio_cache_path.unlink()
            
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "text_entries": len(self._text_cache),
            "audio_entries": len(self._audio_cache),
            "max_age_seconds": self.max_age_seconds
        }
