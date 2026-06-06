"""
Cache manager for the research analyst system.
Implements multi-layer caching for queries, retrieval, and answers.
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

from research_analyst.core.models import (
    Query,
    NormalizedQuery,
    Answer,
    Document,
    PipelineResponse
)
from research_analyst.core.exceptions import CacheError
from research_analyst.config import get_settings
from research_analyst.utils.logger import get_logger
from research_analyst.utils.helpers import generate_hash


logger = get_logger()


class CacheManager:
    """Manage multi-layer caching."""
    
    def __init__(self):
        """Initialize cache manager."""
        self.settings = get_settings()
        self.logger = get_logger()
        
        # Initialize backend
        if self.settings.cache_backend == "redis":
            self._init_redis()
        elif self.settings.cache_backend == "disk":
            self._init_disk()
        else:  # memory
            self._init_memory()
    
    def _init_redis(self):
        """Initialize Redis backend."""
        try:
            import redis
            self.backend = "redis"
            self.redis_client = redis.Redis(
                host=self.settings.redis_host,
                port=self.settings.redis_port,
                db=self.settings.redis_db,
                decode_responses=False  # We'll handle encoding
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info("Initialized Redis cache")
        except ImportError:
            raise CacheError(
                "Redis library not installed",
                details={"required": "pip install redis"}
            )
        except Exception as e:
            raise CacheError(
                f"Failed to connect to Redis: {str(e)}",
                details={
                    "host": self.settings.redis_host,
                    "port": self.settings.redis_port
                }
            )
    
    def _init_disk(self):
        """Initialize disk cache backend."""
        try:
            from diskcache import Cache
            self.backend = "disk"
            cache_dir = Path(self.settings.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.disk_cache = Cache(str(cache_dir))
            self.logger.info(f"Initialized disk cache at {cache_dir}")
        except ImportError:
            raise CacheError(
                "diskcache library not installed",
                details={"required": "pip install diskcache"}
            )
    
    def _init_memory(self):
        """Initialize in-memory cache backend."""
        self.backend = "memory"
        self.memory_cache = {}
        self.logger.info("Initialized in-memory cache")
    
    def _generate_key(self, prefix: str, data: Any) -> str:
        """
        Generate cache key.
        
        Args:
            prefix: Key prefix (e.g., 'query', 'answer')
            data: Data to hash
            
        Returns:
            Cache key string
        """
        # Convert data to string
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, sort_keys=True)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        
        # Generate hash
        data_hash = generate_hash(data_str)
        
        return f"{prefix}:{data_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            if self.backend == "redis":
                value = self.redis_client.get(key)
                if value:
                    self.logger.log_cache_hit(key, "redis")
                    return pickle.loads(value)
                else:
                    self.logger.log_cache_miss(key, "redis")
                    return None
                    
            elif self.backend == "disk":
                value = self.disk_cache.get(key)
                if value is not None:
                    self.logger.log_cache_hit(key, "disk")
                    return value
                else:
                    self.logger.log_cache_miss(key, "disk")
                    return None
                    
            else:  # memory
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    # Check expiration
                    if entry['expires_at'] > datetime.utcnow():
                        self.logger.log_cache_hit(key, "memory")
                        return entry['value']
                    else:
                        # Expired
                        del self.memory_cache[key]
                
                self.logger.log_cache_miss(key, "memory")
                return None
                
        except Exception as e:
            self.logger.warning(
                "Cache get failed",
                key=key,
                error=str(e)
            )
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set item in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.settings.cache_ttl_seconds
        
        try:
            if self.backend == "redis":
                pickled = pickle.dumps(value)
                self.redis_client.setex(key, ttl, pickled)
                
            elif self.backend == "disk":
                self.disk_cache.set(key, value, expire=ttl)
                
            else:  # memory
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                self.memory_cache[key] = {
                    'value': value,
                    'expires_at': expires_at
                }
            
            return True
            
        except Exception as e:
            self.logger.warning(
                "Cache set failed",
                key=key,
                error=str(e)
            )
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        try:
            if self.backend == "redis":
                self.redis_client.delete(key)
            elif self.backend == "disk":
                self.disk_cache.delete(key)
            else:  # memory
                if key in self.memory_cache:
                    del self.memory_cache[key]
            
            return True
            
        except Exception as e:
            self.logger.warning(
                "Cache delete failed",
                key=key,
                error=str(e)
            )
            return False
    
    def clear(self) -> bool:
        """
        Clear all cache.
        
        Returns:
            True if successful
        """
        try:
            if self.backend == "redis":
                self.redis_client.flushdb()
            elif self.backend == "disk":
                self.disk_cache.clear()
            else:  # memory
                self.memory_cache.clear()
            
            self.logger.info("Cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(
                "Cache clear failed",
                error=str(e)
            )
            return False
    
    # ========================================================================
    # High-level caching methods
    # ========================================================================
    
    def cache_normalized_query(
        self,
        query: Query,
        normalized_query: NormalizedQuery
    ) -> bool:
        """
        Cache normalized query.
        
        Args:
            query: Original query
            normalized_query: Normalized query
            
        Returns:
            True if successful
        """
        key = self._generate_key("norm_query", query.text)
        return self.set(key, normalized_query)
    
    def get_normalized_query(self, query: Query) -> Optional[NormalizedQuery]:
        """
        Get cached normalized query.
        
        Args:
            query: Original query
            
        Returns:
            Cached normalized query or None
        """
        key = self._generate_key("norm_query", query.text)
        return self.get(key)
    
    def cache_retrieval_results(
        self,
        query: str,
        documents: list
    ) -> bool:
        """
        Cache retrieval results.
        
        Args:
            query: Query text
            documents: Retrieved documents
            
        Returns:
            True if successful
        """
        key = self._generate_key("retrieval", query)
        # Use shorter TTL for retrieval (information may become stale)
        ttl = min(self.settings.cache_ttl_seconds, 1800)  # Max 30 minutes
        return self.set(key, documents, ttl=ttl)
    
    def get_retrieval_results(self, query: str) -> Optional[list]:
        """
        Get cached retrieval results.
        
        Args:
            query: Query text
            
        Returns:
            Cached documents or None
        """
        key = self._generate_key("retrieval", query)
        return self.get(key)
    
    def cache_answer(
        self,
        query: Query,
        answer: Answer,
        custom_ttl: Optional[int] = None
    ) -> bool:
        """
        Cache answer.
        
        Args:
            query: Original query
            answer: Generated answer
            custom_ttl: Custom TTL (uses default if None)
            
        Returns:
            True if successful
        """
        key = self._generate_key("answer", query.text)
        return self.set(key, answer, ttl=custom_ttl)
    
    def get_answer(self, query: Query) -> Optional[Answer]:
        """
        Get cached answer.
        
        Args:
            query: Original query
            
        Returns:
            Cached answer or None
        """
        key = self._generate_key("answer", query.text)
        return self.get(key)
    
    def cache_pipeline_response(
        self,
        query: Query,
        response: PipelineResponse,
        custom_ttl: Optional[int] = None
    ) -> bool:
        """
        Cache complete pipeline response.
        
        Args:
            query: Original query
            response: Pipeline response
            custom_ttl: Custom TTL
            
        Returns:
            True if successful
        """
        key = self._generate_key("pipeline", query.text)
        return self.set(key, response, ttl=custom_ttl)
    
    def get_pipeline_response(
        self,
        query: Query
    ) -> Optional[PipelineResponse]:
        """
        Get cached pipeline response.
        
        Args:
            query: Original query
            
        Returns:
            Cached pipeline response or None
        """
        key = self._generate_key("pipeline", query.text)
        return self.get(key)
    
    def invalidate_query_cache(self, query: Query) -> bool:
        """
        Invalidate all cache entries for a query.
        
        Args:
            query: Query to invalidate
            
        Returns:
            True if successful
        """
        success = True
        
        # Delete normalized query
        key = self._generate_key("norm_query", query.text)
        success &= self.delete(key)
        
        # Delete retrieval
        key = self._generate_key("retrieval", query.text)
        success &= self.delete(key)
        
        # Delete answer
        key = self._generate_key("answer", query.text)
        success &= self.delete(key)
        
        # Delete pipeline
        key = self._generate_key("pipeline", query.text)
        success &= self.delete(key)
        
        return success
    
    def get_cache_stats(self) -> Dict:
        """
        Get cache statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'backend': self.backend,
        }
        
        try:
            if self.backend == "redis":
                info = self.redis_client.info('stats')
                stats['hits'] = info.get('keyspace_hits', 0)
                stats['misses'] = info.get('keyspace_misses', 0)
                stats['keys'] = self.redis_client.dbsize()
                
            elif self.backend == "disk":
                stats['size'] = self.disk_cache.volume()
                stats['keys'] = len(self.disk_cache)
                
            else:  # memory
                stats['keys'] = len(self.memory_cache)
                # Clean expired entries
                now = datetime.utcnow()
                expired = sum(
                    1 for entry in self.memory_cache.values()
                    if entry['expires_at'] <= now
                )
                stats['expired_keys'] = expired
        
        except Exception as e:
            self.logger.warning(
                "Failed to get cache stats",
                error=str(e)
            )
        
        return stats
    
    def cleanup_expired(self) -> int:
        """
        Cleanup expired cache entries (for memory backend).
        
        Returns:
            Number of entries cleaned
        """
        if self.backend != "memory":
            return 0
        
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.memory_cache.items()
            if entry['expires_at'] <= now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            self.logger.info(
                "Cleaned expired cache entries",
                count=len(expired_keys)
            )
        
        return len(expired_keys)
    
    def warm_cache(self, common_queries: list) -> int:
        """
        Pre-warm cache with common queries.
        
        Args:
            common_queries: List of common query strings
            
        Returns:
            Number of queries warmed
        """
        # This would pre-process common queries
        # For now, just a placeholder
        self.logger.info(
            "Cache warming not implemented yet",
            num_queries=len(common_queries)
        )
        return 0
    
    def __del__(self):
        """Cleanup on deletion."""
        if self.backend == "redis" and hasattr(self, 'redis_client'):
            self.redis_client.close()


    def cache_vector_index(self, query: str, vector_index) -> bool:
        key = self._generate_key("vector", query)
        return self.set(key, vector_index)

    def get_vector_index(self, query: str):
            key = self._generate_key("vector", query)
            return self.get(key)
    
    def cache_knowledge_graph(self, query: str, graph) -> bool:
        key = self._generate_key("graph", query)
        return self.set(key, graph)

    def get_knowledge_graph(self, query: str):
        key = self._generate_key("graph", query)
        return self.get(key)
    
    def cache_subgraph(self, query: str, subgraph) -> bool:
        key = self._generate_key("subgraph", query)
        return self.set(key, subgraph)

    def get_subgraph(self, query: str):
        key = self._generate_key("subgraph", query)
        return self.get(key)


