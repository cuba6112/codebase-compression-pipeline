"""
Redis-based Distributed Caching System
======================================

Provides distributed caching capabilities using Redis for the
codebase compression pipeline, enabling shared cache across
multiple workers and machines.
"""

import json
import pickle
import hashlib
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

try:
    import redis
    from redis.asyncio import Redis as AsyncRedis
    from redis.exceptions import RedisError, ConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

from base_classes import FileMetadata
from enhanced_cache import EnhancedIncrementalCache

logger = logging.getLogger(__name__)


@dataclass
class RedisCacheConfig:
    """Configuration for Redis cache"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 5
    connection_pool_max_connections: int = 50
    decode_responses: bool = False
    
    # Cache settings
    ttl_seconds: int = 86400  # 24 hours default TTL
    key_prefix: str = "codebase_cache"
    enable_compression: bool = True
    max_value_size: int = 512 * 1024 * 1024  # 512MB max value size
    
    # Cluster settings
    enable_cluster: bool = False
    cluster_nodes: List[Dict[str, Any]] = None


class RedisDistributedCache:
    """Distributed cache implementation using Redis"""
    
    def __init__(self, config: RedisCacheConfig):
        self.config = config
        self.client = None
        self.async_client = None
        self._connected = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        if not REDIS_AVAILABLE:
            raise ImportError("Redis library not available. Install with: pip install redis")
            
        self._connect()
        
    def _connect(self):
        """Establish connection to Redis"""
        try:
            if self.config.enable_cluster:
                # Redis cluster mode
                from redis.cluster import RedisCluster
                startup_nodes = self.config.cluster_nodes or [
                    {"host": self.config.host, "port": self.config.port}
                ]
                self.client = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=self.config.decode_responses,
                    skip_full_coverage_check=True,
                    password=self.config.password
                )
            else:
                # Single Redis instance
                self.client = redis.Redis(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    socket_timeout=self.config.socket_timeout,
                    decode_responses=self.config.decode_responses,
                    connection_pool_kwargs={
                        'max_connections': self.config.connection_pool_max_connections
                    }
                )
                
            # Test connection
            self.client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
            
        except (RedisError, ConnectionError) as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            raise
            
    async def _connect_async(self):
        """Establish async connection to Redis"""
        if not self.async_client:
            self.async_client = await AsyncRedis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                decode_responses=self.config.decode_responses
            )
            
    def _make_key(self, key: str) -> str:
        """Create namespaced Redis key"""
        return f"{self.config.key_prefix}:{key}"
        
    def _serialize_metadata(self, metadata: FileMetadata) -> bytes:
        """Serialize FileMetadata to bytes"""
        # Convert to dict and handle special types
        data = asdict(metadata)
        
        # Handle sets by converting to lists
        if 'dependencies' in data and isinstance(data['dependencies'], set):
            data['dependencies'] = list(data['dependencies'])
            
        # Add type information
        data['__type__'] = 'FileMetadata'
        
        # Serialize
        serialized = pickle.dumps(data)
        
        # Compress if enabled and size warrants it
        if self.config.enable_compression and len(serialized) > 1024:
            import lz4.frame
            serialized = b'LZ4:' + lz4.frame.compress(serialized)
            
        return serialized
        
    def _deserialize_metadata(self, data: bytes) -> FileMetadata:
        """Deserialize bytes to FileMetadata"""
        # Decompress if needed
        if data.startswith(b'LZ4:'):
            import lz4.frame
            data = lz4.frame.decompress(data[4:])
            
        # Deserialize
        obj = pickle.loads(data)
        
        # Convert back to FileMetadata
        if isinstance(obj, dict) and obj.get('__type__') == 'FileMetadata':
            obj.pop('__type__')
            # Convert lists back to sets
            if 'dependencies' in obj and isinstance(obj['dependencies'], list):
                obj['dependencies'] = set(obj['dependencies'])
            return FileMetadata(**obj)
        return obj
        
    def get(self, key: str) -> Optional[FileMetadata]:
        """Get item from cache"""
        if not self._connected:
            return None
            
        try:
            redis_key = self._make_key(key)
            data = self.client.get(redis_key)
            
            if data:
                return self._deserialize_metadata(data)
            return None
            
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
            
    async def get_async(self, key: str) -> Optional[FileMetadata]:
        """Async get item from cache"""
        if not self._connected:
            return None
            
        try:
            await self._connect_async()
            redis_key = self._make_key(key)
            data = await self.async_client.get(redis_key)
            
            if data:
                return self._deserialize_metadata(data)
            return None
            
        except Exception as e:
            logger.error(f"Redis async get error for key {key}: {e}")
            return None
            
    def set(self, key: str, value: FileMetadata, ttl: Optional[int] = None) -> bool:
        """Set item in cache"""
        if not self._connected:
            return False
            
        try:
            redis_key = self._make_key(key)
            serialized = self._serialize_metadata(value)
            
            # Check size limit
            if len(serialized) > self.config.max_value_size:
                logger.warning(f"Value too large for key {key}: {len(serialized)} bytes")
                return False
                
            # Set with TTL
            ttl = ttl or self.config.ttl_seconds
            return self.client.setex(redis_key, ttl, serialized)
            
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
            
    async def set_async(self, key: str, value: FileMetadata, ttl: Optional[int] = None) -> bool:
        """Async set item in cache"""
        if not self._connected:
            return False
            
        try:
            await self._connect_async()
            redis_key = self._make_key(key)
            serialized = self._serialize_metadata(value)
            
            # Check size limit
            if len(serialized) > self.config.max_value_size:
                logger.warning(f"Value too large for key {key}: {len(serialized)} bytes")
                return False
                
            # Set with TTL
            ttl = ttl or self.config.ttl_seconds
            return await self.async_client.setex(redis_key, ttl, serialized)
            
        except Exception as e:
            logger.error(f"Redis async set error for key {key}: {e}")
            return False
            
    def delete(self, key: str) -> bool:
        """Delete item from cache"""
        if not self._connected:
            return False
            
        try:
            redis_key = self._make_key(key)
            return bool(self.client.delete(redis_key))
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
            
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        if not self._connected:
            return False
            
        try:
            redis_key = self._make_key(key)
            return bool(self.client.exists(redis_key))
            
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
            
    def get_many(self, keys: List[str]) -> Dict[str, FileMetadata]:
        """Get multiple items from cache"""
        if not self._connected or not keys:
            return {}
            
        try:
            redis_keys = [self._make_key(key) for key in keys]
            values = self.client.mget(redis_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = self._deserialize_metadata(value)
                    except Exception as e:
                        logger.error(f"Failed to deserialize value for key {key}: {e}")
                        
            return result
            
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            return {}
            
    def set_many(self, items: Dict[str, FileMetadata], ttl: Optional[int] = None) -> bool:
        """Set multiple items in cache"""
        if not self._connected or not items:
            return False
            
        try:
            # Use pipeline for atomic operation
            pipe = self.client.pipeline()
            ttl = ttl or self.config.ttl_seconds
            
            for key, value in items.items():
                redis_key = self._make_key(key)
                serialized = self._serialize_metadata(value)
                
                if len(serialized) <= self.config.max_value_size:
                    pipe.setex(redis_key, ttl, serialized)
                else:
                    logger.warning(f"Skipping large value for key {key}")
                    
            pipe.execute()
            return True
            
        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            return False
            
    def clear_prefix(self, prefix: str) -> int:
        """Clear all keys with given prefix"""
        if not self._connected:
            return 0
            
        try:
            pattern = f"{self.config.key_prefix}:{prefix}*"
            count = 0
            
            # Use SCAN to avoid blocking
            for key in self.client.scan_iter(match=pattern, count=100):
                self.client.delete(key)
                count += 1
                
            logger.info(f"Cleared {count} keys with prefix {prefix}")
            return count
            
        except Exception as e:
            logger.error(f"Redis clear_prefix error: {e}")
            return 0
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._connected:
            return {"connected": False}
            
        try:
            info = self.client.info()
            pattern = f"{self.config.key_prefix}:*"
            
            # Count keys (using SCAN to avoid blocking)
            key_count = 0
            for _ in self.client.scan_iter(match=pattern, count=100):
                key_count += 1
                
            return {
                "connected": True,
                "host": self.config.host,
                "port": self.config.port,
                "key_count": key_count,
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0),
                    info.get("keyspace_misses", 0)
                )
            }
            
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {"connected": False, "error": str(e)}
            
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return 0.0
        return round(hits / total * 100, 2)
        
    def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        if not self._connected:
            return False
            
        try:
            return self.client.ping()
        except Exception:
            self._connected = False
            return False
            
    def close(self):
        """Close Redis connections"""
        if self.client:
            self.client.close()
        if self.async_client:
            asyncio.create_task(self.async_client.close())
        self._executor.shutdown(wait=True)
        self._connected = False


class HybridCache:
    """Hybrid cache that uses both local and Redis caching"""
    
    def __init__(self, cache_dir: Path, redis_config: Optional[RedisCacheConfig] = None):
        # Initialize local cache
        self.local_cache = EnhancedIncrementalCache(cache_dir)
        self.cache_dir = cache_dir
        
        self.redis_cache = None
        if redis_config and REDIS_AVAILABLE:
            try:
                self.redis_cache = RedisDistributedCache(redis_config)
                logger.info("Hybrid cache initialized with Redis support")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                logger.info("Falling back to local cache only")
                
    def get_cached_metadata(self, file_path: str, content_hash: str) -> Optional[FileMetadata]:
        """Get metadata with Redis fallback"""
        # Try local cache first
        metadata = self.local_cache.get_cached_metadata(file_path, content_hash)
        if metadata:
            return metadata
            
        # Try Redis if available
        if self.redis_cache:
            key = f"{file_path}:{content_hash}"
            metadata = self.redis_cache.get(key)
            if metadata:
                # Store in local cache for faster access
                self._cache_metadata(file_path, content_hash, metadata)
                return metadata
                
        return None
        
    def _cache_metadata(self, file_path: str, content_hash: str, metadata: FileMetadata):
        """Store metadata in both local and Redis cache"""
        # Store locally
        with self.local_cache._memory_lock:
            if not hasattr(self.local_cache, '_cache'):
                self.local_cache._cache = {}
            self.local_cache._cache[file_path] = {
                'content_hash': content_hash,
                'metadata': metadata,
                'timestamp': time.time()
            }
        
        # Store in Redis if available
        if self.redis_cache:
            key = f"{file_path}:{content_hash}"
            self.redis_cache.set(key, metadata)
            
    def invalidate(self, file_path: Path) -> bool:
        """Invalidate in both caches"""
        # Invalidate in local cache
        local_success = True
        file_str = str(file_path)
        with self.local_cache._memory_lock:
            if hasattr(self.local_cache, '_cache') and file_str in self.local_cache._cache:
                del self.local_cache._cache[file_str]
                
        # Invalidate in Redis
        if self.redis_cache:
            # Need to find all keys for this file (different content hashes)
            pattern = f"{file_str}:*"
            cleared = self.redis_cache.clear_prefix(pattern)
            return local_success and cleared > 0
            
        return local_success
        
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics"""
        stats = {
            "local": {
                "cache_dir": str(self.cache_dir),
                "entries": len(getattr(self.local_cache, '_cache', {}))
            }
        }
        
        if self.redis_cache:
            stats["redis"] = self.redis_cache.get_stats()
            
        return stats
        
    def _save_index(self):
        """Save index to disk (delegate to local cache)"""
        self.local_cache._save_index_safe()
        
    def cleanup(self):
        """Cleanup resources"""
        if self.redis_cache:
            self.redis_cache.close()


def test_redis_cache():
    """Test Redis cache functionality"""
    print("Testing Redis Cache...")
    
    if not REDIS_AVAILABLE:
        print("❌ Redis library not available. Install with: pip install redis")
        return False
        
    # Test configuration
    config = RedisCacheConfig(
        host="localhost",
        port=6379,
        key_prefix="test_cache"
    )
    
    try:
        # Create cache
        cache = RedisDistributedCache(config)
        print("✅ Connected to Redis")
        
        # Test basic operations
        test_metadata = FileMetadata(
            path="/test/file.py",
            size=1000,
            language="python",
            last_modified=time.time(),
            content_hash="abc123"
        )
        
        # Set
        success = cache.set("test_key", test_metadata)
        print(f"✅ Set operation: {success}")
        
        # Get
        retrieved = cache.get("test_key")
        print(f"✅ Get operation: {retrieved is not None}")
        
        # Exists
        exists = cache.exists("test_key")
        print(f"✅ Exists check: {exists}")
        
        # Stats
        stats = cache.get_stats()
        print(f"✅ Stats: {stats}")
        
        # Delete
        deleted = cache.delete("test_key")
        print(f"✅ Delete operation: {deleted}")
        
        # Close
        cache.close()
        print("✅ Cache closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        print("\nMake sure Redis is running:")
        print("  brew services start redis")
        print("  or: redis-server")
        return False


if __name__ == "__main__":
    test_redis_cache()