"""
Enhanced Caching System with Cross-Platform File Locking
========================================================

Implements robust file locking for the IncrementalCache to ensure
thread and process safety across different operating systems.
"""

import asyncio
import json
import os
import platform
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import pickle
import io
import hashlib
import hashlib

logger = logging.getLogger(__name__)


class FileLock:
    """Cross-platform file locking mechanism"""
    
    def __init__(self, lockfile: Path, timeout: float = 30.0):
        self.lockfile = lockfile
        self.timeout = timeout
        self.lock_acquired = False
        self._start_time = None
        
    def acquire(self, blocking: bool = True) -> bool:
        """Acquire the file lock"""
        self._start_time = time.time()
        
        while True:
            try:
                # Try to create lock file exclusively
                fd = os.open(str(self.lockfile), 
                           os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                self.lock_acquired = True
                return True
                
            except FileExistsError:
                if not blocking:
                    return False
                    
                # Check timeout
                if time.time() - self._start_time > self.timeout:
                    # Check if lock file is stale
                    if self._is_stale_lock():
                        self._remove_stale_lock()
                        continue
                    return False
                    
                # Wait before retrying
                time.sleep(0.01)
                
    def release(self):
        """Release the file lock"""
        if self.lock_acquired and self.lockfile.exists():
            try:
                self.lockfile.unlink()
                self.lock_acquired = False
            except OSError as e:
                logger.warning(f"OS error releasing lock: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error releasing lock: {e}", exc_info=True)
                
    def _is_stale_lock(self) -> bool:
        """Check if lock file is stale (older than timeout)"""
        try:
            if self.lockfile.exists():
                mtime = self.lockfile.stat().st_mtime
                return time.time() - mtime > self.timeout
        except OSError as e:
            logger.debug(f"Could not check lock file stats: {e}")
        except Exception as e:
            logger.debug(f"Unexpected error checking lock file stats: {e}", exc_info=True)
        return False
        
    def _remove_stale_lock(self):
        """Remove stale lock file"""
        try:
            self.lockfile.unlink()
            logger.info(f"Removed stale lock file: {self.lockfile}")
        except OSError as e:
            logger.warning(f"Could not remove stale lock: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error removing stale lock: {e}", exc_info=True)
            
    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Could not acquire lock on {self.lockfile}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class EnhancedIncrementalCache:
    """Improved cache with robust cross-platform file locking"""
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400):
        # Validate cache directory
        self.cache_dir = Path(os.path.abspath(cache_dir))
        self.ttl_seconds = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache files
        self.index_file = self.cache_dir / "index.json"
        self.lock_file = self.cache_dir / ".index.lock"
        
        # In-memory locks for thread safety
        self._memory_lock = threading.RLock()
        self._async_lock = asyncio.Lock()
        
        # Load initial index
        self.index = self._load_index_safe()
        
    def _load_index_safe(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index with proper locking"""
        with FileLock(self.lock_file):
            if self.index_file.exists():
                try:
                    with open(self.index_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Cache index corrupted: {e}")
                    self._backup_corrupted_index()
        return {}
        
    def _convert_metadata_to_json(self, metadata: Any) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable format"""
        # This is a simple conversion - extend based on your metadata structure
        if isinstance(metadata, dict):
            return {k: str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v 
                   for k, v in metadata.items()}
        elif hasattr(metadata, '__dict__'):
            return self._convert_metadata_to_json(metadata.__dict__)
        else:
            return {'data': str(metadata)}
    
    def _save_index_safe(self):
        """Save cache index with proper locking"""
        temp_file = self.index_file.with_suffix('.tmp')
        
        with FileLock(self.lock_file):
            try:
                # Write to temp file
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(self.index, f, indent=2, sort_keys=True)
                
                # Atomic rename
                temp_file.replace(self.index_file)
                
            except (OSError, IOError) as e:
                logger.error(f"I/O error saving cache index: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise
            except json.JSONDecodeError as e:
                logger.error(f"JSON error saving cache index: {e}")
                if temp_file.exists():
                    temp_file.unlink()
                raise
            except Exception as e:
                logger.error(f"Unexpected error saving cache index: {e}", exc_info=True)
                if temp_file.exists():
                    temp_file.unlink()
                raise
                
    def _backup_corrupted_index(self):
        """Backup corrupted index file"""
        if self.index_file.exists():
            backup_name = f"index.corrupted.{int(time.time())}.json"
            backup_file = self.cache_dir / backup_name
            try:
                self.index_file.rename(backup_file)
                logger.info(f"Backed up corrupted index to: {backup_file}")
            except OSError as e:
                logger.error(f"OS error backing up corrupted index: {e}")
            except Exception as e:
                logger.error(f"Unexpected error backing up corrupted index: {e}", exc_info=True)
                
    def get_cached_metadata(self, file_path: str, content_hash: str) -> Optional[Any]:
        """Thread-safe retrieval of cached metadata"""
        with self._memory_lock:
            if file_path in self.index:
                entry = self.index[file_path]
                
                # Validate cache entry
                if (entry.get('content_hash') == content_hash and 
                    time.time() - entry.get('timestamp', 0) < self.ttl_seconds):
                    
                    cache_file = self.cache_dir / entry['cache_file']
                    if cache_file.exists():
                        try:
                            # Try to load as JSON first (newer format)
                            json_file = cache_file.with_suffix('.json')
                            if json_file.exists():
                                with open(json_file, 'r') as f:
                                    return json.load(f)
                            
                            # Fall back to pickle for backward compatibility
                            # Use secure pickle loading
                            if cache_file.exists():
                                from security_validation import SecureProcessor
                                return SecureProcessor.safe_pickle_load(cache_file)
                            
                            return None
                        except (OSError, IOError) as e:
                            logger.warning(f"I/O error loading cached data: {e}")
                        except pickle.UnpicklingError as e:
                            logger.warning(f"Unpickling error loading cached data: {e}")
                        except Exception as e:
                            logger.warning(f"Unexpected error loading cached data: {e}", exc_info=True)
                            
        return None
        
    async def get_cached_metadata_async(self, file_path: str, content_hash: str) -> Optional[Any]:
        """Async version of get_cached_metadata"""
        async with self._async_lock:
            return self.get_cached_metadata(file_path, content_hash)
            
    def update_cache(self, file_path: str, content_hash: str, metadata: Any) -> bool:
        """Thread-safe cache update"""
        with self._memory_lock:
            try:
                # Generate cache filename
                cache_filename = hashlib.sha256(file_path.encode()).hexdigest()[:16] + '.pkl'
                cache_file = self.cache_dir / cache_filename
                
                # Save metadata to disk
                # Use JSON for new cache entries (security)
                json_file = cache_file.with_suffix('.json')
                
                # Convert metadata to JSON-serializable format
                json_metadata = self._convert_metadata_to_json(metadata)
                
                # Write atomically
                temp_file = json_file.with_suffix('.tmp')
                try:
                    with open(temp_file, 'w') as f:
                        json.dump(json_metadata, f, indent=2)
                    temp_file.replace(json_file)
                    cache_file = json_file  # Update reference for size calculation
                except Exception:
                    temp_file.unlink(missing_ok=True)
                    raise
                
                # Update index with JSON file reference
                self.index[file_path] = {
                    'content_hash': content_hash,
                    'cache_file': cache_filename.replace('.pkl', '.json'),
                    'timestamp': time.time(),
                    'size': os.path.getsize(cache_file)
                }
                
                # Save index with locking
                self._save_index_safe()
                return True
                
            except (OSError, IOError) as e:
                logger.error(f"I/O error updating cache for {file_path}: {e}")
                return False
            except pickle.PicklingError as e:
                logger.error(f"Pickling error updating cache for {file_path}: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error updating cache for {file_path}: {e}", exc_info=True)
                return False
                
    async def update_cache_async(self, file_path: str, content_hash: str, metadata: Any) -> bool:
        """Async version of update_cache"""
        async with self._async_lock:
            return self.update_cache(file_path, content_hash, metadata)
            
    def clear_expired(self) -> int:
        """Remove expired cache entries"""
        with self._memory_lock:
            current_time = time.time()
            expired_count = 0
            
            with FileLock(self.lock_file):
                # Find expired entries
                expired_files = []
                for file_path, entry in list(self.index.items()):
                    if current_time - entry.get('timestamp', 0) > self.ttl_seconds:
                        expired_files.append(file_path)
                        
                        # Remove cache file
                        cache_file = self.cache_dir / entry['cache_file']
                        if cache_file.exists():
                            try:
                                cache_file.unlink()
                            except OSError as e:
                                logger.warning(f"OS error removing cache file: {e}")
                            except Exception as e:
                                logger.warning(f"Unexpected error removing cache file: {e}", exc_info=True)
                                
                # Update index
                for file_path in expired_files:
                    del self.index[file_path]
                    expired_count += 1
                    
                if expired_count > 0:
                    self._save_index_safe()
                    
            logger.info(f"Cleared {expired_count} expired cache entries")
            return expired_count
            
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._memory_lock:
            total_size = 0
            file_count = 0
            
            for entry in self.index.values():
                total_size += entry.get('size', 0)
                file_count += 1
                
            return {
                'total_files': file_count,
                'total_size_mb': total_size / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
                'ttl_seconds': self.ttl_seconds,
                'oldest_entry': min(
                    (e.get('timestamp', time.time()) for e in self.index.values()),
                    default=None
                )
            }


# Example migration function to upgrade existing cache
def migrate_to_enhanced_cache(old_cache_dir: Path) -> EnhancedIncrementalCache:
    """Migrate from old cache format to enhanced cache"""
    enhanced_cache = EnhancedIncrementalCache(old_cache_dir)
    
    # If old index exists without lock file, it's an old cache
    old_index = old_cache_dir / "index.json"
    if old_index.exists() and not enhanced_cache.lock_file.exists():
        logger.info("Migrating old cache to enhanced format...")
        
        # The enhanced cache will have already loaded the old index
        # Just save it with the new locking mechanism
        enhanced_cache._save_index_safe()
        logger.info("Cache migration complete")
        
    return enhanced_cache