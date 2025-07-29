"""
Incremental Cache
=================

Distributed cache with incremental update support.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Set, Tuple, Optional, Any

from base_classes import FileMetadata
from enhanced_cache import EnhancedIncrementalCache

logger = logging.getLogger(__name__)


class IncrementalCache:
    """Distributed cache with incremental update support - now with enhanced file locking"""
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 86400):
        # Validate cache directory path to prevent path traversal
        cache_dir = Path(os.path.abspath(cache_dir))
        if not str(cache_dir).startswith(os.path.abspath(os.getcwd())):
            # Ensure cache dir is within current working directory or explicitly allowed
            if not os.environ.get('ALLOW_EXTERNAL_CACHE', '').lower() == 'true':
                raise ValueError(f"Cache directory must be within current working directory")
        
        # Use enhanced cache implementation
        self._enhanced_cache = EnhancedIncrementalCache(cache_dir, ttl_seconds)
        
        # Expose properties for compatibility
        self.cache_dir = self._enhanced_cache.cache_dir
        self.ttl_seconds = self._enhanced_cache.ttl_seconds
        self.index_file = self._enhanced_cache.index_file
        self._index_lock = self._enhanced_cache._async_lock
        self._sync_lock = self._enhanced_cache._memory_lock
        self.index = self._enhanced_cache.index
        
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk with proper file locking"""
        # Delegate to enhanced cache
        return self._enhanced_cache._load_index_safe()
    
    def _save_index(self):
        """Save cache index to disk with proper file locking"""
        # Delegate to enhanced cache
        self._enhanced_cache._save_index_safe()
    
    def get_cached_metadata(self, file_path: str, content_hash: str) -> Optional[FileMetadata]:
        """Retrieve cached metadata if valid"""
        # Delegate to enhanced cache and ensure we get FileMetadata type
        data = self._enhanced_cache.get_cached_metadata(file_path, content_hash)
        if data is not None and isinstance(data, FileMetadata):
            return data
        elif data is not None:
            logger.error(f"Invalid cached data type: {type(data)}")
        return None
    
    def cache_metadata(self, metadata: FileMetadata):
        """Cache file metadata with thread safety"""
        logger.debug(f"cache_metadata called for {metadata.path}")
        # Delegate to enhanced cache
        success = self._enhanced_cache.update_cache(
            metadata.path, 
            metadata.content_hash, 
            metadata
        )
        if success:
            logger.debug(f"Successfully cached metadata for {metadata.path}")
            # Update our reference to the index
            self.index = self._enhanced_cache.index
        else:
            logger.error(f"Failed to cache metadata for {metadata.path}")
    
    def get_changed_files(self, 
                         current_files: Dict[str, str]) -> Tuple[Set[str], Set[str], Set[str]]:
        """Identify added, modified, and deleted files"""
        cached_files = set(self.index.keys())
        current_file_set = set(current_files.keys())
        
        added = current_file_set - cached_files
        deleted = cached_files - current_file_set
        
        modified = set()
        for file_path in current_file_set & cached_files:
            if self.index[file_path]['content_hash'] != current_files[file_path]:
                modified.add(file_path)
        
        return added, modified, deleted
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired = []
        
        for file_path, entry in self.index.items():
            if current_time - entry['timestamp'] > self.ttl_seconds:
                expired.append(file_path)
                cache_file = self.cache_dir / entry['cache_file']
                if cache_file.exists():
                    cache_file.unlink()
        
        for file_path in expired:
            del self.index[file_path]
        
        if expired:
            self._save_index()