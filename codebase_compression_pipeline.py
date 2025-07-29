"""
Codebase Compression Pipeline for LLM Context
============================================

A scalable data processing pipeline for transforming large codebases 
into compressed representations optimized for LLM consumption.
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, AsyncIterator, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
import multiprocessing as mp
from queue import Queue
import threading
import time
from datetime import datetime
import pickle
import lz4.frame
import mmh3
from collections import defaultdict, OrderedDict
import ast
import re
import logging
import os
from functools import lru_cache
import aiofiles
import weakref
try:
    from tqdm.asyncio import tqdm as async_tqdm
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logging.info("Note: Install 'tqdm' for progress bars: pip install tqdm")

# Import base classes
from base_classes import FileMetadata, LanguageParser

# Import enhanced cache system
from enhanced_cache import EnhancedIncrementalCache, FileLock
# Import enhanced JavaScript parser
from parsers.enhanced_js_parser import EnhancedJavaScriptParser
# Import Python parser
from parsers.python_parser import PythonParser
# Import TypeScript parser (will be done after logger is initialized)
TYPESCRIPT_PARSER_AVAILABLE = False
TypeScriptParser = None
# Import resilience patterns
from resilience_patterns import (
    with_retry, RetryConfig, CircuitBreaker, CircuitBreakerConfig,
    BulkheadExecutor, TimeoutHandler, FallbackHandler, HealthChecker,
    RetryableError, NonRetryableError
)
# Import security validation
from security_validation import (
    SecurityConfig, SecurityValidator, PathValidator,
    FileValidator, ContentScanner, SecureProcessor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import TypeScript parser after logger is initialized
try:
    from parsers.typescript_parser import TypeScriptParser
    TYPESCRIPT_PARSER_AVAILABLE = True
    logger.info("TypeScript parser loaded successfully")
except ImportError as e:
    TYPESCRIPT_PARSER_AVAILABLE = False
    logger.warning(f"TypeScript parser not available: {e}")
    logger.info("Install Node.js and run: npm install -g @typescript-eslint/parser")

# Try to import Go parser
try:
    from parsers.go_parser import GoParser
    GO_PARSER_AVAILABLE = True
    logger.info("Go parser loaded successfully")
except ImportError as e:
    GO_PARSER_AVAILABLE = False
    logger.warning(f"Go parser not available: {e}")

# Try to import Rust parser
try:
    from parsers.rust_parser import RustParser
    RUST_PARSER_AVAILABLE = True
    logger.info("Rust parser loaded successfully")
except ImportError as e:
    RUST_PARSER_AVAILABLE = False
    logger.warning(f"Rust parser not available: {e}")

# Try to import Redis cache
try:
    from redis_cache import RedisCacheConfig, HybridCache, REDIS_AVAILABLE
    logger.info("Redis cache support loaded successfully")
except ImportError as e:
    REDIS_AVAILABLE = False
    logger.warning(f"Redis cache not available: {e}")

# Import adaptive compression
try:
    from adaptive_compression import (
        AdaptiveCompressionStrategy, AdaptiveCompressionManager,
        CompressionAlgorithm, CompressionProfile
    )
    ADAPTIVE_COMPRESSION_AVAILABLE = True
    logger.info("Adaptive compression loaded successfully")
except ImportError as e:
    ADAPTIVE_COMPRESSION_AVAILABLE = False
    logger.warning(f"Adaptive compression not available: {e}")


# ============================================================================
# STAGE 1: FILE PARSING AND TOKENIZATION
# ============================================================================

# FileMetadata and LanguageParser are now imported from base_classes.py





# ============================================================================
# STAGE 2: PARALLEL PROCESSING ENGINE
# ============================================================================

class ParallelProcessor:
    """Manages parallel processing of files with work stealing"""
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = min(num_workers or mp.cpu_count(), 32)  # Cap at reasonable limit
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.num_workers * 2)
        # Track executors for cleanup
        self._executors = weakref.WeakSet()
        self._executors.add(self.executor)
        self._executors.add(self.thread_executor)
    
    def _cleanup_partial_init(self):
        """Clean up partially initialized components"""
        components = ['processor', 'cache', 'metadata_store', 'health_checker']
        for comp in components:
            if hasattr(self, comp):
                try:
                    obj = getattr(self, comp)
                    if hasattr(obj, 'shutdown'):
                        obj.shutdown()
                    elif hasattr(obj, 'close'):
                        obj.close()
                except AttributeError as e:
                    logger.warning(f"Component {comp} not found during cleanup: {e}")
                except (OSError, IOError) as e:
                    logger.warning(f"I/O error cleaning up {comp}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error cleaning up {comp}: {e}", exc_info=True)
                    
    def _register_health_checks(self):
        """Register component health checks"""
        # Cache health check
        async def check_cache_health():
            try:
                # Test cache operations
                test_key = "__health_check__"
                self.cache.get_cached_metadata(test_key, "test_hash")
                return True
            except (FileNotFoundError, PermissionError, OSError):
                return False
            except Exception:
                logger.debug("Unexpected error in health check", exc_info=True)
                return False
                
        # Metadata store health check
        async def check_metadata_health():
            try:
                # Test metadata query
                self.metadata_store.query({})
                return True
            except (FileNotFoundError, PermissionError, OSError):
                return False
            except Exception:
                logger.debug("Unexpected error in health check", exc_info=True)
                return False
                
        self.health_checker.register_component("cache", check_cache_health)
        self.health_checker.register_component("metadata_store", check_metadata_health)
        
    async def start_health_monitoring(self):
        """Start health monitoring if enabled"""
        if self.enable_resilience:
            await self.health_checker.start()
            
    async def stop_health_monitoring(self):
        """Stop health monitoring if enabled"""
        if self.enable_resilience:
            await self.health_checker.stop()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        self.shutdown()
        return False
    
    
    def shutdown(self):
        """Shutdown all executors"""
        for executor in list(self._executors):
            try:
                executor.shutdown(wait=True)
            except RuntimeError as e:
                logger.error(f"Runtime error shutting down executor: {e}")
            except Exception as e:
                logger.error(f"Unexpected error shutting down executor: {e}", exc_info=True)
        
    async def process_files_parallel(self, 
                                   file_paths: List[Path],
                                   parser_factory: Dict[str, LanguageParser],
                                   batch_size: int = 100) -> AsyncIterator[FileMetadata]:
        """Process files in parallel with batching and work stealing"""
        
        logger.debug(f"Starting process_files_parallel with {len(file_paths)} files, {self.num_workers} workers")
        
        # Handle empty file list
        if not file_paths:
            logger.debug("No files to process, returning early")
            return
        
        # Create work queue with batches
        work_queue = asyncio.Queue()
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            await work_queue.put(batch)
            logger.debug(f"Added batch {i//batch_size + 1} with {len(batch)} files")
        
        # Add sentinel values
        for _ in range(self.num_workers):
            await work_queue.put(None)
        
        # Result queue for workers to push results
        result_queue = asyncio.Queue()
        
        # Worker coroutine
        async def worker():
            while True:
                batch = await work_queue.get()
                if batch is None:
                    break
                
                # Process batch in thread pool
                try:
                    results = await asyncio.get_running_loop().run_in_executor(
                        self.thread_executor,
                        self._process_batch,
                        batch,
                        parser_factory
                    )
                    
                    for result in results:
                        if result:
                            await result_queue.put(result)
                except asyncio.CancelledError:
                    logger.info("Worker task cancelled")
                    raise
                except asyncio.TimeoutError as e:
                    logger.error(f"Worker timeout processing batch: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error in worker: {e}", exc_info=True)
                    # Continue processing other batches
            
            # Signal completion
            await result_queue.put(None)
        
        # Run workers concurrently
        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]
        logger.debug(f"Created {len(workers)} worker tasks")
        
        # Collect results
        completed_workers = 0
        while completed_workers < self.num_workers:
            logger.debug(f"Waiting for results, completed workers: {completed_workers}/{self.num_workers}")
            result = await result_queue.get()
            if result is None:
                completed_workers += 1
                logger.debug(f"Worker completed, total: {completed_workers}")
            else:
                yield result
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
        logger.debug("All workers completed")
    
    def _process_batch(self, 
                      batch: List[Path], 
                      parser_factory: Dict[str, LanguageParser]) -> List[FileMetadata]:
        """Process a batch of files with resilience patterns"""
        results = []
        
        # Configure retry for file operations
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            retryable_exceptions=(IOError, OSError)
        )
        
        @with_retry(retry_config)
        def read_file_with_retry(file_path: Path) -> str:
            """Read file with retry logic"""
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        for file_path in batch:
            try:
                # Read file with retry mechanism
                content = read_file_with_retry(file_path)
                
                # Determine language
                suffix = file_path.suffix.lower()
                language = self._detect_language(suffix)
                
                if language in parser_factory:
                    parser = parser_factory[language]
                    
                    # Parse with error recovery
                    try:
                        metadata = parser.parse(content, str(file_path))
                        results.append(metadata)
                    except SyntaxError as parse_error:
                        # Log syntax error but create minimal metadata
                        logger.error(f"Syntax error parsing {file_path}: {parse_error}")
                        metadata = FileMetadata(
                            path=str(file_path),
                            size=len(content),
                            language=language,
                            last_modified=time.time(),
                            content_hash=hashlib.sha256(content.encode()).hexdigest()
                        )
                        results.append(metadata)
                    except ValueError as parse_error:
                        # Log value error but create minimal metadata
                        logger.error(f"Value error parsing {file_path}: {parse_error}")
                        metadata = FileMetadata(
                            path=str(file_path),
                            size=len(content),
                            language=language,
                            last_modified=time.time(),
                            content_hash=hashlib.sha256(content.encode()).hexdigest()
                        )
                        results.append(metadata)
                    except Exception as parse_error:
                        # Log unexpected parsing error but create minimal metadata
                        logger.error(f"Unexpected parse error for {file_path}: {parse_error}", exc_info=True)
                        metadata = FileMetadata(
                            path=str(file_path),
                            size=len(content),
                            language=language,
                            last_modified=time.time(),
                            content_hash=hashlib.sha256(content.encode()).hexdigest()
                        )
                        results.append(metadata)
                        
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}")
            except PermissionError:
                logger.warning(f"Permission denied: {file_path}")
            except NonRetryableError as e:
                logger.error(f"Non-retryable error for {file_path}: {e}")
            except UnicodeDecodeError as e:
                logger.error(f"Encoding error processing {file_path}: {e}")
            except MemoryError as e:
                logger.error(f"Memory error processing {file_path}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path} after retries: {e}", exc_info=True)
        
        return results
    
    def _detect_language(self, suffix: str) -> str:
        """Detect programming language from file extension"""
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        return language_map.get(suffix, 'unknown')
    
    async def _merge_async_iterators(self, iterators):
        """Merge multiple async iterators"""
        queue = asyncio.Queue()
        finished = set()
        
        async def drain(it, idx):
            async for item in it:
                await queue.put((idx, item))
            finished.add(idx)
        
        # Start draining all iterators
        tasks = []
        for idx, it in enumerate(iterators):
            task = asyncio.create_task(drain(it, idx))
            tasks.append(task)
        
        # Yield items as they arrive
        while len(finished) < len(iterators):
            try:
                idx, item = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield item
            except asyncio.TimeoutError:
                continue


# ============================================================================
# STAGE 3: MEMORY-EFFICIENT STREAMING
# ============================================================================

class StreamingCompressor:
    """Memory-efficient streaming compression with sliding window"""
    
    def __init__(self, window_size: int = 1024 * 1024, chunk_size: int = 64 * 1024, max_hashes: int = 100000, 
                 adaptive_compression: bool = True, cache_dir: Optional[Path] = None):
        # Validate parameters
        if window_size <= 0 or chunk_size <= 0:
            raise ValueError("Window size and chunk size must be positive")
        if chunk_size > window_size:
            raise ValueError("Chunk size cannot be larger than window size")
        
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.buffer = bytearray()
        # Use OrderedDict as LRU cache to prevent unbounded memory growth
        self.seen_hashes = OrderedDict()
        self.max_hashes = max_hashes
        
        # Adaptive compression support
        self.adaptive_compression = adaptive_compression and ADAPTIVE_COMPRESSION_AVAILABLE
        if self.adaptive_compression:
            self.compression_manager = AdaptiveCompressionManager(
                cache_dir or Path('./compression_cache')
            )
        else:
            self.compression_manager = None
        
    async def compress_stream(self, data_stream: AsyncIterator[bytes], 
                            file_path: Optional[Path] = None,
                            metadata: Optional[FileMetadata] = None) -> AsyncIterator[bytes]:
        """Compress data stream with deduplication and adaptive compression"""
        
        async for chunk in data_stream:
            self.buffer.extend(chunk)
            
            # Process buffer when it exceeds window size
            while len(self.buffer) >= self.window_size:
                window = bytes(self.buffer[:self.window_size])
                
                # Deduplicate using content hash
                # Use mmh3 for performance, not security
                content_hash = mmh3.hash128(window)
                # For security-sensitive operations, use SHA-256:
                # content_hash = hashlib.sha256(window).hexdigest()
                if content_hash not in self.seen_hashes:
                    # Implement LRU cache behavior to prevent unbounded growth
                    if len(self.seen_hashes) >= self.max_hashes:
                        # Remove oldest hash (FIFO)
                        self.seen_hashes.popitem(last=False)
                    
                    # Add new hash (marks it as most recently used)
                    self.seen_hashes[content_hash] = True
                    
                    # Compress window
                    if self.adaptive_compression and file_path:
                        # Use adaptive compression
                        compressed, stats = self.compression_manager.compress_file(
                            file_path, window, metadata
                        )
                        yield compressed
                    else:
                        # Fallback to standard LZ4 compression
                        compressed = lz4.frame.compress(window, compression_level=12)
                        yield compressed
                else:
                    # Move to end (mark as recently used)
                    self.seen_hashes.move_to_end(content_hash)
                
                # Slide window
                self.buffer = self.buffer[self.chunk_size:]
        
        # Handle remaining buffer
        if self.buffer:
            if self.adaptive_compression and file_path:
                compressed, stats = self.compression_manager.compress_file(
                    file_path, bytes(self.buffer), metadata
                )
                yield compressed
            else:
                compressed = lz4.frame.compress(bytes(self.buffer))
                yield compressed
    
    def create_chunks(self, 
                     metadata_list: List[FileMetadata], 
                     max_chunk_size: int = 1024 * 1024) -> List[List[FileMetadata]]:
        """Create optimally sized chunks for processing"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Sort by complexity for better compression
        sorted_metadata = sorted(metadata_list, 
                               key=lambda m: m.complexity_score, 
                               reverse=True)
        
        for metadata in sorted_metadata:
            if current_size + metadata.size > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(metadata)
            current_size += metadata.size
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


# ============================================================================
# STAGE 4: CACHING AND INCREMENTAL UPDATES
# ============================================================================

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


# ============================================================================
# STAGE 5: METADATA EXTRACTION AND STORAGE
# ============================================================================

class MetadataStore:
    """Columnar storage for efficient metadata queries"""
    
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Columnar storage for different metadata types
        self.columns = {
            'paths': [],
            'sizes': [],
            'languages': [],
            'complexities': [],
            'token_counts': [],
            'imports': defaultdict(list),
            'exports': defaultdict(list),
            'functions': defaultdict(list),
            'classes': defaultdict(list)
        }
        
        # Inverted indices for fast lookups
        self.import_index = defaultdict(set)
        self.export_index = defaultdict(set)
        self.function_index = defaultdict(set)
        self.class_index = defaultdict(set)
    
    def add_metadata(self, metadata: FileMetadata):
        """Add metadata to columnar store"""
        idx = len(self.columns['paths'])
        
        # Add to columns
        self.columns['paths'].append(metadata.path)
        self.columns['sizes'].append(metadata.size)
        self.columns['languages'].append(metadata.language)
        self.columns['complexities'].append(metadata.complexity_score)
        self.columns['token_counts'].append(metadata.token_count)
        
        # Add to relational columns
        for imp in metadata.imports:
            self.columns['imports'][idx].append(imp)
            self.import_index[imp].add(idx)
        
        for exp in metadata.exports:
            self.columns['exports'][idx].append(exp)
            self.export_index[exp].add(idx)
        
        for func in metadata.functions:
            self.columns['functions'][idx].append(func['name'])
            self.function_index[func['name']].add(idx)
        
        for cls in metadata.classes:
            self.columns['classes'][idx].append(cls['name'])
            self.class_index[cls['name']].add(idx)
    
    def query(self, **criteria) -> List[int]:
        """Query metadata with multiple criteria"""
        result_indices = None
        
        if 'language' in criteria:
            lang_indices = {i for i, lang in enumerate(self.columns['languages']) 
                           if lang == criteria['language']}
            result_indices = lang_indices if result_indices is None else result_indices & lang_indices
        
        if 'imports' in criteria:
            import_indices = self.import_index.get(criteria['imports'], set())
            result_indices = import_indices if result_indices is None else result_indices & import_indices
        
        if 'exports' in criteria:
            export_indices = self.export_index.get(criteria['exports'], set())
            result_indices = export_indices if result_indices is None else result_indices & export_indices
        
        if 'min_complexity' in criteria:
            complex_indices = {i for i, comp in enumerate(self.columns['complexities']) 
                              if comp >= criteria['min_complexity']}
            result_indices = complex_indices if result_indices is None else result_indices & complex_indices
        
        return list(result_indices) if result_indices else []
    
    def get_metadata_by_indices(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Retrieve metadata for given indices"""
        results = []
        
        for idx in indices:
            if idx < len(self.columns['paths']):
                results.append({
                    'path': self.columns['paths'][idx],
                    'size': self.columns['sizes'][idx],
                    'language': self.columns['languages'][idx],
                    'complexity': self.columns['complexities'][idx],
                    'token_count': self.columns['token_counts'][idx],
                    'imports': self.columns['imports'].get(idx, []),
                    'exports': self.columns['exports'].get(idx, []),
                    'functions': self.columns['functions'].get(idx, []),
                    'classes': self.columns['classes'].get(idx, [])
                })
        
        return results
    
    def save(self):
        """Persist metadata store to disk safely"""
        metadata_file = self.store_path / 'metadata.pkl'
        temp_file = metadata_file.with_suffix('.tmp')
        
        try:
            with open(str(temp_file), 'wb') as f:
                pickle.dump({
                    'columns': dict(self.columns),
                    'indices': {
                        'import': dict(self.import_index),
                        'export': dict(self.export_index),
                        'function': dict(self.function_index),
                        'class': dict(self.class_index)
                    }
                }, f)
            # Atomic rename
            temp_file.replace(metadata_file)
        except (OSError, IOError) as e:
            logger.error(f"I/O error saving metadata store: {e}")
            if temp_file.exists():
                temp_file.unlink()
        except pickle.PicklingError as e:
            logger.error(f"Pickling error saving metadata store: {e}")
            if temp_file.exists():
                temp_file.unlink()
        except Exception as e:
            logger.error(f"Unexpected error saving metadata store: {e}", exc_info=True)
            if temp_file.exists():
                temp_file.unlink()
    
    def load(self):
        """Load metadata store from disk"""
        store_file = self.store_path / 'metadata.pkl'
        if store_file.exists():
            with open(str(store_file), 'rb') as f:
                data = pickle.load(f)
                self.columns = defaultdict(list, data['columns'])
                self.import_index = defaultdict(set, data['indices']['import'])
                self.export_index = defaultdict(set, data['indices']['export'])
                self.function_index = defaultdict(set, data['indices']['function'])
                self.class_index = defaultdict(set, data['indices']['class'])


# ============================================================================
# CODEBASE MAP GENERATOR
# ============================================================================

class CodebaseMapGenerator:
    """Generate comprehensive codebase structure and relationships"""
    
    def __init__(self):
        self.file_tree = {}
        self.dependency_graph = {}
        self.statistics = {}
    
    def generate_directory_tree(self, file_paths: List[Path], base_path: Path) -> str:
        """Generate a visual directory tree representation"""
        tree_dict = {}
        
        # Build tree structure
        for file_path in sorted(file_paths):
            try:
                relative_path = file_path.relative_to(base_path)
                parts = relative_path.parts
                current = tree_dict
                
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Add file
                current[parts[-1]] = None
            except ValueError:
                logger.warning(f"File {file_path} is not relative to {base_path}")
                continue
        
        # Convert to string representation
        def build_tree_string(node, prefix="", is_last=True):
            lines = []
            items = sorted(node.items()) if node else []
            
            for i, (name, subtree) in enumerate(items):
                is_last_item = i == len(items) - 1
                
                # Add current item
                connector = "└── " if is_last_item else "├── "
                lines.append(prefix + connector + name)
                
                # Add children
                if subtree is not None:  # It's a directory
                    extension = "    " if is_last_item else "│   "
                    lines.extend(build_tree_string(subtree, prefix + extension, is_last_item))
            
            return lines
        
        tree_lines = [str(base_path.name) + "/"]
        tree_lines.extend(build_tree_string(tree_dict))
        return "\n".join(tree_lines)
    
    def generate_dependency_graph(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Generate import/export relationships between files"""
        graph = {
            "nodes": [],
            "edges": [],
            "clusters": {}
        }
        
        # Create nodes
        for metadata in metadata_list:
            node = {
                "id": metadata.path,
                "label": Path(metadata.path).name,
                "type": metadata.file_type,
                "size": metadata.size,
                "complexity": metadata.complexity_score
            }
            graph["nodes"].append(node)
        
        # Create edges based on imports
        path_to_metadata = {m.path: m for m in metadata_list}
        
        for metadata in metadata_list:
            for dep in metadata.internal_dependencies:
                # Find matching file
                for other_path, other_metadata in path_to_metadata.items():
                    if dep in other_path or Path(other_path).stem == dep:
                        edge = {
                            "source": metadata.path,
                            "target": other_path,
                            "type": "import"
                        }
                        graph["edges"].append(edge)
        
        # Identify clusters (modules)
        modules = {}
        for metadata in metadata_list:
            module_parts = metadata.module_path.split('.')[:-1]
            if module_parts:
                module = '.'.join(module_parts)
                if module not in modules:
                    modules[module] = []
                modules[module].append(metadata.path)
        
        graph["clusters"] = modules
        return graph
    
    def generate_codebase_statistics(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Generate comprehensive statistics about the codebase"""
        stats = {
            "total_files": len(metadata_list),
            "total_size": sum(m.size for m in metadata_list),
            "total_lines": sum(m.token_count for m in metadata_list),  # Approximate
            "languages": {},
            "file_types": {},
            "complexity": {
                "average": 0,
                "max": 0,
                "distribution": {}
            },
            "dependencies": {
                "internal": set(),
                "external": set()
            },
            "top_imports": {},
            "largest_files": [],
            "most_complex_files": []
        }
        
        # Language distribution
        for metadata in metadata_list:
            lang = metadata.language
            stats["languages"][lang] = stats["languages"].get(lang, 0) + 1
            
            file_type = metadata.file_type
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
        
        # Complexity analysis
        complexities = [m.complexity_score for m in metadata_list if m.complexity_score > 0]
        if complexities:
            stats["complexity"]["average"] = sum(complexities) / len(complexities)
            stats["complexity"]["max"] = max(complexities)
            
            # Distribution
            ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, float('inf'))]
            for low, high in ranges:
                key = f"{low}-{high}" if high != float('inf') else f"{low}+"
                count = sum(1 for c in complexities if low <= c < high)
                stats["complexity"]["distribution"][key] = count
        
        # Dependencies
        for metadata in metadata_list:
            stats["dependencies"]["internal"].update(metadata.internal_dependencies)
            stats["dependencies"]["external"].update(metadata.external_dependencies)
        
        stats["dependencies"]["internal"] = len(stats["dependencies"]["internal"])
        stats["dependencies"]["external"] = len(stats["dependencies"]["external"])
        
        # Top imports
        import_counts = {}
        for metadata in metadata_list:
            for imp in metadata.imports:
                import_counts[imp] = import_counts.get(imp, 0) + 1
        
        stats["top_imports"] = dict(sorted(import_counts.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:10])
        
        # Largest and most complex files
        stats["largest_files"] = sorted(metadata_list, 
                                       key=lambda m: m.size, 
                                       reverse=True)[:5]
        stats["most_complex_files"] = sorted(metadata_list, 
                                           key=lambda m: m.complexity_score, 
                                           reverse=True)[:5]
        
        return stats
    
    def create_import_export_matrix(self, metadata_list: List[FileMetadata]) -> Dict[str, Any]:
        """Create a matrix showing import/export relationships"""
        files = [m.path for m in metadata_list]
        matrix = {f: {t: False for t in files} for f in files}
        
        # Build relationships
        for metadata in metadata_list:
            for dep in metadata.internal_dependencies:
                for other in metadata_list:
                    if dep in other.path or Path(other.path).stem == dep:
                        matrix[metadata.path][other.path] = True
        
        return {
            "files": files,
            "matrix": matrix,
            "summary": {
                "total_dependencies": sum(sum(row.values()) for row in matrix.values()),
                "most_imported": [],
                "most_dependent": []
            }
        }


# ============================================================================
# STAGE 6: QUERY-BASED SELECTIVE COMPRESSION
# ============================================================================

class SelectiveCompressor:
    """Intelligent compression based on query patterns and importance"""
    
    def __init__(self, metadata_store: MetadataStore):
        self.metadata_store = metadata_store
        self.compression_strategies = {
            'full': self._compress_full,
            'structural': self._compress_structural,
            'signature': self._compress_signature,
            'summary': self._compress_summary
        }
    
    def compress_by_query(self, 
                         query: Dict[str, Any], 
                         strategy: str = 'structural') -> List[Dict[str, Any]]:
        """Compress files matching query using specified strategy"""
        
        # Find matching files
        indices = self.metadata_store.query(**query)
        metadata_list = self.metadata_store.get_metadata_by_indices(indices)
        
        # Apply compression strategy
        compress_func = self.compression_strategies.get(strategy, self._compress_structural)
        compressed_results = []
        
        for metadata in metadata_list:
            compressed = compress_func(metadata)
            compressed_results.append(compressed)
        
        return compressed_results
    
    def _compress_full(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Full content with minimal compression"""
        return {
            'path': metadata['path'],
            'type': 'full',
            'content': self._load_and_minify(metadata['path']),
            'metadata': metadata
        }
    
    def _compress_structural(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Preserve structure, compress implementation"""
        content = self._load_file(metadata['path'])
        
        # Extract and compress structure
        structure = {
            'imports': metadata['imports'],
            'exports': metadata['exports'],
            'functions': self._extract_function_signatures(content, metadata),
            'classes': self._extract_class_signatures(content, metadata),
            'constants': self._extract_constants(content)
        }
        
        return {
            'path': metadata['path'],
            'type': 'structural',
            'structure': structure,
            'metadata': metadata
        }
    
    def _compress_signature(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Only function/class signatures"""
        content = self._load_file(metadata['path'])
        
        signatures = {
            'functions': self._extract_function_signatures(content, metadata),
            'classes': self._extract_class_signatures(content, metadata)
        }
        
        return {
            'path': metadata['path'],
            'type': 'signature',
            'signatures': signatures,
            'metadata': metadata
        }
    
    def _compress_summary(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """High-level summary only"""
        return {
            'path': metadata['path'],
            'type': 'summary',
            'summary': {
                'language': metadata['language'],
                'size': metadata['size'],
                'complexity': metadata.get('complexity_score', metadata.get('complexity', 0)),
                'imports_count': len(metadata['imports']),
                'exports_count': len(metadata['exports']),
                'functions_count': len(metadata['functions']),
                'classes_count': len(metadata['classes'])
            }
        }
    
    def _load_file(self, path: str) -> str:
        """Load file content safely"""
        try:
            # Validate path
            file_path = Path(os.path.abspath(path))
            # Optional: Add path validation here if needed
            
            return file_path.read_text(encoding='utf-8', errors='ignore')
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return ""
        except PermissionError:
            logger.error(f"Permission denied reading file: {path}")
            return ""
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error loading file {path}: {e}")
            return ""
        except MemoryError as e:
            logger.error(f"Memory error loading file {path}: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error loading file {path}: {e}", exc_info=True)
            return ""
    
    def _load_and_minify(self, path: str) -> str:
        """Load and minify file content"""
        content = self._load_file(path)
        
        # Remove comments and extra whitespace
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                lines.append(line)
        
        return '\n'.join(lines)
    
    def _extract_function_signatures(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract function signatures based on language"""
        signatures = []
        
        if metadata['language'] == 'python':
            # Extract Python function signatures
            pattern = r'def\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^:]+)?:'
            for match in re.finditer(pattern, content):
                signatures.append(match.group(0).rstrip(':'))
        
        elif metadata['language'] == 'javascript':
            # Extract JavaScript function signatures
            patterns = [
                r'function\s+(\w+)\s*\([^)]*\)',
                r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=]+)\s*=>'
            ]
            for pattern in patterns:
                signatures.extend(re.findall(pattern, content))
        
        return signatures
    
    def _extract_class_signatures(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract class signatures based on language"""
        signatures = []
        
        if metadata['language'] == 'python':
            pattern = r'class\s+(\w+)(?:\([^)]*\))?:'
            for match in re.finditer(pattern, content):
                signatures.append(match.group(0).rstrip(':'))
        
        elif metadata['language'] == 'javascript':
            pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
            signatures.extend(re.findall(pattern, content))
        
        return signatures
    
    def _extract_constants(self, content: str) -> List[str]:
        """Extract constant definitions"""
        constants = []
        
        # Common constant patterns
        patterns = [
            r'(?:const|final|static\s+final)\s+\w+\s*=',
            r'[A-Z_]+\s*=\s*[\'"\d]'  # UPPER_CASE constants
        ]
        
        for pattern in patterns:
            constants.extend(re.findall(pattern, content))
        
        return constants[:10]  # Limit to top 10


# ============================================================================
# STAGE 7: OUTPUT FORMATTING AND CHUNKING
# ============================================================================

class OutputFormatter:
    """Format compressed output for LLM consumption"""
    
    def __init__(self, max_context_size: int = 128000):
        self.max_context_size = max_context_size
        self.format_templates = {
            'markdown': self._format_markdown,
            'json': self._format_json,
            'xml': self._format_xml,
            'custom': self._format_custom
        }
    
    def format_output(self, 
                     compressed_data: List[Dict[str, Any]], 
                     format_type: str = 'markdown',
                     chunk_strategy: str = 'semantic',
                     codebase_map: Optional[Dict[str, Any]] = None) -> List[str]:
        """Format compressed data into LLM-ready chunks"""
        
        # Apply formatting
        format_func = self.format_templates.get(format_type, self._format_markdown)
        formatted_items = []
        
        # Add codebase map as first item if provided
        if codebase_map and format_type == 'markdown':
            formatted_items.append(self._format_codebase_map(codebase_map))
        
        # Format each compressed item
        formatted_items.extend([format_func(item) for item in compressed_data])
        
        # Apply chunking strategy
        if chunk_strategy == 'semantic':
            return self._semantic_chunking(formatted_items, compressed_data)
        elif chunk_strategy == 'size':
            return self._size_based_chunking(formatted_items)
        else:
            return self._balanced_chunking(formatted_items, compressed_data)
    
    def _format_codebase_map(self, codebase_map: Dict[str, Any]) -> str:
        """Format codebase map information as markdown"""
        output = []
        
        # Add title
        output.append("# Codebase Analysis\n")
        
        # Add statistics if available
        if 'statistics' in codebase_map:
            stats = codebase_map['statistics']
            output.append("## 📊 Overview\n")
            output.append(f"- **Total Files**: {stats['total_files']}")
            output.append(f"- **Total Size**: {stats['total_size']:,} bytes")
            output.append(f"- **Total Lines**: {stats['total_lines']:,}")
            
            # Language distribution
            if stats['languages']:
                output.append(f"- **Languages**: {', '.join(f'{lang} ({count})' for lang, count in stats['languages'].items())}")
            
            # Complexity
            if stats['complexity']['average'] > 0:
                output.append(f"- **Average Complexity**: {stats['complexity']['average']:.1f}")
            
            output.append("")
        
        # Add directory tree
        if 'directory_tree' in codebase_map:
            output.append("## 🗺️ Project Structure\n")
            output.append("```")
            output.append(codebase_map['directory_tree'])
            output.append("```\n")
        
        # Add dependency graph summary
        if 'dependency_graph' in codebase_map:
            graph = codebase_map['dependency_graph']
            output.append("## 🔗 Dependency Graph\n")
            
            # Show import relationships
            if graph['edges']:
                output.append("### Import Relationships")
                # Group by source
                imports_by_source = {}
                for edge in graph['edges']:
                    source = edge['source']
                    target = edge['target']
                    if source not in imports_by_source:
                        imports_by_source[source] = []
                    imports_by_source[source].append(target)
                
                # Show top importers
                for source, targets in sorted(imports_by_source.items())[:10]:
                    output.append(f"- {Path(source).name} → {', '.join(Path(t).name for t in targets[:3])}")
                    if len(targets) > 3:
                        output.append(f"  (and {len(targets) - 3} more)")
            
            output.append("")
        
        # Add external dependencies
        if 'statistics' in codebase_map and codebase_map['statistics']['dependencies']['external'] > 0:
            output.append("## 📦 External Dependencies\n")
            top_imports = codebase_map['statistics']['top_imports']
            if top_imports:
                output.append("### Most Used Imports")
                for imp, count in list(top_imports.items())[:10]:
                    output.append(f"- {imp} ({count} files)")
            output.append("")
        
        # Add file organization
        output.append("## 📁 File Organization\n")
        
        return '\n'.join(output)
    
    def _format_markdown(self, item: Dict[str, Any]) -> str:
        """Format as markdown"""
        output = [f"## {item['path']}\n"]
        
        if item['type'] == 'full':
            output.append("```")
            output.append(item['content'])
            output.append("```\n")
        
        elif item['type'] == 'structural':
            structure = item['structure']
            
            if structure['imports']:
                output.append("### Imports")
                for imp in structure['imports']:
                    output.append(f"- {imp}")
                output.append("")
            
            if structure['functions']:
                output.append("### Functions")
                for func in structure['functions']:
                    output.append(f"- {func}")
                output.append("")
            
            if structure['classes']:
                output.append("### Classes")
                for cls in structure['classes']:
                    output.append(f"- {cls}")
                output.append("")
        
        elif item['type'] == 'signature':
            output.append("### Signatures")
            for sig in item['signatures'].get('functions', []):
                output.append(f"- {sig}")
            for sig in item['signatures'].get('classes', []):
                output.append(f"- {sig}")
            output.append("")
        
        elif item['type'] == 'summary':
            summary = item['summary']
            output.append(f"- Language: {summary['language']}")
            output.append(f"- Size: {summary['size']} bytes")
            output.append(f"- Complexity: {summary['complexity']:.2f}")
            output.append(f"- Functions: {summary['functions_count']}")
            output.append(f"- Classes: {summary['classes_count']}")
            output.append("")
        
        return '\n'.join(output)
    
    def _format_json(self, item: Dict[str, Any]) -> str:
        """Format as JSON"""
        # Convert sets to lists for JSON serialization
        def convert_sets(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_sets(v) for v in obj]
            return obj
        
        return json.dumps(convert_sets(item), indent=2)
    
    def _format_xml(self, item: Dict[str, Any]) -> str:
        """Format as XML"""
        def dict_to_xml(tag: str, d: Any) -> str:
            if isinstance(d, dict):
                xml = f"<{tag}>"
                for key, value in d.items():
                    xml += dict_to_xml(key, value)
                xml += f"</{tag}>"
                return xml
            elif isinstance(d, list):
                xml = ""
                for item in d:
                    xml += dict_to_xml(tag, item)
                return xml
            else:
                return f"<{tag}>{d}</{tag}>"
        
        return dict_to_xml('file', item)
    
    def _format_custom(self, item: Dict[str, Any]) -> str:
        """Custom format optimized for LLM understanding"""
        output = []
        
        # Header with metadata
        output.append(f"[FILE: {item['path']}]")
        output.append(f"[TYPE: {item['type'].upper()}]")
        
        if 'metadata' in item:
            meta = item['metadata']
            output.append(f"[LANG: {meta['language']} | SIZE: {meta['size']} | COMPLEXITY: {meta['complexity']:.1f}]")
        
        output.append("")
        
        # Content based on type
        if item['type'] == 'structural':
            structure = item['structure']
            
            if structure['imports']:
                output.append("[IMPORTS]")
                output.extend(structure['imports'])
                output.append("")
            
            if structure['functions']:
                output.append("[FUNCTIONS]")
                output.extend(structure['functions'])
                output.append("")
            
            if structure['classes']:
                output.append("[CLASSES]")
                output.extend(structure['classes'])
                output.append("")
        
        return '\n'.join(output)
    
    def _semantic_chunking(self, 
                          formatted_items: List[str], 
                          compressed_data: List[Dict[str, Any]]) -> List[str]:
        """Chunk based on semantic relationships"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Group by import/export relationships
        dependency_graph = self._build_dependency_graph(compressed_data)
        clusters = self._find_clusters(dependency_graph)
        
        # If no clusters found, create individual clusters for each item
        if not clusters:
            clusters = [[i] for i in range(len(formatted_items))]
        
        for cluster in clusters:
            cluster_items = [formatted_items[i] for i in cluster if i < len(formatted_items)]
            cluster_size = sum(len(item) for item in cluster_items)
            
            if current_size + cluster_size > self.max_context_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.extend(cluster_items)
            current_size += cluster_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _size_based_chunking(self, formatted_items: List[str]) -> List[str]:
        """Simple size-based chunking"""
        chunks = []
        current_chunk = []
        current_size = 0
        
        for item in formatted_items:
            item_size = len(item)
            
            if current_size + item_size > self.max_context_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            current_chunk.append(item)
            current_size += item_size
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _balanced_chunking(self, 
                          formatted_items: List[str], 
                          compressed_data: List[Dict[str, Any]]) -> List[str]:
        """Balance chunk sizes while respecting relationships"""
        # Sort by complexity for better distribution
        items_with_complexity = [
            (i, item, compressed_data[i].get('metadata', {}).get('complexity', 0))
            for i, item in enumerate(formatted_items)
        ]
        items_with_complexity.sort(key=lambda x: x[2], reverse=True)
        
        # Distribute into balanced chunks
        num_chunks = max(1, sum(len(item[1]) for item in items_with_complexity) // self.max_context_size + 1)
        chunks = [[] for _ in range(num_chunks)]
        chunk_sizes = [0] * num_chunks
        
        for idx, item, _ in items_with_complexity:
            # Add to smallest chunk
            min_chunk_idx = chunk_sizes.index(min(chunk_sizes))
            chunks[min_chunk_idx].append(item)
            chunk_sizes[min_chunk_idx] += len(item)
        
        return ['\n'.join(chunk) for chunk in chunks if chunk]
    
    def _build_dependency_graph(self, compressed_data: List[Dict[str, Any]]) -> Dict[int, Set[int]]:
        """Build dependency graph from import/export relationships"""
        graph = defaultdict(set)
        
        # Build export index
        export_to_file = {}
        for i, item in enumerate(compressed_data):
            if 'metadata' in item:
                for export in item['metadata'].get('exports', []):
                    export_to_file[export] = i
        
        # Build import relationships
        for i, item in enumerate(compressed_data):
            if 'metadata' in item:
                for import_name in item['metadata'].get('imports', []):
                    if import_name in export_to_file:
                        graph[i].add(export_to_file[import_name])
                        graph[export_to_file[import_name]].add(i)
        
        return graph
    
    def _find_clusters(self, graph: Dict[int, Set[int]]) -> List[List[int]]:
        """Find connected components in dependency graph"""
        visited = set()
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            if node in visited:
                return
            visited.add(node)
            cluster.append(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor, cluster)
        
        # Process nodes that are in the graph
        for node in graph:
            if node not in visited:
                cluster = []
                dfs(node, cluster)
                if cluster:  # Only add non-empty clusters
                    clusters.append(cluster)
        
        return clusters


# ============================================================================
# STAGE 8: PERFORMANCE OPTIMIZATION
# ============================================================================

class PerformanceOptimizer:
    """Optimize pipeline performance through profiling and tuning"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.profiling_enabled = False
    
    def profile_stage(self, stage_name: str):
        """Decorator to profile pipeline stages"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if self.profiling_enabled:
                    start_time = time.time()
                    start_memory = self._get_memory_usage()
                    
                    result = await func(*args, **kwargs)
                    
                    end_time = time.time()
                    end_memory = self._get_memory_usage()
                    
                    self.metrics[stage_name].append({
                        'duration': end_time - start_time,
                        'memory_delta': end_memory - start_memory,
                        'timestamp': datetime.now()
                    })
                    
                    return result
                else:
                    return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def optimize_batch_size(self, 
                           file_sizes: List[int], 
                           available_memory: int) -> int:
        """Calculate optimal batch size based on file sizes and memory"""
        avg_file_size = sum(file_sizes) / len(file_sizes) if file_sizes else 1024 * 1024
        
        # Reserve 50% memory for processing overhead
        usable_memory = available_memory * 0.5
        
        # Calculate batch size with safety margin
        batch_size = max(1, int(usable_memory / (avg_file_size * 2)))
        
        # Cap at reasonable limits
        return min(batch_size, 1000)
    
    def optimize_compression_level(self, 
                                  content_type: str, 
                                  size: int) -> int:
        """Determine optimal compression level"""
        if content_type == 'already_compressed':  # e.g., images, videos
            return 0
        elif size < 1024:  # Small files
            return 1
        elif size < 1024 * 1024:  # Medium files
            return 6
        else:  # Large files
            return 9
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        report = {}
        
        for stage, metrics in self.metrics.items():
            if metrics:
                durations = [m['duration'] for m in metrics]
                memory_deltas = [m['memory_delta'] for m in metrics]
                
                report[stage] = {
                    'avg_duration': sum(durations) / len(durations),
                    'max_duration': max(durations),
                    'min_duration': min(durations),
                    'avg_memory_delta': sum(memory_deltas) / len(memory_deltas),
                    'total_executions': len(metrics)
                }
        
        return report
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class CodebaseCompressionPipeline:
    """Main orchestrator for the compression pipeline with resilience"""
    
    def __init__(self, 
                 cache_dir: Path,
                 output_dir: Path,
                 num_workers: Optional[int] = None,
                 enable_resilience: bool = True,
                 security_config: Optional[SecurityConfig] = None,
                 redis_config: Optional['RedisCacheConfig'] = None,
                 enable_adaptive_compression: bool = True):
        # Initialize security first
        self.security_config = security_config or SecurityConfig()
        self.security_validator = SecurityValidator(self.security_config)
        
        # Validate paths with security
        cache_dir = Path(os.path.abspath(cache_dir))
        output_dir = Path(os.path.abspath(output_dir))
        
        # Validate cache and output directories
        validated_cache = self.security_validator.path_validator.validate_path(cache_dir)
        validated_output = self.security_validator.path_validator.validate_path(output_dir)
        
        if not validated_cache or not validated_output:
            raise ValueError("Invalid cache or output directory paths")
            
        self.cache_dir = validated_cache
        self.output_dir = validated_output
        self.enable_resilience = enable_resilience
        
        # Create directories safely with retry
        @with_retry(RetryConfig(max_attempts=3, retryable_exceptions=(OSError, IOError)))
        def create_directories():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        try:
            create_directories()
        except PermissionError as e:
            logger.error(f"Permission denied creating directories: {e}")
            raise
        except OSError as e:
            logger.error(f"OS error creating directories: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating directories after retries: {e}", exc_info=True)
            raise
        
        # Initialize resilience components
        if enable_resilience:
            self.circuit_breaker = CircuitBreaker(
                "pipeline_main", 
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60)
            )
            self.bulkhead = BulkheadExecutor("pipeline_bulkhead", max_concurrent=20)
            self.health_checker = HealthChecker(check_interval=30.0)
            self.fallback_handler = FallbackHandler()
        
        # Initialize components with error handling
        try:
            self.processor = ParallelProcessor(num_workers)
            
            # Initialize cache (hybrid if Redis is configured)
            if redis_config and REDIS_AVAILABLE:
                try:
                    self.cache = HybridCache(cache_dir, redis_config)
                    logger.info("Using hybrid cache with Redis support")
                except ImportError as e:
                    logger.warning(f"Redis module not available: {e}")
                    logger.info("Falling back to local cache only")
                except ConnectionError as e:
                    logger.warning(f"Failed to connect to Redis: {e}")
                    logger.info("Falling back to local cache only")
                except Exception as e:
                    logger.warning(f"Unexpected error initializing Redis cache: {e}", exc_info=True)
                    logger.info("Falling back to local cache only")
                    self.cache = IncrementalCache(cache_dir)
            else:
                self.cache = IncrementalCache(cache_dir)
                
            self.metadata_store = MetadataStore(cache_dir / 'metadata')
            self.compressor = StreamingCompressor(
                adaptive_compression=enable_adaptive_compression,
                cache_dir=cache_dir / 'adaptive_compression'
            )
            self.selective_compressor = SelectiveCompressor(self.metadata_store)
            self.formatter = OutputFormatter()
            self.optimizer = PerformanceOptimizer()
        except ImportError as e:
            logger.error(f"Import error initializing pipeline components: {e}")
            # Clean up partial initialization
        except AttributeError as e:
            logger.error(f"Attribute error initializing pipeline components: {e}")
            # Clean up partial initialization
        except Exception as e:
            logger.error(f"Unexpected error initializing pipeline components: {e}", exc_info=True)
            # Clean up partial initialization
            self._cleanup_partial_init()
            raise
        
        # Language parsers with fallback
        self.parsers = {
            'python': PythonParser(),
            'javascript': EnhancedJavaScriptParser()
        }
        
        # Add TypeScript parser if available
        if TYPESCRIPT_PARSER_AVAILABLE:
            try:
                self.parsers['typescript'] = TypeScriptParser()
                logger.info("TypeScript parser initialized successfully")
            except ImportError as e:
                logger.warning(f"TypeScript parser module not available: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error initializing TypeScript parser: {e}", exc_info=True)
        else:
            # Fall back to JavaScript parser for TypeScript files
            self.parsers['typescript'] = EnhancedJavaScriptParser()
            
        # Add Go parser if available
        if GO_PARSER_AVAILABLE:
            try:
                self.parsers['go'] = GoParser()
                logger.info("Go parser initialized successfully")
            except ImportError as e:
                logger.warning(f"Go parser module not available: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error initializing Go parser: {e}", exc_info=True)
                
        # Add Rust parser if available
        if RUST_PARSER_AVAILABLE:
            try:
                self.parsers['rust'] = RustParser()
                logger.info("Rust parser initialized successfully")
            except ImportError as e:
                logger.warning(f"Rust parser module not available: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error initializing Rust parser: {e}", exc_info=True)
        
        # Register health checks if resilience enabled
        if enable_resilience:
            self._register_health_checks()
            
    def _cleanup_partial_init(self):
        """Clean up partially initialized components"""
        components = ['processor', 'cache', 'metadata_store', 'health_checker']
        for comp in components:
            if hasattr(self, comp):
                try:
                    obj = getattr(self, comp)
                    if hasattr(obj, 'shutdown'):
                        obj.shutdown()
                    elif hasattr(obj, 'close'):
                        obj.close()
                except AttributeError as e:
                    logger.warning(f"Component {comp} not found during cleanup: {e}")
                except (OSError, IOError) as e:
                    logger.warning(f"I/O error cleaning up {comp}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error cleaning up {comp}: {e}", exc_info=True)
                    
    def _register_health_checks(self):
        """Register component health checks"""
        # Cache health check
        async def check_cache_health():
            try:
                # Test cache operations
                test_key = "__health_check__"
                self.cache.get_cached_metadata(test_key, "test_hash")
                return True
            except (FileNotFoundError, PermissionError, OSError):
                return False
            except Exception:
                logger.debug("Unexpected error in health check", exc_info=True)
                return False
                
        # Metadata store health check
        async def check_metadata_health():
            try:
                # Test metadata query
                self.metadata_store.query({})
                return True
            except (FileNotFoundError, PermissionError, OSError):
                return False
            except Exception:
                logger.debug("Unexpected error in health check", exc_info=True)
                return False
                
        self.health_checker.register_component("cache", check_cache_health)
        self.health_checker.register_component("metadata_store", check_metadata_health)
        
    async def start_health_monitoring(self):
        """Start health monitoring if enabled"""
        if self.enable_resilience:
            await self.health_checker.start()
            
    async def stop_health_monitoring(self):
        """Stop health monitoring if enabled"""
        if self.enable_resilience:
            await self.health_checker.stop()
            
    async def process_codebase_resilient(self,
                                       codebase_path: Path,
                                       output_format: str = 'markdown',
                                       compression_strategy: str = 'structural',
                                       query_filter: Optional[Dict[str, Any]] = None,
                                       ignore_patterns: Optional[List[str]] = None) -> List[Path]:
        """Process codebase with full resilience patterns"""
        if not self.enable_resilience:
            # Fall back to standard processing
            return await self.process_codebase(
                codebase_path, output_format, compression_strategy, 
                query_filter, ignore_patterns
            )
            
        # Start health monitoring
        await self.start_health_monitoring()
        
        # Configure retry for the main operation
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=2.0,
            retryable_exceptions=(RetryableError, ConnectionError, IOError)
        )
        
        @with_retry(retry_config)
        async def process_with_retry():
            # Use circuit breaker for main processing
            return await self.circuit_breaker.call(
                self.process_codebase,
                codebase_path, output_format, compression_strategy,
                query_filter, ignore_patterns
            )
            
        # Setup fallback options
        async def minimal_fallback(*args, **kwargs):
            """Minimal processing fallback"""
            logger.warning("Using minimal fallback processing")
            # Just discover files without full processing
            files = list(codebase_path.rglob('*.py'))[:100]  # Limit to 100 files
            output_file = self.output_dir / f"minimal_{int(time.time())}.txt"
            with open(output_file, 'w') as f:
                f.write(f"Minimal fallback output\n")
                f.write(f"Found {len(files)} Python files\n")
                for file in files:
                    f.write(f"- {file.relative_to(codebase_path)}\n")
            return [output_file]
            
        self.fallback_handler.add_fallback(minimal_fallback)
        
        try:
            # Execute with all resilience patterns
            result = await self.fallback_handler.execute_with_fallback(
                process_with_retry
            )
            
            # Stop health monitoring
            await self.stop_health_monitoring()
            
            return result
            
        except asyncio.CancelledError:
            logger.info("Pipeline processing cancelled")
            await self.stop_health_monitoring()
            raise
        except Exception as e:
            logger.error(f"All resilience measures failed: {e}", exc_info=True)
            await self.stop_health_monitoring()
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup"""
        if hasattr(self, 'processor'):
            self.processor.shutdown()
        if self.enable_resilience and hasattr(self, 'health_checker'):
            # Stop health monitoring synchronously
            try:
                asyncio.create_task(self.stop_health_monitoring())
            except RuntimeError as e:
                # No event loop running - this is expected during shutdown
                logger.debug(f"Could not create task during shutdown: {e}")
        return False
    
    
    def cleanup(self):
        """Clean up all resources"""
        if hasattr(self, 'processor'):
            self.processor.shutdown()
        if hasattr(self, 'cache'):
            self.cache._save_index()  # Ensure cache index is saved
    
    async def _cache_metadata_batch(self, metadata_list: List[FileMetadata]):
        """Cache multiple metadata entries efficiently in async context"""
        # Process metadata in batches to minimize lock contention
        for metadata in metadata_list:
            # Cache synchronously but without blocking the event loop
            await asyncio.get_event_loop().run_in_executor(
                None,  # Use default executor
                self.cache.cache_metadata,
                metadata
            )
            # Add to metadata store synchronously (it's fast)
            self.metadata_store.add_metadata(metadata)
    
    def _populate_metadata_fields(self, metadata: FileMetadata, base_path: Path):
        """Populate additional metadata fields for enhanced mapping"""
        file_path = Path(metadata.path)
        
        # Calculate relative path
        try:
            metadata.relative_path = str(file_path.relative_to(base_path))
        except ValueError:
            metadata.relative_path = metadata.path
        
        # Generate module path for Python files
        if metadata.language == 'python':
            # Convert file path to module notation
            relative_path = Path(metadata.relative_path)
            if relative_path.suffix == '.py':
                # Remove .py extension and convert / to .
                module_parts = list(relative_path.with_suffix('').parts)
                # Remove __init__ from module path
                if module_parts and module_parts[-1] == '__init__':
                    module_parts = module_parts[:-1]
                metadata.module_path = '.'.join(module_parts)
            else:
                metadata.module_path = str(relative_path)
        else:
            metadata.module_path = metadata.relative_path
        
        # Determine file type
        file_name = file_path.name.lower()
        if file_name in ['setup.py', 'setup.cfg', 'pyproject.toml', 'requirements.txt', 
                         'package.json', 'cargo.toml', 'go.mod', 'pom.xml']:
            metadata.file_type = 'config'
        elif file_name in ['readme.md', 'readme.txt', 'readme.rst', 'readme']:
            metadata.file_type = 'doc'
        elif 'test' in file_name or file_path.parent.name == 'tests':
            metadata.file_type = 'test'
        elif file_name == '__init__.py':
            metadata.file_type = 'init'
        else:
            metadata.file_type = 'source'
        
        # Classify dependencies as internal or external
        if metadata.imports:
            for imp in metadata.imports:
                # Simple heuristic: if import starts with project name or is relative, it's internal
                if imp.startswith('.') or imp.startswith(base_path.name):
                    metadata.internal_dependencies.append(imp)
                else:
                    # Check if it's a standard library module
                    if imp.split('.')[0] not in ['os', 'sys', 'io', 'time', 'datetime', 'json', 
                                                 'csv', 'math', 're', 'collections', 'itertools',
                                                 'functools', 'typing', 'pathlib', 'logging']:
                        metadata.external_dependencies.append(imp)
                    else:
                        # Standard library imports could be tracked separately if needed
                        pass
    
    def _validate_path(self, base_path: Path, file_path: Path) -> bool:
        """Validate that file_path is secure and within base_path"""
        try:
            # Use security validator for comprehensive validation
            validation_result = self.security_validator.validate_input(
                file_path, 
                base_path, 
                scan_content=False  # Don't scan content yet
            )
            
            if not validation_result['valid']:
                logger.warning(f"Security validation failed for {file_path}: {validation_result['errors']}")
                return False
                
            # Log any warnings
            if validation_result['warnings']:
                logger.info(f"Security warnings for {file_path}: {validation_result['warnings']}")
                
            return True
            
        except ValueError as e:
            logger.error(f"Invalid path value for {file_path}: {e}")
            return False
        except OSError as e:
            logger.error(f"OS error validating path {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected path validation error for {file_path}: {e}", exc_info=True)
            return False
    
    async def process_codebase(self, 
                             codebase_path: Path,
                             output_format: str = 'markdown',
                             compression_strategy: str = 'structural',
                             query_filter: Optional[Dict[str, Any]] = None,
                             ignore_patterns: Optional[List[str]] = None) -> List[Path]:
        """Process entire codebase through the pipeline"""
        
        # Check rate limiting
        if not self.security_validator.rate_limiter.allow_request():
            logger.error("Rate limit exceeded")
            raise ValueError("Rate limit exceeded. Please try again later.")
        
        # Monitor resources at start
        initial_resources = self.security_validator.check_resource_usage()
        logger.info(f"Initial resource usage: CPU={initial_resources.get('cpu_percent', 0):.1f}%, "
                   f"Memory={initial_resources.get('memory_percent', 0):.1f}%")
        
        # Stage 1: Discovery
        logger.info("Stage 1: Discovering files...")
        
        # Validate base path with security
        try:
            # Security validation for codebase path
            validated_path = self.security_validator.path_validator.validate_path(codebase_path)
            if not validated_path:
                raise ValueError(f"Security validation failed for codebase path: {codebase_path}")
                
            codebase_path = validated_path
            
            if not codebase_path.exists() or not codebase_path.is_dir():
                raise ValueError(f"Invalid codebase path: {codebase_path}")
                
            # Check if path is allowed
            if not any(codebase_path.is_relative_to(allowed) or allowed.is_relative_to(codebase_path) 
                      for allowed in self.security_config.allowed_base_paths):
                raise ValueError(f"Codebase path not in allowed directories: {codebase_path}")
                
        except ValueError as e:
            logger.error(f"Invalid codebase path value: {e}")
            return []
        except OSError as e:
            logger.error(f"OS error with codebase path: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error with codebase path: {e}", exc_info=True)
            return []
        
        # Secure file discovery - filter hidden and sensitive files
        file_paths = []
        for path in codebase_path.rglob('*'):
            # Skip hidden files and directories unless explicitly allowed
            if any(part.startswith('.') for part in path.parts):
                # Skip sensitive hidden directories
                if any(part in {'.git', '.env', '.aws', '.ssh', '.gnupg'} for part in path.parts):
                    logger.debug(f"Skipping sensitive directory: {path}")
                    continue
                    
            # Only include regular files
            if path.is_file():
                file_paths.append(path)
        
        # Apply security limits
        if len(file_paths) > self.security_config.max_total_files:
            logger.error(f"Too many files found: {len(file_paths)} > {self.security_config.max_total_files}")
            return []
        
        # Filter files by extension and validate paths
        supported_extensions = list(self.security_config.allowed_extensions)
        validated_paths = []
        total_size = 0
        
        for f in file_paths:
            if f.is_file() and f.suffix.lower() in supported_extensions:
                if self._validate_path(codebase_path, f):
                    # Check file size
                    try:
                        file_size = f.stat().st_size
                        if file_size > self.security_config.max_file_size:
                            logger.warning(f"Skipping file too large: {f} ({file_size} bytes)")
                            continue
                        total_size += file_size
                        validated_paths.append(f)
                    except OSError as e:
                        logger.warning(f"Could not stat file {f}: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error getting file stats for {f}: {e}", exc_info=True)
                else:
                    logger.warning(f"Skipping invalid path: {f}")
                    
        # Check total size limit (e.g., 1GB)
        max_total_size = 1024 * 1024 * 1024  # 1GB
        if total_size > max_total_size:
            logger.error(f"Total file size too large: {total_size} > {max_total_size}")
            return []
            
        file_paths = validated_paths
        
        logger.info(f"Found {len(file_paths)} source files with supported extensions")
        
        # Apply ignore patterns if provided
        if ignore_patterns:
            filtered_paths = []
            for file_path in file_paths:
                should_ignore = False
                path_str = str(file_path.relative_to(codebase_path))
                
                for pattern in ignore_patterns:
                    # Check if any parent directory matches the pattern
                    if '/' not in pattern and not pattern.startswith('*'):
                        # This is a directory pattern
                        for part in file_path.parts[:-1]:  # Exclude the filename itself
                            if part == pattern:
                                should_ignore = True
                                break
                    # Check file patterns
                    elif pattern.startswith('*.') and file_path.suffix == pattern[1:]:
                        should_ignore = True
                        break
                    # Check if pattern appears in the path
                    elif pattern in str(file_path):
                        should_ignore = True
                        break
                    
                    if should_ignore:
                        break
                
                if not should_ignore:
                    filtered_paths.append(file_path)
            
            file_paths = filtered_paths
            
            # Show statistics about ignored files
            total_files = len(list(codebase_path.rglob('*')))
            ignored_count = total_files - len(file_paths)
            if ignored_count > 0:
                logger.info(f"Ignored {ignored_count} files based on ignore patterns")
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Stage 2: Incremental check
        logger.info("Stage 2: Checking cache for changes...")
        
        # Check resources before heavy processing
        current_resources = self.security_validator.check_resource_usage()
        if current_resources.get('memory_percent', 0) > 80:
            logger.warning(f"High memory usage: {current_resources.get('memory_percent', 0):.1f}%")
            # Consider reducing batch size
            self.batch_size = max(10, self.batch_size // 2)
            logger.info(f"Reduced batch size to {self.batch_size} due to high memory usage")
        
        if TQDM_AVAILABLE and file_paths:
            # Create progress bar for hashing files
            current_files = {}
            with tqdm(total=len(file_paths), desc="Hashing files", unit="files") as pbar:
                for f in file_paths:
                    try:
                        # Read file in chunks for memory efficiency
                        hasher = hashlib.sha256()
                        with open(f, 'rb') as file:
                            while chunk := file.read(8192):
                                hasher.update(chunk)
                        current_files[str(f)] = hasher.hexdigest()
                    except (OSError, IOError) as e:
                        logger.warning(f"I/O error reading {f}: {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error reading {f}: {e}", exc_info=True)
                    pbar.update(1)
        else:
            current_files = {}
            for f in file_paths:
                try:
                    # Read file in chunks for memory efficiency
                    hasher = hashlib.sha256()
                    with open(f, 'rb') as file:
                        while chunk := file.read(8192):
                            hasher.update(chunk)
                    current_files[str(f)] = hasher.hexdigest()
                except (OSError, IOError) as e:
                    logger.warning(f"I/O error reading {f}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error reading {f}: {e}", exc_info=True)
        added, modified, deleted = self.cache.get_changed_files(current_files)
        
        files_to_process = [Path(f) for f in added | modified]
        logger.info(f"Processing {len(files_to_process)} new/modified files")
        
        # Stage 3: Parallel processing
        logger.info("Stage 3: Processing files in parallel...")
        metadata_list = []
        
        # Skip processing if no files to process
        if not files_to_process:
            logger.info("No files to process, skipping parallel processing")
        elif TQDM_AVAILABLE:
            # Create progress bar
            progress_bar = tqdm(
                total=len(files_to_process),
                desc="Processing files",
                unit="files",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
            
            # Collect all metadata first to avoid threading issues in async context
            async for metadata in self.processor.process_files_parallel(files_to_process, self.parsers):
                # Populate new fields
                self._populate_metadata_fields(metadata, codebase_path)
                metadata_list.append(metadata)
                progress_bar.update(1)
            
            progress_bar.close()
            
            # Cache metadata after processing to avoid deadlock
            logger.info("Caching processed metadata...")
            # Use batch caching to minimize lock contention
            await self._cache_metadata_batch(metadata_list)
        else:
            # Collect all metadata first
            async for metadata in self.processor.process_files_parallel(files_to_process, self.parsers):
                # Populate new fields
                self._populate_metadata_fields(metadata, codebase_path)
                metadata_list.append(metadata)
            
            # Cache metadata after processing
            # Use batch caching to minimize lock contention
            await self._cache_metadata_batch(metadata_list)
        
        # Stage 4: Query-based selection
        logger.info("Stage 4: Applying query filters...")
        if query_filter:
            compressed_data = self.selective_compressor.compress_by_query(
                query_filter, 
                compression_strategy
            )
        else:
            # Get all cached metadata for files we didn't just process
            all_cached_metadata = []
            processed_paths = {m.path for m in metadata_list}
            
            for file_path in current_files:
                # Only get cached metadata for files we didn't just process
                if str(file_path) not in processed_paths:
                    cached_meta = self.cache.get_cached_metadata(str(file_path), current_files[file_path])
                    if cached_meta:
                        all_cached_metadata.append(cached_meta)
            
            # Combine newly processed and cached metadata (no duplicates)
            all_metadata_list = metadata_list + all_cached_metadata
            
            # Compress all files based on strategy
            compress_func = self.selective_compressor.compression_strategies.get(
                compression_strategy, 
                self.selective_compressor._compress_structural
            )
            compressed_data = []
            for metadata in all_metadata_list:
                try:
                    # Convert FileMetadata to dict if needed
                    if hasattr(metadata, '__dict__'):
                        metadata_dict = {
                            'path': metadata.path,
                            'size': metadata.size,
                            'language': metadata.language,
                            'last_modified': metadata.last_modified,
                            'content_hash': metadata.content_hash,
                            'imports': list(metadata.imports) if hasattr(metadata.imports, '__iter__') else metadata.imports,
                            'exports': list(metadata.exports) if hasattr(metadata.exports, '__iter__') else metadata.exports,
                            'functions': metadata.functions,
                            'classes': metadata.classes,
                            'dependencies': list(metadata.dependencies) if isinstance(metadata.dependencies, set) else metadata.dependencies,
                            'complexity_score': metadata.complexity_score,
                            'token_count': metadata.token_count
                        }
                    else:
                        metadata_dict = metadata
                    
                    compressed = compress_func(metadata_dict)
                    if compressed:
                        compressed_data.append(compressed)
                except AttributeError as e:
                    logger.warning(f"Invalid metadata structure for compression: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error compressing {getattr(metadata, 'path', metadata.get('path', 'unknown'))}: {e}", exc_info=True)
        
        logger.info(f"Compressed {len(compressed_data)} files")
        
        # Generate codebase map
        logger.info("Generating codebase map...")
        map_generator = CodebaseMapGenerator()
        
        # Generate directory tree
        all_file_paths = [Path(m.path) for m in all_metadata_list]
        directory_tree = map_generator.generate_directory_tree(all_file_paths, codebase_path)
        
        # Generate statistics
        statistics = map_generator.generate_codebase_statistics(all_metadata_list)
        
        # Generate dependency graph
        dependency_graph = map_generator.generate_dependency_graph(all_metadata_list)
        
        codebase_map = {
            'directory_tree': directory_tree,
            'statistics': statistics,
            'dependency_graph': dependency_graph
        }
        
        # Stage 5: Format and chunk output
        logger.info("Stage 5: Formatting output...")
        chunks = self.formatter.format_output(
            compressed_data,
            output_format,
            chunk_strategy='semantic',
            codebase_map=codebase_map
        )
        
        # Stage 6: Write output files
        logger.info("Stage 6: Writing output files...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_files = []
        
        if not chunks:
            logger.warning("No chunks generated from compressed data")
            # Debug: try to save raw compressed data
            if compressed_data:
                debug_file = self.output_dir / f"debug_compressed.json"
                with open(debug_file, 'w') as f:
                    json.dump(compressed_data[:5], f, indent=2)  # Save first 5 for debugging
                logger.debug(f"Saved sample compressed data to {debug_file}")
        
        for i, chunk in enumerate(chunks):
            output_file = self.output_dir / f"compressed_{i:04d}.{output_format}"
            output_file.write_text(chunk)
            output_files.append(output_file)
        
        logger.info(f"Created {len(output_files)} output files")
        
        # Save metadata store
        self.metadata_store.save()
        
        # Generate performance report
        if self.optimizer.profiling_enabled:
            report = self.optimizer.get_performance_report()
            report_file = self.output_dir / 'performance_report.json'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return output_files


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

async def main():
    """Example usage of the pipeline"""
    
    # Configure pipeline
    pipeline = CodebaseCompressionPipeline(
        cache_dir=Path('./cache'),
        output_dir=Path('./output'),
        num_workers=4
    )
    
    # Process codebase with specific query
    query = {
        'language': 'python',
        'min_complexity': 5.0,
        'imports': 'numpy'
    }
    
    output_files = await pipeline.process_codebase(
        codebase_path=Path('./my_project'),
        output_format='markdown',
        compression_strategy='structural',
        query_filter=query
    )
    
    logger.info(f"Pipeline completed. Output files: {output_files}")


if __name__ == "__main__":
    asyncio.run(main())