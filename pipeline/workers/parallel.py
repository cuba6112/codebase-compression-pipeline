"""
Parallel Processor
==================

Manages parallel processing of files with work stealing.
"""

import asyncio
import hashlib
import logging
import multiprocessing as mp
import time
import weakref
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator

from base_classes import FileMetadata, LanguageParser
from resilience_patterns import (
    with_retry, RetryConfig, NonRetryableError
)

logger = logging.getLogger(__name__)


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
                except Exception as e:
                    logger.warning(f"Error cleaning up {comp}: {e}")
                    
    def _register_health_checks(self):
        """Register component health checks"""
        # Cache health check
        async def check_cache_health():
            try:
                # Test cache operations
                test_key = "__health_check__"
                self.cache.get_cached_metadata(test_key, "test_hash")
                return True
            except Exception:
                return False
                
        # Metadata store health check
        async def check_metadata_health():
            try:
                # Test metadata query
                self.metadata_store.query({})
                return True
            except Exception:
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
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")
        
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
                except Exception as e:
                    logger.error(f"Error in worker: {e}")
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
                    except Exception as parse_error:
                        # Log parsing error but create minimal metadata
                        logger.error(f"Parse error for {file_path}: {parse_error}")
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
            except Exception as e:
                logger.error(f"Failed to process {file_path} after retries: {e}")
        
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