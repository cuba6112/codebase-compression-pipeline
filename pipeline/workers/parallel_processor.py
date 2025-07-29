"""
Parallel processing manager with work stealing for efficient file processing.
"""

import asyncio
import hashlib
import logging
import time
import weakref
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, AsyncIterator

from base_classes import FileMetadata, LanguageParser
from resilience_patterns import with_retry, RetryConfig, NonRetryableError

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """Manages parallel processing of files with work stealing and resilience."""
    
    def __init__(self, num_workers: Optional[int] = None):
        """Initialize parallel processor with specified number of workers."""
        self.num_workers = min(num_workers or mp.cpu_count(), 32)  # Cap at reasonable limit
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.num_workers * 2)
        
        # Track executors for cleanup
        self._executors = weakref.WeakSet()
        self._executors.add(self.executor)
        self._executors.add(self.thread_executor)
        
        logger.info(f"Initialized ParallelProcessor with {self.num_workers} workers")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.shutdown()
        return False
    
    
    def shutdown(self):
        """Shutdown all executors gracefully."""
        for executor in list(self._executors):
            try:
                executor.shutdown(wait=True)
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")
    
    async def process_files_parallel(self, 
                                   file_paths: List[Path],
                                   parser_factory: Dict[str, LanguageParser],
                                   batch_size: int = 100) -> AsyncIterator[FileMetadata]:
        """
        Process files in parallel with batching and work stealing.
        
        Args:
            file_paths: List of file paths to process
            parser_factory: Dictionary mapping language to parser instances
            batch_size: Number of files to process in each batch
            
        Yields:
            FileMetadata objects as files are processed
        """
        logger.debug(f"Starting parallel processing of {len(file_paths)} files with batch size {batch_size}")
        
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
        
        # Add sentinel values for worker termination
        for _ in range(self.num_workers):
            await work_queue.put(None)
        
        # Result queue for workers to push results
        result_queue = asyncio.Queue()
        
        # Worker coroutine
        async def worker():
            """Process batches from work queue until sentinel is received."""
            while True:
                batch = await work_queue.get()
                if batch is None:
                    break
                
                # Process batch in thread pool to avoid blocking
                try:
                    results = await asyncio.get_running_loop().run_in_executor(
                        self.thread_executor,
                        self._process_batch,
                        batch,
                        parser_factory
                    )
                    
                    # Push results to result queue
                    for result in results:
                        if result:
                            await result_queue.put(result)
                except Exception as e:
                    logger.error(f"Error in worker processing batch: {e}")
                    # Continue processing other batches
            
            # Signal worker completion
            await result_queue.put(None)
        
        # Create and start workers
        workers = [asyncio.create_task(worker()) for _ in range(self.num_workers)]
        logger.debug(f"Created {len(workers)} worker tasks")
        
        # Collect results as they become available
        completed_workers = 0
        while completed_workers < self.num_workers:
            result = await result_queue.get()
            if result is None:
                completed_workers += 1
                logger.debug(f"Worker completed, total: {completed_workers}/{self.num_workers}")
            else:
                yield result
        
        # Wait for all workers to complete
        await asyncio.gather(*workers)
        logger.debug("All workers completed")
    
    def _process_batch(self, 
                      batch: List[Path], 
                      parser_factory: Dict[str, LanguageParser]) -> List[FileMetadata]:
        """
        Process a batch of files with resilience patterns.
        
        Args:
            batch: List of file paths to process
            parser_factory: Dictionary mapping language to parser instances
            
        Returns:
            List of FileMetadata objects for successfully processed files
        """
        results = []
        
        # Configure retry for file operations
        retry_config = RetryConfig(
            max_attempts=3,
            initial_delay=0.5,
            retryable_exceptions=(IOError, OSError)
        )
        
        @with_retry(retry_config)
        def read_file_with_retry(file_path: Path) -> str:
            """Read file with retry logic for transient errors."""
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        for file_path in batch:
            try:
                # Read file with retry mechanism
                content = read_file_with_retry(file_path)
                
                # Determine language from file extension
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
                else:
                    logger.debug(f"No parser available for language: {language}")
                    
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
        """
        Detect programming language from file extension.
        
        Args:
            suffix: File extension (e.g., '.py')
            
        Returns:
            Language identifier string
        """
        language_map = {
            '.py': 'python',
            '.pyw': 'python',
            '.pyi': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.mjs': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.cs': 'csharp',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.R': 'r',
            '.m': 'matlab',
            '.lua': 'lua',
            '.pl': 'perl',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            '.ps1': 'powershell',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.sql': 'sql',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.tex': 'latex',
            '.dockerfile': 'dockerfile',
            '.Dockerfile': 'dockerfile',
            '.makefile': 'makefile',
            '.Makefile': 'makefile',
            '.cmake': 'cmake',
            '.vim': 'vim',
            '.vimrc': 'vim',
        }
        return language_map.get(suffix, 'unknown')
    
    async def merge_async_iterators(self, *iterators: AsyncIterator) -> AsyncIterator:
        """
        Merge multiple async iterators into a single stream.
        
        Args:
            *iterators: Variable number of async iterators to merge
            
        Yields:
            Items from all iterators as they become available
        """
        queue = asyncio.Queue()
        finished = set()
        
        async def drain(it: AsyncIterator, idx: int):
            """Drain an iterator and put items in the queue."""
            try:
                async for item in it:
                    await queue.put((idx, item))
            finally:
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
                # Check if all iterators are finished
                if len(finished) == len(iterators):
                    break
                continue
        
        # Ensure all tasks complete
        await asyncio.gather(*tasks)
        
        # Drain any remaining items in queue
        while not queue.empty():
            try:
                idx, item = queue.get_nowait()
                yield item
            except asyncio.QueueEmpty:
                break