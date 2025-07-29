"""
Memory-efficient streaming compression with sliding window and deduplication.
"""

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List, AsyncIterator, Tuple

import lz4.frame
import mmh3

from base_classes import FileMetadata

logger = logging.getLogger(__name__)

# Try to import adaptive compression
ADAPTIVE_COMPRESSION_AVAILABLE = False
AdaptiveCompressionManager = None

try:
    from adaptive_compression import AdaptiveCompressionManager
    ADAPTIVE_COMPRESSION_AVAILABLE = True
    logger.info("Adaptive compression support loaded")
except ImportError:
    logger.info("Adaptive compression not available, using standard LZ4 compression")


class StreamingCompressor:
    """
    Memory-efficient streaming compression with sliding window deduplication.
    
    This compressor processes data in chunks, maintaining a sliding window
    for deduplication and applying compression to unique chunks only.
    """
    
    def __init__(self, 
                 window_size: int = 1024 * 1024,  # 1MB window
                 chunk_size: int = 64 * 1024,     # 64KB chunks
                 max_hashes: int = 100000,        # Maximum cached hashes
                 adaptive_compression: bool = True,
                 cache_dir: Optional[Path] = None):
        """
        Initialize the streaming compressor.
        
        Args:
            window_size: Size of the sliding window for deduplication
            chunk_size: Size of chunks to process at a time
            max_hashes: Maximum number of content hashes to cache
            adaptive_compression: Whether to use adaptive compression if available
            cache_dir: Directory for adaptive compression cache
        """
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
            
        logger.debug(f"Initialized StreamingCompressor with window_size={window_size}, "
                    f"chunk_size={chunk_size}, adaptive={self.adaptive_compression}")
    
    async def compress_stream(self, 
                            data_stream: AsyncIterator[bytes], 
                            file_path: Optional[Path] = None,
                            metadata: Optional[FileMetadata] = None) -> AsyncIterator[bytes]:
        """
        Compress data stream with deduplication and adaptive compression.
        
        Args:
            data_stream: Async iterator yielding data chunks
            file_path: Optional file path for adaptive compression hints
            metadata: Optional file metadata for compression optimization
            
        Yields:
            Compressed data chunks
        """
        total_input = 0
        total_output = 0
        duplicates_found = 0
        
        async for chunk in data_stream:
            self.buffer.extend(chunk)
            total_input += len(chunk)
            
            # Process buffer when it exceeds window size
            while len(self.buffer) >= self.window_size:
                window = bytes(self.buffer[:self.window_size])
                
                # Deduplicate using content hash
                content_hash = mmh3.hash128(window)
                
                if content_hash not in self.seen_hashes:
                    # Implement LRU cache behavior to prevent unbounded growth
                    if len(self.seen_hashes) >= self.max_hashes:
                        # Remove oldest hash (FIFO)
                        self.seen_hashes.popitem(last=False)
                    
                    # Add new hash (marks it as most recently used)
                    self.seen_hashes[content_hash] = True
                    
                    # Compress window
                    if self.adaptive_compression and file_path:
                        # Use adaptive compression based on file type and content
                        compressed, stats = self.compression_manager.compress_file(
                            file_path, window, metadata
                        )
                        total_output += len(compressed)
                        logger.debug(f"Adaptive compression stats: {stats}")
                        yield compressed
                    else:
                        # Fallback to standard LZ4 compression
                        compressed = lz4.frame.compress(window, compression_level=12)
                        total_output += len(compressed)
                        yield compressed
                else:
                    # Duplicate found - move to end (mark as recently used)
                    self.seen_hashes.move_to_end(content_hash)
                    duplicates_found += 1
                
                # Slide window
                self.buffer = self.buffer[self.chunk_size:]
        
        # Handle remaining buffer
        if self.buffer:
            if self.adaptive_compression and file_path:
                compressed, stats = self.compression_manager.compress_file(
                    file_path, bytes(self.buffer), metadata
                )
                total_output += len(compressed)
                yield compressed
            else:
                compressed = lz4.frame.compress(bytes(self.buffer))
                total_output += len(compressed)
                yield compressed
        
        # Log compression statistics
        if total_input > 0:
            compression_ratio = (1 - total_output / total_input) * 100
            logger.info(f"Compression complete: {total_input} -> {total_output} bytes "
                       f"({compression_ratio:.1f}% reduction), {duplicates_found} duplicates skipped")
    
    def create_chunks(self, 
                     metadata_list: List[FileMetadata], 
                     max_chunk_size: int = 1024 * 1024) -> List[List[FileMetadata]]:
        """
        Create optimally sized chunks for processing.
        
        Groups files into chunks that don't exceed max_chunk_size,
        sorting by complexity for better compression.
        
        Args:
            metadata_list: List of file metadata objects
            max_chunk_size: Maximum size of each chunk in bytes
            
        Returns:
            List of chunks, where each chunk is a list of FileMetadata
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Sort by complexity for better compression
        # Higher complexity files are processed first
        sorted_metadata = sorted(metadata_list, 
                               key=lambda m: m.complexity_score, 
                               reverse=True)
        
        for metadata in sorted_metadata:
            # Check if adding this file would exceed chunk size
            if current_size + metadata.size > max_chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            
            current_chunk.append(metadata)
            current_size += metadata.size
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        logger.debug(f"Created {len(chunks)} chunks from {len(metadata_list)} files")
        return chunks
    
    def reset_deduplication_cache(self):
        """Reset the deduplication hash cache."""
        self.seen_hashes.clear()
        logger.debug("Deduplication cache reset")
    
    def get_cache_stats(self) -> dict:
        """
        Get statistics about the deduplication cache.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'cache_size': len(self.seen_hashes),
            'max_cache_size': self.max_hashes,
            'cache_utilization': len(self.seen_hashes) / self.max_hashes * 100
        }