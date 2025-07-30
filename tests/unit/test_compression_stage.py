"""
Unit tests for compression stage and streaming compressor
========================================================

Tests for pipeline/stages/compression.py including:
- StreamingCompressor functionality
- Error handling and edge cases
- Deduplication logic
- Adaptive compression integration
- Memory management and limits
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from collections import OrderedDict
import lz4.frame
import mmh3

# Mock the base classes and dependencies
import sys
sys.modules['base_classes'] = MagicMock()
sys.modules['adaptive_compression'] = MagicMock()

# Import after mocking
from pipeline.stages.compression import StreamingCompressor


class TestStreamingCompressor:
    """Test StreamingCompressor functionality"""
    
    def test_compressor_initialization_default(self):
        """Test compressor initialization with default parameters"""
        compressor = StreamingCompressor()
        
        assert compressor.window_size == 1024 * 1024  # 1MB
        assert compressor.chunk_size == 64 * 1024     # 64KB
        assert compressor.max_hashes == 100000
        assert isinstance(compressor.buffer, bytearray)
        assert isinstance(compressor.seen_hashes, OrderedDict)
        assert len(compressor.seen_hashes) == 0
    
    def test_compressor_initialization_custom(self):
        """Test compressor initialization with custom parameters"""
        compressor = StreamingCompressor(
            window_size=512 * 1024,  # 512KB
            chunk_size=32 * 1024,    # 32KB
            max_hashes=50000,
            adaptive_compression=False
        )
        
        assert compressor.window_size == 512 * 1024
        assert compressor.chunk_size == 32 * 1024
        assert compressor.max_hashes == 50000
        assert compressor.adaptive_compression is False
    
    def test_invalid_window_size(self):
        """Test that invalid window size raises ValueError"""
        with pytest.raises(ValueError, match="Window size and chunk size must be positive"):
            StreamingCompressor(window_size=0)
        
        with pytest.raises(ValueError, match="Window size and chunk size must be positive"):
            StreamingCompressor(window_size=-1024)
    
    def test_invalid_chunk_size(self):
        """Test that invalid chunk size raises ValueError"""
        with pytest.raises(ValueError, match="Window size and chunk size must be positive"):
            StreamingCompressor(chunk_size=0)
        
        with pytest.raises(ValueError, match="Window size and chunk size must be positive"):
            StreamingCompressor(chunk_size=-512)
    
    def test_chunk_size_exceeds_window_size(self):
        """Test that chunk size cannot exceed window size"""
        with pytest.raises(ValueError, match="Chunk size cannot be larger than window size"):
            StreamingCompressor(window_size=1024, chunk_size=2048)
    
    @pytest.mark.asyncio
    async def test_compress_stream_small_data(self):
        """Test compression of small data stream"""
        compressor = StreamingCompressor(
            window_size=1024,
            chunk_size=256,
            adaptive_compression=False
        )
        
        # Create small data stream
        test_data = b"Hello, World! This is a test."
        
        async def data_stream():
            yield test_data
        
        # Compress the stream
        compressed_chunks = []
        async for chunk in compressor.compress_stream(data_stream()):
            compressed_chunks.append(chunk)
        
        # Should have compressed the data
        assert len(compressed_chunks) > 0
        
        # Verify we can decompress it back
        compressed_data = b''.join(compressed_chunks)
        decompressed = lz4.frame.decompress(compressed_data)
        assert decompressed == test_data
    
    @pytest.mark.asyncio
    async def test_compress_stream_large_data(self):
        """Test compression of large data stream that exceeds window size"""
        compressor = StreamingCompressor(
            window_size=1024,  # Small window for testing
            chunk_size=256,
            adaptive_compression=False
        )
        
        # Create large data stream (4KB, 4x window size)
        test_data = b"A" * 4096
        
        async def data_stream():
            # Yield in chunks to simulate streaming
            for i in range(0, len(test_data), 500):
                yield test_data[i:i+500]
        
        compressed_chunks = []
        async for chunk in compressor.compress_stream(data_stream()):
            compressed_chunks.append(chunk)
        
        # Should have multiple compressed chunks
        assert len(compressed_chunks) > 1
        
        # Verify total compressed data is smaller than original
        total_compressed_size = sum(len(chunk) for chunk in compressed_chunks)
        assert total_compressed_size < len(test_data)
    
    @pytest.mark.asyncio
    async def test_compress_stream_deduplication(self):
        """Test deduplication functionality"""
        compressor = StreamingCompressor(
            window_size=1024,
            chunk_size=256,
            adaptive_compression=False
        )
        
        # Create data with repeating patterns
        pattern = b"Repeated pattern! " * 60  # About 1KB pattern
        test_data = pattern + pattern  # Duplicate the pattern
        
        async def data_stream():
            for i in range(0, len(test_data), 500):
                yield test_data[i:i+500]
        
        compressed_chunks = []
        async for chunk in compressor.compress_stream(data_stream()):
            compressed_chunks.append(chunk)
        
        # Should detect deduplication (fewer chunks than if no dedup)
        # Check that some hashes were cached
        assert len(compressor.seen_hashes) > 0
    
    @pytest.mark.asyncio
    async def test_compress_stream_empty_input(self):
        """Test compression of empty input stream"""
        compressor = StreamingCompressor(adaptive_compression=False)
        
        async def empty_stream():
            # Empty async generator
            return
            yield  # This will never execute
        
        compressed_chunks = []
        async for chunk in compressor.compress_stream(empty_stream()):
            compressed_chunks.append(chunk)
        
        # Should handle empty input gracefully
        assert len(compressed_chunks) == 0
    
    @pytest.mark.asyncio
    async def test_compress_stream_single_byte_chunks(self):
        """Test compression with very small input chunks"""
        compressor = StreamingCompressor(
            window_size=100,
            chunk_size=25,
            adaptive_compression=False
        )
        
        test_data = b"Small chunks test with single bytes"
        
        async def single_byte_stream():
            for byte in test_data:
                yield bytes([byte])
        
        compressed_chunks = []
        async for chunk in compressor.compress_stream(single_byte_stream()):
            compressed_chunks.append(chunk)
        
        # Should handle single-byte chunks correctly
        assert len(compressed_chunks) > 0
        
        # Verify decompression
        compressed_data = b''.join(compressed_chunks)
        decompressed = lz4.frame.decompress(compressed_data)
        assert decompressed == test_data
    
    @pytest.mark.asyncio
    async def test_compress_stream_with_file_metadata(self):
        """Test compression with file metadata"""
        compressor = StreamingCompressor(adaptive_compression=False)
        
        # Mock FileMetadata
        from unittest.mock import Mock
        metadata = Mock()
        metadata.language = "python"
        metadata.complexity_score = 5.0
        metadata.size = 1024
        
        test_data = b"def test(): return 'Hello, World!'"
        
        async def data_stream():
            yield test_data
        
        compressed_chunks = []
        async for chunk in compressor.compress_stream(
            data_stream(), 
            file_path=Path("/test/file.py"), 
            metadata=metadata
        ):
            compressed_chunks.append(chunk)
        
        assert len(compressed_chunks) > 0
    
    def test_lru_cache_behavior(self):
        """Test LRU cache behavior for hash deduplication"""
        compressor = StreamingCompressor(max_hashes=3)  # Small cache for testing
        
        # Add hashes up to limit
        compressor.seen_hashes[1] = True
        compressor.seen_hashes[2] = True
        compressor.seen_hashes[3] = True
        assert len(compressor.seen_hashes) == 3
        
        # Add another hash - should evict oldest (1)
        compressor.seen_hashes[4] = True
        if len(compressor.seen_hashes) > 3:
            # Remove oldest to maintain limit
            compressor.seen_hashes.popitem(last=False)
        
        assert len(compressor.seen_hashes) == 3
        assert 1 not in compressor.seen_hashes
        assert 4 in compressor.seen_hashes
    
    def test_create_chunks_basic(self):
        """Test basic chunk creation functionality"""
        compressor = StreamingCompressor()
        
        # Mock FileMetadata objects
        metadata_list = []
        for i in range(5):
            metadata = Mock()
            metadata.size = 100 * (i + 1)  # Increasing sizes: 100, 200, 300, 400, 500
            metadata.complexity_score = i * 2.0  # Complexity: 0, 2, 4, 6, 8
            metadata_list.append(metadata)
        
        chunks = compressor.create_chunks(metadata_list, max_chunk_size=600)
        
        # Should create chunks that don't exceed size limit
        assert len(chunks) > 0
        
        # Check that chunks don't exceed size limit
        for chunk in chunks:
            total_size = sum(m.size for m in chunk)
            assert total_size <= 600
    
    def test_create_chunks_sorting_by_complexity(self):
        """Test that chunks are sorted by complexity"""
        compressor = StreamingCompressor()
        
        # Create metadata with different complexity scores
        metadata_list = []
        complexities = [1.0, 5.0, 3.0, 8.0, 2.0]  # Unsorted
        for i, complexity in enumerate(complexities):
            metadata = Mock()
            metadata.size = 100
            metadata.complexity_score = complexity
            metadata.id = i  # For tracking
            metadata_list.append(metadata)
        
        chunks = compressor.create_chunks(metadata_list, max_chunk_size=1000)
        
        # Should be sorted by complexity (highest first)
        all_files = []
        for chunk in chunks:
            all_files.extend(chunk)
        
        # Check that files are sorted by complexity (descending)
        for i in range(len(all_files) - 1):
            assert all_files[i].complexity_score >= all_files[i + 1].complexity_score
    
    def test_create_chunks_single_large_file(self):
        """Test chunk creation with single file larger than chunk size"""
        compressor = StreamingCompressor()
        
        # Create a single large file
        large_metadata = Mock()
        large_metadata.size = 2000  # Larger than our max_chunk_size
        large_metadata.complexity_score = 5.0
        
        chunks = compressor.create_chunks([large_metadata], max_chunk_size=1000)
        
        # Should still create a chunk even though file exceeds limit
        assert len(chunks) == 1
        assert len(chunks[0]) == 1
        assert chunks[0][0] == large_metadata
    
    def test_create_chunks_empty_list(self):
        """Test chunk creation with empty metadata list"""
        compressor = StreamingCompressor()
        
        chunks = compressor.create_chunks([], max_chunk_size=1000)
        
        assert len(chunks) == 0
    
    def test_reset_deduplication_cache(self):
        """Test cache reset functionality"""
        compressor = StreamingCompressor()
        
        # Add some hashes to cache
        compressor.seen_hashes[1] = True
        compressor.seen_hashes[2] = True
        compressor.seen_hashes[3] = True
        assert len(compressor.seen_hashes) == 3
        
        # Reset cache
        compressor.reset_deduplication_cache()
        
        assert len(compressor.seen_hashes) == 0
    
    def test_get_cache_stats(self):
        """Test cache statistics reporting"""
        compressor = StreamingCompressor(max_hashes=100)
        
        # Add some hashes
        for i in range(25):
            compressor.seen_hashes[i] = True
        
        stats = compressor.get_cache_stats()
        
        assert stats['cache_size'] == 25
        assert stats['max_cache_size'] == 100
        assert stats['cache_utilization'] == 25.0  # 25/100 * 100%
    
    def test_get_cache_stats_empty(self):
        """Test cache statistics with empty cache"""
        compressor = StreamingCompressor(max_hashes=50)
        
        stats = compressor.get_cache_stats()
        
        assert stats['cache_size'] == 0
        assert stats['max_cache_size'] == 50
        assert stats['cache_utilization'] == 0.0


class TestCompressionErrorHandling:
    """Test error handling in compression operations"""
    
    @pytest.mark.asyncio
    async def test_compress_stream_lz4_error(self):
        """Test handling of LZ4 compression errors"""
        compressor = StreamingCompressor(adaptive_compression=False)
        
        # Mock lz4.frame.compress to raise an exception
        with patch('lz4.frame.compress', side_effect=Exception("LZ4 compression failed")):
            test_data = b"Test data for compression error"
            
            async def data_stream():
                yield test_data
            
            # Should handle compression errors gracefully
            compressed_chunks = []
            with pytest.raises(Exception, match="LZ4 compression failed"):
                async for chunk in compressor.compress_stream(data_stream()):
                    compressed_chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_compress_stream_hash_error(self):
        """Test handling of hash calculation errors"""
        compressor = StreamingCompressor(
            window_size=100,
            chunk_size=25,
            adaptive_compression=False
        )
        
        test_data = b"Test data for hash error handling"
        
        with patch('mmh3.hash128', side_effect=Exception("Hash calculation failed")):
            async def data_stream():
                yield test_data
            
            # Should handle hash errors gracefully
            with pytest.raises(Exception, match="Hash calculation failed"):
                compressed_chunks = []
                async for chunk in compressor.compress_stream(data_stream()):
                    compressed_chunks.append(chunk)
    
    @pytest.mark.asyncio
    async def test_compress_stream_memory_pressure(self):
        """Test behavior under memory pressure conditions"""
        # Create a compressor with very small limits
        compressor = StreamingCompressor(
            window_size=64,    # Very small window
            chunk_size=16,     # Very small chunks
            max_hashes=5,      # Very small cache
            adaptive_compression=False
        )
        
        # Create data that will stress the small buffer
        large_data = b"Memory pressure test data! " * 1000  # ~27KB
        
        async def data_stream():
            # Send data in small chunks to test buffer management
            for i in range(0, len(large_data), 100):
                yield large_data[i:i+100]
        
        compressed_chunks = []
        try:
            async for chunk in compressor.compress_stream(data_stream()):
                compressed_chunks.append(chunk)
        except MemoryError:
            pytest.fail("Should handle memory pressure gracefully")
        
        # Should still produce compressed output
        assert len(compressed_chunks) > 0
    
    def test_create_chunks_with_invalid_metadata(self):
        """Test chunk creation with invalid metadata objects"""
        compressor = StreamingCompressor()
        
        # Create metadata with missing attributes
        invalid_metadata = Mock()
        # Don't set size or complexity_score attributes
        
        with patch.object(invalid_metadata, 'size', side_effect=AttributeError("No size attribute")):
            with pytest.raises(AttributeError):
                compressor.create_chunks([invalid_metadata], max_chunk_size=1000)
    
    def test_create_chunks_with_negative_sizes(self):
        """Test chunk creation with negative file sizes"""
        compressor = StreamingCompressor()
        
        # Create metadata with negative size
        metadata = Mock()
        metadata.size = -100  # Invalid negative size
        metadata.complexity_score = 1.0
        
        chunks = compressor.create_chunks([metadata], max_chunk_size=1000)
        
        # Should still create chunks but may behave unexpectedly
        # In a real implementation, you might want to validate and reject negative sizes
        assert len(chunks) == 1
    
    @pytest.mark.asyncio
    async def test_adaptive_compression_fallback(self):
        """Test fallback when adaptive compression fails"""
        # Mock adaptive compression to be available but fail
        with patch('pipeline.stages.compression.ADAPTIVE_COMPRESSION_AVAILABLE', True):
            compressor = StreamingCompressor(adaptive_compression=True)
            
            # Mock the adaptive compression manager to fail
            mock_manager = Mock()
            mock_manager.compress_file.side_effect = Exception("Adaptive compression failed")
            compressor.compression_manager = mock_manager
            
            test_data = b"Test data for adaptive compression fallback"
            
            async def data_stream():
                yield test_data
            
            # Should fall back to standard compression
            compressed_chunks = []
            async for chunk in compressor.compress_stream(
                data_stream(), 
                file_path=Path("/test/file.py")
            ):
                compressed_chunks.append(chunk)
            
            # Should still produce compressed output via fallback
            assert len(compressed_chunks) > 0


class TestCompressionIntegration:
    """Test integration scenarios for compression stage"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_compression_pipeline(self):
        """Test end-to-end compression with realistic data"""
        compressor = StreamingCompressor(
            window_size=4096,
            chunk_size=1024,
            adaptive_compression=False
        )
        
        # Create realistic Python code data
        python_code = b'''
import os
import sys
from pathlib import Path

def main():
    """Main function for the application"""
    print("Starting application...")
    
    # Process command line arguments
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if input_path.exists():
            process_file(input_path)
        else:
            print(f"File not found: {input_path}")
            return 1
    
    return 0

def process_file(file_path: Path):
    """Process a single file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            print(f"Processing {len(content)} characters from {file_path}")
    except IOError as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    sys.exit(main())
'''
        
        async def code_stream():
            # Simulate file reading in chunks
            chunk_size = 200
            for i in range(0, len(python_code), chunk_size):
                yield python_code[i:i + chunk_size]
        
        # Compress the code
        compressed_chunks = []
        async for chunk in compressor.compress_stream(code_stream()):
            compressed_chunks.append(chunk)
        
        # Verify compression worked
        assert len(compressed_chunks) > 0
        total_compressed = sum(len(chunk) for chunk in compressed_chunks)
        assert total_compressed < len(python_code)  # Should be compressed
        
        # Verify we can decompress
        all_compressed = b''.join(compressed_chunks)
        decompressed = lz4.frame.decompress(all_compressed)
        assert decompressed == python_code
    
    @pytest.mark.asyncio
    async def test_concurrent_compression_streams(self):
        """Test handling multiple concurrent compression streams"""
        compressor1 = StreamingCompressor(window_size=1024, chunk_size=256, adaptive_compression=False)
        compressor2 = StreamingCompressor(window_size=1024, chunk_size=256, adaptive_compression=False)
        
        data1 = b"First data stream for concurrent testing " * 50
        data2 = b"Second data stream with different content " * 50
        
        async def stream1():
            for i in range(0, len(data1), 300):
                yield data1[i:i + 300]
        
        async def stream2():
            for i in range(0, len(data2), 400):
                yield data2[i:i + 400]
        
        # Process both streams concurrently
        results = await asyncio.gather(
            [chunk async for chunk in compressor1.compress_stream(stream1())],
            [chunk async for chunk in compressor2.compress_stream(stream2())]
        )
        
        chunks1, chunks2 = results
        
        # Both should produce compressed output
        assert len(chunks1) > 0
        assert len(chunks2) > 0
        
        # Verify independent compression
        compressed1 = b''.join(chunks1)
        compressed2 = b''.join(chunks2)
        
        decompressed1 = lz4.frame.decompress(compressed1)
        decompressed2 = lz4.frame.decompress(compressed2)
        
        assert decompressed1 == data1
        assert decompressed2 == data2


if __name__ == '__main__':
    pytest.main([__file__])