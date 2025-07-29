"""
Adaptive Compression Strategy System
====================================

Provides intelligent compression strategy selection based on file type,
content characteristics, and system resources. This module analyzes files
and selects optimal compression algorithms and parameters.
"""

import lz4.frame
import zlib
import bz2
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import numpy as np
from collections import Counter
import time

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    logging.warning("zstandard not available. Install with: pip install zstandard")

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    logging.warning("brotli not available. Install with: pip install brotli")

from base_classes import FileMetadata

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Available compression algorithms"""
    LZ4 = "lz4"
    ZLIB = "zlib"
    BZIP2 = "bzip2"
    ZSTD = "zstd"
    BROTLI = "brotli"
    NONE = "none"


@dataclass
class CompressionProfile:
    """Compression profile for a specific file type"""
    algorithm: CompressionAlgorithm
    level: int  # Compression level (algorithm-specific)
    block_size: int = 65536  # Block size for streaming
    dictionary: Optional[bytes] = None  # Pre-trained dictionary
    priority: str = "balanced"  # "speed", "ratio", "balanced"
    
    
@dataclass
class ContentAnalysis:
    """Analysis results for file content"""
    entropy: float = 0.0  # Shannon entropy
    repetition_score: float = 0.0  # How repetitive the content is
    structure_score: float = 0.0  # How structured (vs random)
    compressibility_estimate: float = 0.0  # Estimated compression ratio
    dominant_patterns: List[str] = field(default_factory=list)
    byte_distribution: Dict[int, float] = field(default_factory=dict)
    is_likely_compressed: bool = False
    is_binary: bool = False
    

class AdaptiveCompressionStrategy:
    """Adaptive compression strategy selector"""
    
    def __init__(self):
        # Define compression profiles for different file types
        self.profiles = {
            # Source code - high structure, good compression
            "python": CompressionProfile(
                algorithm=CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=6,
                priority="balanced"
            ),
            "javascript": CompressionProfile(
                algorithm=CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=5,
                priority="balanced"  
            ),
            "typescript": CompressionProfile(
                algorithm=CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=5,
                priority="balanced"
            ),
            "go": CompressionProfile(
                algorithm=CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.LZ4,
                level=5,
                priority="speed"
            ),
            "rust": CompressionProfile(
                algorithm=CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.LZ4,
                level=5,
                priority="speed"
            ),
            
            # Web assets - balance between size and speed
            "html": CompressionProfile(
                algorithm=CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=6,
                priority="ratio"
            ),
            "css": CompressionProfile(
                algorithm=CompressionAlgorithm.BROTLI if BROTLI_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=6,
                priority="ratio"
            ),
            
            # Data files - maximize compression
            "json": CompressionProfile(
                algorithm=CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=9,
                priority="ratio"
            ),
            "xml": CompressionProfile(
                algorithm=CompressionAlgorithm.BZIP2,
                level=9,
                priority="ratio"
            ),
            "csv": CompressionProfile(
                algorithm=CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.ZLIB,
                level=7,
                priority="ratio"
            ),
            
            # Documentation - good compression, readable
            "markdown": CompressionProfile(
                algorithm=CompressionAlgorithm.ZLIB,
                level=6,
                priority="balanced"
            ),
            "text": CompressionProfile(
                algorithm=CompressionAlgorithm.ZLIB,
                level=6,
                priority="balanced"
            ),
            
            # Binary/Already compressed - minimal compression
            "binary": CompressionProfile(
                algorithm=CompressionAlgorithm.LZ4,
                level=1,
                priority="speed"
            ),
            "image": CompressionProfile(
                algorithm=CompressionAlgorithm.NONE,
                level=0,
                priority="speed"
            ),
            "compressed": CompressionProfile(
                algorithm=CompressionAlgorithm.NONE,
                level=0,
                priority="speed"
            )
        }
        
        # Extension to type mapping
        self.extension_map = {
            # Source code
            '.py': 'python',
            '.js': 'javascript', 
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            
            # Web
            '.html': 'html',
            '.htm': 'html', 
            '.css': 'css',
            '.scss': 'css',
            '.sass': 'css',
            
            # Data
            '.json': 'json',
            '.xml': 'xml',
            '.csv': 'csv',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            
            # Docs
            '.md': 'markdown',
            '.txt': 'text',
            '.rst': 'text',
            
            # Binary/Compressed
            '.zip': 'compressed',
            '.gz': 'compressed',
            '.tar': 'compressed',
            '.rar': 'compressed',
            '.7z': 'compressed',
            '.png': 'image',
            '.jpg': 'image',
            '.jpeg': 'image',
            '.gif': 'image',
            '.webp': 'image',
            '.pdf': 'binary',
            '.exe': 'binary',
            '.dll': 'binary',
            '.so': 'binary',
            '.dylib': 'binary'
        }
        
        # Initialize compressors
        self._init_compressors()
        
    def _init_compressors(self):
        """Initialize compression objects"""
        self.compressors = {}
        
        # LZ4 - always available
        self.compressors[CompressionAlgorithm.LZ4] = lz4.frame
        
        # ZLIB - always available
        self.compressors[CompressionAlgorithm.ZLIB] = zlib
        
        # BZIP2 - always available
        self.compressors[CompressionAlgorithm.BZIP2] = bz2
        
        # ZSTD - optional
        if ZSTD_AVAILABLE:
            self.compressors[CompressionAlgorithm.ZSTD] = zstd
            
        # Brotli - optional
        if BROTLI_AVAILABLE:
            self.compressors[CompressionAlgorithm.BROTLI] = brotli
            
    def analyze_content(self, content: bytes, sample_size: int = 8192) -> ContentAnalysis:
        """Analyze content characteristics"""
        analysis = ContentAnalysis()
        
        # Sample content if too large
        if len(content) > sample_size:
            # Take samples from beginning, middle, and end
            samples = [
                content[:sample_size//3],
                content[len(content)//2 - sample_size//6:len(content)//2 + sample_size//6],
                content[-sample_size//3:]
            ]
            sample = b''.join(samples)
        else:
            sample = content
            
        # Calculate byte frequency
        byte_counts = Counter(sample)
        total_bytes = len(sample)
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in byte_counts.values():
            if count > 0:
                p = count / total_bytes
                entropy -= p * math.log2(p)
                
        analysis.entropy = entropy
        
        # Normalize entropy (8 bits max)
        normalized_entropy = entropy / 8.0
        
        # Check if likely already compressed (high entropy)
        analysis.is_likely_compressed = normalized_entropy > 0.95
        
        # Calculate repetition score
        # Check for repeated sequences
        repetitions = 0
        chunk_size = 16
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample)-chunk_size, chunk_size)]
        unique_chunks = len(set(chunks))
        if len(chunks) > 0:
            analysis.repetition_score = 1.0 - (unique_chunks / len(chunks))
            
        # Structure score - based on byte distribution
        # Structured data tends to use limited byte ranges
        used_bytes = len(byte_counts)
        analysis.structure_score = 1.0 - (used_bytes / 256.0)
        
        # Detect if binary
        # Check for null bytes and non-printable characters
        null_bytes = byte_counts.get(0, 0)
        printable_count = sum(byte_counts.get(b, 0) for b in range(32, 127))
        analysis.is_binary = (null_bytes > total_bytes * 0.01) or (printable_count < total_bytes * 0.75)
        
        # Estimate compressibility
        # Based on entropy, repetition, and structure
        analysis.compressibility_estimate = (
            (1.0 - normalized_entropy) * 0.4 +
            analysis.repetition_score * 0.3 +
            analysis.structure_score * 0.3
        )
        
        # Store byte distribution
        analysis.byte_distribution = {
            byte: count/total_bytes 
            for byte, count in byte_counts.most_common(10)
        }
        
        return analysis
        
    def select_strategy(
        self, 
        file_path: Path,
        metadata: Optional[FileMetadata] = None,
        content_sample: Optional[bytes] = None,
        system_load: float = 0.5
    ) -> CompressionProfile:
        """Select optimal compression strategy"""
        
        # Get file extension
        ext = file_path.suffix.lower()
        
        # Check if we have a predefined profile
        file_type = self.extension_map.get(ext)
        if file_type and file_type in self.profiles:
            base_profile = self.profiles[file_type]
        else:
            # Default profile
            base_profile = CompressionProfile(
                algorithm=CompressionAlgorithm.LZ4,
                level=3,
                priority="balanced"
            )
            
        # If we have content, analyze it
        if content_sample:
            analysis = self.analyze_content(content_sample)
            
            # Adjust based on analysis
            if analysis.is_likely_compressed:
                # Don't compress already compressed data
                return CompressionProfile(
                    algorithm=CompressionAlgorithm.NONE,
                    level=0,
                    priority="speed"
                )
                
            # Adjust compression level based on compressibility
            if analysis.compressibility_estimate > 0.7:
                # Highly compressible - use stronger compression
                if base_profile.algorithm == CompressionAlgorithm.LZ4:
                    base_profile.algorithm = CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.ZLIB
                base_profile.level = min(base_profile.level + 2, 9)
                
            elif analysis.compressibility_estimate < 0.3:
                # Poorly compressible - use faster compression
                base_profile.algorithm = CompressionAlgorithm.LZ4
                base_profile.level = max(base_profile.level - 2, 1)
                
        # Adjust based on file size (from metadata if available)
        if metadata:
            if metadata.size > 10 * 1024 * 1024:  # > 10MB
                # Large files - prioritize speed
                if base_profile.algorithm in [CompressionAlgorithm.BZIP2, CompressionAlgorithm.BROTLI]:
                    base_profile.algorithm = CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.LZ4
                base_profile.level = max(base_profile.level - 1, 1)
                
            elif metadata.size < 10 * 1024:  # < 10KB
                # Small files - can afford stronger compression
                base_profile.level = min(base_profile.level + 1, 9)
                
        # Adjust based on system load
        if system_load > 0.8:
            # High load - prioritize speed
            base_profile.algorithm = CompressionAlgorithm.LZ4
            base_profile.level = 1
            base_profile.priority = "speed"
            
        elif system_load < 0.3:
            # Low load - can use stronger compression
            if base_profile.algorithm == CompressionAlgorithm.LZ4:
                base_profile.algorithm = CompressionAlgorithm.ZSTD if ZSTD_AVAILABLE else CompressionAlgorithm.ZLIB
            base_profile.level = min(base_profile.level + 1, 9)
            
        return base_profile
        
    def compress(
        self,
        data: bytes,
        profile: CompressionProfile
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress data using selected profile"""
        start_time = time.time()
        stats = {
            "algorithm": profile.algorithm.value,
            "level": profile.level,
            "original_size": len(data),
            "compressed_size": 0,
            "compression_ratio": 0.0,
            "compression_time": 0.0
        }
        
        if profile.algorithm == CompressionAlgorithm.NONE:
            stats["compressed_size"] = len(data)
            stats["compression_ratio"] = 1.0
            return data, stats
            
        compressed = data
        
        try:
            if profile.algorithm == CompressionAlgorithm.LZ4:
                compressed = lz4.frame.compress(
                    data,
                    compression_level=profile.level,
                    block_size=lz4.frame.BLOCKSIZE_MAX4MB
                )
                
            elif profile.algorithm == CompressionAlgorithm.ZLIB:
                compressed = zlib.compress(data, level=profile.level)
                
            elif profile.algorithm == CompressionAlgorithm.BZIP2:
                compressed = bz2.compress(data, compresslevel=profile.level)
                
            elif profile.algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                cctx = zstd.ZstdCompressor(level=profile.level)
                compressed = cctx.compress(data)
                
            elif profile.algorithm == CompressionAlgorithm.BROTLI and BROTLI_AVAILABLE:
                compressed = brotli.compress(data, quality=profile.level)
                
        except Exception as e:
            logger.error(f"Compression failed with {profile.algorithm}: {e}")
            # Fall back to LZ4
            compressed = lz4.frame.compress(data, compression_level=1)
            stats["algorithm"] = "lz4 (fallback)"
            
        stats["compressed_size"] = len(compressed)
        stats["compression_ratio"] = len(compressed) / len(data) if len(data) > 0 else 1.0
        stats["compression_time"] = time.time() - start_time
        
        return compressed, stats
        
    def decompress(
        self,
        data: bytes,
        algorithm: CompressionAlgorithm
    ) -> bytes:
        """Decompress data"""
        if algorithm == CompressionAlgorithm.NONE:
            return data
            
        try:
            if algorithm == CompressionAlgorithm.LZ4:
                return lz4.frame.decompress(data)
                
            elif algorithm == CompressionAlgorithm.ZLIB:
                return zlib.decompress(data)
                
            elif algorithm == CompressionAlgorithm.BZIP2:
                return bz2.decompress(data)
                
            elif algorithm == CompressionAlgorithm.ZSTD and ZSTD_AVAILABLE:
                dctx = zstd.ZstdDecompressor()
                return dctx.decompress(data)
                
            elif algorithm == CompressionAlgorithm.BROTLI and BROTLI_AVAILABLE:
                return brotli.decompress(data)
                
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
            
        return data
        
    def get_algorithm_info(self, algorithm: CompressionAlgorithm) -> Dict[str, Any]:
        """Get information about compression algorithm"""
        info = {
            "name": algorithm.value,
            "available": algorithm in self.compressors or algorithm == CompressionAlgorithm.NONE,
            "speed": "unknown",
            "ratio": "unknown",
            "memory_usage": "unknown"
        }
        
        # Algorithm characteristics
        characteristics = {
            CompressionAlgorithm.LZ4: {
                "speed": "very fast",
                "ratio": "moderate",
                "memory_usage": "low"
            },
            CompressionAlgorithm.ZLIB: {
                "speed": "moderate",
                "ratio": "good",
                "memory_usage": "moderate"
            },
            CompressionAlgorithm.BZIP2: {
                "speed": "slow",
                "ratio": "very good",
                "memory_usage": "high"
            },
            CompressionAlgorithm.ZSTD: {
                "speed": "fast",
                "ratio": "very good",
                "memory_usage": "moderate"
            },
            CompressionAlgorithm.BROTLI: {
                "speed": "slow",
                "ratio": "excellent",
                "memory_usage": "high"
            },
            CompressionAlgorithm.NONE: {
                "speed": "instant",
                "ratio": "none",
                "memory_usage": "none"
            }
        }
        
        if algorithm in characteristics:
            info.update(characteristics[algorithm])
            
        return info
        
    def benchmark_algorithms(
        self,
        data: bytes,
        algorithms: Optional[List[CompressionAlgorithm]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark different compression algorithms on data"""
        if algorithms is None:
            algorithms = [
                algo for algo in CompressionAlgorithm 
                if algo != CompressionAlgorithm.NONE
            ]
            
        results = {}
        
        for algo in algorithms:
            if algo not in self.compressors:
                continue
                
            # Test different compression levels
            for level in [1, 5, 9]:
                profile = CompressionProfile(algorithm=algo, level=level)
                
                try:
                    compressed, stats = self.compress(data, profile)
                    
                    # Decompress to verify
                    decompression_start = time.time()
                    decompressed = self.decompress(compressed, algo)
                    decompression_time = time.time() - decompression_start
                    
                    # Verify integrity
                    if len(decompressed) != len(data):
                        logger.error(f"Decompression size mismatch for {algo}")
                        continue
                        
                    key = f"{algo.value}_level_{level}"
                    results[key] = {
                        **stats,
                        "decompression_time": decompression_time,
                        "total_time": stats["compression_time"] + decompression_time
                    }
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {algo} level {level}: {e}")
                    
        return results


class AdaptiveCompressionManager:
    """Manages adaptive compression for the pipeline"""
    
    def __init__(self, cache_dir: Path):
        self.strategy = AdaptiveCompressionStrategy()
        self.cache_dir = cache_dir
        self.stats_file = cache_dir / "compression_stats.json"
        self.stats = self._load_stats()
        
    def _load_stats(self) -> Dict[str, Any]:
        """Load compression statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "total_files": 0,
            "total_original_size": 0,
            "total_compressed_size": 0,
            "algorithm_usage": {},
            "file_type_stats": {}
        }
        
    def _save_stats(self):
        """Save compression statistics"""
        try:
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save compression stats: {e}")
            
    def compress_file(
        self,
        file_path: Path,
        content: bytes,
        metadata: Optional[FileMetadata] = None,
        system_load: float = 0.5
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Compress file with adaptive strategy"""
        # Select strategy
        profile = self.strategy.select_strategy(
            file_path,
            metadata,
            content[:8192] if len(content) > 8192 else content,
            system_load
        )
        
        # Compress
        compressed, stats = self.strategy.compress(content, profile)
        
        # Update statistics
        self._update_stats(file_path, stats)
        
        return compressed, stats
        
    def _update_stats(self, file_path: Path, compression_stats: Dict[str, Any]):
        """Update compression statistics"""
        self.stats["total_files"] += 1
        self.stats["total_original_size"] += compression_stats["original_size"]
        self.stats["total_compressed_size"] += compression_stats["compressed_size"]
        
        # Algorithm usage
        algo = compression_stats["algorithm"]
        if algo not in self.stats["algorithm_usage"]:
            self.stats["algorithm_usage"][algo] = 0
        self.stats["algorithm_usage"][algo] += 1
        
        # File type stats
        ext = file_path.suffix.lower()
        if ext not in self.stats["file_type_stats"]:
            self.stats["file_type_stats"][ext] = {
                "count": 0,
                "original_size": 0,
                "compressed_size": 0,
                "algorithms_used": {}
            }
            
        type_stats = self.stats["file_type_stats"][ext]
        type_stats["count"] += 1
        type_stats["original_size"] += compression_stats["original_size"]
        type_stats["compressed_size"] += compression_stats["compressed_size"]
        
        if algo not in type_stats["algorithms_used"]:
            type_stats["algorithms_used"][algo] = 0
        type_stats["algorithms_used"][algo] += 1
        
        # Save periodically
        if self.stats["total_files"] % 100 == 0:
            self._save_stats()
            
    def get_compression_report(self) -> str:
        """Generate compression statistics report"""
        if self.stats["total_files"] == 0:
            return "No compression statistics available yet."
            
        report = []
        report.append("=== Adaptive Compression Report ===\n")
        
        # Overall stats
        total_ratio = self.stats["total_compressed_size"] / self.stats["total_original_size"]
        saved_bytes = self.stats["total_original_size"] - self.stats["total_compressed_size"]
        saved_percentage = (1 - total_ratio) * 100
        
        report.append(f"Total files processed: {self.stats['total_files']:,}")
        report.append(f"Original size: {self.stats['total_original_size']:,} bytes")
        report.append(f"Compressed size: {self.stats['total_compressed_size']:,} bytes")
        report.append(f"Space saved: {saved_bytes:,} bytes ({saved_percentage:.1f}%)")
        report.append(f"Average compression ratio: {total_ratio:.3f}")
        
        # Algorithm usage
        report.append("\n--- Algorithm Usage ---")
        for algo, count in sorted(self.stats["algorithm_usage"].items(), 
                                 key=lambda x: x[1], reverse=True):
            percentage = (count / self.stats["total_files"]) * 100
            report.append(f"{algo}: {count} files ({percentage:.1f}%)")
            
        # File type performance
        report.append("\n--- File Type Performance ---")
        type_perfs = []
        for ext, stats in self.stats["file_type_stats"].items():
            if stats["original_size"] > 0:
                ratio = stats["compressed_size"] / stats["original_size"]
                type_perfs.append((ext, stats["count"], ratio, stats))
                
        # Sort by compression ratio (best first)
        type_perfs.sort(key=lambda x: x[2])
        
        for ext, count, ratio, stats in type_perfs[:10]:  # Top 10
            saved_pct = (1 - ratio) * 100
            report.append(f"{ext}: {count} files, {ratio:.3f} ratio ({saved_pct:.1f}% saved)")
            
        return "\n".join(report)


def test_adaptive_compression():
    """Test adaptive compression system"""
    print("Testing Adaptive Compression...")
    
    import tempfile
    import os
    
    # Create temporary cache directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir)
        
        # Create manager
        manager = AdaptiveCompressionManager(cache_dir)
        
        # Test different file types
        test_cases = [
            # Python code - should compress well
            ("test.py", b"import os\nimport sys\n\ndef main():\n    print('Hello')\n" * 100),
            
            # JSON - highly compressible
            ("data.json", json.dumps({"key": "value", "items": list(range(1000))}).encode() * 10),
            
            # Binary data - less compressible
            ("binary.dat", bytes(range(256)) * 100),
            
            # Already compressed - should skip
            ("compressed.gz", lz4.frame.compress(b"already compressed")),
            
            # Small file
            ("small.txt", b"Hello, World!"),
            
            # Large repetitive file
            ("large.log", b"2024-01-01 INFO: Application started\n" * 10000),
        ]
        
        print("\n📊 Compression Test Results:")
        print("-" * 80)
        print(f"{'File':<20} {'Original':<12} {'Compressed':<12} {'Ratio':<8} {'Algorithm':<15} {'Time (ms)':<10}")
        print("-" * 80)
        
        for filename, content in test_cases:
            file_path = Path(filename)
            compressed, stats = manager.compress_file(file_path, content)
            
            print(f"{filename:<20} {stats['original_size']:<12,} {stats['compressed_size']:<12,} "
                  f"{stats['compression_ratio']:<8.3f} {stats['algorithm']:<15} "
                  f"{stats['compression_time']*1000:<10.1f}")
                  
        # Test content analysis
        print("\n📈 Content Analysis Test:")
        strategy = AdaptiveCompressionStrategy()
        
        # Analyze different content types
        contents = {
            "Random": os.urandom(1024),
            "Repetitive": b"AAAA" * 256,
            "Structured": b'{"key": "value"}\n' * 64,
            "Text": b"The quick brown fox jumps over the lazy dog. " * 20,
        }
        
        for name, content in contents.items():
            analysis = strategy.analyze_content(content)
            print(f"\n{name}:")
            print(f"  Entropy: {analysis.entropy:.2f}")
            print(f"  Repetition: {analysis.repetition_score:.2f}")
            print(f"  Structure: {analysis.structure_score:.2f}")
            print(f"  Compressibility: {analysis.compressibility_estimate:.2f}")
            print(f"  Is compressed: {analysis.is_likely_compressed}")
            print(f"  Is binary: {analysis.is_binary}")
            
        # Print report
        print("\n" + manager.get_compression_report())
        
        print("\n✅ Adaptive compression test completed!")
        
        # Benchmark algorithms
        print("\n⚡ Benchmarking Compression Algorithms...")
        test_data = b"Sample text for compression testing. " * 1000
        results = strategy.benchmark_algorithms(test_data)
        
        print(f"\n{'Algorithm':<20} {'Ratio':<8} {'Compress (ms)':<15} {'Decompress (ms)':<15}")
        print("-" * 60)
        
        for name, stats in sorted(results.items(), key=lambda x: x[1]['compression_ratio']):
            print(f"{name:<20} {stats['compression_ratio']:<8.3f} "
                  f"{stats['compression_time']*1000:<15.1f} "
                  f"{stats['decompression_time']*1000:<15.1f}")


if __name__ == "__main__":
    test_adaptive_compression()