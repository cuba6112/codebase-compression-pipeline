"""
Optimized Pipeline Configurations for Different Use Cases
========================================================

This module provides pre-configured pipeline settings optimized
for various scenarios and constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration settings for the compression pipeline"""
    
    # Processing settings
    num_workers: Optional[int] = None
    batch_size: int = 100
    file_extensions: Optional[List[str]] = None
    
    # Memory settings
    window_size: int = 1024 * 1024  # 1MB
    chunk_size: int = 64 * 1024     # 64KB
    max_memory_per_worker: int = 512 * 1024 * 1024  # 512MB
    
    # Cache settings
    cache_ttl_seconds: int = 86400  # 24 hours
    max_cache_size_gb: float = 10.0
    enable_cache: bool = True
    
    # Compression settings
    compression_level: int = 12
    deduplication_enabled: bool = True
    compression_strategy: str = 'structural'
    
    # Output settings
    max_context_size: int = 128000
    output_format: str = 'markdown'
    chunk_strategy: str = 'semantic'
    
    # Performance settings
    enable_profiling: bool = False
    parallel_io: bool = True
    prefetch_files: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters"""
        if self.file_extensions is None:
            self.file_extensions = [
                '.py', '.js', '.ts', '.jsx', '.tsx',
                '.java', '.cpp', '.c', '.h', '.hpp',
                '.go', '.rs', '.rb', '.php', '.swift'
            ]
        
        # Validate numeric parameters
        if self.num_workers is not None and self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_size > self.window_size:
            raise ValueError("chunk_size cannot exceed window_size")
        if self.max_memory_per_worker <= 0:
            raise ValueError("max_memory_per_worker must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        if self.max_cache_size_gb < 0:
            raise ValueError("max_cache_size_gb cannot be negative")
        if self.compression_level < 0 or self.compression_level > 12:
            raise ValueError("compression_level must be between 0 and 12")
        if self.max_context_size <= 0:
            raise ValueError("max_context_size must be positive")
        if self.compression_strategy not in ['full', 'structural', 'signature', 'summary']:
            raise ValueError(f"Invalid compression_strategy: {self.compression_strategy}")
        if self.output_format not in ['markdown', 'json', 'xml', 'custom']:
            raise ValueError(f"Invalid output_format: {self.output_format}")
        if self.chunk_strategy not in ['semantic', 'size', 'balanced']:
            raise ValueError(f"Invalid chunk_strategy: {self.chunk_strategy}")


class ConfigPresets:
    """Pre-configured settings for common use cases"""
    
    @staticmethod
    def large_codebase() -> PipelineConfig:
        """
        Optimized for large codebases (>100k files)
        - Aggressive caching
        - Memory-efficient streaming
        - Distributed processing
        """
        return PipelineConfig(
            num_workers=None,  # Use all CPUs
            batch_size=500,
            window_size=512 * 1024,  # 512KB windows
            chunk_size=32 * 1024,    # 32KB chunks
            max_memory_per_worker=256 * 1024 * 1024,  # 256MB per worker
            cache_ttl_seconds=7 * 24 * 3600,  # 1 week
            max_cache_size_gb=50.0,
            compression_level=6,  # Faster compression
            deduplication_enabled=True,
            compression_strategy='structural',
            chunk_strategy='balanced',
            enable_profiling=True,
            parallel_io=True,
            prefetch_files=True
        )
    
    @staticmethod
    def memory_constrained() -> PipelineConfig:
        """
        Optimized for systems with limited memory (<4GB)
        - Small buffers
        - Aggressive compression
        - Single worker processing
        """
        return PipelineConfig(
            num_workers=2,
            batch_size=10,
            window_size=128 * 1024,  # 128KB windows
            chunk_size=16 * 1024,    # 16KB chunks
            max_memory_per_worker=128 * 1024 * 1024,  # 128MB
            cache_ttl_seconds=3600,  # 1 hour
            max_cache_size_gb=1.0,
            compression_level=12,  # Maximum compression
            deduplication_enabled=True,
            compression_strategy='summary',  # Minimal memory
            chunk_strategy='size',  # Simple chunking
            enable_profiling=False,
            parallel_io=False,
            prefetch_files=False
        )
    
    @staticmethod
    def real_time_processing() -> PipelineConfig:
        """
        Optimized for low latency processing
        - Fast compression
        - Minimal caching
        - Parallel everything
        """
        return PipelineConfig(
            num_workers=None,  # Use all CPUs
            batch_size=50,
            window_size=256 * 1024,  # 256KB
            chunk_size=64 * 1024,    # 64KB
            max_memory_per_worker=1024 * 1024 * 1024,  # 1GB
            cache_ttl_seconds=300,  # 5 minutes
            max_cache_size_gb=5.0,
            compression_level=1,  # Fastest compression
            deduplication_enabled=False,  # Skip dedup for speed
            compression_strategy='signature',  # Fast extraction
            chunk_strategy='size',  # Simple chunking
            enable_profiling=False,
            parallel_io=True,
            prefetch_files=True
        )
    
    @staticmethod
    def high_quality_output() -> PipelineConfig:
        """
        Optimized for best compression quality
        - Full semantic analysis
        - Maximum compression
        - Intelligent chunking
        """
        return PipelineConfig(
            num_workers=None,
            batch_size=100,
            window_size=2 * 1024 * 1024,  # 2MB windows
            chunk_size=128 * 1024,         # 128KB chunks
            max_memory_per_worker=2 * 1024 * 1024 * 1024,  # 2GB
            cache_ttl_seconds=24 * 3600,
            max_cache_size_gb=20.0,
            compression_level=12,
            deduplication_enabled=True,
            compression_strategy='structural',
            chunk_strategy='semantic',  # Best chunking
            output_format='custom',     # LLM-optimized format
            enable_profiling=True,
            parallel_io=True,
            prefetch_files=True
        )
    
    @staticmethod
    def development_mode() -> PipelineConfig:
        """
        Optimized for development and debugging
        - Verbose output
        - No caching
        - Single-threaded for debugging
        """
        return PipelineConfig(
            num_workers=1,
            batch_size=1,
            window_size=1024 * 1024,
            chunk_size=64 * 1024,
            cache_ttl_seconds=0,  # No caching
            enable_cache=False,
            compression_level=6,
            deduplication_enabled=True,
            compression_strategy='full',  # Keep everything
            chunk_strategy='size',
            output_format='json',  # Easy to inspect
            enable_profiling=True,
            parallel_io=False,
            prefetch_files=False
        )
    
    @staticmethod
    def cloud_storage() -> PipelineConfig:
        """
        Optimized for cloud storage backends
        - Large batches for network efficiency
        - Compressed output
        - Parallel uploads
        """
        return PipelineConfig(
            num_workers=None,
            batch_size=1000,
            window_size=4 * 1024 * 1024,  # 4MB windows
            chunk_size=256 * 1024,         # 256KB chunks
            cache_ttl_seconds=3600,
            max_cache_size_gb=5.0,
            compression_level=9,
            deduplication_enabled=True,
            compression_strategy='structural',
            chunk_strategy='balanced',
            max_context_size=256000,  # Larger chunks for cloud
            enable_profiling=True,
            parallel_io=True,
            prefetch_files=True
        )


class AdaptiveConfig:
    """Dynamically adjust configuration based on system resources"""
    
    @staticmethod
    def auto_configure(codebase_path: Path) -> PipelineConfig:
        """Automatically configure based on codebase and system characteristics"""
        import psutil
        import os
        
        # Get system resources
        cpu_count = os.cpu_count() or 4
        total_memory = psutil.virtual_memory().total
        available_memory = psutil.virtual_memory().available
        
        # Analyze codebase
        file_count = 0
        total_size = 0
        try:
            for f in codebase_path.rglob('*'):
                if f.is_file():
                    try:
                        file_count += 1
                        total_size += f.stat().st_size
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Cannot access file {f}: {e}")
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}")
            file_count = 1  # Fallback
            total_size = 1024  # Fallback
        
        avg_file_size = total_size / file_count if file_count > 0 else 1024
        
        # Choose base configuration
        if total_memory < 4 * 1024 * 1024 * 1024:  # < 4GB RAM
            config = ConfigPresets.memory_constrained()
        elif file_count > 100000:  # Large codebase
            config = ConfigPresets.large_codebase()
        elif avg_file_size < 10 * 1024:  # Small files
            config = ConfigPresets.real_time_processing()
        else:
            config = ConfigPresets.high_quality_output()
        
        # Adjust based on available resources
        config.num_workers = min(cpu_count, max(1, int(available_memory / (512 * 1024 * 1024))))
        config.batch_size = min(1000, max(10, int(available_memory / (avg_file_size * 10))))
        config.window_size = min(4 * 1024 * 1024, int(available_memory / (config.num_workers * 4)))
        
        return config
    
    @staticmethod
    def optimize_for_query(base_config: PipelineConfig, 
                          query: Dict[str, Any]) -> PipelineConfig:
        """Optimize configuration based on query characteristics"""
        config = PipelineConfig(**base_config.__dict__)
        
        # Adjust based on query complexity
        if 'min_complexity' in query and query['min_complexity'] > 10:
            # Complex files need more memory
            config.window_size *= 2
            config.compression_strategy = 'full'
        
        if 'language' in query:
            # Language-specific optimizations
            if query['language'] in ['python', 'javascript']:
                config.compression_strategy = 'structural'
            elif query['language'] in ['c', 'cpp']:
                config.compression_strategy = 'signature'
        
        if 'imports' in query or 'exports' in query:
            # Dependency analysis needs semantic chunking
            config.chunk_strategy = 'semantic'
        
        return config


class CompressionProfiles:
    """Compression strategy profiles for different content types"""
    
    @staticmethod
    def get_profile(file_type: str) -> Dict[str, Any]:
        """Get compression profile for file type"""
        profiles = {
            'documentation': {
                'strategy': 'summary',
                'compression_level': 12,
                'preserve': ['headings', 'code_blocks']
            },
            'test_files': {
                'strategy': 'signature',
                'compression_level': 9,
                'preserve': ['test_names', 'assertions']
            },
            'configuration': {
                'strategy': 'full',
                'compression_level': 6,
                'preserve': ['all']
            },
            'source_code': {
                'strategy': 'structural',
                'compression_level': 9,
                'preserve': ['imports', 'exports', 'signatures']
            },
            'generated': {
                'strategy': 'summary',
                'compression_level': 12,
                'preserve': ['metadata']
            }
        }
        
        # Detect file type
        if any(pattern in file_type for pattern in ['test', 'spec']):
            return profiles['test_files']
        elif file_type in ['.md', '.rst', '.txt']:
            return profiles['documentation']
        elif file_type in ['.json', '.yaml', '.toml', '.ini']:
            return profiles['configuration']
        elif file_type in ['.min.js', '.bundle.js', '.dist.js']:
            return profiles['generated']
        else:
            return profiles['source_code']


class PerformanceTuning:
    """Performance tuning utilities"""
    
    @staticmethod
    def benchmark_config(config: PipelineConfig, 
                        sample_files: list) -> Dict[str, float]:
        """Benchmark configuration on sample files"""
        import time
        import tempfile
        from pathlib import Path
        
        metrics = {}
        
        # Test compression speed
        start = time.time()
        # Simulate compression
        for _ in range(100):
            import lz4.frame
            data = b'x' * config.window_size
            compressed = lz4.frame.compress(data, compression_level=config.compression_level)
        metrics['compression_speed_mbps'] = (100 * config.window_size / 1024 / 1024) / (time.time() - start)
        
        # Test parsing speed (simplified)
        start = time.time()
        for _ in range(10):
            # Simulate parsing
            import ast
            try:
                ast.parse("def foo(): pass")
            except Exception as e:
                logger.error(f"Error in benchmark parsing: {e}")
        metrics['parsing_speed_files_per_sec'] = 10 / (time.time() - start)
        
        # Estimate memory usage
        metrics['estimated_memory_gb'] = (
            config.num_workers * config.max_memory_per_worker / 1024 / 1024 / 1024
        )
        
        # Estimate throughput
        metrics['estimated_throughput_files_per_min'] = (
            config.num_workers * config.batch_size * 60 / 
            (config.window_size / metrics['compression_speed_mbps'] / 1024 / 1024)
        )
        
        return metrics
    
    @staticmethod
    def suggest_improvements(config: PipelineConfig, 
                           metrics: Dict[str, float]) -> list:
        """Suggest configuration improvements based on metrics"""
        suggestions = []
        
        if metrics['compression_speed_mbps'] < 100:
            suggestions.append(
                "Consider reducing compression_level for better throughput"
            )
        
        if metrics['estimated_memory_gb'] > 8:
            suggestions.append(
                "High memory usage detected. Consider reducing num_workers or window_size"
            )
        
        if metrics['estimated_throughput_files_per_min'] < 1000:
            suggestions.append(
                "Low throughput. Consider increasing batch_size or num_workers"
            )
        
        if config.compression_strategy == 'full' and config.max_context_size < 64000:
            suggestions.append(
                "Full compression with small context. Consider using 'structural' strategy"
            )
        
        return suggestions


# Example usage function
def get_optimal_config(codebase_path: Path, 
                      constraints: Optional[Dict[str, Any]] = None) -> PipelineConfig:
    """
    Get optimal configuration for a codebase with optional constraints
    
    Args:
        codebase_path: Path to the codebase
        constraints: Optional constraints like max_memory, max_time, etc.
    
    Returns:
        Optimized PipelineConfig
    """
    # Start with auto-configuration
    config = AdaptiveConfig.auto_configure(codebase_path)
    
    # Apply constraints if provided
    if constraints:
        if 'max_memory_gb' in constraints:
            max_mem_bytes = constraints['max_memory_gb'] * 1024 * 1024 * 1024
            config.num_workers = min(
                config.num_workers,
                int(max_mem_bytes / config.max_memory_per_worker)
            )
        
        if 'max_time_minutes' in constraints:
            # Prefer speed over compression quality
            config.compression_level = min(config.compression_level, 6)
            config.deduplication_enabled = False
            config.chunk_strategy = 'size'
        
        if 'output_size_mb' in constraints:
            # Maximize compression
            config.compression_level = 12
            config.compression_strategy = 'summary'
    
    return config