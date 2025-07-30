"""
Unit tests for adaptive configuration and performance tuning
============================================================

Tests for pipeline_configs.py including:
- AdaptiveConfig dynamic configuration
- CompressionProfiles selection  
- PerformanceTuning benchmarking
- Configuration validation and optimization
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import psutil

# Import the classes to test
from pipeline_configs import (
    PipelineConfig, ConfigPresets, AdaptiveConfig, 
    CompressionProfiles, PerformanceTuning, get_optimal_config
)


class TestPipelineConfig:
    """Test PipelineConfig validation and initialization"""
    
    def test_default_config_valid(self):
        """Test that default configuration is valid"""
        config = PipelineConfig()
        assert config.num_workers is None
        assert config.batch_size == 100
        assert config.compression_level == 12
        assert config.compression_strategy == 'structural'
        assert config.output_format == 'markdown'
        assert len(config.file_extensions) > 0
    
    def test_invalid_num_workers(self):
        """Test that invalid num_workers raises ValueError"""
        with pytest.raises(ValueError, match="num_workers must be positive"):
            PipelineConfig(num_workers=0)
        
        with pytest.raises(ValueError, match="num_workers must be positive"):
            PipelineConfig(num_workers=-1)
    
    def test_invalid_batch_size(self):
        """Test that invalid batch_size raises ValueError"""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            PipelineConfig(batch_size=0)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            PipelineConfig(batch_size=-5)
    
    def test_invalid_window_size(self):
        """Test that invalid window_size raises ValueError"""
        with pytest.raises(ValueError, match="window_size must be positive"):
            PipelineConfig(window_size=0)
        
        with pytest.raises(ValueError, match="window_size must be positive"):
            PipelineConfig(window_size=-1024)
    
    def test_chunk_size_exceeds_window_size(self):
        """Test that chunk_size cannot exceed window_size"""
        with pytest.raises(ValueError, match="chunk_size cannot exceed window_size"):
            PipelineConfig(window_size=1024, chunk_size=2048)
    
    def test_invalid_compression_level(self):
        """Test that invalid compression_level raises ValueError"""
        with pytest.raises(ValueError, match="compression_level must be between 0 and 12"):
            PipelineConfig(compression_level=-1)
        
        with pytest.raises(ValueError, match="compression_level must be between 0 and 12"):
            PipelineConfig(compression_level=13)
    
    def test_invalid_compression_strategy(self):
        """Test that invalid compression_strategy raises ValueError"""
        with pytest.raises(ValueError, match="Invalid compression_strategy"):
            PipelineConfig(compression_strategy='invalid_strategy')
    
    def test_invalid_output_format(self):
        """Test that invalid output_format raises ValueError"""
        with pytest.raises(ValueError, match="Invalid output_format"):
            PipelineConfig(output_format='invalid_format')
    
    def test_invalid_chunk_strategy(self):
        """Test that invalid chunk_strategy raises ValueError"""
        with pytest.raises(ValueError, match="Invalid chunk_strategy"):
            PipelineConfig(chunk_strategy='invalid_chunk')
    
    def test_negative_cache_ttl(self):
        """Test that negative cache_ttl_seconds raises ValueError"""
        with pytest.raises(ValueError, match="cache_ttl_seconds cannot be negative"):
            PipelineConfig(cache_ttl_seconds=-1)
    
    def test_negative_cache_size(self):
        """Test that negative max_cache_size_gb raises ValueError"""
        with pytest.raises(ValueError, match="max_cache_size_gb cannot be negative"):
            PipelineConfig(max_cache_size_gb=-1.0)


class TestConfigPresets:
    """Test configuration presets"""
    
    def test_large_codebase_preset(self):
        """Test large codebase configuration preset"""
        config = ConfigPresets.large_codebase()
        assert config.batch_size == 500
        assert config.compression_level == 6  # Optimized for speed
        assert config.compression_strategy == 'structural'
        assert config.chunk_strategy == 'balanced'
        assert config.enable_profiling is True
        assert config.parallel_io is True
    
    def test_memory_constrained_preset(self):
        """Test memory constrained configuration preset"""
        config = ConfigPresets.memory_constrained()
        assert config.num_workers == 2
        assert config.batch_size == 10
        assert config.window_size == 128 * 1024  # Small windows
        assert config.chunk_size == 16 * 1024   # Small chunks
        assert config.compression_level == 12   # Maximum compression
        assert config.compression_strategy == 'summary'
        assert config.parallel_io is False
    
    def test_real_time_processing_preset(self):
        """Test real-time processing configuration preset"""
        config = ConfigPresets.real_time_processing()
        assert config.compression_level == 1  # Fastest compression
        assert config.deduplication_enabled is False  # Skip for speed
        assert config.compression_strategy == 'signature'
        assert config.chunk_strategy == 'size'
        assert config.cache_ttl_seconds == 300  # 5 minutes
    
    def test_high_quality_output_preset(self):
        """Test high quality output configuration preset"""
        config = ConfigPresets.high_quality_output()
        assert config.compression_level == 12
        assert config.compression_strategy == 'structural'
        assert config.chunk_strategy == 'semantic'
        assert config.output_format == 'custom'
        assert config.window_size == 2 * 1024 * 1024  # Large windows
    
    def test_development_mode_preset(self):
        """Test development mode configuration preset"""
        config = ConfigPresets.development_mode()
        assert config.num_workers == 1
        assert config.batch_size == 1
        assert config.enable_cache is False
        assert config.cache_ttl_seconds == 0
        assert config.compression_strategy == 'full'
        assert config.output_format == 'json'
        assert config.parallel_io is False
    
    def test_cloud_storage_preset(self):
        """Test cloud storage configuration preset"""
        config = ConfigPresets.cloud_storage()
        assert config.batch_size == 1000
        assert config.window_size == 4 * 1024 * 1024  # Large windows
        assert config.chunk_size == 256 * 1024
        assert config.max_context_size == 256000
        assert config.compression_level == 9
        assert config.parallel_io is True


class TestAdaptiveConfig:
    """Test adaptive configuration logic"""
    
    @patch('psutil.virtual_memory')
    @patch('os.cpu_count')
    def test_auto_configure_high_memory_system(self, mock_cpu_count, mock_virtual_memory):
        """Test auto-configuration for high memory system"""
        # Mock system with 16GB RAM and 8 CPUs
        mock_cpu_count.return_value = 8
        mock_memory = Mock()
        mock_memory.total = 16 * 1024 * 1024 * 1024  # 16GB
        mock_memory.available = 12 * 1024 * 1024 * 1024  # 12GB available
        mock_virtual_memory.return_value = mock_memory
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test files
            (temp_path / "file1.py").write_text("print('hello')")
            (temp_path / "file2.js").write_text("console.log('world')")
            
            config = AdaptiveConfig.auto_configure(temp_path)
            
            # Should choose high quality config for normal sized codebase
            assert config.compression_strategy in ['structural', 'full']
            assert config.num_workers <= 8
            assert config.window_size > 0
    
    @patch('psutil.virtual_memory')
    @patch('os.cpu_count')
    def test_auto_configure_low_memory_system(self, mock_cpu_count, mock_virtual_memory):
        """Test auto-configuration for low memory system"""
        # Mock system with 2GB RAM and 2 CPUs
        mock_cpu_count.return_value = 2
        mock_memory = Mock()
        mock_memory.total = 2 * 1024 * 1024 * 1024  # 2GB
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB available
        mock_virtual_memory.return_value = mock_memory
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "file1.py").write_text("print('hello')")
            
            config = AdaptiveConfig.auto_configure(temp_path)
            
            # Should choose memory constrained config
            assert config.compression_strategy == 'summary'
            assert config.num_workers <= 2
            assert config.window_size <= 512 * 1024  # Small windows
    
    def test_auto_configure_large_codebase(self):
        """Test auto-configuration for large codebase"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create many small files to simulate large codebase
            for i in range(150):  # Create enough to trigger large codebase logic
                (temp_path / f"file{i}.py").write_text(f"# File {i}")
            
            with patch('psutil.virtual_memory') as mock_mem, \
                 patch('os.cpu_count', return_value=8):
                mock_memory = Mock()
                mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
                mock_memory.available = 6 * 1024 * 1024 * 1024  # 6GB
                mock_mem.return_value = mock_memory
                
                config = AdaptiveConfig.auto_configure(temp_path)
                
                # Should use settings optimized for large codebase
                assert config.batch_size >= 100
                assert config.enable_profiling is True
    
    def test_auto_configure_with_permission_errors(self):
        """Test auto-configuration handles permission errors gracefully"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a file that we'll mock to raise permission error
            test_file = temp_path / "test.py"
            test_file.write_text("print('test')")
            
            with patch.object(Path, 'stat', side_effect=PermissionError("Access denied")):
                # Should not crash, should use fallback values
                config = AdaptiveConfig.auto_configure(temp_path)
                assert isinstance(config, PipelineConfig)
    
    def test_optimize_for_query_complex_files(self):
        """Test query optimization for complex files"""
        base_config = ConfigPresets.balanced()
        query = {'min_complexity': 15.0}
        
        optimized = AdaptiveConfig.optimize_for_query(base_config, query)
        
        assert optimized.window_size == base_config.window_size * 2
        assert optimized.compression_strategy == 'full'
    
    def test_optimize_for_query_python_language(self):
        """Test query optimization for Python language"""
        base_config = ConfigPresets.balanced()
        query = {'language': 'python'}
        
        optimized = AdaptiveConfig.optimize_for_query(base_config, query)
        
        assert optimized.compression_strategy == 'structural'
    
    def test_optimize_for_query_cpp_language(self):
        """Test query optimization for C++ language"""
        base_config = ConfigPresets.balanced()
        query = {'language': 'cpp'}
        
        optimized = AdaptiveConfig.optimize_for_query(base_config, query)
        
        assert optimized.compression_strategy == 'signature'
    
    def test_optimize_for_query_dependency_analysis(self):
        """Test query optimization for dependency analysis"""
        base_config = ConfigPresets.balanced()
        query = {'imports': True, 'exports': True}
        
        optimized = AdaptiveConfig.optimize_for_query(base_config, query)
        
        assert optimized.chunk_strategy == 'semantic'


class TestCompressionProfiles:
    """Test compression profile selection"""
    
    def test_test_file_profile(self):
        """Test profile for test files"""
        profile = CompressionProfiles.get_profile('test_example.py')
        assert profile['strategy'] == 'signature'
        assert profile['compression_level'] == 9
        assert 'test_names' in profile['preserve']
        assert 'assertions' in profile['preserve']
    
    def test_spec_file_profile(self):
        """Test profile for spec files"""
        profile = CompressionProfiles.get_profile('example.spec.js')
        assert profile['strategy'] == 'signature'
        assert 'test_names' in profile['preserve']
    
    def test_documentation_profile(self):
        """Test profile for documentation files"""
        profile = CompressionProfiles.get_profile('.md')
        assert profile['strategy'] == 'summary'
        assert profile['compression_level'] == 12
        assert 'headings' in profile['preserve']
        assert 'code_blocks' in profile['preserve']
    
    def test_configuration_profile(self):
        """Test profile for configuration files"""
        profile = CompressionProfiles.get_profile('.json')
        assert profile['strategy'] == 'full'
        assert profile['compression_level'] == 6
        assert 'all' in profile['preserve']
    
    def test_generated_file_profile(self):
        """Test profile for generated files"""
        profile = CompressionProfiles.get_profile('.min.js')
        assert profile['strategy'] == 'summary'
        assert profile['compression_level'] == 12
        assert 'metadata' in profile['preserve']
    
    def test_source_code_profile(self):
        """Test profile for source code files"""
        profile = CompressionProfiles.get_profile('.py')
        assert profile['strategy'] == 'structural'
        assert profile['compression_level'] == 9
        assert 'imports' in profile['preserve']
        assert 'exports' in profile['preserve']
        assert 'signatures' in profile['preserve']


class TestPerformanceTuning:
    """Test performance tuning and benchmarking"""
    
    def test_benchmark_config(self):
        """Test configuration benchmarking"""
        config = ConfigPresets.balanced()
        sample_files = []  # Empty for this test
        
        metrics = PerformanceTuning.benchmark_config(config, sample_files)
        
        # Check that all expected metrics are present
        assert 'compression_speed_mbps' in metrics
        assert 'parsing_speed_files_per_sec' in metrics
        assert 'estimated_memory_gb' in metrics
        assert 'estimated_throughput_files_per_min' in metrics
        
        # Check that metrics have reasonable values
        assert metrics['compression_speed_mbps'] > 0
        assert metrics['parsing_speed_files_per_sec'] > 0
        assert metrics['estimated_memory_gb'] > 0
        assert metrics['estimated_throughput_files_per_min'] > 0
    
    def test_suggest_improvements_slow_compression(self):
        """Test suggestions for slow compression"""
        config = ConfigPresets.balanced()
        metrics = {
            'compression_speed_mbps': 50,  # Slow compression
            'parsing_speed_files_per_sec': 100,
            'estimated_memory_gb': 2,
            'estimated_throughput_files_per_min': 5000
        }
        
        suggestions = PerformanceTuning.suggest_improvements(config, metrics)
        
        assert len(suggestions) > 0
        assert any('compression_level' in suggestion for suggestion in suggestions)
    
    def test_suggest_improvements_high_memory(self):
        """Test suggestions for high memory usage"""
        config = ConfigPresets.balanced()
        metrics = {
            'compression_speed_mbps': 200,
            'parsing_speed_files_per_sec': 100,
            'estimated_memory_gb': 12,  # High memory usage
            'estimated_throughput_files_per_min': 5000
        }
        
        suggestions = PerformanceTuning.suggest_improvements(config, metrics)
        
        assert len(suggestions) > 0
        assert any('memory usage' in suggestion for suggestion in suggestions)
    
    def test_suggest_improvements_low_throughput(self):
        """Test suggestions for low throughput"""
        config = ConfigPresets.balanced()
        metrics = {
            'compression_speed_mbps': 200,
            'parsing_speed_files_per_sec': 100,
            'estimated_memory_gb': 2,
            'estimated_throughput_files_per_min': 500  # Low throughput
        }
        
        suggestions = PerformanceTuning.suggest_improvements(config, metrics)
        
        assert len(suggestions) > 0
        assert any('throughput' in suggestion for suggestion in suggestions)
    
    def test_suggest_improvements_suboptimal_strategy(self):
        """Test suggestions for suboptimal compression strategy"""
        config = PipelineConfig(
            compression_strategy='full',
            max_context_size=32000  # Small context with full compression
        )
        metrics = {
            'compression_speed_mbps': 200,
            'parsing_speed_files_per_sec': 100,
            'estimated_memory_gb': 2,
            'estimated_throughput_files_per_min': 5000
        }
        
        suggestions = PerformanceTuning.suggest_improvements(config, metrics)
        
        assert len(suggestions) > 0
        assert any('structural' in suggestion for suggestion in suggestions)


class TestOptimalConfig:
    """Test optimal configuration selection"""
    
    def test_get_optimal_config_no_constraints(self):
        """Test optimal configuration without constraints"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('hello')")
            
            with patch('psutil.virtual_memory') as mock_mem, \
                 patch('os.cpu_count', return_value=4):
                mock_memory = Mock()
                mock_memory.total = 8 * 1024 * 1024 * 1024
                mock_memory.available = 6 * 1024 * 1024 * 1024
                mock_mem.return_value = mock_memory
                
                config = get_optimal_config(temp_path)
                assert isinstance(config, PipelineConfig)
    
    def test_get_optimal_config_memory_constraint(self):
        """Test optimal configuration with memory constraint"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('hello')")
            
            constraints = {'max_memory_gb': 2}
            
            with patch('psutil.virtual_memory') as mock_mem, \
                 patch('os.cpu_count', return_value=8):
                mock_memory = Mock()
                mock_memory.total = 16 * 1024 * 1024 * 1024
                mock_memory.available = 12 * 1024 * 1024 * 1024
                mock_mem.return_value = mock_memory
                
                config = get_optimal_config(temp_path, constraints)
                
                # Should limit workers based on memory constraint
                max_workers_for_constraint = int(2 * 1024 * 1024 * 1024 / config.max_memory_per_worker)
                assert config.num_workers <= max_workers_for_constraint
    
    def test_get_optimal_config_time_constraint(self):
        """Test optimal configuration with time constraint"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('hello')")
            
            constraints = {'max_time_minutes': 5}
            
            config = get_optimal_config(temp_path, constraints)
            
            # Should optimize for speed
            assert config.compression_level <= 6
            assert config.deduplication_enabled is False
            assert config.chunk_strategy == 'size'
    
    def test_get_optimal_config_output_size_constraint(self):
        """Test optimal configuration with output size constraint"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "test.py").write_text("print('hello')")
            
            constraints = {'output_size_mb': 10}
            
            config = get_optimal_config(temp_path, constraints)
            
            # Should maximize compression
            assert config.compression_level == 12
            assert config.compression_strategy == 'summary'


if __name__ == '__main__':
    pytest.main([__file__])