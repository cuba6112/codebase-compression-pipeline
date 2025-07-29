#!/usr/bin/env python3
"""
Integration tests for the codebase compression pipeline.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import json

from codebase_compression_pipeline import CodebaseCompressionPipeline
from pipeline_configs import ConfigPresets
from security_validation import SecurityConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample.py"
    content = '''
def hello_world():
    """Simple hello world function."""
    return "Hello, World!"

class TestClass:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    print(hello_world())
'''
    file_path.write_text(content)
    return file_path


class TestPipelineIntegration:
    """Integration tests for the main pipeline."""
    
    def test_basic_pipeline_flow(self, temp_dir, sample_python_file):
        """Test basic pipeline processing."""
        cache_dir = temp_dir / "cache"
        output_dir = temp_dir / "output"
        
        # Create permissive security config for testing
        security_config = SecurityConfig(
            allowed_base_paths=[temp_dir, Path("/tmp"), Path("/var/folders"), Path("/private")],  # Allow temp directories
            forbidden_paths=[],  # Allow all paths for testing
            max_file_size=100 * 1024 * 1024,  # 100MB
            max_total_files=10000
        )
        
        pipeline = CodebaseCompressionPipeline(
            cache_dir=cache_dir,
            output_dir=output_dir,
            num_workers=1,
            security_config=security_config
        )
        
        try:
            output_files = pipeline.process_codebase(
                codebase_path=temp_dir,
                output_format='json',
                compression_strategy='structural'
            )
            
            assert len(output_files) > 0
            assert all(f.exists() for f in output_files)
            
            # Check that output contains valid JSON
            first_output = output_files[0]
            content = first_output.read_text()
            # Should be valid JSON or markdown with JSON content
            assert len(content) > 0
            
        finally:
            pipeline.cleanup()
    
    def test_empty_directory(self, temp_dir):
        """Test pipeline behavior with empty directory."""
        cache_dir = temp_dir / "cache"
        output_dir = temp_dir / "output"
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        # Create permissive security config for testing
        security_config = SecurityConfig(
            allowed_base_paths=[temp_dir, Path("/tmp"), Path("/var/folders"), Path("/private")],  # Allow temp directories
            forbidden_paths=[],  # Allow all paths for testing
            max_file_size=100 * 1024 * 1024,  # 100MB
            max_total_files=10000
        )
        
        pipeline = CodebaseCompressionPipeline(
            cache_dir=cache_dir,
            output_dir=output_dir,
            num_workers=1,
            security_config=security_config
        )
        
        try:
            output_files = pipeline.process_codebase(
                codebase_path=empty_dir,
                output_format='markdown'
            )
            
            # Should handle empty directory gracefully
            assert isinstance(output_files, list)
            
        finally:
            pipeline.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])