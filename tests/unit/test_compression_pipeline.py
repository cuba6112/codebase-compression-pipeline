"""
Unit tests for main compression pipeline
=======================================

Tests for codebase_compression_pipeline.py including:
- CodebaseCompressionPipeline main functionality
- Error handling and resilience 
- Async processing capabilities
- Stream processing edge cases
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

# Mock the imports to avoid dependency issues in tests
import sys
from unittest.mock import MagicMock

# Mock the base classes and dependencies
sys.modules['base_classes'] = MagicMock()
sys.modules['enhanced_cache'] = MagicMock()
sys.modules['parsers.enhanced_js_parser'] = MagicMock()
sys.modules['parsers.python_parser'] = MagicMock()
sys.modules['pipeline.stages.metadata'] = MagicMock()
sys.modules['resilience_patterns'] = MagicMock()
sys.modules['security_validation'] = MagicMock()

# Import after mocking
from codebase_compression_pipeline import (
    CodebaseCompressionPipeline, FileMetadata, PipelineMonitor
)


class TestFileMetadata:
    """Test FileMetadata dataclass functionality"""
    
    def test_file_metadata_creation(self):
        """Test basic FileMetadata creation"""
        metadata = FileMetadata(
            path=Path("/test/file.py"),
            size=1024,
            language="python",
            content_hash="abc123",
            complexity_score=5.0,
            imports=["os", "sys"],
            exports=["main_function"],
            functions=["func1", "func2"],
            classes=["TestClass"],
            last_modified=1234567890.0
        )
        
        assert metadata.path == Path("/test/file.py")
        assert metadata.size == 1024
        assert metadata.language == "python"
        assert metadata.content_hash == "abc123"
        assert metadata.complexity_score == 5.0
        assert len(metadata.imports) == 2
        assert len(metadata.exports) == 1
        assert len(metadata.functions) == 2
        assert len(metadata.classes) == 1
        assert metadata.last_modified == 1234567890.0
    
    def test_file_metadata_defaults(self):
        """Test FileMetadata with minimal required fields"""
        metadata = FileMetadata(
            path=Path("/test/simple.txt"),
            size=100,
            language="text"
        )
        
        assert metadata.content_hash is None
        assert metadata.complexity_score == 0.0
        assert metadata.imports == []
        assert metadata.exports == []
        assert metadata.functions == []
        assert metadata.classes == []
        assert metadata.last_modified is None
    
    def test_file_metadata_str_representation(self):
        """Test FileMetadata string representation"""
        metadata = FileMetadata(
            path=Path("/test/file.py"),
            size=1024,
            language="python"
        )
        
        str_repr = str(metadata)
        assert "file.py" in str_repr
        assert "python" in str_repr
        assert "1024" in str_repr


class TestPipelineMonitor:
    """Test PipelineMonitor functionality"""
    
    @pytest.fixture
    def monitor(self):
        """Create a PipelineMonitor instance for testing"""
        return PipelineMonitor()
    
    def test_monitor_initialization(self, monitor):
        """Test PipelineMonitor initialization"""
        assert monitor.start_time is not None
        assert monitor.files_processed == 0
        assert monitor.total_size == 0
        assert monitor.errors == []
        assert monitor.stage_metrics == {}
    
    def test_record_file(self, monitor):
        """Test recording processed file"""
        monitor.record_file(Path("/test/file.py"), 1024)
        
        assert monitor.files_processed == 1
        assert monitor.total_size == 1024
    
    def test_record_error(self, monitor):
        """Test recording errors"""
        test_error = Exception("Test error")
        monitor.record_error("test_stage", test_error)
        
        assert len(monitor.errors) == 1
        assert monitor.errors[0]["stage"] == "test_stage"
        assert "Test error" in str(monitor.errors[0]["error"])
    
    def test_record_stage_metric(self, monitor):
        """Test recording stage metrics"""
        monitor.record_stage_metric("parsing", "duration", 5.0)
        
        assert "parsing" in monitor.stage_metrics
        assert monitor.stage_metrics["parsing"]["duration"] == 5.0
    
    def test_get_summary(self, monitor):
        """Test getting pipeline summary"""
        monitor.record_file(Path("/test/file1.py"), 1024)
        monitor.record_file(Path("/test/file2.js"), 2048)
        monitor.record_error("parsing", Exception("Parse error"))
        
        summary = monitor.get_summary()
        
        assert summary["files_processed"] == 2
        assert summary["total_size"] == 3072
        assert summary["error_count"] == 1
        assert "duration" in summary
        assert "throughput_files_per_sec" in summary


class TestCodebaseCompressionPipeline:
    """Test main CodebaseCompressionPipeline functionality"""
    
    @pytest.fixture
    def temp_codebase(self):
        """Create a temporary codebase for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create sample files
            (temp_path / "main.py").write_text("""
import os
import sys

def main():
    print("Hello World")

class TestClass:
    def method(self):
        return "test"

if __name__ == "__main__":
    main()
""")
            
            (temp_path / "utils.js").write_text("""
const fs = require('fs');

function readFile(path) {
    return fs.readFileSync(path, 'utf8');
}

module.exports = { readFile };
""")
            
            (temp_path / "config.json").write_text("""
{
    "name": "test-project",
    "version": "1.0.0",
    "dependencies": {}
}
""")
            
            # Create subdirectory
            sub_dir = temp_path / "src"
            sub_dir.mkdir()
            (sub_dir / "helper.py").write_text("def helper(): return True")
            
            yield temp_path
    
    @pytest.fixture
    def mock_pipeline(self):
        """Create a mock pipeline for testing"""
        with patch('codebase_compression_pipeline.EnhancedIncrementalCache'), \
             patch('codebase_compression_pipeline.SecurityValidator'), \
             patch('codebase_compression_pipeline.AsyncMetadataStore'):
            
            pipeline = CodebaseCompressionPipeline(
                cache_dir=Path("/tmp/test_cache"),
                output_dir=Path("/tmp/test_output"),
                num_workers=2
            )
            yield pipeline
    
    def test_pipeline_initialization(self, mock_pipeline):
        """Test pipeline initialization"""
        assert mock_pipeline.cache_dir == Path("/tmp/test_cache")
        assert mock_pipeline.output_dir == Path("/tmp/test_output")
        assert mock_pipeline.num_workers == 2
        assert isinstance(mock_pipeline.monitor, PipelineMonitor)
    
    @pytest.mark.asyncio
    async def test_discover_files(self, mock_pipeline, temp_codebase):
        """Test file discovery functionality"""
        with patch.object(mock_pipeline, '_should_process_file', return_value=True):
            files = await mock_pipeline._discover_files(temp_codebase)
            
            assert len(files) >= 3  # At least main.py, utils.js, helper.py
            assert any(f.name == "main.py" for f in files)
            assert any(f.name == "utils.js" for f in files)
            assert any(f.name == "helper.py" for f in files)
    
    def test_should_process_file(self, mock_pipeline):
        """Test file filtering logic"""
        # Should process Python files
        assert mock_pipeline._should_process_file(Path("/test/file.py"))
        assert mock_pipeline._should_process_file(Path("/test/script.js"))
        
        # Should skip ignored patterns
        assert not mock_pipeline._should_process_file(Path("/test/__pycache__/file.pyc"))
        assert not mock_pipeline._should_process_file(Path("/test/node_modules/package.json"))
        assert not mock_pipeline._should_process_file(Path("/test/.git/config"))
        assert not mock_pipeline._should_process_file(Path("/test/dist/bundle.js"))
    
    def test_get_ignore_patterns(self, mock_pipeline):
        """Test ignore patterns generation"""
        patterns = mock_pipeline._get_ignore_patterns()
        
        assert "__pycache__" in patterns
        assert "node_modules" in patterns
        assert ".git" in patterns
        assert "*.pyc" in patterns
        assert "dist" in patterns
        assert "build" in patterns
    
    @pytest.mark.asyncio
    async def test_process_single_file_success(self, mock_pipeline):
        """Test successful single file processing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            f.flush()
            
            file_path = Path(f.name)
            
            with patch.object(mock_pipeline, '_parse_file') as mock_parse:
                mock_parse.return_value = FileMetadata(
                    path=file_path,
                    size=100,
                    language="python",
                    functions=["test"]
                )
                
                metadata = await mock_pipeline._process_single_file(file_path)
                
                assert metadata is not None
                assert metadata.path == file_path
                assert metadata.language == "python"
                assert "test" in metadata.functions
    
    @pytest.mark.asyncio
    async def test_process_single_file_error_handling(self, mock_pipeline):
        """Test error handling in single file processing"""
        non_existent_file = Path("/non/existent/file.py")
        
        # Should handle file not found gracefully
        metadata = await mock_pipeline._process_single_file(non_existent_file)
        assert metadata is None
        
        # Should record error in monitor
        assert len(mock_pipeline.monitor.errors) > 0
    
    @pytest.mark.asyncio
    async def test_process_single_file_permission_error(self, mock_pipeline):
        """Test handling permission errors"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test(): pass")
            f.flush()
            file_path = Path(f.name)
        
        with patch('pathlib.Path.read_text', side_effect=PermissionError("Access denied")):
            metadata = await mock_pipeline._process_single_file(file_path)
            assert metadata is None
            
            # Should record permission error
            assert any("permission" in str(error["error"]).lower() 
                      for error in mock_pipeline.monitor.errors)
    
    def test_detect_language(self, mock_pipeline):
        """Test language detection"""
        assert mock_pipeline._detect_language(Path("test.py")) == "python"
        assert mock_pipeline._detect_language(Path("test.js")) == "javascript"
        assert mock_pipeline._detect_language(Path("test.ts")) == "typescript"
        assert mock_pipeline._detect_language(Path("test.go")) == "go"
        assert mock_pipeline._detect_language(Path("test.rs")) == "rust"
        assert mock_pipeline._detect_language(Path("test.java")) == "java"
        assert mock_pipeline._detect_language(Path("test.cpp")) == "cpp"
        assert mock_pipeline._detect_language(Path("test.unknown")) == "text"
    
    def test_parse_file_python(self, mock_pipeline):
        """Test Python file parsing"""
        python_content = """
import os
import sys

def main():
    '''Main function'''
    print("Hello World")

class TestClass:
    '''Test class'''
    def method(self):
        return "test"
"""
        
        with patch('codebase_compression_pipeline.PythonParser') as MockParser:
            mock_parser = Mock()
            mock_parser.parse.return_value = {
                'imports': ['os', 'sys'],
                'functions': ['main'],
                'classes': ['TestClass'],
                'complexity_score': 3.0
            }
            MockParser.return_value = mock_parser
            
            metadata = mock_pipeline._parse_file(
                Path("/test/file.py"), 
                python_content, 
                "python"
            )
            
            assert metadata.language == "python"
            assert metadata.imports == ['os', 'sys']
            assert metadata.functions == ['main']
            assert metadata.classes == ['TestClass']
            assert metadata.complexity_score == 3.0
    
    def test_parse_file_javascript(self, mock_pipeline):
        """Test JavaScript file parsing"""
        js_content = """
const fs = require('fs');
const path = require('path');

function readFile(filePath) {
    return fs.readFileSync(filePath, 'utf8');
}

module.exports = { readFile };
"""
        
        with patch('codebase_compression_pipeline.EnhancedJavaScriptParser') as MockParser:
            mock_parser = Mock()
            mock_parser.parse.return_value = {
                'imports': ['fs', 'path'],
                'functions': ['readFile'],
                'exports': ['readFile'],
                'complexity_score': 2.0
            }
            MockParser.return_value = mock_parser
            
            metadata = mock_pipeline._parse_file(
                Path("/test/file.js"), 
                js_content, 
                "javascript"
            )
            
            assert metadata.language == "javascript"
            assert metadata.imports == ['fs', 'path']
            assert metadata.functions == ['readFile']
            assert metadata.exports == ['readFile']
            assert metadata.complexity_score == 2.0
    
    def test_parse_file_unsupported_language(self, mock_pipeline):
        """Test parsing unsupported language file"""
        metadata = mock_pipeline._parse_file(
            Path("/test/file.unknown"), 
            "some content", 
            "unknown"
        )
        
        assert metadata.language == "unknown"
        assert metadata.imports == []
        assert metadata.functions == []
        assert metadata.classes == []
        assert metadata.complexity_score == 0.0
    
    def test_parse_file_error_handling(self, mock_pipeline):
        """Test parse file error handling"""
        with patch('codebase_compression_pipeline.PythonParser') as MockParser:
            mock_parser = Mock()
            mock_parser.parse.side_effect = Exception("Parse error")
            MockParser.return_value = mock_parser
            
            metadata = mock_pipeline._parse_file(
                Path("/test/file.py"), 
                "invalid python", 
                "python"
            )
            
            # Should return basic metadata even on parse error
            assert metadata.language == "python"
            assert metadata.complexity_score == 0.0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, mock_pipeline):
        """Test batch file processing"""
        files = [
            Path("/test/file1.py"),
            Path("/test/file2.js"),
            Path("/test/file3.py")
        ]
        
        with patch.object(mock_pipeline, '_process_single_file') as mock_process:
            mock_process.side_effect = [
                FileMetadata(path=files[0], size=100, language="python"),
                FileMetadata(path=files[1], size=200, language="javascript"),
                FileMetadata(path=files[2], size=150, language="python")
            ]
            
            results = await mock_pipeline._process_batch(files)
            
            assert len(results) == 3
            assert all(metadata is not None for metadata in results)
            assert mock_process.call_count == 3
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_errors(self, mock_pipeline):
        """Test batch processing with some file errors"""
        files = [
            Path("/test/file1.py"),
            Path("/test/error_file.py"),
            Path("/test/file3.js")
        ]
        
        with patch.object(mock_pipeline, '_process_single_file') as mock_process:
            mock_process.side_effect = [
                FileMetadata(path=files[0], size=100, language="python"),
                None,  # Error case
                FileMetadata(path=files[2], size=200, language="javascript")
            ]
            
            results = await mock_pipeline._process_batch(files)
            
            # Should return valid results and filter out None values
            valid_results = [r for r in results if r is not None]
            assert len(valid_results) == 2
    
    def test_calculate_content_hash(self, mock_pipeline):
        """Test content hash calculation"""
        content1 = "def test(): pass"
        content2 = "def test(): pass"
        content3 = "def other(): pass"
        
        hash1 = mock_pipeline._calculate_content_hash(content1)
        hash2 = mock_pipeline._calculate_content_hash(content2)
        hash3 = mock_pipeline._calculate_content_hash(content3)
        
        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be hex string
        assert isinstance(hash1, str)
        assert len(hash1) == 8  # mmh3 hash is 32-bit, 8 hex chars
    
    def test_cleanup(self, mock_pipeline):
        """Test pipeline cleanup"""
        # Should not raise any exceptions
        mock_pipeline.cleanup()
    
    @pytest.mark.asyncio
    async def test_process_codebase_integration(self, mock_pipeline, temp_codebase):
        """Test full codebase processing integration"""
        with patch.object(mock_pipeline, '_parse_file') as mock_parse, \
             patch.object(mock_pipeline, 'cache') as mock_cache, \
             patch.object(mock_pipeline, '_generate_output') as mock_output:
            
            # Mock parse to return basic metadata
            mock_parse.return_value = FileMetadata(
                path=Path("/test/file.py"),
                size=100,
                language="python"
            )
            
            mock_cache.get.return_value = None  # No cached data
            mock_cache.set = Mock()
            
            mock_output.return_value = ["output_file.md"]
            
            # Run the full process
            output_files = await mock_pipeline.process_codebase(
                codebase_path=temp_codebase,
                output_format='markdown',
                compression_strategy='structural'
            )
            
            assert isinstance(output_files, list)
            assert len(output_files) > 0


class TestPipelineErrorHandling:
    """Test error handling and resilience patterns"""
    
    @pytest.mark.asyncio
    async def test_file_processing_with_encoding_errors(self):
        """Test handling of file encoding errors"""
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.py', delete=False) as f:
            # Write invalid UTF-8 bytes
            f.write(b'\xff\xfe\x00\x00invalid utf-8')
            f.flush()
            file_path = Path(f.name)
        
        with patch('codebase_compression_pipeline.EnhancedIncrementalCache'), \
             patch('codebase_compression_pipeline.SecurityValidator'), \
             patch('codebase_compression_pipeline.AsyncMetadataStore'):
            
            pipeline = CodebaseCompressionPipeline(
                cache_dir=Path("/tmp/test_cache"),
                output_dir=Path("/tmp/test_output")
            )
            
            # Should handle encoding error gracefully
            metadata = await pipeline._process_single_file(file_path)
            
            # Should either return None or basic metadata depending on fallback
            if metadata is not None:
                assert metadata.path == file_path
    
    @pytest.mark.asyncio
    async def test_concurrent_file_processing_limits(self):
        """Test that concurrent processing respects limits"""
        files = [Path(f"/test/file{i}.py") for i in range(100)]
        
        with patch('codebase_compression_pipeline.EnhancedIncrementalCache'), \
             patch('codebase_compression_pipeline.SecurityValidator'), \
             patch('codebase_compression_pipeline.AsyncMetadataStore'):
            
            pipeline = CodebaseCompressionPipeline(
                cache_dir=Path("/tmp/test_cache"),
                output_dir=Path("/tmp/test_output"),
                num_workers=4
            )
            
            with patch.object(pipeline, '_process_single_file') as mock_process:
                mock_process.return_value = FileMetadata(
                    path=Path("/test/file.py"),
                    size=100,
                    language="python"
                )
                
                # Process batch - should limit concurrency
                results = await pipeline._process_batch(files[:10])
                
                assert len(results) == 10
                assert mock_process.call_count == 10
    
    def test_monitor_thread_safety(self):
        """Test that monitor operations are thread-safe"""
        import threading
        import time
        
        monitor = PipelineMonitor()
        errors = []
        
        def worker(worker_id):
            try:
                for i in range(100):
                    monitor.record_file(Path(f"/test/file_{worker_id}_{i}.py"), 1024)
                    monitor.record_stage_metric("parsing", f"worker_{worker_id}_metric", i)
                    if i % 10 == 0:
                        monitor.record_error("test", Exception(f"Error from worker {worker_id}"))
            except Exception as e:
                errors.append(e)
        
        # Run multiple workers concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should not have any thread safety errors
        assert len(errors) == 0
        
        # Should have processed all files
        assert monitor.files_processed == 500
        assert monitor.total_size == 500 * 1024


if __name__ == '__main__':
    pytest.main([__file__])