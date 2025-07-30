"""
Unit tests for pipeline stages
==============================

Tests for pipeline/stages/ including:
- OutputFormatter functionality
- SelectiveCompressor query-based compression
- Pipeline stage integration and error handling
- Output formatting and chunking strategies
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
import xml.etree.ElementTree as ET

# Mock the dependencies
import sys
sys.modules['pipeline.stages.metadata'] = MagicMock()

# Import the classes to test
from pipeline.stages.formatting import OutputFormatter
from pipeline.stages.selection import SelectiveCompressor


class TestOutputFormatter:
    """Test OutputFormatter functionality"""
    
    @pytest.fixture
    def formatter(self):
        """Create an OutputFormatter instance for testing"""
        return OutputFormatter(max_context_size=10000)
    
    @pytest.fixture
    def sample_compressed_data(self):
        """Create sample compressed data for testing"""
        return [
            {
                'path': '/test/file1.py',
                'language': 'python',
                'size': 1024,
                'complexity_score': 5.5,
                'imports': ['os', 'sys'],
                'exports': ['main_function'],
                'functions': ['main_function', 'helper'],
                'classes': ['TestClass'],
                'content_summary': 'Main application file with core functionality',
                'code_signature': 'def main_function(): ...',
                'compressed_content': 'import os\ndef main_function():\n    return "hello"'
            },
            {
                'path': '/test/utils.js',
                'language': 'javascript',
                'size': 512,
                'complexity_score': 2.0,
                'imports': ['fs', 'path'],
                'exports': ['readFile', 'writeFile'],
                'functions': ['readFile', 'writeFile'],
                'classes': [],
                'content_summary': 'Utility functions for file operations',
                'code_signature': 'function readFile(path) { ... }',
                'compressed_content': 'const fs = require("fs");\nfunction readFile(path) { return fs.readFileSync(path); }'
            },
            {
                'path': '/test/config.json',
                'language': 'json',
                'size': 256,
                'complexity_score': 0.0,
                'imports': [],
                'exports': [],
                'functions': [],
                'classes': [],
                'content_summary': 'Configuration file with app settings',
                'code_signature': '{"name": "test-app", ...}',
                'compressed_content': '{"name": "test-app", "version": "1.0.0"}'
            }
        ]
    
    @pytest.fixture
    def sample_codebase_map(self):
        """Create sample codebase map for testing"""
        return {
            'total_files': 3,
            'total_size': 1792,
            'languages': {'python': 1, 'javascript': 1, 'json': 1},
            'directory_structure': {
                '/test': ['file1.py', 'utils.js', 'config.json']
            },
            'dependency_graph': {
                '/test/file1.py': ['os', 'sys'],
                '/test/utils.js': ['fs', 'path']
            }
        }
    
    def test_formatter_initialization(self, formatter):
        """Test formatter initialization"""
        assert formatter.max_context_size == 10000
        assert 'markdown' in formatter.format_templates
        assert 'json' in formatter.format_templates
        assert 'xml' in formatter.format_templates
        assert 'custom' in formatter.format_templates
    
    def test_format_output_markdown_default(self, formatter, sample_compressed_data):
        """Test markdown formatting (default)"""
        chunks = formatter.format_output(sample_compressed_data, format_type='markdown')
        
        assert len(chunks) > 0
        assert isinstance(chunks[0], str)
        
        # Should contain markdown formatting
        markdown_content = chunks[0]
        assert '# ' in markdown_content or '## ' in markdown_content
        assert 'file1.py' in markdown_content
        assert 'python' in markdown_content
    
    def test_format_output_json(self, formatter, sample_compressed_data):
        """Test JSON formatting"""
        chunks = formatter.format_output(sample_compressed_data, format_type='json')
        
        assert len(chunks) > 0
        
        # Should be valid JSON
        json_content = json.loads(chunks[0])
        assert 'files' in json_content
        assert len(json_content['files']) == 3
        assert json_content['files'][0]['path'] == '/test/file1.py'
    
    def test_format_output_xml(self, formatter, sample_compressed_data):
        """Test XML formatting"""
        chunks = formatter.format_output(sample_compressed_data, format_type='xml')
        
        assert len(chunks) > 0
        
        # Should be valid XML
        root = ET.fromstring(chunks[0])
        assert root.tag == 'codebase'
        
        files = root.findall('file')
        assert len(files) == 3
        assert files[0].get('path') == '/test/file1.py'
    
    def test_format_output_with_codebase_map(self, formatter, sample_compressed_data, sample_codebase_map):
        """Test formatting with codebase map"""
        chunks = formatter.format_output(
            sample_compressed_data, 
            format_type='markdown',
            codebase_map=sample_codebase_map
        )
        
        assert len(chunks) > 0
        
        # Should include codebase overview
        content = chunks[0]
        assert 'Codebase Overview' in content or 'total_files' in content
        assert '3' in content  # Total files count
    
    def test_format_markdown_single_file(self, formatter):
        """Test markdown formatting for single file"""
        file_data = {
            'path': '/test/sample.py',
            'language': 'python',
            'size': 500,
            'complexity_score': 3.0,
            'imports': ['os'],
            'functions': ['main'],
            'classes': ['App'],
            'content_summary': 'Sample Python application',
            'compressed_content': 'def main():\n    print("Hello")'
        }
        
        formatted = formatter._format_markdown(file_data)
        
        assert '## /test/sample.py' in formatted
        assert 'python' in formatted
        assert 'complexity_score: 3.0' in formatted
        assert 'def main():' in formatted
    
    def test_format_json_single_file(self, formatter):
        """Test JSON formatting for single file"""
        file_data = {
            'path': '/test/sample.js',
            'language': 'javascript',
            'functions': ['test'],
            'compressed_content': 'function test() { return true; }'
        }
        
        formatted = formatter._format_json(file_data)
        json_data = json.loads(formatted)
        
        assert json_data['path'] == '/test/sample.js'
        assert json_data['language'] == 'javascript'
        assert 'test' in json_data['functions']
    
    def test_format_xml_single_file(self, formatter):
        """Test XML formatting for single file"""
        file_data = {
            'path': '/test/sample.cpp',
            'language': 'cpp',
            'classes': ['TestClass'],
            'compressed_content': 'class TestClass {};'
        }
        
        formatted = formatter._format_xml(file_data)
        root = ET.fromstring(formatted)
        
        assert root.tag == 'file'
        assert root.get('path') == '/test/sample.cpp'
        assert root.get('language') == 'cpp'
    
    def test_format_custom_single_file(self, formatter):
        """Test custom formatting for single file"""
        file_data = {
            'path': '/test/sample.go',
            'language': 'go',
            'complexity_score': 4.5,
            'imports': ['fmt'],
            'functions': ['main'],
            'compressed_content': 'package main\nfunc main() {}'
        }
        
        formatted = formatter._format_custom(file_data)
        
        # Custom format should be concise and LLM-optimized
        assert '/test/sample.go' in formatted
        assert 'go' in formatted
        assert 'fmt' in formatted
        assert 'main' in formatted
    
    def test_semantic_chunking(self, formatter, sample_compressed_data):
        """Test semantic chunking strategy"""
        formatted_items = [formatter._format_markdown(item) for item in sample_compressed_data]
        
        chunks = formatter._semantic_chunking(formatted_items, sample_compressed_data)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
    
    def test_size_based_chunking(self, formatter, sample_compressed_data):
        """Test size-based chunking strategy"""
        formatted_items = [formatter._format_markdown(item) for item in sample_compressed_data]
        
        chunks = formatter._size_based_chunking(formatted_items)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
    
    def test_balanced_chunking(self, formatter, sample_compressed_data):
        """Test balanced chunking strategy"""
        formatted_items = [formatter._format_markdown(item) for item in sample_compressed_data]
        
        chunks = formatter._balanced_chunking(formatted_items, sample_compressed_data)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
    
    def test_chunking_with_oversized_content(self, formatter):
        """Test chunking with content that exceeds max context size"""
        # Create oversized content
        large_content = "x" * (formatter.max_context_size + 1000)
        large_data = [{
            'path': '/test/large.py',
            'language': 'python',
            'compressed_content': large_content
        }]
        
        chunks = formatter.format_output(large_data, chunk_strategy='size')
        
        # Should split oversized content
        assert len(chunks) > 1
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
    
    def test_format_codebase_map(self, formatter, sample_codebase_map):
        """Test codebase map formatting"""
        formatted = formatter._format_codebase_map(sample_codebase_map)
        
        assert 'Codebase Overview' in formatted
        assert 'Total Files: 3' in formatted
        assert 'python: 1' in formatted
        assert 'javascript: 1' in formatted
    
    def test_format_output_empty_data(self, formatter):
        """Test formatting with empty data"""
        chunks = formatter.format_output([])
        
        # Should handle empty data gracefully
        assert isinstance(chunks, list)
        # May be empty or contain placeholder content
    
    def test_format_output_invalid_format_type(self, formatter, sample_compressed_data):
        """Test formatting with invalid format type"""
        # Should fall back to markdown
        chunks = formatter.format_output(sample_compressed_data, format_type='invalid_format')
        
        assert len(chunks) > 0
        # Should contain markdown-style formatting as fallback
        assert '#' in chunks[0] or '*' in chunks[0]
    
    def test_format_output_invalid_chunk_strategy(self, formatter, sample_compressed_data):
        """Test formatting with invalid chunk strategy"""
        # Should fall back to semantic chunking
        chunks = formatter.format_output(
            sample_compressed_data, 
            chunk_strategy='invalid_strategy'
        )
        
        assert len(chunks) > 0


class TestSelectiveCompressor:
    """Test SelectiveCompressor functionality"""
    
    @pytest.fixture
    def mock_metadata_store(self):
        """Create a mock MetadataStore for testing"""
        store = Mock()
        
        # Mock query results
        store.query.return_value = [0, 1, 2]  # Indices
        
        # Mock metadata
        sample_metadata = [
            {
                'path': '/test/file1.py',
                'language': 'python',
                'size': 1024,
                'complexity_score': 5.5,
                'imports': ['os', 'sys'],
                'functions': ['main', 'helper'],
                'classes': ['App'],
                'content': 'import os\ndef main():\n    pass\nclass App:\n    pass'
            },
            {
                'path': '/test/utils.js',
                'language': 'javascript',
                'size': 512,
                'complexity_score': 2.0,
                'imports': ['fs'],
                'functions': ['readFile'],
                'classes': [],
                'content': 'const fs = require("fs");\nfunction readFile(path) { return fs.readFileSync(path); }'
            },
            {
                'path': '/test/config.json',
                'language': 'json',
                'size': 128,
                'complexity_score': 0.0,
                'imports': [],
                'functions': [],
                'classes': [],
                'content': '{"name": "test-app", "version": "1.0.0"}'
            }
        ]
        
        store.get_metadata_by_indices.return_value = sample_metadata
        
        return store
    
    @pytest.fixture
    def compressor(self, mock_metadata_store):
        """Create a SelectiveCompressor instance for testing"""
        return SelectiveCompressor(mock_metadata_store)
    
    def test_compressor_initialization(self, compressor, mock_metadata_store):
        """Test compressor initialization"""
        assert compressor.metadata_store == mock_metadata_store
        assert 'full' in compressor.compression_strategies
        assert 'structural' in compressor.compression_strategies
        assert 'signature' in compressor.compression_strategies
        assert 'summary' in compressor.compression_strategies
    
    def test_compress_by_query_default_strategy(self, compressor):
        """Test compression by query with default strategy"""
        query = {'language': 'python', 'min_complexity': 5.0}
        
        results = compressor.compress_by_query(query)
        
        assert len(results) == 3  # Should return 3 compressed files
        assert all('path' in result for result in results)
        
        # Should use structural compression by default
        compressor.metadata_store.query.assert_called_once_with(language='python', min_complexity=5.0)
    
    def test_compress_by_query_full_strategy(self, compressor):
        """Test compression with full strategy"""
        query = {'language': 'python'}
        
        results = compressor.compress_by_query(query, strategy='full')
        
        assert len(results) == 3
        
        # Full compression should preserve original content
        for result in results:
            assert 'content' in result
            assert 'path' in result
            assert result['compression_strategy'] == 'full'
    
    def test_compress_by_query_structural_strategy(self, compressor):
        """Test compression with structural strategy"""
        query = {'complexity_score__gt': 1.0}
        
        results = compressor.compress_by_query(query, strategy='structural')
        
        assert len(results) == 3
        
        # Structural compression should preserve structure
        for result in results:
            assert 'imports' in result
            assert 'functions' in result
            assert 'classes' in result
            assert result['compression_strategy'] == 'structural'
    
    def test_compress_by_query_signature_strategy(self, compressor):
        """Test compression with signature strategy"""
        query = {'language': 'javascript'}
        
        results = compressor.compress_by_query(query, strategy='signature')
        
        assert len(results) == 3
        
        # Signature compression should extract key signatures
        for result in results:
            assert 'code_signature' in result
            assert result['compression_strategy'] == 'signature'
    
    def test_compress_by_query_summary_strategy(self, compressor):
        """Test compression with summary strategy"""
        query = {'size__lt': 1000}
        
        results = compressor.compress_by_query(query, strategy='summary')
        
        assert len(results) == 3
        
        # Summary compression should provide high-level overview
        for result in results:
            assert 'content_summary' in result
            assert result['compression_strategy'] == 'summary'
    
    def test_compress_full_single_file(self, compressor):
        """Test full compression for single file"""
        metadata = {
            'path': '/test/sample.py',
            'language': 'python',
            'size': 500,
            'content': 'def main():\n    print("Hello World")\n\nif __name__ == "__main__":\n    main()'
        }
        
        result = compressor._compress_full(metadata)
        
        assert result['path'] == '/test/sample.py'
        assert result['compression_strategy'] == 'full'
        assert result['content'] == metadata['content']
        assert result['size'] == 500
    
    def test_compress_structural_single_file(self, compressor):
        """Test structural compression for single file"""
        metadata = {
            'path': '/test/sample.py',
            'language': 'python',
            'imports': ['os', 'sys'],
            'functions': ['main', 'helper'],
            'classes': ['TestClass'],
            'content': 'import os\nimport sys\n\ndef main():\n    pass\n\ndef helper():\n    pass\n\nclass TestClass:\n    def method(self):\n        pass'
        }
        
        result = compressor._compress_structural(metadata)
        
        assert result['compression_strategy'] == 'structural'
        assert result['imports'] == ['os', 'sys']
        assert result['functions'] == ['main', 'helper']
        assert result['classes'] == ['TestClass']
        assert 'structural_summary' in result
    
    def test_compress_signature_single_file(self, compressor):
        """Test signature compression for single file"""
        metadata = {
            'path': '/test/sample.js',
            'language': 'javascript',
            'functions': ['readFile', 'writeFile'],
            'content': 'function readFile(path) {\n    return fs.readFileSync(path);\n}\n\nfunction writeFile(path, data) {\n    fs.writeFileSync(path, data);\n}'
        }
        
        result = compressor._compress_signature(metadata)
        
        assert result['compression_strategy'] == 'signature'
        assert 'code_signature' in result
        assert 'readFile' in result['code_signature']
        assert 'writeFile' in result['code_signature']
    
    def test_compress_summary_single_file(self, compressor):
        """Test summary compression for single file"""
        metadata = {
            'path': '/test/config.json',
            'language': 'json',
            'size': 256,
            'complexity_score': 0.0,
            'content': '{"name": "test-app", "version": "1.0.0", "dependencies": {}}'
        }
        
        result = compressor._compress_summary(metadata)
        
        assert result['compression_strategy'] == 'summary'
        assert 'content_summary' in result
        assert 'key_characteristics' in result
        assert result['size'] == 256
    
    def test_compress_by_query_empty_results(self, compressor):
        """Test compression with query that returns no results"""
        # Mock empty query results
        compressor.metadata_store.query.return_value = []
        compressor.metadata_store.get_metadata_by_indices.return_value = []
        
        query = {'language': 'nonexistent'}
        results = compressor.compress_by_query(query)
        
        assert len(results) == 0
    
    def test_compress_by_query_invalid_strategy(self, compressor):
        """Test compression with invalid strategy"""
        query = {'language': 'python'}
        
        # Should fall back to structural compression
        results = compressor.compress_by_query(query, strategy='invalid_strategy')
        
        assert len(results) == 3
        # Should use structural as fallback
        for result in results:
            assert result['compression_strategy'] == 'structural'
    
    def test_extract_function_signatures(self, compressor):
        """Test function signature extraction"""
        python_content = """
def simple_function():
    pass

def function_with_args(arg1, arg2):
    return arg1 + arg2

class TestClass:
    def method(self, param):
        return param * 2
    
    async def async_method(self):
        await something()
"""
        
        signatures = compressor._extract_function_signatures(python_content, 'python')
        
        assert 'simple_function()' in signatures
        assert 'function_with_args(arg1, arg2)' in signatures
        assert 'method(self, param)' in signatures
        assert 'async_method(self)' in signatures
    
    def test_extract_function_signatures_javascript(self, compressor):
        """Test JavaScript function signature extraction"""
        js_content = """
function regularFunction(param1, param2) {
    return param1 + param2;
}

const arrowFunction = (x, y) => {
    return x * y;
};

class MyClass {
    constructor(name) {
        this.name = name;
    }
    
    method(param) {
        return this.name + param;
    }
}
"""
        
        signatures = compressor._extract_function_signatures(js_content, 'javascript')
        
        assert 'regularFunction(param1, param2)' in signatures
        assert 'arrowFunction(x, y)' in signatures
        assert 'constructor(name)' in signatures
        assert 'method(param)' in signatures
    
    def test_generate_content_summary(self, compressor):
        """Test content summary generation"""
        metadata = {
            'path': '/test/app.py',
            'language': 'python',
            'size': 2048,
            'complexity_score': 8.5,
            'imports': ['os', 'sys', 'json', 'logging'],
            'functions': ['main', 'process_data', 'save_results'],
            'classes': ['DataProcessor', 'ResultsHandler'],
            'content': 'A complex Python application with data processing capabilities'
        }
        
        summary = compressor._generate_content_summary(metadata)
        
        assert 'python' in summary.lower()
        assert 'functions' in summary.lower()
        assert 'classes' in summary.lower()
        assert str(metadata['size']) in summary
    
    def test_error_handling_in_compression(self, compressor):
        """Test error handling during compression"""
        # Test with malformed metadata
        invalid_metadata = {
            'path': '/test/broken.py',
            # Missing required fields
        }
        
        # Should handle missing fields gracefully
        result = compressor._compress_structural(invalid_metadata)
        
        assert 'path' in result
        assert result['compression_strategy'] == 'structural'
        # Should provide default values for missing fields
        assert 'imports' in result
        assert 'functions' in result
        assert 'classes' in result


class TestPipelineStageIntegration:
    """Test integration between pipeline stages"""
    
    def test_formatter_compressor_integration(self):
        """Test integration between SelectiveCompressor and OutputFormatter"""
        # Mock metadata store
        mock_store = Mock()
        mock_store.query.return_value = [0]
        mock_store.get_metadata_by_indices.return_value = [{
            'path': '/test/integration.py',
            'language': 'python',
            'size': 1024,
            'complexity_score': 5.0,
            'imports': ['os'],
            'functions': ['main'],
            'classes': ['App'],
            'content': 'import os\ndef main():\n    pass\nclass App:\n    pass'
        }]
        
        # Create compressor and get compressed data
        compressor = SelectiveCompressor(mock_store)
        compressed_data = compressor.compress_by_query({'language': 'python'}, strategy='structural')
        
        # Format the compressed data
        formatter = OutputFormatter(max_context_size=5000)
        formatted_chunks = formatter.format_output(compressed_data, format_type='markdown')
        
        assert len(formatted_chunks) > 0
        assert 'integration.py' in formatted_chunks[0]
        assert 'python' in formatted_chunks[0]
    
    def test_stage_error_propagation(self):
        """Test error propagation between stages"""
        # Test formatter with invalid data from compressor
        formatter = OutputFormatter()
        
        invalid_compressed_data = [
            {
                'path': '/test/invalid.py',
                # Missing required fields for formatting
            }
        ]
        
        # Should handle missing fields gracefully
        chunks = formatter.format_output(invalid_compressed_data)
        
        assert isinstance(chunks, list)
        # Should still produce some output even with invalid data
    
    def test_memory_efficiency_across_stages(self):
        """Test memory efficiency when processing large datasets across stages"""
        # Create large dataset
        large_dataset = []
        for i in range(1000):
            large_dataset.append({
                'path': f'/test/file_{i}.py',
                'language': 'python',
                'size': 1024,
                'complexity_score': i % 10,
                'imports': ['os', 'sys'],
                'functions': [f'function_{i}'],
                'classes': [f'Class_{i}'],
                'content': f'def function_{i}():\n    pass\nclass Class_{i}:\n    pass'
            })
        
        # Process through formatter with size-based chunking
        formatter = OutputFormatter(max_context_size=50000)  # Smaller chunks
        chunks = formatter.format_output(large_dataset, chunk_strategy='size')
        
        # Should create multiple chunks to manage memory
        assert len(chunks) > 1
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
        
        # Total content should be preserved (approximately)
        total_chunk_length = sum(len(chunk) for chunk in chunks)
        assert total_chunk_length > 0


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases in pipeline stages"""
    
    def test_formatter_with_none_values(self):
        """Test formatter handling None values in data"""
        formatter = OutputFormatter()
        
        data_with_nones = [{
            'path': '/test/file.py',
            'language': None,
            'imports': None,
            'functions': None,
            'content': None
        }]
        
        chunks = formatter.format_output(data_with_nones)
        
        assert len(chunks) > 0
        # Should handle None values gracefully
        assert 'file.py' in chunks[0]
    
    def test_formatter_with_empty_strings(self):
        """Test formatter handling empty string values"""
        formatter = OutputFormatter()
        
        data_with_empty = [{
            'path': '',
            'language': '',
            'content': '',
            'imports': [],
            'functions': []
        }]
        
        chunks = formatter.format_output(data_with_empty)
        
        assert len(chunks) > 0
        # Should handle empty values gracefully
    
    def test_compressor_with_malformed_content(self):
        """Test compressor handling malformed content"""
        mock_store = Mock()
        mock_store.query.return_value = [0]
        mock_store.get_metadata_by_indices.return_value = [{
            'path': '/test/malformed.py',
            'language': 'python',
            'content': 'def incomplete_function(\n    # Malformed Python code'
        }]
        
        compressor = SelectiveCompressor(mock_store)
        
        # Should handle malformed content gracefully
        results = compressor.compress_by_query({'language': 'python'}, strategy='signature')
        
        assert len(results) > 0
        assert 'path' in results[0]
    
    def test_unicode_handling(self):
        """Test handling of unicode content"""
        formatter = OutputFormatter()
        
        unicode_data = [{
            'path': '/test/unicode.py',
            'language': 'python',
            'content': 'def greet():\n    print("Hello 世界! 🌍")\n    return "Ñice dây"',
            'functions': ['greet']
        }]
        
        chunks = formatter.format_output(unicode_data)
        
        assert len(chunks) > 0
        # Should preserve unicode characters
        assert '世界' in chunks[0]
        assert '🌍' in chunks[0]
        assert 'Ñice' in chunks[0]
    
    def test_very_large_single_file(self):
        """Test handling of very large single files"""
        formatter = OutputFormatter(max_context_size=1000)  # Small limit
        
        large_content = "# Very large file\n" + "x = 1\n" * 1000  # Large file
        large_file_data = [{
            'path': '/test/large.py',
            'language': 'python',
            'content': large_content,
            'size': len(large_content)
        }]
        
        chunks = formatter.format_output(large_file_data, chunk_strategy='size')
        
        # Should split large file into multiple chunks
        assert len(chunks) > 1
        assert all(len(chunk) <= formatter.max_context_size for chunk in chunks)
    
    def test_concurrent_access_safety(self):
        """Test thread safety of formatters and compressors"""
        import threading
        import time
        
        formatter = OutputFormatter()
        results = []
        errors = []
        
        def worker(worker_id):
            try:
                data = [{
                    'path': f'/test/worker_{worker_id}.py',
                    'language': 'python',
                    'content': f'def worker_{worker_id}():\n    pass'
                }]
                chunks = formatter.format_output(data)
                results.append((worker_id, len(chunks)))
            except Exception as e:
                errors.append((worker_id, e))
        
        # Run multiple workers concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 10


if __name__ == '__main__':
    pytest.main([__file__])