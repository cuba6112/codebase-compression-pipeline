#!/usr/bin/env python3
"""
Unit tests for Python parser functionality.
"""

import pytest
from parsers.python_parser import PythonParser


class TestPythonParser:
    """Unit tests for PythonParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PythonParser()
    
    def test_basic_function_parsing(self):
        """Test parsing of basic Python functions."""
        content = '''
def test_function(arg1, arg2=None):
    """Test function docstring."""
    return arg1 + (arg2 or 0)
'''
        
        metadata = self.parser.parse(content, "test.py")
        
        assert metadata is not None
        assert metadata.language == "python"
        assert len(metadata.functions) > 0
        
        func = metadata.functions[0]
        assert func['name'] == 'test_function'
    
    def test_class_parsing(self):
        """Test parsing of Python classes."""
        content = '''
class TestClass:
    """Test class docstring."""
    
    def __init__(self, value):
        self.value = value
    
    def get_value(self):
        return self.value
'''
        
        metadata = self.parser.parse(content, "test.py")
        
        assert metadata is not None
        assert len(metadata.classes) > 0
        
        cls = metadata.classes[0]
        assert cls['name'] == 'TestClass'
    
    def test_import_extraction(self):
        """Test extraction of import statements."""
        content = '''
import os
import sys
from pathlib import Path
from typing import Dict, List
'''
        
        metadata = self.parser.parse(content, "test.py")
        
        assert metadata is not None
        assert len(metadata.imports) > 0
        
        # Should find multiple imports
        import_names = [imp for imp in metadata.imports]
        assert 'os' in str(import_names)
        assert 'sys' in str(import_names)
    
    def test_complexity_calculation(self):
        """Test basic complexity calculation."""
        simple_content = '''
def simple_function():
    return True
'''
        
        complex_content = '''
def complex_function(x, y):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                if y > i:
                    return i * y
                else:
                    continue
            else:
                break
    return None
'''
        
        simple_metadata = self.parser.parse(simple_content, "simple.py")
        complex_metadata = self.parser.parse(complex_content, "complex.py")
        
        assert simple_metadata.complexity_score > 0
        assert complex_metadata.complexity_score > simple_metadata.complexity_score


if __name__ == "__main__":
    pytest.main([__file__])