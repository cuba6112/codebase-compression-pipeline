#!/usr/bin/env python3
"""
Unit tests for the parser registry system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

from parsers.registry import ParserRegistry, ParserInfo, get_parser_registry
from base_classes import LanguageParser, FileMetadata


class MockParser(LanguageParser):
    """Mock parser for testing"""
    LANGUAGE = "mock"
    EXTENSIONS = [".mock"]
    DESCRIPTION = "Mock parser for testing"
    VERSION = "1.0.0"
    PRIORITY = 5
    
    def parse(self, content: str, file_path: str) -> FileMetadata:
        return FileMetadata(
            path=file_path,
            size=len(content),
            language=self.LANGUAGE,
            last_modified=0,
            content_hash="mock_hash"
        )


class HighPriorityMockParser(LanguageParser):
    """High priority mock parser for testing conflicts"""
    LANGUAGE = "mock" 
    EXTENSIONS = [".mock"]
    DESCRIPTION = "High priority mock parser"
    VERSION = "2.0.0"
    PRIORITY = 15
    
    def parse(self, content: str, file_path: str) -> FileMetadata:
        return FileMetadata(
            path=file_path,
            size=len(content),
            language=self.LANGUAGE,
            last_modified=0,
            content_hash="high_priority_hash"
        )


class TestParserRegistry:
    """Test cases for ParserRegistry"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.registry = ParserRegistry()
    
    def test_registry_initialization(self):
        """Test registry initializes properly"""
        assert self.registry._parsers == {}
        assert self.registry._language_map == {}
        assert self.registry._extension_map == {}
        assert not self.registry._loaded
    
    def test_register_custom_parser(self):
        """Test registering a custom parser"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock",
            description="Test parser"
        )
        
        assert "mock" in self.registry._parsers
        assert "mock" in self.registry._language_map
        assert ".mock" in self.registry._extension_map
        
        parser_info = self.registry._parsers["mock"]
        assert parser_info.name == "mock"
        assert parser_info.parser_class == MockParser
        assert parser_info.language == "mock"
        assert parser_info.source == "custom"
    
    def test_parser_priority_conflict_resolution(self):
        """Test that higher priority parsers override lower priority ones"""
        # Register low priority parser first
        self.registry.register_custom_parser(
            name="mock_low",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock",
            priority=5
        )
        
        # Register high priority parser for same extension
        self.registry.register_custom_parser(
            name="mock_high", 
            parser_class=HighPriorityMockParser,
            extensions=[".mock"],
            language="mock",
            priority=15
        )
        
        # High priority parser should win
        parser_class = self.registry.get_parser_for_extension(".mock")
        assert parser_class == HighPriorityMockParser
        
        # But low priority should be rejected if registered after
        self.registry.register_custom_parser(
            name="mock_low2",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock",
            priority=3
        )
        
        # Should still be high priority parser
        parser_class = self.registry.get_parser_for_extension(".mock")
        assert parser_class == HighPriorityMockParser
    
    def test_get_parser_for_extension(self):
        """Test retrieving parser by file extension"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock", ".test"],
            language="mock"
        )
        
        # Test both extensions
        assert self.registry.get_parser_for_extension(".mock") == MockParser
        assert self.registry.get_parser_for_extension(".test") == MockParser
        
        # Test case insensitive
        assert self.registry.get_parser_for_extension(".MOCK") == MockParser
        
        # Test unknown extension
        assert self.registry.get_parser_for_extension(".unknown") is None
    
    def test_get_parser_for_language(self):
        """Test retrieving parser by language"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock"
        )
        
        assert self.registry.get_parser_for_language("mock") == MockParser
        assert self.registry.get_parser_for_language("unknown") is None
    
    def test_list_parsers(self):
        """Test listing all registered parsers"""
        self.registry.register_custom_parser(
            name="mock1",
            parser_class=MockParser,
            extensions=[".mock1"],
            language="mock1"
        )
        
        self.registry.register_custom_parser(
            name="mock2", 
            parser_class=MockParser,
            extensions=[".mock2"],
            language="mock2"
        )
        
        parsers = self.registry.list_parsers()
        assert len(parsers) == 2
        
        parser_names = [p.name for p in parsers]
        assert "mock1" in parser_names
        assert "mock2" in parser_names
    
    def test_list_supported_extensions(self):
        """Test listing supported extensions"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock1", ".mock2", ".test"],
            language="mock"
        )
        
        extensions = self.registry.list_supported_extensions()
        assert ".mock1" in extensions
        assert ".mock2" in extensions
        assert ".test" in extensions
    
    def test_detect_language(self):
        """Test language detection from file path"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock_lang"
        )
        
        assert self.registry.detect_language("test.mock") == "mock_lang"
        assert self.registry.detect_language("test.unknown") == "unknown"
        assert self.registry.detect_language("/path/to/file.mock") == "mock_lang"
    
    def test_create_parser_factory(self):
        """Test creating parser factory dictionary"""
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock_lang"
        )
        
        factory = self.registry.create_parser_factory()
        
        # Should have language mapping
        assert "mock_lang" in factory
        assert isinstance(factory["mock_lang"], MockParser)
        
        # Should also have extension mapping
        assert ".mock" in factory
        assert isinstance(factory[".mock"], MockParser)
    
    def test_get_parser_stats(self):
        """Test getting registry statistics"""
        self.registry.register_custom_parser(
            name="mock1",
            parser_class=MockParser,
            extensions=[".mock1"],
            language="mock1"
        )
        
        self.registry.register_custom_parser(
            name="mock2",
            parser_class=MockParser,
            extensions=[".mock2", ".test"], 
            language="mock2"
        )
        
        stats = self.registry.get_parser_stats()
        
        assert stats["total_parsers"] == 2
        assert stats["supported_extensions"] == 3  # .mock1, .mock2, .test
        assert stats["supported_languages"] == 2
        assert "custom" in stats["by_source"]
        assert stats["by_source"]["custom"] == 2
        
        # Check parser details
        parser_names = [p["name"] for p in stats["parsers"]]
        assert "mock1" in parser_names
        assert "mock2" in parser_names
    
    @patch('parsers.registry.ENTRY_POINTS_AVAILABLE', True)
    def test_load_builtin_parsers(self):
        """Test loading built-in parsers"""
        # This test checks that the registry can load built-in parsers
        # without actually importing them (they might not be available in test env)
        
        with patch('parsers.registry.logger') as mock_logger:
            self.registry._load_builtin_parsers()
            
            # Should attempt to load parsers
            # The exact behavior depends on what parsers are available
            # At minimum, it should not crash
            assert True  # If we get here, no exception was thrown
    
    @patch('parsers.registry.ENTRY_POINTS_AVAILABLE', True)
    @patch('parsers.registry.entry_points')
    def test_load_plugin_parsers(self, mock_entry_points):
        """Test loading parsers from entry points"""
        # Mock entry point
        mock_ep = Mock()
        mock_ep.name = "test_plugin"
        mock_ep.load.return_value = MockParser
        
        mock_entry_points.return_value = [mock_ep]
        
        self.registry._load_plugin_parsers()
        
        # Should have loaded the plugin parser
        assert "test_plugin" in self.registry._parsers
        parser_info = self.registry._parsers["test_plugin"]
        assert parser_info.source == "plugin"
        assert parser_info.parser_class == MockParser
    
    @patch('parsers.registry.ENTRY_POINTS_AVAILABLE', False)
    def test_load_plugin_parsers_no_entry_points(self):
        """Test behavior when entry points not available"""
        with patch('parsers.registry.logger') as mock_logger:
            self.registry._load_plugin_parsers()
            
            # Should log warning about entry points not available
            mock_logger.warning.assert_called_once()
    
    def test_load_parsers_idempotent(self):
        """Test that load_parsers is idempotent"""
        # Mock the loading methods to track calls
        with patch.object(self.registry, '_load_builtin_parsers') as mock_builtin, \
             patch.object(self.registry, '_load_plugin_parsers') as mock_plugin:
            
            # First call should load
            self.registry.load_parsers()
            assert mock_builtin.called
            assert mock_plugin.called
            assert self.registry._loaded
            
            # Reset mocks
            mock_builtin.reset_mock()
            mock_plugin.reset_mock()
            
            # Second call should not load again
            self.registry.load_parsers()
            assert not mock_builtin.called
            assert not mock_plugin.called
    
    def test_load_parsers_force_reload(self):
        """Test force reloading parsers"""
        # Add a parser
        self.registry.register_custom_parser(
            name="mock",
            parser_class=MockParser,
            extensions=[".mock"],
            language="mock"
        )
        
        assert len(self.registry._parsers) == 1
        
        # Force reload should clear existing parsers
        with patch.object(self.registry, '_load_builtin_parsers'), \
             patch.object(self.registry, '_load_plugin_parsers'):
            
            self.registry.load_parsers(force_reload=True)
            
            # Should have cleared parsers (since we mocked the loading methods)
            assert len(self.registry._parsers) == 0


class TestGlobalRegistry:
    """Test the global registry functions"""
    
    def test_get_parser_registry_singleton(self):
        """Test that get_parser_registry returns the same instance"""
        registry1 = get_parser_registry()
        registry2 = get_parser_registry()
        
        assert registry1 is registry2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])