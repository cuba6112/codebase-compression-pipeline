"""
Pluggable Parser Registry System
===============================

Dynamic parser loading and management system using setuptools entry points.
Supports both built-in parsers and external plugins.
"""

import logging
import warnings
from typing import Dict, Optional, Type, List, Any, Callable
from pathlib import Path
import importlib
from dataclasses import dataclass

try:
    from importlib.metadata import entry_points
    ENTRY_POINTS_AVAILABLE = True
except ImportError:
    try:
        from importlib_metadata import entry_points
        ENTRY_POINTS_AVAILABLE = True
    except ImportError:
        ENTRY_POINTS_AVAILABLE = False
        entry_points = None

from base_classes import LanguageParser, FileMetadata

logger = logging.getLogger(__name__)


@dataclass
class ParserInfo:
    """Information about a parser plugin"""
    name: str
    parser_class: Type[LanguageParser]
    extensions: List[str]
    language: str
    priority: int = 0  # Higher priority parsers override lower priority ones
    source: str = "builtin"  # "builtin", "plugin", "custom"
    description: str = ""
    version: str = "1.0.0"


class ParserRegistry:
    """
    Central registry for language parsers with plugin support.
    
    Supports parser discovery via:
    1. Built-in parsers (hardcoded)
    2. Entry points (setuptools plugins)
    3. Runtime registration (custom parsers)
    """
    
    def __init__(self):
        self._parsers: Dict[str, ParserInfo] = {}
        self._language_map: Dict[str, str] = {}  # language -> parser_name
        self._extension_map: Dict[str, str] = {}  # extension -> parser_name
        self._loaded = False
        
    def _load_builtin_parsers(self):
        """Load built-in parsers"""
        builtin_parsers = []
        
        # Python parser (always available)
        try:
            from .python_parser import PythonParser
            builtin_parsers.append(ParserInfo(
                name="python",
                parser_class=PythonParser,
                extensions=[".py", ".pyw", ".pyi"],
                language="python",
                source="builtin",
                description="Built-in Python AST parser",
                version="1.0.0"
            ))
        except ImportError as e:
            logger.error(f"Failed to load built-in Python parser: {e}")
        
        # JavaScript parser
        try:
            from .enhanced_js_parser import EnhancedJavaScriptParser
            builtin_parsers.append(ParserInfo(
                name="javascript",
                parser_class=EnhancedJavaScriptParser,
                extensions=[".js", ".jsx", ".mjs", ".cjs"],
                language="javascript",
                source="builtin",
                description="Enhanced JavaScript parser with JSX support",
                version="1.0.0"
            ))
        except ImportError:
            logger.info("Enhanced JavaScript parser not available")
        
        # TypeScript parser
        try:
            from .typescript_parser import TypeScriptParser
            builtin_parsers.append(ParserInfo(
                name="typescript",
                parser_class=TypeScriptParser,
                extensions=[".ts", ".tsx"],
                language="typescript", 
                source="builtin",
                description="TypeScript parser with type information",
                version="1.0.0"
            ))
        except ImportError:
            logger.info("TypeScript parser not available")
        
        # Go parser
        try:
            from .go_parser import GoParser
            builtin_parsers.append(ParserInfo(
                name="go",
                parser_class=GoParser,
                extensions=[".go"],
                language="go",
                source="builtin", 
                description="Go language parser",
                version="1.0.0"
            ))
        except ImportError:
            logger.info("Go parser not available")
        
        # Rust parser
        try:
            from .rust_parser import RustParser
            builtin_parsers.append(ParserInfo(
                name="rust",
                parser_class=RustParser,
                extensions=[".rs"],
                language="rust",
                source="builtin",
                description="Rust language parser", 
                version="1.0.0"
            ))
        except ImportError:
            logger.info("Rust parser not available")
        
        # Register all built-in parsers
        for parser_info in builtin_parsers:
            self._register_parser(parser_info)
            
    def _load_plugin_parsers(self):
        """Load parsers from entry points"""
        if not ENTRY_POINTS_AVAILABLE:
            logger.warning("Entry points not available - plugin parsers cannot be loaded")
            return
            
        try:
            # Get parser entry points
            parser_eps = entry_points(group='codebase_compression.parsers')
            
            for ep in parser_eps:
                try:
                    # Load the parser class
                    parser_class = ep.load()
                    
                    # Validate it's a proper parser
                    if not issubclass(parser_class, LanguageParser):
                        logger.error(f"Plugin parser {ep.name} is not a LanguageParser subclass")
                        continue
                    
                    # Get parser metadata
                    extensions = getattr(parser_class, 'EXTENSIONS', [f".{ep.name}"])
                    language = getattr(parser_class, 'LANGUAGE', ep.name)
                    description = getattr(parser_class, 'DESCRIPTION', f"Plugin parser for {language}")
                    version = getattr(parser_class, 'VERSION', "1.0.0")
                    priority = getattr(parser_class, 'PRIORITY', 10)  # Plugins have higher priority
                    
                    parser_info = ParserInfo(
                        name=ep.name,
                        parser_class=parser_class,
                        extensions=extensions,
                        language=language,
                        priority=priority,
                        source="plugin",
                        description=description,
                        version=version
                    )
                    
                    self._register_parser(parser_info)
                    logger.info(f"Loaded plugin parser: {ep.name} v{version}")
                    
                except Exception as e:
                    logger.error(f"Failed to load plugin parser {ep.name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading plugin parsers: {e}")
            
    def _register_parser(self, parser_info: ParserInfo):
        """Register a parser with the registry"""
        name = parser_info.name
        
        # Check for conflicts
        if name in self._parsers:
            existing = self._parsers[name]
            if parser_info.priority <= existing.priority:
                logger.warning(f"Parser {name} already registered with higher priority")
                return
            else:
                logger.info(f"Replacing parser {name} (priority {existing.priority} -> {parser_info.priority})")
        
        # Register parser
        self._parsers[name] = parser_info
        self._language_map[parser_info.language] = name
        
        # Register extensions
        for ext in parser_info.extensions:
            if ext in self._extension_map:
                existing_parser = self._parsers[self._extension_map[ext]]
                if parser_info.priority > existing_parser.priority:
                    logger.info(f"Extension {ext}: {existing_parser.name} -> {name}")
                    self._extension_map[ext] = name
                else:
                    logger.debug(f"Extension {ext} already handled by higher priority parser")
            else:
                self._extension_map[ext] = name
                
    def load_parsers(self, force_reload: bool = False):
        """Load all available parsers"""
        if self._loaded and not force_reload:
            return
            
        logger.info("Loading parsers...")
        
        # Clear existing registrations if reloading
        if force_reload:
            self._parsers.clear()
            self._language_map.clear()
            self._extension_map.clear()
            
        # Load parsers in order
        self._load_builtin_parsers()
        self._load_plugin_parsers()
        
        self._loaded = True
        logger.info(f"Loaded {len(self._parsers)} parsers for {len(set(self._extension_map.values()))} languages")
        
    def get_parser_for_extension(self, extension: str) -> Optional[Type[LanguageParser]]:
        """Get parser class for file extension"""
        if not self._loaded:
            self.load_parsers()
            
        extension = extension.lower()
        if extension in self._extension_map:
            parser_name = self._extension_map[extension]
            return self._parsers[parser_name].parser_class
        return None
        
    def get_parser_for_language(self, language: str) -> Optional[Type[LanguageParser]]:
        """Get parser class for language"""
        if not self._loaded:
            self.load_parsers()
            
        if language in self._language_map:
            parser_name = self._language_map[language]
            return self._parsers[parser_name].parser_class
        return None
        
    def get_parser_info(self, name: str) -> Optional[ParserInfo]:
        """Get information about a parser"""
        if not self._loaded:
            self.load_parsers()
        return self._parsers.get(name)
        
    def list_parsers(self) -> List[ParserInfo]:
        """List all registered parsers"""
        if not self._loaded:
            self.load_parsers()
        return list(self._parsers.values())
        
    def list_supported_extensions(self) -> List[str]:
        """List all supported file extensions"""
        if not self._loaded:
            self.load_parsers()
        return list(self._extension_map.keys())
        
    def register_custom_parser(self, 
                             name: str,
                             parser_class: Type[LanguageParser],
                             extensions: List[str],
                             language: str,
                             priority: int = 20,
                             description: str = "",
                             version: str = "1.0.0"):
        """Register a custom parser at runtime"""
        parser_info = ParserInfo(
            name=name,
            parser_class=parser_class,
            extensions=extensions,
            language=language,
            priority=priority,
            source="custom",
            description=description or f"Custom {language} parser",
            version=version
        )
        
        self._register_parser(parser_info)
        logger.info(f"Registered custom parser: {name}")
        
    def create_parser_factory(self) -> Dict[str, LanguageParser]:
        """Create a factory dictionary for backwards compatibility"""
        if not self._loaded:
            self.load_parsers()
            
        factory = {}
        
        # Create instances of all parsers
        for parser_info in self._parsers.values():
            try:
                parser_instance = parser_info.parser_class()
                factory[parser_info.language] = parser_instance
                
                # Also map by extensions for direct lookup
                for ext in parser_info.extensions:
                    if ext not in factory:  # Don't override language mappings
                        factory[ext] = parser_instance
                        
            except Exception as e:
                logger.error(f"Failed to instantiate parser {parser_info.name}: {e}")
                
        return factory
        
    def detect_language(self, file_path: str) -> str:
        """Detect language from file path"""
        if not self._loaded:
            self.load_parsers()
            
        extension = Path(file_path).suffix.lower()
        if extension in self._extension_map:
            parser_name = self._extension_map[extension]
            return self._parsers[parser_name].language
        return "unknown"
        
    def get_parser_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        if not self._loaded:
            self.load_parsers()
            
        stats = {
            "total_parsers": len(self._parsers),
            "supported_extensions": len(self._extension_map),
            "supported_languages": len(self._language_map),
            "by_source": {},
            "parsers": []
        }
        
        # Group by source
        for parser_info in self._parsers.values():
            source = parser_info.source
            if source not in stats["by_source"]:
                stats["by_source"][source] = 0
            stats["by_source"][source] += 1
            
            stats["parsers"].append({
                "name": parser_info.name,
                "language": parser_info.language,
                "extensions": parser_info.extensions,
                "source": parser_info.source,
                "priority": parser_info.priority,
                "description": parser_info.description,
                "version": parser_info.version
            })
            
        return stats


# Global registry instance
_global_registry = ParserRegistry()


def get_parser_registry() -> ParserRegistry:
    """Get the global parser registry"""
    return _global_registry


def get_parser_for_file(file_path: str) -> Optional[Type[LanguageParser]]:
    """Convenience function to get parser for a file"""
    extension = Path(file_path).suffix.lower()
    return _global_registry.get_parser_for_extension(extension)


def list_available_parsers() -> List[ParserInfo]:
    """Convenience function to list all parsers"""
    return _global_registry.list_parsers()


def create_parser_factory() -> Dict[str, LanguageParser]:
    """Create parser factory for backwards compatibility"""
    return _global_registry.create_parser_factory()