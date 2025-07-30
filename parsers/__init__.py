"""
Language parsers for the codebase compression pipeline.

New Pluggable Architecture:
- Parsers are now loaded dynamically via a registry system
- Supports built-in parsers, entry point plugins, and runtime registration
- Use get_parser_registry() for advanced parser management
- Use create_parser_factory() for backwards compatibility
"""

from .base import ParserMixin, TokenizerMixin, ImportExtractorMixin, StructureExtractorMixin
from .registry import (
    ParserRegistry, 
    ParserInfo, 
    get_parser_registry, 
    get_parser_for_file,
    list_available_parsers,
    create_parser_factory
)

# Import individual parsers for direct access if needed
from .python_parser import PythonParser

# Enhanced parsers - import with error handling for backwards compatibility
try:
    from .enhanced_js_parser import EnhancedJavaScriptParser
    JS_PARSER_AVAILABLE = True
except ImportError:
    EnhancedJavaScriptParser = None
    JS_PARSER_AVAILABLE = False

try:
    from .typescript_parser import TypeScriptParser
    TS_PARSER_AVAILABLE = True
except ImportError:
    TypeScriptParser = None
    TS_PARSER_AVAILABLE = False

try:
    from .go_parser import GoParser
    GO_PARSER_AVAILABLE = True
except ImportError:
    GoParser = None
    GO_PARSER_AVAILABLE = False

try:
    from .rust_parser import RustParser
    RUST_PARSER_AVAILABLE = True
except ImportError:
    RustParser = None
    RUST_PARSER_AVAILABLE = False

__all__ = [
    # Parser mixins and base classes
    'ParserMixin',
    'TokenizerMixin', 
    'ImportExtractorMixin',
    'StructureExtractorMixin',
    
    # Registry system (new pluggable architecture)
    'ParserRegistry',
    'ParserInfo',
    'get_parser_registry',
    'get_parser_for_file',
    'list_available_parsers',
    'create_parser_factory',
    
    # Individual parsers (for direct access)
    'PythonParser',
    'EnhancedJavaScriptParser',
    'TypeScriptParser',
    'GoParser',
    'RustParser',
]

# Legacy parser registry for backwards compatibility
# DEPRECATED: Use create_parser_factory() instead
PARSER_REGISTRY = {
    '.py': PythonParser,
    '.js': EnhancedJavaScriptParser if JS_PARSER_AVAILABLE else None,
    '.jsx': EnhancedJavaScriptParser if JS_PARSER_AVAILABLE else None,
    '.ts': TypeScriptParser if TS_PARSER_AVAILABLE else None,
    '.tsx': TypeScriptParser if TS_PARSER_AVAILABLE else None,
    '.go': GoParser if GO_PARSER_AVAILABLE else None,
    '.rs': RustParser if RUST_PARSER_AVAILABLE else None,
}

def get_parser_factory():
    """
    Get parser factory using the new registry system.
    
    Returns:
        Dict mapping languages and extensions to parser instances
    """
    return create_parser_factory()

# Initialize the global registry on import
_registry = get_parser_registry()
_registry.load_parsers()