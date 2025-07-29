"""
Language parsers for the codebase compression pipeline.
"""

from .base import ParserMixin, TokenizerMixin, ImportExtractorMixin, StructureExtractorMixin
from .python_parser import PythonParser

# Enhanced parsers - import with error handling
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
    'ParserMixin',
    'TokenizerMixin', 
    'ImportExtractorMixin',
    'StructureExtractorMixin',
    'PythonParser',
    'EnhancedJavaScriptParser',
    'TypeScriptParser',
    'GoParser',
    'RustParser',
]

# Parser registry for factory pattern
PARSER_REGISTRY = {
    '.py': PythonParser,
    '.js': EnhancedJavaScriptParser if JS_PARSER_AVAILABLE else None,
    '.jsx': EnhancedJavaScriptParser if JS_PARSER_AVAILABLE else None,
    '.ts': TypeScriptParser if TS_PARSER_AVAILABLE else None,
    '.tsx': TypeScriptParser if TS_PARSER_AVAILABLE else None,
    '.go': GoParser if GO_PARSER_AVAILABLE else None,
    '.rs': RustParser if RUST_PARSER_AVAILABLE else None,
}