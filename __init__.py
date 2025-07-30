"""
Codebase Compression Pipeline
============================

A high-performance pipeline for compressing large codebases into 
optimized representations for Large Language Model consumption.

Core Features:
- Multi-language parsing (Python, JavaScript, TypeScript, Go, Rust)
- Streaming compression with LZ4
- Incremental caching with content-based hashing
- Security validation and sandboxing
- Configurable output formats (Markdown, JSON, XML)
- Parallel processing with work-stealing queues
- Performance monitoring and adaptive configuration

Usage:
    from codebase_compression_pipeline import CodebaseCompressionPipeline
    
    pipeline = CodebaseCompressionPipeline(
        cache_dir="./cache",
        output_dir="./output"
    )
    
    files = pipeline.process_codebase(
        codebase_path="./my_project",
        output_format="markdown",
        compression_strategy="structural"
    )
"""

__version__ = "1.0.0"
__author__ = "Project Think"
__email__ = ""
__description__ = "High-performance pipeline for compressing codebases for LLM context optimization"

# Try relative imports first (when installed as package), then absolute (when running from source)
try:
    # Main exports
    from .codebase_compression_pipeline import (
        CodebaseCompressionPipeline,
        FileMetadata
    )

    # Configuration exports
    from .pipeline_configs import (
        ConfigPresets,
        AdaptiveConfig,
        PipelineConfig,
        CompressionProfiles,
        PerformanceTuning
    )
    
    # Monitoring exports
    from .pipeline_monitoring import PipelineMonitor

    # Security exports
    from .security_validation import (
        SecurityConfig,
        SecurityValidator,
        PathValidator,
        FileValidator,
        ContentScanner
    )

    # Parser exports
    from .base_classes import LanguageParser
    from .parsers.python_parser import PythonParser
    from .parsers.enhanced_js_parser import EnhancedJavaScriptParser

    # Try to import optional parsers
    try:
        from .parsers.typescript_parser import TypeScriptParser
        _has_typescript = True
    except ImportError:
        _has_typescript = False

    try:
        from .parsers.go_parser import GoParser
        _has_go = True
    except ImportError:
        _has_go = False

    try:
        from .parsers.rust_parser import RustParser
        _has_rust = True
    except ImportError:
        _has_rust = False

    # Pipeline stage exports
    from .pipeline.stages.compression import StreamingCompressor
    from .pipeline.stages.metadata import MetadataStore
    from .pipeline.stages.cache import IncrementalCache

    # Enhanced caching
    from .enhanced_cache import EnhancedIncrementalCache

    # Resilience patterns
    from .resilience_patterns import (
        with_retry,
        RetryConfig,
        CircuitBreaker,
        TimeoutHandler
    )

except ImportError:
    # Fallback to absolute imports (running from source directory)
    try:
        from codebase_compression_pipeline import (
            CodebaseCompressionPipeline,
            FileMetadata
        )

        from pipeline_configs import (
            ConfigPresets,
            AdaptiveConfig,
            PipelineConfig,
            CompressionProfiles,
            PerformanceTuning
        )
        
        from pipeline_monitoring import PipelineMonitor

        from security_validation import (
            SecurityConfig,
            SecurityValidator,
            PathValidator,
            FileValidator,
            ContentScanner
        )

        from base_classes import LanguageParser
        from parsers.python_parser import PythonParser
        from parsers.enhanced_js_parser import EnhancedJavaScriptParser

        # Try to import optional parsers
        try:
            from parsers.typescript_parser import TypeScriptParser
            _has_typescript = True
        except ImportError:
            _has_typescript = False

        try:
            from parsers.go_parser import GoParser
            _has_go = True
        except ImportError:
            _has_go = False

        try:
            from parsers.rust_parser import RustParser
            _has_rust = True
        except ImportError:
            _has_rust = False

        from pipeline.stages.compression import StreamingCompressor
        from pipeline.stages.metadata import MetadataStore
        from pipeline.stages.cache import IncrementalCache
        from enhanced_cache import EnhancedIncrementalCache

        from resilience_patterns import (
            with_retry,
            RetryConfig,
            CircuitBreaker,
            TimeoutHandler
        )

    except ImportError as e:
        # If we can't import anything, provide minimal functionality
        import warnings
        warnings.warn(f"Could not import core modules: {e}. Some functionality may be limited.")
        
        # Define minimal classes
        class CodebaseCompressionPipeline:
            def __init__(self, *args, **kwargs):
                raise ImportError("Core pipeline module not available")
        
        _has_typescript = False
        _has_go = False
        _has_rust = False

# Define constants for strategies and formats (available regardless of import success)
class CompressionStrategy:
    """Available compression strategies"""
    FULL = 'full'
    STRUCTURAL = 'structural'
    SIGNATURE = 'signature'
    SUMMARY = 'summary'

class OutputFormat:
    """Available output formats"""
    MARKDOWN = 'markdown'
    JSON = 'json'
    XML = 'xml'
    CUSTOM = 'custom'

# Build __all__ dynamically based on successful imports
__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    
    # Convenience function
    'compress_codebase',
]

# Add main classes if available
_main_classes = [
    'CodebaseCompressionPipeline',
    'FileMetadata', 
    'PipelineMonitor',
    'AdaptiveConfig'
]

# Add configuration classes if available  
_config_classes = [
    'ConfigPresets',
    'CompressionStrategy', 
    'OutputFormat'
]

# Add security classes if available
_security_classes = [
    'SecurityConfig',
    'SecurityValidator',
    'PathValidator',
    'FileValidator', 
    'ContentScanner'
]

# Add parser classes if available
_parser_classes = [
    'LanguageParser',
    'PythonParser',
    'EnhancedJavaScriptParser'
]

# Add pipeline components if available
_pipeline_classes = [
    'StreamingCompressor',
    'MetadataStore',
    'IncrementalCache',
    'EnhancedIncrementalCache'
]

# Add resilience classes if available
_resilience_classes = [
    'with_retry',
    'RetryConfig', 
    'CircuitBreaker',
    'TimeoutHandler'
]

# Check which classes are actually available and add to __all__
for class_list in [_main_classes, _config_classes, _security_classes, 
                   _parser_classes, _pipeline_classes, _resilience_classes]:
    for class_name in class_list:
        if class_name in globals():
            __all__.append(class_name)

# Add optional parsers if available
if _has_typescript and 'TypeScriptParser' in globals():
    __all__.append('TypeScriptParser')
if _has_go and 'GoParser' in globals():
    __all__.append('GoParser') 
if _has_rust and 'RustParser' in globals():
    __all__.append('RustParser')

# Convenience function for quick processing
def compress_codebase(codebase_path, output_dir="./compressed_output", 
                     output_format="markdown", compression_strategy="structural",
                     **kwargs):
    """
    Convenience function for quick codebase compression.
    
    Args:
        codebase_path: Path to the codebase to compress
        output_dir: Directory for compressed output
        output_format: Output format ('markdown', 'json', 'xml')
        compression_strategy: Strategy ('full', 'structural', 'signature', 'summary')
        **kwargs: Additional arguments passed to CodebaseCompressionPipeline
        
    Returns:
        List of output file paths
    """
    from pathlib import Path
    
    pipeline = CodebaseCompressionPipeline(
        cache_dir=Path(output_dir) / "cache",
        output_dir=output_dir,
        **kwargs
    )
    
    try:
        return pipeline.process_codebase(
            codebase_path=codebase_path,
            output_format=output_format,
            compression_strategy=compression_strategy
        )
    finally:
        pipeline.cleanup()