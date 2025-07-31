# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installing Dependencies
Install required dependencies:
```bash
pip install -r requirements.txt
```

Core dependencies include:
- `lz4>=4.0.0` - High-speed compression
- `mmh3>=3.0.0` - MurmurHash3 for content hashing
- `psutil>=5.9.0` - System resource monitoring
- `tqdm>=4.65.0` - Progress bars
- `aiofiles>=23.0.0` - Async file operations

Optional dependencies:
```bash
pip install redis numpy brotli zstandard  # Enhanced caching and compression
npm install typescript @typescript-eslint/parser  # TypeScript support
```

### Running Tests
Use the custom test runner:
```bash
# Run all tests
python run_tests.py

# Run with coverage report
python run_tests.py --coverage

# Run specific test by pattern
python run_tests.py --test "test_compression"

# Skip slow performance tests
python run_tests.py --fast

# Run tests in parallel
python run_tests.py --parallel 4

# Install test dependencies first
python run_tests.py --install-deps
```

### Running the Pipeline

Using the shell wrapper (recommended):
```bash
# Basic usage (compress current directory)
./compress.sh

# Compress specific directory
./compress.sh /path/to/project

# With custom format and strategy
./compress.sh /path/to/project json full

# Clear cache before compression
./compress.sh --reset-cache /path/to/project

# Show help
./compress.sh --help
```

Direct Python execution:
```bash
# Run the pipeline directly
python codebase_compression_pipeline.py

# Monitor performance in real-time
python pipeline_monitoring.py

# Run with specific compression strategy
python codebase_compression_pipeline.py --compression-strategy structural

# Enable profiling for performance analysis
python codebase_compression_pipeline.py --enable-profiling

# Process specific file types only
python codebase_compression_pipeline.py --extensions .py,.js,.ts
```

### Development Commands
Code quality and formatting:
```bash
black .                    # Format code
flake8 .                   # Lint code
mypy .                     # Type checking
pre-commit run --all-files # Run pre-commit hooks
```

Note: The shell scripts automatically detect Python 3 from PATH, conda, or system locations

## High-Level Architecture

This codebase implements a **Code Shrinking Algorithm for LLM Context Optimization**. The system compresses large codebases into smaller representations while preserving semantic meaning, enabling LLMs to process more code within their context windows.

### Core Components

1. **CodebaseCompressionPipeline** (`codebase_compression_pipeline.py`): Main orchestrator that processes codebases through 8 distinct stages:
   - File Parsing & Tokenization (language-specific parsers in `parsers/`)
   - Parallel Processing Engine (work-stealing queue in `pipeline/workers/`)
   - Memory-Efficient Streaming Compression (using LZ4)
   - Caching & Incremental Updates (mmh3 content hashing, enhanced with Redis support)
   - Metadata Storage & Indexing (columnar storage in `pipeline/stages/`)
   - Query-Based Selective Compression
   - Output Formatting & Chunking
   - Performance Optimization (psutil-based monitoring)

2. **Language Parsers** (`parsers/`): Extensible parser system supporting:
   - **Python** (`python_parser.py`): AST-based parsing with complexity analysis
   - **JavaScript/TypeScript** (`enhanced_js_parser.py`, `typescript_parser.py`): Enhanced parsing with TypeScript support
   - **Go** (`go_parser.py`): Go language support with AST parsing
   - **Rust** (`rust_parser.py`): Rust language support
   - **Base Parser** (`base.py`): Common interface and utilities

3. **Pipeline Stages** (`pipeline/stages/`): Modular processing stages:
   - **Cache** (`cache.py`): Caching layer with content hashing
   - **Compression** (`compression.py`): LZ4 compression with multiple strategies
   - **Formatting** (`formatting.py`): Output format generation (Markdown, JSON, XML)
   - **Mapping** (`mapping.py`): File and metadata mapping
   - **Metadata** (`metadata.py`): Metadata extraction and storage
   - **Optimization** (`optimization.py`): Performance optimization
   - **Selection** (`selection.py`): Query-based content selection

4. **Pipeline Configuration** (`pipeline_configs.py`): Pre-configured settings for different scenarios:
   - Compression strategies: 'full', 'structural', 'signature', 'summary'
   - Output formats: 'markdown', 'json', 'xml', 'custom'
   - Memory and performance tuning parameters
   - Dynamic configuration based on system resources

5. **Pipeline Monitoring** (`pipeline_monitoring.py`): Real-time performance tracking:
   - Stage-level performance metrics (throughput, latency)
   - Resource utilization tracking (CPU, memory via psutil)
   - Bottleneck detection and optimization suggestions
   - System-aware auto-tuning

6. **Security & Resilience**:
   - **Security Validation** (`security_validation.py`): Input validation and security scanning
   - **Resilience Patterns** (`resilience_patterns.py`): Circuit breakers, retry logic, fallback mechanisms

### Key Design Patterns

1. **Streaming Architecture**: Processes files in chunks to maintain constant memory usage regardless of codebase size
2. **Language-Specific Parsers**: AST-based parsing for Python, JavaScript, TypeScript, Go, and Rust (extensible architecture)
3. **Columnar Storage**: Metadata stored in column format for vectorized queries and efficient memory usage
4. **Work-Stealing Queue**: Dynamic load balancing for parallel processing across multiple workers
5. **Incremental Caching**: Only processes changed files, with content hash validation using mmh3
6. **Async Processing**: Full async/await support with aiofiles for non-blocking I/O operations
7. **Modular Pipeline Stages**: Each processing stage is independent and can be configured/replaced

### Data Flow

```
Raw Files → Language Detection → Parsing → Metadata Extraction → 
Compression (LZ4) → Query Filtering → Output Formatting → Chunked Output
```

### Key Libraries

- **lz4**: High-speed compression (500+ MB/s) with frame format, using compression level 12
- **mmh3**: MurmurHash3 for content-based deduplication and cache validation
- **psutil**: System resource monitoring for adaptive pipeline configuration
- **tqdm**: Progress bars for user feedback (optional, auto-installed)
- **aiofiles**: Async file operations for non-blocking I/O
- **redis**: Optional distributed caching for multi-instance deployments
- **pytest**: Comprehensive testing framework with async support

### Usage Example

```python
import asyncio
from pathlib import Path
from codebase_compression_pipeline import CodebaseCompressionPipeline

async def compress_project():
    pipeline = CodebaseCompressionPipeline(
        cache_dir=Path('./cache'),
        output_dir=Path('./compressed_output'),
        num_workers=4
    )

    output_files = await pipeline.process_codebase_async(
        codebase_path=Path('./my_project'),
        output_format='markdown',
        compression_strategy='structural',
        query_filter={'language': 'python', 'min_complexity': 5.0}
    )
    
    pipeline.cleanup()
    return output_files

# Run the pipeline
asyncio.run(compress_project())
```

## Configuration Presets

The system includes pre-configured settings via `ConfigPresets`:
- `large_codebase()`: Optimized for >100k files
- `memory_constrained()`: For systems with <4GB RAM
- `real_time()`: Low-latency processing for interactive use
- `maximum_compression()`: Best compression ratio (slower)
- `balanced()`: Default balanced configuration

## Important Files and Locations

- **Cache Directory**: `./cache/` - Contains pickled metadata and index.json
- **Output Directory**: `./compressed_output/` - Generated compressed files
- **Test Directory**: `./tests/` - Unit tests (`tests/unit/`) and integration tests (`tests/integration/`)
- **Parser Directory**: `./parsers/` - Language-specific parsers with base classes
- **Pipeline Directory**: `./pipeline/` - Modular pipeline stages and worker implementations
- **Configuration Files**: 
  - `requirements.txt` - Python dependencies
  - `setup.py` - Package configuration
  - `package.json` - Node.js dependencies (for TypeScript support)
- **Shell Scripts**: 
  - `compress.sh` - Main compression wrapper script
  - `install.sh`, `setup_macos.sh` - Installation scripts
- **Ignore Patterns**: Automatically ignores common directories (node_modules, .git, __pycache__, dist, build, etc.)

## Performance Considerations

- **Memory Usage**: Streaming processing ensures constant memory overhead
- **Parallelization**: Work-stealing queues for optimal CPU utilization
- **Caching**: Content-hash based incremental processing
- **Monitoring**: Real-time performance tracking with bottleneck detection

## Project Context

This is a proof-of-concept implementation for creating an algorithm that shrinks codebases for better LLM context utilization. The pipeline architecture (detailed in `pipeline_architecture.md`) emphasizes scalability, configurability, and semantic preservation.

Key capabilities:
- **Multi-language support**: Python, JavaScript, TypeScript, Go, Rust
- **Performance optimization**: 500+ files/second processing with constant memory usage
- **Compression ratios**: 10:1 to 50:1 depending on strategy
- **Security**: Built-in validation and sanitization
- **Resilience**: Circuit breakers, retry logic, fallback mechanisms

## Additional Documentation

- `README.md`: Comprehensive project overview with features and usage
- `pipeline_architecture.md`: Detailed technical architecture
- `brainstorm.md`: Initial design brainstorming notes
- `security-report.md`: Security analysis and validation documentation
- `exception_handling_improvements.md`: Error handling and resilience patterns

## Development Workflow

When working with this codebase:
1. Install dependencies: `pip install -r requirements.txt`
2. Run tests before making changes: `python run_tests.py`
3. Use the shell wrapper for testing: `./compress.sh /path/to/test/project`
4. Format code before committing: `black . && flake8 . && mypy .`
5. Run coverage analysis: `python run_tests.py --coverage`