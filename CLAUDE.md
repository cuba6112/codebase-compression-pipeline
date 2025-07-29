# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Installing Dependencies
The project requires the following external dependencies:
```bash
pip install lz4 mmh3 psutil tqdm
```

### Running the Pipeline

Using the global `compress` command (available after sourcing ~/.zshrc):
```bash
# Basic usage (compress current directory)
compress

# Compress specific directory
compress /path/to/project

# With custom format and strategy
compress /path/to/project json full

# Clear cache before compression
compress --reset-cache /path/to/project

# Show help
compress --help
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

Note: The project uses conda Python at `/opt/anaconda3/bin/python`

## High-Level Architecture

This codebase implements a **Code Shrinking Algorithm for LLM Context Optimization**. The system compresses large codebases into smaller representations while preserving semantic meaning, enabling LLMs to process more code within their context windows.

### Core Components

1. **CodebaseCompressionPipeline** (`codebase_compression_pipeline.py`): Main orchestrator that processes codebases through 8 distinct stages:
   - File Parsing & Tokenization (language-specific parsers)
   - Parallel Processing Engine (work-stealing queue)
   - Memory-Efficient Streaming Compression (using LZ4)
   - Caching & Incremental Updates (mmh3 content hashing)
   - Metadata Storage & Indexing (columnar storage)
   - Query-Based Selective Compression
   - Output Formatting & Chunking
   - Performance Optimization (psutil-based monitoring)

2. **Pipeline Configuration** (`pipeline_configs.py`): Pre-configured settings for different scenarios:
   - Compression strategies: 'full', 'structural', 'signature', 'summary'
   - Output formats: 'markdown', 'json', 'xml', 'custom'
   - Memory and performance tuning parameters
   - Dynamic configuration based on system resources

3. **Pipeline Monitoring** (`pipeline_monitoring.py`): Real-time performance tracking:
   - Stage-level performance metrics (throughput, latency)
   - Resource utilization tracking (CPU, memory via psutil)
   - Bottleneck detection and optimization suggestions
   - System-aware auto-tuning

### Key Design Patterns

1. **Streaming Architecture**: Processes files in chunks to maintain constant memory usage regardless of codebase size
2. **Language-Specific Parsers**: Uses AST for Python, regex for JavaScript (extensible for other languages)
3. **Columnar Storage**: Metadata stored in column format for vectorized queries and efficient memory usage
4. **Work-Stealing Queue**: Dynamic load balancing for parallel processing
5. **Incremental Caching**: Only processes changed files, with content hash validation using mmh3

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

### Usage Example

```python
pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./compressed_output'),
    num_workers=4
)

output_files = await pipeline.process_codebase(
    codebase_path=Path('./my_project'),
    output_format='markdown',
    compression_strategy='structural',
    query_filter={'language': 'python', 'min_complexity': 5.0}
)
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
- **Ignore Patterns**: Automatically ignores common directories (node_modules, .git, __pycache__, etc.)

## Performance Considerations

- **Memory Usage**: Streaming processing ensures constant memory overhead
- **Parallelization**: Work-stealing queues for optimal CPU utilization
- **Caching**: Content-hash based incremental processing
- **Monitoring**: Real-time performance tracking with bottleneck detection

## Project Context

This is a proof-of-concept implementation for creating an algorithm that shrinks codebases for better LLM context utilization. The pipeline architecture (detailed in `pipeline_architecture.md`) emphasizes scalability, configurability, and semantic preservation.

Additional documentation:
- `pipeline_architecture.md`: Detailed technical architecture
- `brainstorm.md`: Initial design brainstorming notes