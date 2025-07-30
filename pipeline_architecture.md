# Codebase Compression Pipeline Architecture

## Overview

This pipeline transforms large codebases into compressed representations optimized for LLM context windows. It implements a multi-stage, streaming architecture with intelligent caching, parallel processing, and query-based selective compression.

## Pipeline Stages and Data Flow

### Stage 1: File Parsing and Tokenization

**Purpose**: Extract structured metadata and tokenize source files

**Data Transformations**:
```
Raw File → Parser → FileMetadata {
    path: str
    size: int
    language: str
    content_hash: str
    imports/exports: List[str]
    functions/classes: List[Dict]
    complexity_score: float
    token_count: int
}
```

**Key Features**:
- Language-specific parsers (Python uses AST, JavaScript uses regex)
- Parallel file processing with work stealing
- Metadata extraction includes:
  - Import/export relationships
  - Function/class signatures
  - Cyclomatic complexity calculation
  - Token counting for size estimation

**Optimization Techniques**:
- Lazy parsing - only parse when needed
- Streaming tokenization to avoid loading entire files
- Cache parsed ASTs for reuse

### Stage 2: Parallel Processing Engine

**Purpose**: Distribute parsing workload across CPU cores

**Data Flow**:
```
File Paths → Batch Queue → Worker Pool → Parsed Metadata Stream
```

**Key Features**:
- Dynamic batch sizing based on file sizes
- Work stealing for load balancing
- Async processing with ProcessPoolExecutor
- Error isolation - failures don't crash pipeline

**Optimization Techniques**:
- Batch size optimization: `batch_size = min(1000, usable_memory / (avg_file_size * 2))`
- CPU affinity for cache locality
- Memory-mapped files for large codebases

### Stage 3: Memory-Efficient Streaming

**Purpose**: Process data without loading entire codebase into memory

**Data Transformations**:
```
Metadata Stream → Sliding Window → Deduplication → LZ4 Compression → Compressed Chunks
```

**Key Features**:
- Sliding window compression (default 1MB window)
- Content-based deduplication using MurmurHash3
- Streaming LZ4 compression with level 12
- Chunk creation based on complexity sorting

**Memory Management**:
```python
# Window processing
while len(buffer) >= window_size:
    window = buffer[:window_size]
    if hash(window) not in seen:
        yield compress(window)
    buffer = buffer[chunk_size:]  # Slide window
```

### Stage 4: Caching and Incremental Updates

**Purpose**: Avoid reprocessing unchanged files

**Cache Structure**:
```
cache/
├── index.json          # File path → cache entry mapping
├── {hash}.pkl          # Cached FileMetadata objects
└── metadata/           # Columnar metadata store
```

**Incremental Processing**:
```python
# Identify changes
added = current_files - cached_files
modified = {f for f in current & cached if hash_changed(f)}
deleted = cached_files - current_files

# Process only changes
process_files(added | modified)
remove_from_cache(deleted)
```

**Cache Invalidation**:
- TTL-based expiration (default 24 hours)
- Content hash validation
- Automatic cleanup of orphaned entries

### Stage 5: Metadata Storage (Columnar Store)

**Purpose**: Enable fast queries over file metadata

**Storage Layout**:
```
Columnar Storage:
- paths:        [path1, path2, ...]
- sizes:        [size1, size2, ...]
- languages:    [lang1, lang2, ...]
- complexities: [comp1, comp2, ...]

Inverted Indices:
- import_index:   {import_name: Set[file_indices]}
- function_index: {func_name: Set[file_indices]}
- class_index:    {class_name: Set[file_indices]}
```

**Query Performance**:
- O(1) lookup for imports/exports via inverted indices
- Columnar layout enables vectorized operations
- Memory-efficient storage using primitive arrays

### Stage 6: Query-Based Selective Compression

**Purpose**: Compress only relevant files based on queries

**Compression Strategies**:

1. **Full Compression** (minimal):
   - Minify code (remove comments/whitespace)
   - Preserve all implementation details

2. **Structural Compression** (default):
   - Extract imports/exports
   - Function/class signatures only
   - Remove implementation bodies
   - Preserve architectural relationships

3. **Signature Compression** (aggressive):
   - Only function/class signatures
   - No implementation or imports

4. **Summary Compression** (maximum):
   - Statistical summary only
   - File metrics and counts

**Query Examples**:
```python
# Find complex Python files using numpy
query = {
    'language': 'python',
    'imports': 'numpy',
    'min_complexity': 10.0
}

# Find all files exporting specific function
query = {
    'exports': 'processData',
    'language': 'javascript'
}
```

### Stage 7: Output Formatting and Chunking

**Purpose**: Format compressed data for optimal LLM consumption

**Chunking Strategies**:

1. **Semantic Chunking**:
   - Groups related files by import/export dependencies
   - Builds dependency graph and finds connected components
   - Keeps related code together

2. **Size-Based Chunking**:
   - Simple division by token count
   - Ensures chunks fit in context window

3. **Balanced Chunking**:
   - Distributes files evenly across chunks
   - Sorts by complexity for better distribution
   - Minimizes chunk size variance

**Output Formats**:
```
Markdown:
## path/to/file.py
### Imports
- numpy
- pandas
### Functions
- def process_data(df: DataFrame) -> DataFrame
- def calculate_metrics(data: np.array) -> Dict

JSON:
{
  "path": "path/to/file.py",
  "type": "structural",
  "structure": {
    "imports": ["numpy", "pandas"],
    "functions": [...]
  }
}

Custom (LLM-optimized):
[FILE: path/to/file.py]
[TYPE: STRUCTURAL]
[LANG: python | SIZE: 2048 | COMPLEXITY: 8.5]

[IMPORTS]
numpy
pandas

[FUNCTIONS]
process_data(df: DataFrame) -> DataFrame
calculate_metrics(data: np.array) -> Dict
```

### Stage 8: Performance Optimization

**Purpose**: Monitor and optimize pipeline performance

**Metrics Tracked**:
- Stage execution time
- Memory usage delta
- Cache hit rates
- Compression ratios

**Auto-Tuning Parameters**:
```python
# Dynamic batch sizing
batch_size = optimize_batch_size(file_sizes, available_memory)

# Adaptive compression levels
compression_level = optimize_compression_level(content_type, size)

# Worker pool sizing
num_workers = min(cpu_count(), len(files) // min_files_per_worker)
```

## Data Transformation Pipeline

```
Raw Codebase
    ↓
[Discovery] → File paths with basic metadata
    ↓
[Parsing] → Structured FileMetadata objects
    ↓
[Caching] → Deduplicated metadata (skip unchanged)
    ↓
[Storage] → Columnar format with indices
    ↓
[Query] → Filtered file set
    ↓
[Compression] → Strategy-based transformation
    ↓
[Formatting] → LLM-ready text format
    ↓
[Chunking] → Context-window-sized chunks
    ↓
Output Files
```

## Performance Characteristics

### Time Complexity
- File parsing: O(n) where n is file size
- Metadata storage: O(1) for lookups, O(m) for queries
- Compression: O(n) with deduplication overhead
- Chunking: O(k log k) for semantic clustering

### Space Complexity
- Streaming: O(window_size) memory per worker
- Cache: O(num_files * metadata_size)
- Indices: O(unique_symbols * avg_occurrences)

### Throughput Optimizations
1. **Parallel parsing**: Linear speedup with CPU cores
2. **Streaming compression**: Constant memory usage
3. **Incremental updates**: Process only changes
4. **Columnar queries**: Vectorized operations
5. **Adaptive batching**: Optimal memory utilization

## Configuration and Tuning

### Key Parameters
```python
# Processing
num_workers = cpu_count()
batch_size = 100  # files per batch
window_size = 1MB  # compression window
chunk_size = 64KB  # sliding increment

# Caching
cache_ttl = 24 * 3600  # seconds
max_cache_size = 10GB

# Compression
compression_level = 12  # LZ4 level
dedup_threshold = 0.9  # similarity threshold

# Output
max_context_size = 128000  # tokens
chunk_strategy = 'semantic'  # or 'size', 'balanced'
```

### Scaling Considerations

**For Large Codebases (>1M files)**:
- Use distributed cache (Redis)
- Implement sharded metadata store
- Enable streaming output to cloud storage
- Use memory-mapped files for parsing

**For High Throughput**:
- Increase worker pool size
- Enable GPU acceleration for tokenization
- Use faster compression (LZ4 level 1)
- Implement read-ahead caching

**For Memory Constraints**:
- Reduce window size
- Enable on-disk metadata store
- Use aggressive compression strategies
- Implement file sampling

## Example Usage Patterns

### 1. Full Codebase Analysis
```python
# Process entire codebase with structural compression
output = await pipeline.process_codebase_async(
    codebase_path=Path('./project'),
    compression_strategy='structural'
)
```

### 2. Targeted Extraction
```python
# Extract only complex Python files
output = await pipeline.process_codebase_async(
    codebase_path=Path('./project'),
    query_filter={
        'language': 'python',
        'min_complexity': 10.0
    },
    compression_strategy='full'
)
```

### 3. Dependency Analysis
```python
# Find all files importing specific module
output = await pipeline.process_codebase_async(
    codebase_path=Path('./project'),
    query_filter={
        'imports': 'tensorflow'
    },
    compression_strategy='signature'
)
```

### 4. Incremental Updates
```python
# Process only changes since last run
pipeline.cache.cleanup_expired()  # Clean old entries
output = await pipeline.process_codebase_async(
    codebase_path=Path('./project'),
    compression_strategy='structural'
)
```

## Monitoring and Debugging

### Performance Monitoring
```python
# Enable profiling
pipeline.optimizer.profiling_enabled = True

# Process codebase
await pipeline.process_codebase_async(...)

# Get performance report
report = pipeline.optimizer.get_performance_report()
```

### Debug Output
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Track memory usage
import tracemalloc
tracemalloc.start()
```

### Health Checks
```python
# Verify cache integrity
cache.verify_index()

# Check metadata store consistency
metadata_store.validate_indices()

# Monitor compression ratios
compressor.get_compression_stats()
```