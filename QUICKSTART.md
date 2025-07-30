# Quick Start Guide - Enhanced Codebase Compression Pipeline

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import lz4, mmh3, psutil; print('Dependencies installed successfully!')"
```

## Basic Usage

### Simple Compression

```python
import asyncio
from pathlib import Path
from codebase_compression_pipeline import CodebaseCompressionPipeline

async def compress_project():
    # Create pipeline
    pipeline = CodebaseCompressionPipeline(
        cache_dir=Path('./cache'),
        output_dir=Path('./compressed_output')
    )
    
    # Process codebase
    output_files = await pipeline.process_codebase_async(
        codebase_path=Path('./my_project'),
        output_format='markdown'  # or 'json', 'xml'
    )
    
    print(f"Compressed files: {output_files}")

# Run
asyncio.run(compress_project())
```

### Using the Command Line

```bash
# Basic usage
compress /path/to/project

# With custom format and strategy
compress /path/to/project json structural

# Clear cache before compression
compress --reset-cache /path/to/project
```

## Advanced Features

### 1. With Security Configuration

```python
from security_validation import SecurityConfig

# Configure security
security_config = SecurityConfig(
    allowed_base_paths=[Path.home() / 'projects'],
    max_file_size=50 * 1024 * 1024,  # 50MB limit
    enable_content_scanning=True  # Scan for sensitive data
)

pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./output'),
    security_config=security_config
)
```

### 2. With Resilience Features

```python
# Enable all resilience features
pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./output'),
    enable_resilience=True  # Enables retry, circuit breaker, health checks
)

# Use resilient processing
output_files = await pipeline.process_codebase_resilient(
    codebase_path=Path('./my_project')
)
```

### 3. With Custom Ignore Patterns

```python
output_files = await pipeline.process_codebase_async(
    codebase_path=Path('./my_project'),
    ignore_patterns=[
        'node_modules',
        'venv',
        '*.log',
        'build',
        'dist',
        '__pycache__'
    ]
)
```

### 4. With Query Filters

```python
# Process only Python files with high complexity
output_files = await pipeline.process_codebase_async(
    codebase_path=Path('./my_project'),
    query_filter={
        'language': 'python',
        'min_complexity': 10.0
    }
)
```

### 5. Adaptive Configuration

```python
from pipeline_configs import AdaptiveConfig

# Automatically tune settings based on system resources
config = AdaptiveConfig.auto_configure(Path('./my_project'))

pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./output'),
    num_workers=config.num_workers,
    batch_size=config.batch_size
)
```

## Output Formats

### 1. Markdown (Default)
- Human-readable format
- Great for documentation
- Includes code snippets and metadata

### 2. JSON
- Machine-readable format
- Complete metadata
- Easy to parse and analyze

### 3. XML
- Structured format
- Compatible with enterprise tools
- Includes all metadata

## Compression Strategies

### 1. Full
- Complete code and metadata
- No information loss
- Largest output size

### 2. Structural (Default)
- Code structure and signatures
- Removes implementation details
- Balanced size/information

### 3. Signature
- Only function/class signatures
- Minimal size
- API documentation focus

### 4. Summary
- High-level overview
- Smallest size
- Quick understanding

## Testing

### Run All Tests
```bash
python run_tests.py -v
```

### Run with Coverage
```bash
python run_tests.py -c
```

### Run Specific Tests
```bash
python run_tests.py -t test_security
```

## Monitoring Performance

```python
from pipeline_monitoring import PipelineMonitor

# Create monitor
monitor = PipelineMonitor()
monitor.start_monitoring()

# Process with monitoring
await pipeline.process_codebase_async(Path('./project'))

# Get performance metrics
metrics = monitor.get_summary()
print(f"Processing speed: {metrics['throughput_mb_per_sec']} MB/s")
```

## Common Use Cases

### 1. Analyzing Large Codebases
```python
# Optimized for large projects
from pipeline_configs import ConfigPresets

config = ConfigPresets.large_codebase()
pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./output'),
    num_workers=config.num_workers
)
```

### 2. Memory-Constrained Systems
```python
config = ConfigPresets.memory_constrained()
# Uses streaming and minimal memory
```

### 3. Real-Time Processing
```python
config = ConfigPresets.real_time()
# Optimized for low latency
```

## Troubleshooting

### Issue: "Too many files" error
**Solution**: Increase file limit in security config
```python
security_config = SecurityConfig(max_total_files=200000)
```

### Issue: "File too large" error
**Solution**: Increase file size limit
```python
security_config = SecurityConfig(max_file_size=200*1024*1024)  # 200MB
```

### Issue: Permission denied
**Solution**: Check file permissions and allowed paths
```python
security_config = SecurityConfig(
    allowed_base_paths=[Path.cwd(), Path.home()]
)
```

### Issue: Slow processing
**Solution**: Increase workers and check cache
```python
pipeline = CodebaseCompressionPipeline(
    cache_dir=Path('./cache'),
    output_dir=Path('./output'),
    num_workers=8  # Increase workers
)
```

## Best Practices

1. **Always use absolute paths** for reliability
2. **Enable caching** for repeated processing
3. **Configure security** based on your environment
4. **Use resilient processing** for production
5. **Monitor performance** for large codebases
6. **Run tests** after configuration changes

## Next Steps

- See `CLAUDE.md` for AI assistant integration
- Check `pipeline_architecture.md` for technical architecture details
- Explore example configurations in `pipeline_configs.py`
- Review the test suite in `tests/` for usage examples

## Getting Help

- Run tests to verify setup: `python run_tests.py`
- Check logs in the output directory
- Review error messages for security validation
- Enable debug logging for detailed information

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```