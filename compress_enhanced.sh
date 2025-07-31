#!/bin/bash

# Enhanced Codebase Compression Pipeline Wrapper
# Supports mode flags like -full, -structural, -signature, -summary

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path - try to detect the best available Python 3
if command -v python3 &> /dev/null; then
    PYTHON_PATH="python3"
elif command -v python &> /dev/null && python --version 2>&1 | grep -q "Python 3"; then
    PYTHON_PATH="python"
elif [ -f "/opt/anaconda3/bin/python" ]; then
    PYTHON_PATH="/opt/anaconda3/bin/python"
else
    echo "Error: Python 3 not found. Please install Python 3 or add it to PATH."
    exit 1
fi

# Check if compress_realtime.py exists for best progress bar support
if [ -f "$SCRIPT_DIR/compress_realtime.py" ]; then
    exec "$PYTHON_PATH" "$SCRIPT_DIR/compress_realtime.py" "$@"
# Fall back to compress.py if available
elif [ -f "$SCRIPT_DIR/compress.py" ]; then
    exec "$PYTHON_PATH" "$SCRIPT_DIR/compress.py" "$@"
fi

# Default values
CODEBASE_PATH="."
OUTPUT_FORMAT="markdown"
COMPRESSION_STRATEGY="structural"
RESET_CACHE=false

# Help function
show_help() {
    echo "Usage: compress [OPTIONS] [CODEBASE_PATH]"
    echo ""
    echo "Compression Modes:"
    echo "  -full         Complete code with minimal compression"
    echo "  -structural   Preserve structure, compress implementation (default)"
    echo "  -signature    Function/class signatures only"
    echo "  -summary      High-level overview and key components"
    echo ""
    echo "Output Formats:"
    echo "  -markdown     Human-readable format for LLMs (default)"
    echo "  -json         Structured data for programmatic processing"
    echo "  -xml          Hierarchical representation with metadata"
    echo ""
    echo "Other Options:"
    echo "  -h, --help    Show this help message"
    echo "  --reset-cache Clear the cache before processing"
    echo ""
    echo "Examples:"
    echo "  compress                      # Compress current directory"
    echo "  compress -full                # Full compression of current dir"
    echo "  compress -summary /path       # Summary of specific path"
    echo "  compress -json -full .        # Full compression in JSON"
    echo "  compress --reset-cache -full  # Clear cache and full compress"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --reset-cache)
            RESET_CACHE=true
            shift
            ;;
        -full)
            COMPRESSION_STRATEGY="full"
            shift
            ;;
        -structural)
            COMPRESSION_STRATEGY="structural"
            shift
            ;;
        -signature)
            COMPRESSION_STRATEGY="signature"
            shift
            ;;
        -summary)
            COMPRESSION_STRATEGY="summary"
            shift
            ;;
        -markdown)
            OUTPUT_FORMAT="markdown"
            shift
            ;;
        -json)
            OUTPUT_FORMAT="json"
            shift
            ;;
        -xml)
            OUTPUT_FORMAT="xml"
            shift
            ;;
        *)
            # Assume it's the codebase path
            CODEBASE_PATH="$1"
            shift
            ;;
    esac
done

# Check if tqdm is installed
"$PYTHON_PATH" -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing tqdm for progress bars..."
    "$PYTHON_PATH" -m pip install tqdm --quiet
fi

# Create temporary Python script
TEMP_SCRIPT=$(mktemp -t compress_wrapper.XXXXXX.py)

if [ -z "$TEMP_SCRIPT" ] || [ ! -f "$TEMP_SCRIPT" ]; then
    echo "Error: Failed to create temporary file"
    exit 1
fi

cat > "$TEMP_SCRIPT" << 'EOF'
import asyncio
import sys
from pathlib import Path

# Add the pipeline directory to Python path
sys.path.insert(0, sys.argv[1])

from codebase_compression_pipeline import CodebaseCompressionPipeline

async def main():
    # Get arguments
    script_dir = sys.argv[1]
    codebase_path = Path(sys.argv[2])
    output_format = sys.argv[3]
    compression_strategy = sys.argv[4]
    reset_cache = sys.argv[5] == 'true'
    
    # Resolve paths
    if not codebase_path.is_absolute():
        codebase_path = Path.cwd() / codebase_path
    
    cache_dir = Path(script_dir) / 'cache'
    
    # Reset cache if requested
    if reset_cache:
        print("Clearing cache...")
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cache cleared successfully")
    
    # Configure pipeline
    pipeline = CodebaseCompressionPipeline(
        cache_dir=cache_dir,
        output_dir=Path.cwd() / 'compressed_output',
        num_workers=4
    )
    
    print(f"Compressing: {codebase_path}")
    print(f"Mode: {compression_strategy}")
    print(f"Format: {output_format}")
    print(f"Output: {Path.cwd() / 'compressed_output'}")
    print("-" * 50)
    
    # Configure ignore patterns
    ignore_patterns = [
        'node_modules',
        '__pycache__',
        '.git',
        '.github',
        '.venv',
        'venv',
        'env',
        '.env',
        'dist',
        'build',
        'target',
        '.next',
        '.cache',
        'coverage',
        '.pytest_cache',
        '.mypy_cache',
        '*.pyc',
        '*.pyo',
        '*.egg-info',
        '.DS_Store',
        'compressed_output'
    ]
    
    # Process codebase
    output_files = await pipeline.process_codebase(
        codebase_path=codebase_path,
        output_format=output_format,
        compression_strategy=compression_strategy,
        ignore_patterns=ignore_patterns
    )
    
    print("\n✅ Compression complete!")
    print(f"📁 Generated {len(output_files)} output files in ./compressed_output/")
    
    # Show file sizes
    total_size = 0
    for file in output_files:
        size = file.stat().st_size
        total_size += size
        print(f"   - {file.name}: {size:,} bytes")
    
    if len(output_files) > 0:
        print(f"📊 Total output size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Set up trap to ensure cleanup
trap "rm -f '$TEMP_SCRIPT'" EXIT

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Run the Python script with unbuffered output for real-time progress
# Direct execution for better progress bar support
PYTHONUNBUFFERED=1 exec "$PYTHON_PATH" "$TEMP_SCRIPT" "$SCRIPT_DIR" "$CODEBASE_PATH" "$OUTPUT_FORMAT" "$COMPRESSION_STRATEGY" "$RESET_CACHE"
EXIT_CODE=$?

# Exit with the same code as the Python script
exit $EXIT_CODE