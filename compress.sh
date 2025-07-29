#!/bin/bash

# Codebase Compression Pipeline Wrapper
# This script provides a convenient way to run the compression pipeline from any directory

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set Python path (using the conda environment detected earlier)
PYTHON_PATH="/opt/anaconda3/bin/python"

# Default values
CODEBASE_PATH="${1:-.}"
OUTPUT_FORMAT="${2:-markdown}"
COMPRESSION_STRATEGY="${3:-structural}"

# Help function
show_help() {
    echo "Usage: compress [OPTIONS] [CODEBASE_PATH] [OUTPUT_FORMAT] [COMPRESSION_STRATEGY]"
    echo ""
    echo "Arguments:"
    echo "  CODEBASE_PATH         Path to compress (default: current directory)"
    echo "  OUTPUT_FORMAT         Output format: markdown, json, xml, custom (default: markdown)"
    echo "  COMPRESSION_STRATEGY  Strategy: full, structural, signature, summary (default: structural)"
    echo ""
    echo "Options:"
    echo "  -h, --help           Show this help message"
    echo "  --reset-cache        Clear the cache before processing"
    echo ""
    echo "Examples:"
    echo "  compress                     # Compress current directory"
    echo "  compress /path/to/project    # Compress specific directory"
    echo "  compress . json full         # Full compression in JSON format"
    echo "  compress --reset-cache .     # Clear cache and compress current dir"
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

# Check for cache reset flag
RESET_CACHE=false
if [[ "$1" == "--reset-cache" ]]; then
    RESET_CACHE=true
    shift  # Remove the flag from arguments
    CODEBASE_PATH="${1:-.}"
    OUTPUT_FORMAT="${2:-markdown}"
    COMPRESSION_STRATEGY="${3:-structural}"
fi

# Check if tqdm is installed and install if missing
"$PYTHON_PATH" -c "import tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing tqdm for progress bars..."
    "$PYTHON_PATH" -m pip install tqdm --quiet
fi

# Create temporary Python script that accepts command line arguments
# Use a more unique template and ensure proper creation
TEMP_SCRIPT=$(mktemp -t compress_wrapper.XXXXXX.py)

# Check if mktemp succeeded
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
    print(f"Output format: {output_format}")
    print(f"Compression strategy: {compression_strategy}")
    print(f"Output directory: {Path.cwd() / 'compressed_output'}")
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
        '.DS_Store'
    ]
    
    # Process codebase
    output_files = await pipeline.process_codebase(
        codebase_path=codebase_path,
        output_format=output_format,
        compression_strategy=compression_strategy,
        ignore_patterns=ignore_patterns
    )
    
    print("\nCompression complete!")
    print(f"Generated {len(output_files)} output files in ./compressed_output/")

if __name__ == "__main__":
    asyncio.run(main())
EOF

# Set up trap to ensure cleanup happens
trap "rm -f '$TEMP_SCRIPT'" EXIT

# Ensure Rust is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# Run the Python script
"$PYTHON_PATH" "$TEMP_SCRIPT" "$SCRIPT_DIR" "$CODEBASE_PATH" "$OUTPUT_FORMAT" "$COMPRESSION_STRATEGY" "$RESET_CACHE"
EXIT_CODE=$?

# Exit with the same code as the Python script
exit $EXIT_CODE