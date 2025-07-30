#!/usr/bin/env python3
"""
Enhanced wrapper for the codebase compression pipeline with proper progress bar support.
"""

import asyncio
import sys
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from codebase_compression_pipeline import CodebaseCompressionPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress codebases for LLM context optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  compress.py                      # Compress current directory
  compress.py -full                # Full compression of current dir
  compress.py -summary /path       # Summary of specific path
  compress.py -json -full .        # Full compression in JSON
  compress.py --reset-cache -full  # Clear cache and full compress
        """
    )
    
    # Compression modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-full', action='store_const', dest='strategy', const='full',
                           help='Complete code with minimal compression')
    mode_group.add_argument('-structural', action='store_const', dest='strategy', const='structural',
                           help='Preserve structure, compress implementation (default)')
    mode_group.add_argument('-signature', action='store_const', dest='strategy', const='signature',
                           help='Function/class signatures only')
    mode_group.add_argument('-summary', action='store_const', dest='strategy', const='summary',
                           help='High-level overview and key components')
    
    # Output formats
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument('-markdown', action='store_const', dest='format', const='markdown',
                             help='Human-readable format for LLMs (default)')
    format_group.add_argument('-json', action='store_const', dest='format', const='json',
                             help='Structured data for programmatic processing')
    format_group.add_argument('-xml', action='store_const', dest='format', const='xml',
                             help='Hierarchical representation with metadata')
    
    # Other options
    parser.add_argument('--reset-cache', action='store_true',
                       help='Clear the cache before processing')
    parser.add_argument('path', nargs='?', default='.',
                       help='Path to codebase (default: current directory)')
    
    # Set defaults
    parser.set_defaults(strategy='structural', format='markdown')
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Resolve paths
    codebase_path = Path(args.path).resolve()
    script_dir = Path(__file__).parent
    cache_dir = script_dir / 'cache'
    output_dir = Path.cwd() / 'compressed_output'
    
    # Reset cache if requested
    if args.reset_cache:
        print("Clearing cache...")
        import shutil
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cache cleared successfully")
        # Recreate cache directory
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure pipeline
    pipeline = CodebaseCompressionPipeline(
        cache_dir=cache_dir,
        output_dir=output_dir,
        num_workers=4
    )
    
    print(f"Compressing: {codebase_path}")
    print(f"Mode: {args.strategy}")
    print(f"Format: {args.format}")
    print(f"Output: {output_dir}")
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
    try:
        output_files = await pipeline.process_codebase(
            codebase_path=codebase_path,
            output_format=args.format,
            compression_strategy=args.strategy,
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
        
        if output_files:
            print(f"📊 Total output size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())