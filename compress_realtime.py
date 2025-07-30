#!/usr/bin/env python3
"""
Real-time wrapper for the compression pipeline that properly handles progress bars.
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Run compression with real-time output."""
    # Get script directory
    script_dir = Path(__file__).parent
    python_path = "/opt/anaconda3/bin/python"
    
    # Build command
    cmd = [python_path, str(script_dir / "compress.py")] + sys.argv[1:]
    
    # Set environment for unbuffered output
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    
    # Run with real-time output
    try:
        # Use subprocess.Popen for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,  # Line buffered
            env=env
        )
        
        # Read and print output line by line
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        
        # Wait for process to complete
        return_code = process.wait()
        return return_code
        
    except KeyboardInterrupt:
        if process:
            process.terminate()
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())