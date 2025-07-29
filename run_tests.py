#!/usr/bin/env python3
"""
Test Runner for Codebase Compression Pipeline
============================================

Runs the comprehensive test suite with various options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(args):
    """Run pytest with specified options"""
    cmd = ["pytest", "tests/"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add coverage
    if args.coverage:
        cmd.extend(["--cov=.", "--cov-report=html", "--cov-report=term"])
    
    # Run specific test
    if args.test:
        cmd.extend(["-k", args.test])
    
    # Run only fast tests (exclude slow/performance tests)
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    # Show output
    if args.show_output:
        cmd.append("-s")
    
    # Number of parallel workers
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Run with pytest
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for codebase compression pipeline")
    
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Verbose test output")
    parser.add_argument("-c", "--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("-t", "--test", type=str,
                       help="Run specific test by name pattern")
    parser.add_argument("-f", "--fast", action="store_true",
                       help="Skip slow/performance tests")
    parser.add_argument("-s", "--show-output", action="store_true",
                       help="Show print statements during tests")
    parser.add_argument("-p", "--parallel", type=int,
                       help="Run tests in parallel with N workers")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install test dependencies first")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        print("Installing test dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run tests
    exit_code = run_tests(args)
    
    # Show coverage report location
    if args.coverage and exit_code == 0:
        print("\nCoverage report generated in htmlcov/index.html")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()