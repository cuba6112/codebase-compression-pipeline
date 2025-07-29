#!/usr/bin/env python3
"""
Setup configuration for Codebase Compression Pipeline.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements from requirements.txt
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="codebase-compression-pipeline",
    version="1.0.0",
    author="Project Think",
    author_email="",
    description="High-performance pipeline for compressing codebases for LLM context optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/codebase-compression-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Documentation",
        "Topic :: Text Processing :: Markup",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "redis": ["redis>=4.0.0"],
        "advanced": ["numpy", "brotli", "zstandard"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ],
        "typescript": ["nodejs"],
    },
    entry_points={
        "console_scripts": [
            "compress-codebase=codebase_compression_pipeline:main",
            "run-pipeline-tests=run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.md",
            "*.txt",
            "*.json",
            "*.js",
            "*.go",
            "*.rs",
        ],
    },
    keywords=[
        "code-compression",
        "llm",
        "ast-parsing",
        "codebase-analysis",
        "documentation",
        "code-review",
        "large-language-models",
        "code-optimization",
        "token-reduction",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/codebase-compression-pipeline/issues",
        "Source": "https://github.com/your-username/codebase-compression-pipeline",
        "Documentation": "https://github.com/your-username/codebase-compression-pipeline/wiki",
    },
)