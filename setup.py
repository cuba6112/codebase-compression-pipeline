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
    packages=find_packages(include=['codebase_compression_pipeline*', 'parsers*', 'pipeline*']),
    py_modules=[
        'codebase_compression_pipeline',
        'pipeline_configs', 
        'pipeline_monitoring',
        'security_validation',
        'resilience_patterns',
        'enhanced_cache',
        'adaptive_compression',
        'base_classes',
        'run_tests'
    ],
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
        "security": ["python-magic"],
        "monitoring": ["prometheus-client>=0.14.0"],
        "typescript": ["nodejs"],  # Note: requires Node.js to be installed separately
        "go": [],  # Go parser requires Go compiler to be installed
        "rust": [],  # Rust parser requires Rust compiler to be installed
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov",
            "pytest-asyncio",
            "pytest-mock",
        ],
        "all": [
            "redis>=4.0.0",
            "numpy",
            "brotli", 
            "zstandard",
            "python-magic",
            "prometheus-client>=0.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "compress-codebase=codebase_compression_pipeline:main",
            "run-pipeline-tests=run_tests:main",
            "pipeline-monitor=pipeline_monitoring:main",
        ],
        "codebase_compression.parsers": [
            "python=parsers.python_parser:PythonParser",
            "javascript=parsers.enhanced_js_parser:EnhancedJavaScriptParser",
            "typescript=parsers.typescript_parser:TypeScriptParser [typescript]",
            "go=parsers.go_parser:GoParser [go]", 
            "rust=parsers.rust_parser:RustParser [rust]",
        ],
        "codebase_compression.output_formats": [
            "markdown=pipeline.stages.formatting:MarkdownFormatter",
            "json=pipeline.stages.formatting:JsonFormatter",
            "xml=pipeline.stages.formatting:XmlFormatter",
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