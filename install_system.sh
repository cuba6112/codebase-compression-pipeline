#!/bin/bash
# Install system dependencies for macOS

echo "🚀 Installing system dependencies..."

# Make script executable
chmod +x install_system.sh

# Install Redis Python library
echo "📦 Installing Redis Python library..."
pip3 install redis

# Install Homebrew packages if not already installed
if ! command -v redis-cli &> /dev/null; then
    echo "📦 Installing Redis..."
    brew install redis
    brew services start redis
fi

if ! command -v go &> /dev/null; then
    echo "📦 Installing Go..."
    brew install go
fi

if ! command -v rustc &> /dev/null; then
    echo "📦 Installing Rust..."
    brew install rust
fi

# Install Python packages
echo "🐍 Installing Python packages..."
pip3 install python-magic watchdog py-tree-sitter

# Build Go parser if Go is available
if command -v go &> /dev/null; then
    echo "🔨 Building Go parser..."
    if [ -f "go_ast_parser.go" ]; then
        go build -o go_parser go_ast_parser.go
        echo "✅ Go parser built"
    fi
fi

# Build Rust parser if Rust is available
if command -v rustc &> /dev/null; then
    echo "🔨 Building Rust parser..."
    if [ -f "rust_ast_parser.rs" ]; then
        rustc rust_ast_parser.rs -o rust_parser
        echo "✅ Rust parser built"
    fi
fi

echo "✅ All dependencies installed!"