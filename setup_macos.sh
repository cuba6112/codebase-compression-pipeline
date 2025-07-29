#!/bin/bash
# Setup script for macOS development environment
# For the Codebase Compression Pipeline project

set -e  # Exit on error

echo "🚀 Setting up macOS development environment..."

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Please install from https://brew.sh"
    exit 1
fi

echo "✅ Homebrew found: $(brew --version | head -n1)"

# Update Homebrew
echo "📦 Updating Homebrew..."
brew update

# Install missing tools
echo "🔧 Installing development tools..."

# Go
if ! command -v go &> /dev/null; then
    echo "📦 Installing Go..."
    brew install go
else
    echo "✅ Go already installed: $(go version)"
fi

# Rust
if ! command -v rustc &> /dev/null; then
    echo "📦 Installing Rust..."
    brew install rust
else
    echo "✅ Rust already installed: $(rustc --version)"
fi

# Redis
if ! command -v redis-cli &> /dev/null; then
    echo "📦 Installing Redis..."
    brew install redis
else
    echo "✅ Redis already installed: $(redis-cli --version)"
fi

# Python packages
echo "🐍 Installing Python packages..."
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Additional packages for new features
pip3 install redis watchdog py-tree-sitter

# Node.js packages for TypeScript parser
echo "📦 Installing Node.js packages..."
npm install -g @typescript-eslint/parser @typescript-eslint/typescript-estree typescript

# Optional: Install MLX for GPU acceleration on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo "🎯 Detected Apple Silicon - Installing MLX for GPU acceleration..."
    pip3 install mlx
fi

# Create project directories if needed
mkdir -p cache compressed_output logs

# Start Redis service
echo "🚀 Starting Redis service..."
brew services start redis

# Verify installations
echo "✅ Verifying installations..."
echo "---"
command -v go && go version || echo "❌ Go not found"
command -v rustc && rustc --version || echo "❌ Rust not found"
command -v redis-cli && redis-cli ping || echo "❌ Redis not responding"
command -v tsc && tsc --version || echo "❌ TypeScript not found"
echo "---"

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Run tests: python run_tests.py -v"
echo "2. Start implementing TypeScript parser"
echo "3. Check Redis: redis-cli ping"
echo ""
echo "💡 For GPU acceleration on Apple Silicon, we installed MLX."
echo "   Use it instead of CUDA for Metal GPU compute."