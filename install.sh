#!/bin/bash
chmod +x install.sh
echo "Installing Redis and dependencies..."

# Install Redis Python library
pip3 install redis

# Install system packages if Homebrew is available
if command -v brew &> /dev/null; then
    echo "Installing Redis with Homebrew..."
    brew install redis
    brew services start redis
    
    echo "Installing Go..."
    brew install go
else
    echo "Homebrew not found. Please install Redis manually:"
    echo "Visit: https://redis.io/docs/getting-started/installation/"
fi

# Install Python packages
pip3 install python-magic watchdog py-tree-sitter

echo "✅ Redis Python library installed"
echo "✅ Dependencies updated"