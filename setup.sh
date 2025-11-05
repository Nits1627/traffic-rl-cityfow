#!/bin/bash
# Setup script for Traffic RL SUMO project

set -e  # Exit on any error

echo "🚦 Setting up Traffic RL SUMO project..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This project is designed for macOS only"
    exit 1
fi

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "❌ Python 3.10+ is required. Found: $python_version"
    echo "   Please install Python 3.10 or 3.11"
    exit 1
fi

echo "✅ Python version: $python_version"

# Check if SUMO is installed
echo "🔍 Checking SUMO installation..."
if ! command -v sumo &> /dev/null; then
    echo "❌ SUMO not found in PATH"
    echo "   Please install SUMO from: https://eclipse.org/sumo/"
    echo "   And set SUMO_HOME environment variable"
    exit 1
fi

if [[ -z "$SUMO_HOME" ]]; then
    echo "❌ SUMO_HOME environment variable not set"
    echo "   Please add to your ~/.zshrc or ~/.bash_profile:"
    echo "   export SUMO_HOME=\"/Library/Frameworks/sumo.framework/Versions/Current\""
    echo "   export PATH=\"\$SUMO_HOME/bin:\$PATH\""
    echo "   export PYTHONPATH=\"\$SUMO_HOME/tools:\$PYTHONPATH\""
    exit 1
fi

echo "✅ SUMO found: $SUMO_HOME"

# Create virtual environment
echo "📦 Creating virtual environment..."
if [[ ! -d ".venv" ]]; then
    python3 -m venv .venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/*.py

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p network routes sumo_configs output models logs

# Generate network files
echo "🌐 Generating SUMO network files..."
python scripts/generate_network.py

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🚀 Next steps:"
echo "   1. Activate virtual environment: source .venv/bin/activate"
echo "   2. Train the agent: python scripts/train.py"
echo "   3. Evaluate the agent: python scripts/evaluate.py --model models/ppo_traffic_light/final_model.zip"
echo "   4. Launch GUI: python scripts/launch_gui.py"
echo ""
echo "📚 For more information, see README.md"
