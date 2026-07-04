#!/bin/bash
# Quick setup script for macOS/Linux

echo "================================================"
echo "LLM Project Setup"
echo "================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

echo ""
echo "Step 2: Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Step 3: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. Verify setup: python verify_setup.py"
echo "  2. Train model: python train.py"
echo "  3. Chat with model: python inference.py"
echo ""
