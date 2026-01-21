#!/bin/bash

# Setup script for Research Paper Chat Assistant

set -e

echo "=================================="
echo "Research Paper Chat Assistant Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python -m venv venv
    echo "✓ Virtual environment created"
else
    echo ""
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take several minutes..."
pip install -r requirements.txt

# Install Detectron2
echo ""
echo "Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git' || echo "Warning: Detectron2 installation failed. You may need to install it manually."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your OpenRouter API key!"
else
    echo ""
    echo ".env file already exists"
fi

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p data/processed
mkdir -p data/embeddings
mkdir -p logs
mkdir -p chroma_db
echo "✓ Directories created"

# Summary
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OpenRouter API key"
echo "2. Place PDF files in the Data/ directory"
echo "3. Run: python scripts/process_documents.py"
echo "4. Run: streamlit run app.py"
echo ""
echo "For more information, see README.md"
