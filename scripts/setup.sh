#!/bin/bash
# Healthcare IDP System Setup Script

set -e

echo "================================================"
echo "Healthcare IDP System - Setup Script"
echo "================================================"

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.9"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi
echo "Python version OK: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model
echo ""
echo "Downloading spaCy language model..."
python -m spacy download en_core_web_lg

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/samples
mkdir -p models/classification
mkdir -p models/ner
mkdir -p logs

# Copy environment file
echo ""
echo "Setting up environment file..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env file created from template. Please update with your credentials."
else
    echo ".env file already exists."
fi

# Run tests
echo ""
echo "Running tests..."
pytest tests/ -v --tb=short || echo "Some tests failed. Please check the output above."

# Summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Update .env with your AWS credentials"
echo "2. Run the pipeline: python -m src.pipeline"
echo "3. Start the API: uvicorn api.main:app --reload"
echo "4. Access API docs: http://localhost:8000/docs"
echo ""
