#!/bin/bash

# Setup script for Long Video to Shorts

echo "Setting up Long Video to Shorts..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit the .env file and add your API keys."
fi

echo "Setup complete! You can now run the highlight generator."
echo "To activate the virtual environment in the future, run: source venv/bin/activate"
echo "To run the highlight generator, run: python highlight_generator.py <input_video>" 