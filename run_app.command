#!/bin/bash

# YouTube ID & Transcript Processor Launcher
# This script sets up the environment and launches the Streamlit app

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Change to the project directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Launch the Streamlit app
echo "Starting YouTube ID & Transcript Processor..."
echo "The app will open in your browser at http://localhost:8501"
streamlit run integrated_app.py --server.port 8501