#!/bin/bash

# KyleBot Launcher Script
# This script helps you start KyleBot with the web interface

echo "🤖 Welcome to KyleBot!"
echo "======================"

# Check if virtual environment exists
if [ ! -d "kylebot_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv kylebot_env"
    echo "Then run: source kylebot_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source kylebot_env/bin/activate

# Check if Gradio is installed
if ! python -c "import gradio" 2>/dev/null; then
    echo "📦 Installing Gradio..."
    pip install gradio
fi

echo "🚀 Starting KyleBot Web Interface..."
echo "📱 Open your browser to: http://localhost:7860"
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Start the Gradio interface
python kylebot_gradio.py 