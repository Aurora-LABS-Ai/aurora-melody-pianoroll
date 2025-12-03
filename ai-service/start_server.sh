#!/bin/bash
# ============================================================================
# Aurora Melody AI Server Startup Script (Linux/macOS)
# ============================================================================
#
# The server loads configuration from (priority order):
#   1. Command-line arguments (if provided)
#   2. Environment variables / .env file
#   3. config.yaml
#
# Quick Start:
#   1. Copy config.yaml.example to config.yaml (or edit config.yaml)
#   2. Update the model paths in config.yaml
#   3. Run: ./start_server.sh
#
# Or with arguments:
#   ./start_server.sh --model_path model.safetensors --config_path config.json --vocab_path vocab.pkl
# ============================================================================

set -e

echo ""
echo "============================================="
echo "  Aurora Melody AI Server"
echo "============================================="
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if config.yaml exists
if [ -f "config.yaml" ]; then
    echo "Config: config.yaml found"
else
    echo "Note: config.yaml not found - using defaults or CLI args"
fi

# Check if .env exists
if [ -f ".env" ]; then
    echo "Env:    .env found"
fi

echo ""
echo "Starting server..."
echo ""

# Pass all arguments to the Python script
python3 melody_server.py "$@"
