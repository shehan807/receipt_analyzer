#!/bin/bash
# Setup script for receipt_analyzer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up receipt_analyzer..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Setup complete!"
echo ""
echo "To use the tool:"
echo "  1. Activate the virtual environment:"
echo "     source $VENV_DIR/bin/activate"
echo ""
echo "  2. Set your API key:"
echo "     export ANTHROPIC_API_KEY=\"sk-...\""
echo ""
echo "  3. Run the analyzer:"
echo "     python $SCRIPT_DIR/analyze_receipts.py /path/to/receipts"
echo ""
echo "  4. Or use dry-run mode first:"
echo "     python $SCRIPT_DIR/analyze_receipts.py /path/to/receipts --dry-run"
