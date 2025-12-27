#!/usr/bin/env bash
#
# Process a Twos export file through the complete setup pipeline.
#
# Usage: ./setup_wizard.sh <path-to-twos-export.md> [path-to-claude-config.json]

set -e

if [ $# -eq 0 ]; then
    echo "ERROR: No Twos export file specified"
    echo ""
    echo "Usage: ./setup_wizard.sh <path-to-twos-export.md> [path-to-claude-config.json]"
    echo ""
    echo "Example:"
    echo "  ./setup_wizard.sh /path/to/twos_export.md"
    echo "  ./setup_wizard.sh /path/to/twos_export.md /path/to/claude_desktop_config.json"
    echo ""
    echo "This script will:"
    echo "  1. Copy your export to data/raw/twos_export.md"
    echo "  2. Activate the virtual environment"
    echo "  3. Run the setup wizard (convert, groom, load to SQLite)"
    exit 1
fi

EXPORT_FILE="$1"
CLAUDE_CONFIG_PATH="$2"

if [ ! -f "$EXPORT_FILE" ]; then
    echo "ERROR: File not found: $EXPORT_FILE"
    exit 1
fi

if [ -n "$CLAUDE_CONFIG_PATH" ] && [ ! -f "$CLAUDE_CONFIG_PATH" ]; then
    echo "ERROR: Claude config file not found: $CLAUDE_CONFIG_PATH"
    exit 1
fi

echo "Processing Twos export: $EXPORT_FILE"
echo ""

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found at .venv"
    echo ""
    read -p "Would you like to create it now? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
        echo "Virtual environment created successfully"
        echo ""
        echo "Installing dependencies..."
        .venv/bin/pip install -e .
        echo ""
    else
        echo "Setup cancelled. Please create a virtual environment manually with:"
        echo "  python3 -m venv .venv"
        echo "  source .venv/bin/activate"
        echo "  pip install -e ."
        exit 0
    fi
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Run setup wizard
echo "Running setup wizard..."
echo ""
if [ -n "$CLAUDE_CONFIG_PATH" ]; then
    python3 scripts/setup_wizard.py --export-file "$EXPORT_FILE" --claude-config "$CLAUDE_CONFIG_PATH"
else
    python3 scripts/setup_wizard.py --export-file "$EXPORT_FILE"
fi
