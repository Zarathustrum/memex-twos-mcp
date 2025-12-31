#!/usr/bin/env bash
#
# Wrapper script for setup_wizard.py with argument passthrough support.
#
# Usage:
#   ./setup_wizard.sh <path-to-twos-export.md> [path-to-claude-config.json] [--flags...]
#   ./setup_wizard.sh --jump [--flags...]
#   ./setup_wizard.sh --help

set -e

# Parse arguments
EXPORT_FILE=""
CLAUDE_CONFIG_PATH=""
EXTRA_ARGS=()

# Check if first arg is a flag (starts with --)
if [ $# -eq 0 ] || [[ "$1" == "--help" ]]; then
    echo "Setup Wizard Wrapper - Activates venv and runs setup_wizard.py"
    echo ""
    echo "Usage:"
    echo "  ./setup_wizard.sh <export-file> [claude-config] [--flags...]"
    echo "  ./setup_wizard.sh --jump [--flags...]"
    echo "  ./setup_wizard.sh --help"
    echo ""
    echo "Examples:"
    echo "  # Standard setup with export file"
    echo "  ./setup_wizard.sh /path/to/twos_export.md"
    echo ""
    echo "  # With Claude config"
    echo "  ./setup_wizard.sh export.md /path/to/claude_config.json"
    echo ""
    echo "  # Jump mode for testing (no export file needed)"
    echo "  ./setup_wizard.sh --jump"
    echo ""
    echo "  # Verbose mode with flags"
    echo "  ./setup_wizard.sh export.md --verbose --no-color"
    echo ""
    echo "  # Pass any setup_wizard.py flags through"
    echo "  ./setup_wizard.sh export.md --groom --ai-analysis"
    echo ""
    echo "All flags are passed through to scripts/setup_wizard.py"
    echo "See: python3 scripts/setup_wizard.py --help"
    exit 0
fi

# Parse positional arguments and flags
while [ $# -gt 0 ]; do
    if [[ "$1" == --* ]]; then
        # It's a flag - add to extra args
        EXTRA_ARGS+=("$1")
        shift
    else
        # It's a positional argument
        if [ -z "$EXPORT_FILE" ]; then
            EXPORT_FILE="$1"
            shift
        elif [ -z "$CLAUDE_CONFIG_PATH" ]; then
            CLAUDE_CONFIG_PATH="$1"
            shift
        else
            # Extra positional arg - treat as flag
            EXTRA_ARGS+=("$1")
            shift
        fi
    fi
done

# Validate export file if provided
if [ -n "$EXPORT_FILE" ] && [ ! -f "$EXPORT_FILE" ]; then
    echo "ERROR: Export file not found: $EXPORT_FILE"
    exit 1
fi

# Validate claude config if provided
if [ -n "$CLAUDE_CONFIG_PATH" ] && [ ! -f "$CLAUDE_CONFIG_PATH" ]; then
    echo "ERROR: Claude config file not found: $CLAUDE_CONFIG_PATH"
    exit 1
fi

# Show what we're processing
if [ -n "$EXPORT_FILE" ]; then
    echo "Processing Twos export: $EXPORT_FILE"
else
    echo "Running setup wizard (no export file specified)"
fi
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

# Build command with all arguments
echo "Running setup wizard..."
echo ""

CMD_ARGS=()

# Add export file if provided
if [ -n "$EXPORT_FILE" ]; then
    CMD_ARGS+=(--export-file "$EXPORT_FILE")
fi

# Add claude config if provided
if [ -n "$CLAUDE_CONFIG_PATH" ]; then
    CMD_ARGS+=(--claude-config "$CLAUDE_CONFIG_PATH")
fi

# Add all extra flags
CMD_ARGS+=("${EXTRA_ARGS[@]}")

# Run Python script with all arguments
python3 scripts/setup_wizard.py "${CMD_ARGS[@]}"
