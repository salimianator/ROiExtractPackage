#!/bin/bash
# Launch cell_editor_gui.py using the local venv.
#
# Usage:
#   ./run_gui.sh                                         # auto-loads last pipeline run
#   ./run_gui.sh path/to/your_imaging_file.tif           # explicit TIFF path
#   ./run_gui.sh path/to/your_imaging_file.tif path/to/labeled_mask.npy

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found at $SCRIPT_DIR/venv. Run ./setup_env.sh first."
    exit 1
fi

# Clear Anaconda PYTHONPATH contamination so venv packages are used exclusively
unset PYTHONPATH
unset PYTHONHOME

echo "Using: $($PYTHON --version) from $PYTHON"
cd "$SCRIPT_DIR"
"$PYTHON" cell_editor_gui.py "$@"
