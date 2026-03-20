#!/bin/bash
# Run SigProcessingPipeline.py using the local venv.
#
# Usage:
#   ./run.sh path/to/your_imaging_file.tif
#   ./run.sh path/to/your_imaging_file.tif --outdir path/to/output

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$SCRIPT_DIR/venv/bin/python3"

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: venv not found at $SCRIPT_DIR/venv. Run ./setup_env.sh first."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Usage: $0 path/to/imaging_file.tif [--outdir path/to/output]"
    exit 1
fi

# Clear Anaconda PYTHONPATH contamination so venv packages are used exclusively
unset PYTHONPATH
unset PYTHONHOME

echo "Using: $($PYTHON --version) from $PYTHON"
cd "$SCRIPT_DIR"
"$PYTHON" SigProcessingPipeline.py "$@"
