#!/bin/bash
# Creates a virtual environment and installs all packages needed for pipeline_improved.py

set -e  # exit on any error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python3"

echo "===================================="
echo " Setting up virtual environment"
echo "===================================="

# Use the system Python (Xcode), NOT anaconda
/usr/bin/python3 -m venv --clear "$VENV_DIR"

echo "Virtual environment created at: $VENV_DIR"
echo "Using python: $(/usr/bin/python3 --version)"

echo "Installing packages..."

# Use the venv pip directly by full path — avoids any Anaconda pip contamination
"$PIP" install --upgrade pip

"$PIP" install \
    numpy \
    tifffile \
    imagecodecs \
    matplotlib \
    scikit-image \
    opencv-python-headless \
    imageio \
    imageio-ffmpeg

echo ""
echo "===================================="
echo " Verifying installs"
echo "===================================="
"$PYTHON" -c "import numpy, tifffile, matplotlib, skimage, cv2, imageio; print('All packages imported successfully')"

echo ""
echo "===================================="
echo " Done! To run the pipeline:"
echo "===================================="
echo ""
echo "  source venv/bin/activate"
echo "  python3 pipeline_improved.py"
echo ""
