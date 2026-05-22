#!/bin/bash
# setup-python.sh - Install Python dependencies for Render deployment

set -e

echo "========================================="
echo "Setting up PyTorch for CropSense Backend"
echo "========================================="

if ! command -v python3 &> /dev/null; then
    echo "python3 not found!"
    exit 1
fi

echo "Python found: $(python3 --version)"
echo ""

echo "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet

echo "Installing PyTorch (CPU-only) and dependencies..."
echo "This may take 2-3 minutes..."

if python3 -m pip install -r requirements.txt --quiet; then
    echo "Dependencies installed successfully"
else
    echo "First install attempt failed, retrying with --no-cache-dir..."
    python3 -m pip install -r requirements.txt --no-cache-dir --quiet
fi

echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} successfully installed')" || {
    echo "PyTorch verification failed!"
    exit 1
}

python3 -c "import torchvision; print('TorchVision installed')"
python3 -c "import PIL; print('Pillow installed')"
python3 -c "import pymongo; print('PyMongo installed')"

echo ""
echo "========================================="
echo "All Python dependencies ready!"
echo "========================================="
