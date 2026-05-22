#!/bin/bash
# setup-python.sh - Install Python dependencies for Render deployment

set -e

echo "========================================="
echo "Setting up PyTorch for CropSense Backend"
echo "========================================="

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ python3 not found!"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet

# Install requirements with CPU PyTorch
echo "Installing PyTorch (CPU-only) and dependencies..."
echo "This may take 2-3 minutes..."

# Main install with CPU index
if python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --quiet; then
    echo "✓ Dependencies installed successfully"
else
    echo "⚠ First install attempt failed, retrying with --no-cache-dir..."
    python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --no-cache-dir --quiet
fi

# Verify PyTorch
echo ""
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} successfully installed')" || {
    echo "❌ PyTorch verification failed!"
    exit 1
}

# Verify other critical packages
python3 -c "import torchvision; print(f'✓ TorchVision installed')"
python3 -c "import PIL; print(f'✓ Pillow installed')"
python3 -c "import pymongo; print(f'✓ PyMongo installed')"

echo ""
echo "========================================="
echo "✓ All Python dependencies ready!"
echo "========================================="
