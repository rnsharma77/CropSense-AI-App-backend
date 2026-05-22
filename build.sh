#!/bin/bash
# CropSense AI — Render Build Script
# This ensures all dependencies are properly installed

set -e

echo "====== CropSense Build Script ======"
echo "Current directory: $(pwd)"
echo "Node version: $(node --version)"
echo "npm version: $(npm --version)"
echo "Python version: $(python3 --version)"
echo ""

# Step 1: Install Node dependencies
echo "Step 1: Installing Node.js dependencies..."
npm install --verbose

# Step 2: Upgrade pip, setuptools, and wheel
echo "Step 2: Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

# Step 3: Install Python ML dependencies (using CPU-only PyTorch index)
echo "Step 3: Installing Python ML dependencies..."
echo "Contents of requirements.txt:"
cat requirements.txt
echo ""
echo "Installing with PyTorch CPU index URL..."
python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --verbose

# Verify torch installation
echo "Step 4: Verifying torch installation..."
python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully on CPU')" || {
  echo "❌ PyTorch import failed!"
  echo "Troubleshooting: Check if Render build timed out or ran out of disk space"
  echo "If so, consider using a Docker container instead"
  exit 1
}

# Verify other key packages
python3 -c "import torchvision; print(f'✓ TorchVision {torchvision.__version__} installed')"
python3 -c "import sklearn; print(f'✓ Scikit-learn installed')"
python3 -c "import PIL; print(f'✓ Pillow installed')"

echo ""
echo "====== Build completed successfully ======"
