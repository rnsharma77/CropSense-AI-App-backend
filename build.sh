#!/bin/bash
# CropSense AI - Render Build Script

set -e

echo "====== CropSense Build Script ======"
echo "Current directory: $(pwd)"
echo "Node version: $(node --version)"
echo "npm version: $(npm --version)"
echo "Python version: $(python3 --version)"
echo ""

echo "Step 1: Installing Node.js dependencies..."
npm install --verbose

echo "Step 2: Upgrading pip, setuptools, and wheel..."
python3 -m pip install --upgrade pip setuptools wheel --quiet

echo "Step 3: Installing Python ML dependencies..."
echo "Contents of requirements.txt:"
cat requirements.txt
echo ""
echo "Installing with PyTorch CPU wheels enabled via requirements.txt..."
python3 -m pip install -r requirements.txt --verbose

echo "Step 4: Verifying torch installation..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully on CPU')" || {
  echo "PyTorch import failed!"
  echo "Troubleshooting: Check if Render build timed out or ran out of disk space"
  echo "If so, consider using a Docker container instead"
  exit 1
}

python3 -c "import torchvision; print(f'TorchVision {torchvision.__version__} installed')"
python3 -c "import sklearn; print('Scikit-learn installed')"
python3 -c "import PIL; print('Pillow installed')"

echo ""
echo "====== Build completed successfully ======"
