FROM node:22-alpine as base

# Set working directory
WORKDIR /app

# Install system dependencies for Python and ML libraries
RUN apk add --no-cache \
    python3 \
    py3-pip \
    gcc \
    g++ \
    musl-dev \
    python3-dev \
    linux-headers \
    libffi-dev \
    openssl-dev

# Copy package.json files
COPY package.json package-lock.json ./
COPY requirements.txt ./

# Build stage: Install Node dependencies
FROM base as builder
RUN npm ci --only=production

# Production stage
FROM base as production

# Copy Node modules from builder
COPY --from=builder /app/node_modules ./node_modules

# Install Python dependencies with CPU-only PyTorch
# Using the PyTorch CPU index to avoid GPU wheels and reduce image size
RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu

# Verify torch installation
RUN python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed successfully')" && \
    python3 -c "import torchvision; print(f'✓ TorchVision installed')"

# Copy application code
COPY . .

# Expose port
EXPOSE 5050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node -e "fetch('http://localhost:5050/api/health').catch(() => process.exit(1))"

# Start application
CMD ["npm", "start"]
