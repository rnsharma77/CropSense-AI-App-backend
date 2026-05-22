FROM node:22-bookworm-slim as base

WORKDIR /app

# Use a Debian-based image so PyTorch CPU wheels install cleanly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json ./
COPY requirements.txt ./

FROM base as builder
RUN npm ci --omit=dev

FROM base as production

COPY --from=builder /app/node_modules ./node_modules

# requirements.txt already includes the PyTorch CPU wheel index as an extra index.
RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    python3 -m pip install -r requirements.txt --break-system-packages

RUN python3 -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')" && \
    python3 -c "import torchvision; print('TorchVision installed')"

COPY . .

EXPOSE 5050

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node -e "fetch('http://localhost:5050/api/health').catch(() => process.exit(1))"

CMD ["npm", "start"]
