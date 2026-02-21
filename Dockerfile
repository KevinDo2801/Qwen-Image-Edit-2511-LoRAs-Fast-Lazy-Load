FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip3 install --upgrade pip

# Install PyTorch with CUDA 12.1 support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY handler.py .
COPY qwenimage/ ./qwenimage/

# Command to run the handler
CMD ["python3", "-u", "handler.py"]
