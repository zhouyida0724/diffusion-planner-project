# Dockerfile for Diffusion Planner (Minimal)
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.1.0 with CUDA 12.1
RUN pip3 install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install core Python dependencies
RUN pip3 install \
    numpy \
    pandas \
    scipy \
    matplotlib \
    transformers \
    timm \
    opencv-python \
    pillow \
    tqdm \
    setuptools \
    wheel

# Install nuplan-devkit (without conflicting dependencies)
RUN pip3 install nuplan-devkit --no-deps

# Set default working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
