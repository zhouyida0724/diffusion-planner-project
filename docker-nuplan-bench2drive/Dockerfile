# nuPlan + Bench2Drive Docker Environment
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CARLA_VERSION=0.9.15

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.1.0 with CUDA 11.8
RUN pip3 install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies
RUN pip3 install \
    "numpy<2" \
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

# Install nuplan-devkit
RUN pip3 install nuplan-devkit

# Create workspace directory
RUN mkdir -p /workspace/data

# Note: Download CARLA 0.9.15 from https://github.com/carla-simulator/carla/releases/tag/0.9.15
# and place the extracted folder in /workspace/CARLA/

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
