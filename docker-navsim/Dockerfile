# NavSim Docker Environment
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 2.0.0 (CUDA 11.8)
RUN pip3 install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html

# Install Python dependencies (without l5kit first to avoid conflicts)
RUN pip3 install \
    numpy \
    pandas \
    scipy \
    matplotlib \
    transformers \
    timm \
    opencv-python \
    pillow \
    tqdm

# Install l5kit with --no-deps to avoid torch version conflict
RUN pip3 install l5kit --no-deps

# Install l5kit dependencies manually (compatible versions)
RUN pip3 install \
    pyyaml \
    typing-extensions \
    zarr \
    ptable \
    transforms3d \
    protobuf \
    shapely \
    pymap3d \
    imageio \
    ipywidgets \
    bokeh \
    tornado \
    jupyterlab \
    jupyter-server \
    notebook-shim \
    jupyterlab-server

# Install nuscenes-devkit (with retry)
RUN pip3 install nuscenes-devkit || pip3 install nuscenes-devkit

# Create workspace directory
RUN mkdir -p /workspace/data

WORKDIR /workspace

CMD ["/bin/bash"]
