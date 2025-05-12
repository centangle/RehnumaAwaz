# Base CUDA image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install system dependencies in a single step
RUN apt update && apt install -y \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY piper /app/piper

# Set working directory
WORKDIR /app/piper/src/python

# Install dependencies globally (no venv)
RUN pip install --upgrade pip wheel setuptools && \
    pip install -e .

# Build monotonic aligner
RUN bash build_monotonic_align.sh

# Install PyTorch (CUDA 12 compatible)
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Set default command to keep container running
CMD [ "tail", "-f", "/dev/null" ]
