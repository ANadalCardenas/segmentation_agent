
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn9-runtime-ubuntu24.04

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        ffmpeg \
        libsm6 \
        libxext6 \
        libasound2 \
        libasound2-dev \
        portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR $WORKSPACE

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install PyTorch with CUDA first (YOLOv5 recommendation)
# Adjust CUDA version if needed.
RUN pip3 install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project requirements (whisper, opencv, etc.)
RUN pip3 install --no-cache-dir -r requirements.txt

# Clone YOLOv5 and install its requirements (excluding torch, already installed)
RUN git clone https://github.com/ultralytics/yolov5.git \
    && pip3 install --no-cache-dir -r yolov5/requirements.txt

ENV PYTHONUNBUFFERED=1
