FROM ubuntu:24.04

# Install Python, PortAudio, Qt, etc.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ca-certificates git \
    libgl1 libglib2.0-0 pyqt5-dev-tools \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
ARG WORKSPACE=/workspace/segmentation_agent
RUN mkdir -p $WORKSPACE
WORKDIR $WORKSPACE

# Install requirements
COPY requirements.txt /tmp/requirements.txt
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=1
