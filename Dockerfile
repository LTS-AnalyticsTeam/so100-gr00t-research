FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt update && \
    apt install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt install -y netcat dnsutils && \
    apt-get update && \
    apt-get install -y libgl1-mesa-glx git libvulkan-dev \
    zip unzip wget curl git git-lfs build-essential cmake \
    vim less sudo htop ca-certificates man tmux ffmpeg \
    # Add OpenCV system dependencies
    libglib2.0-0 libsm6 libxext6 libxrender-dev

WORKDIR /workspace/so100-gr00t-research

CMD [ "/bin/bash" ]
