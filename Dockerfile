ARG FROM_IMAGE=nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM $FROM_IMAGE

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list.d/cuda.list
#     && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get upgrade -y
# Editors and so on
RUN apt-get install -y vim git tmux bash-completion sudo locales curl gnupg2 lsb-release python3-argcomplete  python3-pip
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8


ARG PYTHON_VERSION=3.8
RUN pip install torch==1.12.1 torchvision==0.13.0 torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt .
RUN pip install -r requirements.txt
# RUN pip install cmake

RUN apt update && \
    apt install -y libsm6 && \
    pip install -U pip && \
    pip install catkin_pkg && \
    pip install empy && \
    pip install lark && \     
    python3 -m pip cache purge


ADD . /workspace/
WORKDIR /workspace
