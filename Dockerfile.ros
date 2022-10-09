ARG FROM_IMAGE=nvidia/cudagl:11.3.0-devel-ubuntu20.04
FROM $FROM_IMAGE

ENV DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=foxy
# ENV ROS_DISTRO=$ROS_DISTRO

RUN rm /etc/apt/sources.list.d/cuda.list
#     && rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get upgrade -y
# Editors and so on
RUN apt-get install -y vim git tmux bash-completion sudo locales curl gnupg2 lsb-release python3-argcomplete
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb [arch=amd64,arm64] http://packages.ros.org/ros2/ubuntu `lsb_release -cs` main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN apt-get update
RUN apt-get install -y ros-${ROS_DISTRO}-desktop python3-colcon-common-extensions

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    git \
    python3-colcon-common-extensions \
    python3-pip \
    lcov \
    # libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/mzahana/containers/blob/master/docker/Dockerfile_ros-melodic-cuda10.1/Dockerfile
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ros-$ROS_DISTRO-pcl-conversions \
    ros-$ROS_DISTRO-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# install packages
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    libglfw3-dev \
    libassimp-dev \
    # iwyu \
    && rm -rf /var/lib/apt/lists/*

# https://github.com/mzahana/containers/blob/master/docker/Dockerfile_ros-melodic-cuda10.1/Dockerfile
# https://github.com/larics/uav_nuc_setup/blob/master/scripts/installation.sh
# https://github.com/ctu-mrs/uav_core/blob/master/installation/dependencies/general.sh
# https://github.com/tynguyen/arm64v8_dockers/blob/master/Dockerfile
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ros-$ROS_DISTRO-tf2-ros \
    ros-$ROS_DISTRO-tf2-eigen \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-image-transport \
    && rm -rf /var/lib/apt/lists/*


#### ROS2 Yak is for TSDF fusion, doesnt seem to work on 20.04

# WORKDIR /root/ros2_yak/src

# RUN git clone https://github.com/ros-industrial/cmake_common_scripts -b master
# RUN git clone https://github.com/ros-industrial/yak -b devel
# RUN git clone https://github.com/schornakj/gl_depth_sim -b feature/pure-cmake
# RUN git clone https://github.com/ros-industrial/yak_ros2 -b master

# WORKDIR /root/ros2_yak

# RUN /bin/bash -c 'source /opt/ros/$ROS_DISTRO/setup.sh && \
#     colcon build --symlink-install --cmake-args "-DCMAKE_LIBRARY_PATH=/usr/local/cuda-11.0/targets/x86_64-linux/lib/stubs/" "-DBUILD_DEMO=True" "-DCUDA_ARCH_AND_PTX=7.5"'

# RUN echo "source /root/ros2_yak/install/setup.bash" >> /root/.bashrc
####

# install GLX-Gears (for debugging)
RUN apt-get update && apt-get install -y \
   mesa-utils && \
   rm -rf /var/lib/apt/lists/*


ARG PYTHON_VERSION=3.8
# RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
#     # /opt/conda/bin/python -mpip install -r requirements.txt && \
#     /opt/conda/bin/conda clean -ya

# RUN echo 'export PATH=/opt/conda/bin:$PATH' >> ~/.bashrc && \
#     echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib' >> ~/.bashrc && \
#     /opt/conda/bin/conda init

# ENV \
#     PATH="/opt/conda/bin:$PATH" \
#     LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/lib" \
#     MAKEFLAGS="-j1"

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt update && \
    #curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    #echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    #sed -i -e 's/ubuntu .* main/ubuntu focal main/g' /etc/apt/sources.list.d/ros2.list && \
    #apt update && \
    #apt install -y ros-foxy-ros-base && \
    apt install -y python3-colcon-common-extensions && \
    apt install -y ros-foxy-v4l2-camera && \
    apt install -y ros-foxy-cv-bridge && \
    apt install -y ros-foxy-usb-cam && \
    apt install -y ros-foxy-camera-calibration && \
    rm -rf /var/lib/apt/lists/*  &&\
    apt -y clean && \
    pip install -U pip && \
    pip install catkin_pkg && \
    pip install empy && \
    pip install lark && \     
    python3 -m pip cache purge


ADD . /workspace/
WORKDIR /workspace

RUN pip install cmake

# RUN colcon build --symlink-install --packages-select yolox_ros_py bboxes_ex_msgs

RUN echo 'source /opt/ros/foxy/setup.bash' >> ~/.bashrc  
RUN echo 'source /workspace/install/setup.bash' >> ~/.bashrc  

# RUN colcon build --base-paths image_pipeline/ --packages-select camera_calibration
# RUN colcon build --packages-select ros2_shared opencv_cam

# RUN echo 'source /workspace/image_pipeline/install/local_setup.bash' >> ~/.bashrc  

WORKDIR /workspace
RUN mkdir -p /root/.ros/camera_info/
RUN cp ./calibration/* /root/.ros/camera_info/