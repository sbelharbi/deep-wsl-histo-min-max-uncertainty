# base container
FROM nvidia/cuda::10.0-devel-ubuntu18.04

# update
ENV DEBIAN_FRONTEND "noninteractive"
RUN apt-get update -y
RUN apt-get -y \
    -o Dpkg::Options::="--force-confdef" \
    -o Dpkg::Options::="--force-confold" dist-upgrade

# install basic
RUN apt-get install -y --no-install-recommends \
    less sudo ssh \
    build-essential \
    unzip git curl wget vim tree htop \
    python3-dev python3-tk \
    ninja-build

# python libs
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip3 install \
    future six cffi numpy pillow tqdm Cython awscli ninja

# install pytorch
RUN pip3 install https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl
RUN pip3 install torchvision

# clean up
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y
RUN apt-get clean -y && apt-get autoclean -y
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# create mountpoint from host
RUN mkdir -p /workspace

# create non-root user
ARG user_name=ubuntu
ARG user_id=1000
ARG group_name=ubuntu
ARG group_id=1000
RUN groupadd -g ${group_id} ${group_name}
RUN useradd -u ${user_id} -g ${group_id} -d /home/${user_name} --create-home --shell /bin/bash ${user_name}
RUN echo "${user_name} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
RUN chown -R ${user_name}:${group_name} /home/${user_name}
RUN chown -R ${user_name}:${group_name} /workspace
RUN chsh -s /bin/bash ${user_name}
USER ubuntu
WORKDIR /home/ubuntu
ENV HOME /home/ubuntu

