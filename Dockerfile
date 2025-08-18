FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

RUN apt-get update 

RUN apt-get upgrade -yqq

RUN apt-get install software-properties-common git curl ffmpeg libsm6 libxext6 pkg-config cmake libcairo2-dev -yqq

RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get install python3.10 python3.10-distutils python3.10-dev -yqq

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 0

RUN curl https://bootstrap.pypa.io/get-pip.py | python

RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 -q

RUN pip install wandb timm coloredlogs opencv-python matplotlib einops kornia -q
