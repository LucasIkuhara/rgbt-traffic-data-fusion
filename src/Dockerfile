FROM --platform=linux/amd64 nvidia/cuda:12.6.0-devel-ubuntu24.04

RUN apt update && apt install -y libxcb1 libgl1 unzip just python3 python3-pip

RUN pip3 install --break-system-packages poetry
