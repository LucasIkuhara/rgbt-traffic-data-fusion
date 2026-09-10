FROM --platform=linux/amd64 nvidia/cuda:12.6.0-devel-ubuntu24.04

RUN apt update && apt install -y libxcb1 libgl1 unzip just python3

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

