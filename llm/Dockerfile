FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04 as base
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

COPY . /workspace

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install fschat-0.2.36-py3-none-any.whl