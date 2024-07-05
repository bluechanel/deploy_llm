FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

COPY . /workspace
ADD fschat-0.2.36-py3-none-any.whl /workspace

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=api_requirements.txt,target=api_requirements.txt \
    pip install -r api_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple