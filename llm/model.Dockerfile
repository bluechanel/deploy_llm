FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as base
RUN apt-get update -y \
    && apt-get install -y python3-pip

WORKDIR /workspace

COPY . /workspace

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=model_requirements.txt,target=model_requirements.txt \
    pip install -r model_requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple