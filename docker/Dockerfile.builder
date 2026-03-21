FROM nvcr.io/nvidia/tensorrt:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl \
 && ln -sf $(which python3) /usr/bin/python \
 && ln -sf $(which pip3) /usr/bin/pip \
 && pip install --upgrade --ignore-installed pip setuptools wheel \
 && pip install \
    torch torchvision torchaudio \
    transformers pycuda pyyaml numpy \
    onnx onnxscript yq onnxruntime onnx-graphsurgeon openai-clip \
    fastapi uvicorn redis pydantic-settings httpx \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY builder ./builder

ENV MODEL_REPOSITORY=/models
ENV REDIS_URL=redis://redis:6379/0

EXPOSE 8090

CMD ["uvicorn", "builder.main:app", "--host", "0.0.0.0", "--port", "8090"]
