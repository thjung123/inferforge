# Triton Inference Service

A production-ready inference serving platform built on **FastAPI** and **NVIDIA Triton Inference Server**.  

It provides an easy-to-use API gateway, automated model conversion, and performance tools — enabling you to deploy and scale multimodal models (e.g., image & text) in production with minimal effort.

---

## Overview

This project aims to simplify end-to-end inference deployment:

- **FastAPI Gateway** – Entry point for REST API requests with validation, retry, and structured logging  
- **Triton Inference Server** – High-performance serving of ONNX/TensorRT models with ensemble support  
- **Model Conversion Tools** – Export PyTorch models to ONNX → TensorRT with precision optimization

---

## Prerequisites

Before you start, make sure you have the following installed:

- Python 3.12+
- uv – Python environment & dependency manager  
- Docker & Docker Compose  
- NVIDIA GPU driver (for TensorRT & GPU inference)

---

## Local Development Setup

### 1. Clone the repository

```bash
git clone https://github.com/thjung123/triton-inference-service.git
cd triton-inference-service
```

### 2. Initialize the Python environment

We use uv for environment and dependency management.

```bash
uv init
uv sync
```

This will create a virtual environment and install all required dependencies.

## Run Locally (with Docker Compose)

Before running the containers, make sure the TensorRT `.plan` files and `config.pbtxt` are built.  
These files are **not included in the repository**.  
You can generate them by running the following script:

```bash
bash model_builder/scripts/convert_all.sh
```
This script will:
- Download model weights (.pt)
- Export them to ONNX
- Build optimized TensorRT engines
- Generate Triton config.pbtxt files under model_repository/

Once the models are ready, you can start the full stack:

This will start:
- FastAPI gateway on http://localhost:8080
- Redis (for rate limiting) on localhost:6379
- Triton Inference Server on http://localhost:8001

```bash
docker-compose -f docker/docker-compose.yml up --build
```



Then, you can test the API:
```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -H "x-api-key: test-key" \
  -d '{
    "model_name": "clip_ensemble",
    "inputs": {
      "image_urls": [
        "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/4/47/Golden_Retriever_Carlos_%2810591010556%29.jpg"
      ],
      "texts": [
        "a photo of a cat",
        "a photo of a dog"
      ]
    }
  }' | jq .
```

### Running Tests
Run unit and integration tests using pytest:
```bash
uv run pytest -v
```