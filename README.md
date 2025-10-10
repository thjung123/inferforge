# Triton Inference Service

A production-ready inference serving platform built on **FastAPI** and **NVIDIA Triton Inference Server**.  

It provides an easy-to-use API gateway, automated model conversion, and performance tools — enabling you to deploy and scale multimodal models (e.g., image & text) in production with minimal effort.

---

## Overview

This project aims to simplify end-to-end inference deployment:

- **FastAPI Gateway** – Entry point for REST API requests with validation, retry, and structured logging  
- **Triton Inference Server** – High-performance serving of ONNX/TensorRT models with ensemble support  
- **Model Conversion Tools** – Export PyTorch models to ONNX → TensorRT with precision optimization  
- **Benchmarking Suite** – Automate performance sweeps and latency/throughput measurement  
- **Deployment-Ready** – Docker, Compose, and Helm support for production rollout

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

This will start:
- FastAPI gateway on http://localhost:8080
- Redis (for rate limiting) on localhost:6379
- Triton Inference Server on http://localhost:8001

```bash
docker-compose -f docker/docker-compose.yml up --build
```

Then, you can test the API:
```bash

```

## Running Tests
Run unit and integration tests using pytest:
```bash
uv run pytest -v
```

or run them directly with Docker:
```bash
docker-compose run --rm gateway pytest
```