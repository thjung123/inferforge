# Triton Performance Benchmark Suite

This directory contains benchmarking scripts for evaluating **NVIDIA Triton Inference Server** performance using the `perf_analyzer` tool.

---

## Environment

- **Base image**: `nvcr.io/nvidia/tritonserver:25.06-py3-sdk`
- **Target model**: `clip_text_encoder` (text model only)
- **Backend**: TensorRT (FP16)
- **GPU**: NVIDIA T4 (16GB)
- **Test matrix**: defined in [`performance/config_matrix.json`](./config_matrix.json)
- **Output directories**:
  - Raw CSVs → `/workspace/perf_out`
  - Processed results → `performance/perf_image/`

---

## Usage

### 1. Run full benchmark sweep

```bash
python performance/sweep_perf.py
```

- Iterates through all batch sizes, concurrency levels, and dynamic batching configs from config_matrix.json.
- Runs perf_analyzer for each configuration.
- Automatically:
  - Inserts or restores model configs (config.pbtxt)
  - Reloads models using Triton HTTP API 
  - Saves results as .csv files under /workspace/perf_out

Note:

Before running sweep_perf.py, make sure Triton is launched with explicit model control mode:

```
tritonserver --model-control-mode=explicit
```

### 2. Analyze and visualize results

```bash
python performance/analyze_results.py
```

- Aggregates all generated .csv result files.
- Computes throughput, latency, efficiency, and GPU utilization summaries.
- Generates visualizations under performance/perf_image/:
  - 01_batch_scaling_dynamic_b1.png 
  - 02_dynamic_vs_static_b64_c1.png 
  - 03_concurrency_vs_latency_dynamic_b64.png 
  - 04_gpu_util_vs_throughput.png
- Also exports a summary.csv file for factual data reference.
