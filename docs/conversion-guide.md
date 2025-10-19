# Model Conversion & Deployment Strategy

This document defines how our PyTorch-based models (e.g., BERT, CLIP) are converted, optimized, and deployed in production for NVIDIA Triton Inference Server.

---

## 1. Overview

In production environments, models are not manually stored in `weights/`.  
Instead, we fetch pretrained checkpoints (usually from Hugging Face Hub or OpenAI CLIP repository) and convert them into formats compatible with Triton Inference Server.

| Stage | Purpose | Tool/Script |
|-------|----------|-------------|
| PyTorch → ONNX | Framework-agnostic representation | convert_to_onnx.py |
| ONNX → TensorRT Engine | GPU-optimized inference graph | build_trt_engine.sh |
| Validation | Ensure numerical correctness | validate_model.py |
| Triton Config Generation | Create serving metadata | generate_triton_config.py |

---

## 2. Model Build Pipeline

### Step 1 — Convert PyTorch → ONNX

Command:
```
python model_builder/scripts/convert_to_onnx.py --config model_builder/model_configs/bert_config.yaml
```

- Loads pretrained weights (from Hugging Face or local cache)
- Exports to `model_repository/<model_name>/model.onnx`
- Supports dynamic shape configuration (`min/opt/max`) from YAML

> In production, the conversion is done inside a container build step (e.g. CI/CD runner), using consistent PyTorch and CUDA versions to ensure deterministic graph export.

---

### Step 2 — Optimize ONNX → TensorRT

> **Build Environment:**  
> TensorRT engine must be built inside the same CUDA/TensorRT container version used for deployment. 
> Current build image: nvcr.io/nvidia/tensorrt:25.06-py3  
> (This ensures engine compatibility with Triton 25.06 runtime.)

Command:
```
bash model_builder/scripts/build_trt_engine.sh <ONNX_PATH> [fp16|int8]
```
Example:
```
bash model_builder/scripts/build_trt_engine.sh model_repository/bert_encoder/bert_encoder.onnx fp16
```

This script internally calls trtexec to convert the ONNX model into a TensorRT engine (.plan file).

It automatically handles:
- Dynamic-shape models (e.g., BERT): applies --minShapes / --optShapes / --maxShapes
- Static-shape models (e.g., CLIP text/image): omits shape options automatically

Output
```
model_repository/<model_name>/1/model.plan
```

---

### Step 3 — Validate Outputs

Command:
```
python model_builder/scripts/validate_model.py --config model_builder/model_configs/bert_config.yaml
```

- Loads both PyTorch and TensorRT engines
- Runs inference on sample inputs
- Compares numerical outputs (L2 error, max_diff)
- Logs validation summary

output:
```
[INFO] Saved traced TorchScript model at weights/clip_image_encoder_traced.pt
[+] Mean absolute diff: 0.001193
Validation passed successfully.

[INFO] Saved traced TorchScript model at weights/clip_text_encoder_traced.pt
[+] Mean absolute diff: 0.001625
Validation passed successfully.

[INFO] Saved traced TorchScript model at weights/bert_encoder_traced.pt
[+] Mean absolute diff: 0.000897
Validation passed successfully.
```
> All encoder models validated successfully.  
> Mean absolute differences < 0.002 confirm numerical equivalence between PyTorch and TensorRT.


---

### Step 4 — Generate Triton Configuration

Command:
```
python model_builder/scripts/generate_triton_config.py --config model_builder/model_configs/bert_config.yaml
```

- Reads YAML model spec
- Generates `model_repository/<model_name>/config.pbtxt` automatically
- Includes input/output tensors, precision, batching, and instance settings

> config.pbtxt is required by Triton to understand model metadata and runtime behavior.  
> For ensemble models (e.g., CLIP), the generator can define inter-model connections and cosine similarity fusion logic automatically.

---

## 3. Model Repository Structure

```
model_repository/
├── bert_encoder/
│   ├── 1/
│   │   └── model.plan
│   ├── model.onnx
│   └── config.pbtxt
│
├── text_encoder/
│   ├── 1/
│   │   └── model.plan
│   ├── clip_text_encoder.onnx
│   └── config.pbtxt
│
├── image_encoder/
│   ├── 1/
│   │   └── model.plan
│   ├── clip_image_encoder.onnx
│   └── config.pbtxt
│
└── ensemble/
    └── config.pbtxt    # optional (for multimodal fusion)
```

---

## 4. Model Source & Weight Management

| Model | Source | Download Method | Example |
|--------|---------|----------------|----------|
| BERT (bert-base-uncased) | Hugging Face | transformers.BertModel.from_pretrained() | weights/bert-base-uncased.pt |
| CLIP (ViT-B/32) | OpenAI | clip.load("ViT-B/32") | weights/clip_text_encoder.pt, weights/clip_image_encoder.pt |

In production:
- We don’t manually commit `.pt` files.
- Instead, each build container downloads the pretrained weights dynamically.
- A utility script (`download_weights.py`) handles caching and offline reuse.

Example automated weight fetch:
```
from transformers import BertModel
import clip, torch, os

os.makedirs("weights", exist_ok=True)

bert = BertModel.from_pretrained("bert-base-uncased")
torch.save(bert.state_dict(), "weights/bert-base-uncased.pt")

clip_model, _ = clip.load("ViT-B/32", device="cpu")
torch.save(clip_model.transformer.state_dict(), "weights/clip_text_encoder.pt")
torch.save(clip_model.visual.state_dict(), "weights/clip_image_encoder.pt")
```

---

## 5. Deployment Flow (End-to-End)

```
model_configs/bert_config.yaml
│
├─▶ convert_to_onnx.py
│ └── model_repository/bert_encoder/model.onnx
│
├─▶ build_trt_engine.sh
│ └── model_repository/bert_encoder/1/model.plan
│
├─▶ validate_model.py
│ └── PyTorch ↔ TensorRT
│
└─▶ generate_triton_config.py
└── model_repository/bert_encoder/config.pbtxt
```


> This entire chain can be automated in CI/CD (e.g., GitHub Actions or Jenkins). Once complete, model_repository/ is published or mounted by Triton Server.

---

## 6. Production Recommendations

| Area | Recommendation |
|-------|----------------|
| Versioning | Increment Triton model version for every new build (1/, 2/, …). |
| Precision | Default fp16 unless precision-critical; use calibration for int8. |
| Validation | Always validate outputs numerically post-conversion. |
| Storage | Use S3, NFS, or Artifact Registry for model_repository synchronization. |
| Monitoring | Integrate Triton metrics (/metrics) with Prometheus + Grafana. |
| Automation | Run the 4-step build in CI/CD for reproducible pipelines. |
| Environment | Use the same container image (`nvcr.io/nvidia/tensorrt:25.06-py3`) for both engine build and Triton deployment. |

---

## 7. Example CI/CD Integration

steps:

name: Convert PyTorch to ONNX
```
run: python model_builder/scripts/convert_to_onnx.py --config model_builder/model_configs/bert_config.yaml
```
name: Build TensorRT Engine
```
run: bash model_builder/scripts/build_trt_engine.sh model_repository/bert_encoder/model.onnx fp16
```
name: Validate Model
```
run: python model_builder/scripts/validate_model.py --config model_builder/model_configs/bert_config.yaml
```
name: Generate Triton Config
```
run: python model_builder/scripts/generate_triton_config.py --config model_builder/model_configs/bert_config.yaml
```



---

## Summary

| Stage | Description | Output |
|--------|--------------|---------|
| convert_to_onnx.py | Converts PyTorch → ONNX | model.onnx |
| build_trt_engine.sh | Optimizes ONNX → TensorRT | 1/model.plan |
| validate_model.py | Ensures consistency | Validation logs |
| generate_triton_config.py | Generates config.pbtxt | Triton metadata |
