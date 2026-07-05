# Build pipeline

HuggingFace → ONNX → TensorRT, then config generation and Triton deploy.
`run_build_pipeline` pulls the source model, exports ONNX (opset 18), builds the
engine, and checks fp16 precision before the artifacts are published.

## fp16 precision check

fp16 halves the engine's memory and speeds up inference, but the math can drift
from fp32. Before an engine is accepted the builder runs the same inputs through
the fp16 engine and an onnxruntime fp32 reference, then gates on per-row cosine
similarity and max absolute error (`services/precision_validator.py`).

bert-base-uncased, `last_hidden_state`, 8 samples at seq len 128, on an A40 with
TensorRT 10.13:

| engine                | cosine (min) | max abs err | mean abs err |
|-----------------------|--------------|-------------|--------------|
| TRT fp32 vs ONNX fp32 | 0.999999     | 0.0046      | 0.00047      |
| TRT fp16 vs ONNX fp32 | 0.999995     | 0.0219      | 0.00145      |

The fp32 row is the baseline: it separates kernel and layout differences from
the precision cost, and it comes out near zero, so the engine itself tracks the
reference. The fp16 gap on top of that is small — cosine holds at 0.99999 and
the worst element is off by 0.022, well inside the gate (min cosine 0.999, max
abs err 0.05). For this model fp16 is essentially free, so it's the default.
