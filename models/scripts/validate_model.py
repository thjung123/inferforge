import argparse
from typing import Any

import numpy as np
import torch
import yaml
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
from pathlib import Path
from transformers import BertModel, CLIPModel


# ============================================================
# CLIP Helper Modules
# ============================================================


class CLIPTextEncoder(torch.nn.Module):
    """CLIP text encoder with projection head (output: [B, 512])"""

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection

    def forward(self, input_ids):
        out = self.text_model(input_ids)
        pooled = out["pooler_output"]  # (B, 512)
        return self.text_projection(pooled)


class CLIPImageEncoder(torch.nn.Module):
    """CLIP vision encoder with projection head (output: [B, 512])"""

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values):
        out = self.vision_model(pixel_values)
        pooled = out["pooler_output"]  # (B, 768)
        return self.visual_projection(pooled)  # (B, 512)


# ============================================================
# TensorRT inference utils
# ============================================================


def load_engine(engine_path: Path):
    trt_logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def infer_trt(engine, input_data: list[np.ndarray]):
    """Run inference on TensorRT engine."""
    context = engine.create_execution_context()
    all_tensors = [n for n in engine]
    input_names = [
        n for n in all_tensors if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        n for n in all_tensors if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
    ]

    for name, inp in zip(input_names, input_data):
        engine_shape = tuple(engine.get_tensor_shape(name))
        shape = inp.shape if -1 in engine_shape else engine_shape
        context.set_input_shape(name, shape)

    allocs = {}
    for name in all_tensors:
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        buf = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        allocs[name] = buf

        if name in input_names:
            idx = input_names.index(name)
            cuda.memcpy_htod(buf, input_data[idx].astype(dtype))

    bindings = [int(allocs[name]) for name in all_tensors]
    context.execute_v2(bindings)

    results = []
    for name in output_names:
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        out = np.empty(shape, dtype=dtype)
        cuda.memcpy_dtoh(out, allocs[name])
        results.append(out)
    return results


# ============================================================
# Torch models utils
# ============================================================


def load_or_create_torchscript(model_path: Path, model_name: str):
    try:
        model = torch.jit.load(model_path)
        print(f"[INFO] Loaded TorchScript models from {model_path}")
        return model
    except Exception:
        print(f"[WARN] {model_path.name} is not TorchScript. Attempting to convert...")
        example: tuple[Any, ...]
        if "bert" in model_name.lower():
            model = BertModel.from_pretrained("bert-base-uncased")
            vocab_size = model.config.vocab_size
            example = (
                torch.randint(0, vocab_size, (1, 128)),
                torch.ones((1, 128), dtype=torch.long),
                torch.zeros((1, 128), dtype=torch.long),
            )

        elif "clip_text" in model_name.lower():
            model = CLIPTextEncoder()
            example = (torch.randint(0, 49408, (1, 77)),)

        elif "clip_image" in model_name.lower():
            model = CLIPImageEncoder()
            example = (torch.randn((1, 3, 224, 224)),)

        else:
            raise ValueError(f"Unknown models type: {model_name}")

        model.eval()
        traced = torch.jit.trace(model, example, strict=False)
        ts_path = model_path.parent / f"{model_name}_traced.pt"
        traced.save(ts_path)
        print(f"[INFO] Saved traced TorchScript models at {ts_path}")
        return traced


# ============================================================
# Dummy input generator
# ============================================================


def build_dummy_input(cfg):
    name = cfg["model_name"]

    if "bert" in name.lower():
        vocab_size = 30522
        return (
            torch.randint(0, vocab_size, (1, 128)),
            torch.ones((1, 128), dtype=torch.long),
            torch.zeros((1, 128), dtype=torch.long),
        )

    elif "clip_text" in name.lower():
        return (torch.randint(0, 49408, (1, 77)),)

    elif "clip_image" in name.lower():
        return (torch.randn((1, 3, 224, 224)),)

    else:
        raise ValueError(f"Unknown models type: {name}")


# ============================================================
# Validation main
# ============================================================


def validate(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    print(f"[INFO] Validating {model_name} outputs...")

    model_path = Path(cfg["paths"]["source_model"])
    model = (
        load_or_create_torchscript(model_path, model_name)
        if model_path.exists()
        else None
    )
    if model is None:
        print(f"[WARN] No valid models found at {model_path}")
        return

    dummy_inputs = build_dummy_input(cfg)
    with torch.no_grad():
        torch_out = model(*dummy_inputs)
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]
        if isinstance(torch_out, dict):
            torch_out = next(iter(torch_out.values()))

    engine_path = Path(cfg["paths"]["engine_model_dir"]) / "1" / "models.plan"
    engine = load_engine(engine_path)
    trt_out = infer_trt(engine, [x.numpy() for x in dummy_inputs])

    torch_main = torch_out.detach().cpu().numpy()
    diff = np.abs(trt_out[0] - torch_main).mean()
    print(f"[+] Mean absolute diff: {diff:.6f}")
    assert diff < 5e-2, f"Model outputs diverge too much! diff={diff:.4f}"
    print("Validation passed successfully.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate TensorRT models output vs PyTorch"
    )
    parser.add_argument("--config", required=True, help="Path to models YAML config")
    args = parser.parse_args()
    validate(args.config)
