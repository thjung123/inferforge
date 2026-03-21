import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("builder")

_DATATYPE_TO_TORCH = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "INT32": torch.int32,
    "INT64": torch.int64,
}


class _CLIPTextEncoder(nn.Module):
    def __init__(self, source: str):
        super().__init__()
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained(source)
        self.text_model = clip.text_model
        self.text_projection = clip.text_projection

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        out = self.text_model(input_ids, attention_mask=attention_mask)
        return self.text_projection(out["pooler_output"])


class _CLIPImageEncoder(nn.Module):
    def __init__(self, source: str):
        super().__init__()
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained(source)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values)
        return self.visual_projection(out["pooler_output"])


_CLIP_EXPORTERS: dict[str, type[nn.Module]] = {
    "text_model": _CLIPTextEncoder,
    "vision_model": _CLIPImageEncoder,
}


def _load_model(cfg: dict[str, Any]) -> torch.nn.Module:
    source = cfg["source"]
    model_type = cfg["model_type"]

    if model_type == "bert":
        from transformers import AutoModel

        return AutoModel.from_pretrained(source).eval()

    if model_type == "clip":
        export_target = cfg.get("export_target")
        if export_target not in _CLIP_EXPORTERS:
            raise ValueError(
                f"Unknown export_target '{export_target}' for clip. "
                f"Expected one of {list(_CLIP_EXPORTERS)}"
            )
        return _CLIP_EXPORTERS[export_target](source).eval()

    raise ValueError(f"Unsupported model_type: {model_type}")


def _build_dummy_inputs(cfg: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    inputs = []
    for inp in cfg["inputs"]:
        shape = [abs(d) if d != -1 else 8 for d in inp["shape"]]
        torch_dtype = _DATATYPE_TO_TORCH[inp["datatype"]]
        if torch_dtype in (torch.int32, torch.int64):
            inputs.append(torch.randint(0, 1000, shape, dtype=torch_dtype))
        else:
            inputs.append(torch.randn(shape, dtype=torch_dtype))
    return tuple(inputs)


def _build_dynamic_axes(cfg: dict[str, Any]) -> dict[str, dict[int, str]] | None:
    dynamic = cfg.get("dynamic_shapes", {})
    if not dynamic.get("enabled"):
        return None

    axes: dict[str, dict[int, str]] = {}
    for inp in cfg["inputs"]:
        name = inp["name"]
        dim_labels = {}
        for i, d in enumerate(inp["shape"]):
            if d == -1:
                dim_labels[i] = f"dim_{i}"
        if dim_labels:
            axes[name] = dim_labels

    for out in cfg.get("outputs", []):
        name = out["name"]
        dim_labels = {}
        for i, d in enumerate(out["shape"]):
            if d == -1:
                dim_labels[i] = f"dim_{i}"
        if dim_labels:
            axes[name] = dim_labels

    return axes if axes else None


def export_onnx(cfg: dict[str, Any], onnx_path: Path) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    model = _load_model(cfg)
    dummy = _build_dummy_inputs(cfg)
    dynamic_axes = _build_dynamic_axes(cfg)

    input_names = [inp["name"] for inp in cfg["inputs"]]
    output_names = [out["name"] for out in cfg.get("outputs", [])]

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    logger.info(f"Exported ONNX → {onnx_path}")
    return onnx_path
