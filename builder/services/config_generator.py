import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("builder")

_DATATYPE_MAP = {
    "FP32": "TYPE_FP32",
    "FP16": "TYPE_FP16",
    "INT32": "TYPE_INT32",
    "INT64": "TYPE_INT64",
}


def generate_config_pbtxt(cfg: dict[str, Any], output_dir: Path) -> Path:
    model_name = cfg["model_name"]
    max_batch_size = cfg.get("triton", {}).get("max_batch_size", 0)
    precision = cfg.get("precision", {}).get("default", "fp16").upper()

    lines = [
        f'name: "{model_name}"',
        'backend: "tensorrt_plan"',
        f"max_batch_size: {max_batch_size}",
        "",
    ]

    if "inputs" in cfg:
        lines.append("input [")
        for i, inp in enumerate(cfg["inputs"]):
            dims = ", ".join(str(d) for d in inp["shape"][1:])
            dtype = _DATATYPE_MAP[inp["datatype"]]
            comma = "," if i < len(cfg["inputs"]) - 1 else ""
            lines.append(
                f'  {{ name: "{inp["name"]}" data_type: {dtype} dims: [{dims}] }}{comma}'
            )
        lines.append("]")
        lines.append("")

    if "outputs" in cfg:
        lines.append("output [")
        for i, out in enumerate(cfg["outputs"]):
            dims = ", ".join(str(d) for d in out["shape"][1:])
            dtype = _DATATYPE_MAP[out["datatype"]]
            comma = "," if i < len(cfg["outputs"]) - 1 else ""
            lines.append(
                f'  {{ name: "{out["name"]}" data_type: {dtype} dims: [{dims}] }}{comma}'
            )
        lines.append("]")
        lines.append("")

    instance_count = cfg.get("triton", {}).get("instance_count", 1)
    lines.append("instance_group [")
    lines.append(f"  {{ kind: KIND_GPU count: {instance_count} }}")
    lines.append("]")
    lines.append("")

    lines.append("optimization {")
    lines.append("  execution_accelerators {")
    lines.append("    gpu_execution_accelerator: [")
    lines.append(
        f'      {{ name: "tensorrt" parameters {{ key: "precision_mode" value: "{precision}" }} }}'
    )
    lines.append("    ]")
    lines.append("  }")
    lines.append("}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pbtxt_path = output_dir / "config.pbtxt"
    pbtxt_path.write_text("\n".join(lines))

    logger.info(f"Generated config → {pbtxt_path}")
    return pbtxt_path
