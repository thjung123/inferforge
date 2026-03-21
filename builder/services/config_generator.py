import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("builder")

_DATATYPE_MAP = {
    "FP32": "TYPE_FP32",
    "FP16": "TYPE_FP16",
    "INT32": "TYPE_INT32",
    "INT64": "TYPE_INT64",
    "STRING": "TYPE_STRING",
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


def _render_io_block(block_type: str, items: list[dict[str, Any]]) -> list[str]:
    lines = [f"{block_type} ["]
    for i, item in enumerate(items):
        dims = ", ".join(str(d) for d in item["dims"])
        dtype = _DATATYPE_MAP[item["datatype"]]
        comma = "," if i < len(items) - 1 else ""
        lines.append(
            f'  {{ name: "{item["name"]}" data_type: {dtype} dims: [{dims}] }}{comma}'
        )
    lines.append("]")
    return lines


def generate_processor_config(
    step: dict[str, Any], output_dir: Path, max_batch_size: int = 256
) -> Path:
    model_name = step["model_name"]
    instance_kind = step.get("instance_kind", "KIND_CPU")

    lines = [
        f'name: "{model_name}"',
        'backend: "python"',
        f"max_batch_size: {max_batch_size}",
        "",
    ]
    lines.extend(_render_io_block("input", step["inputs"]))
    lines.append("")

    outputs = step["outputs"]
    max_length = step.get("params", {}).get("max_length")
    if max_length is not None:
        outputs = [
            {**o, "dims": [max_length]} if o["datatype"] != "STRING" else o
            for o in outputs
        ]

    lines.extend(_render_io_block("output", outputs))
    lines.append("")
    lines.append("instance_group [")
    lines.append(f"  {{ kind: {instance_kind} }}")
    lines.append("]")

    params = step.get("params", {})
    if params:
        lines.append("")
        lines.append("parameters {")
        for key, value in params.items():
            lines.append(f'  {{ key: "{key}" value: {{ string_value: "{value}" }} }}')
        lines.append("}")

    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    pbtxt_path = model_dir / "config.pbtxt"
    pbtxt_path.write_text("\n".join(lines))

    logger.info(f"Generated processor config → {pbtxt_path}")
    return pbtxt_path


def generate_ensemble_config(ensemble_cfg: dict[str, Any], output_dir: Path) -> Path:
    name = ensemble_cfg["name"]
    max_batch_size = ensemble_cfg.get("max_batch_size", 256)

    lines = [
        f'name: "{name}"',
        'platform: "ensemble"',
        f"max_batch_size: {max_batch_size}",
        "",
    ]
    lines.extend(_render_io_block("input", ensemble_cfg["inputs"]))
    lines.append("")
    lines.extend(_render_io_block("output", ensemble_cfg["outputs"]))
    lines.append("")

    lines.append("ensemble_scheduling {")
    lines.append("  step [")
    for si, step in enumerate(ensemble_cfg["steps"]):
        lines.append("    {")
        lines.append(f'      model_name: "{step["model_name"]}"')
        lines.append("      model_version: -1")
        for key, value in step["input_map"].items():
            lines.append("      input_map {")
            lines.append(f'        key: "{key}"')
            lines.append(f'        value: "{value}"')
            lines.append("      }")
        for key, value in step["output_map"].items():
            lines.append("      output_map {")
            lines.append(f'        key: "{key}"')
            lines.append(f'        value: "{value}"')
            lines.append("      }")
        comma = "," if si < len(ensemble_cfg["steps"]) - 1 else ""
        lines.append(f"    }}{comma}")
    lines.append("  ]")
    lines.append("}")

    model_dir = output_dir / name
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "1").mkdir(exist_ok=True)
    pbtxt_path = model_dir / "config.pbtxt"
    pbtxt_path.write_text("\n".join(lines))

    logger.info(f"Generated ensemble config → {pbtxt_path}")
    return pbtxt_path
