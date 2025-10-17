import argparse
import yaml
from pathlib import Path

DTYPE_MAP = {
    "float32": "TYPE_FP32",
    "float16": "TYPE_FP16",
    "int32": "TYPE_INT32",
    "int64": "TYPE_INT64",
}


def generate_triton_config(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    backend = (
        "ensemble" if cfg.get("ensemble", {}).get("enabled", False) else "tensorrt_plan"
    )
    max_batch_size = cfg.get("triton", {}).get("max_batch_size", 1)
    precision = cfg.get("precision", {}).get("default", "fp16").upper()

    lines = [
        f'name: "{model_name}"',
        f'platform: "{backend}"',
        f"max_batch_size: {max_batch_size}",
        "",
    ]

    if "inputs" in cfg:
        lines.append("input [")
        for i, inp in enumerate(cfg["inputs"]):
            dims = ", ".join(str(d) for d in inp["shape"][1:])
            dtype = DTYPE_MAP[inp["dtype"]]
            comma = "," if i < len(cfg["inputs"]) - 1 else ""
            lines.append(
                f'  {{ name: "{inp["name"]}" data_type: {dtype} dims: [{dims}] }}{comma}'
            )
        lines.append("]\n")

    if "outputs" in cfg:
        lines.append("output [")
        for i, out in enumerate(cfg["outputs"]):
            dims = ", ".join(str(d) for d in out["shape"][1:])
            dtype = DTYPE_MAP[out["dtype"]]
            comma = "," if i < len(cfg["outputs"]) - 1 else ""
            lines.append(
                f'  {{ name: "{out["name"]}" data_type: {dtype} dims: [{dims}] }}{comma}'
            )
        lines.append("]\n")

    lines.append("instance_group [")
    lines.append("  { kind: KIND_GPU count: 1 }")
    lines.append("]\n")

    lines.append("optimization {")
    lines.append("  execution_accelerators {")
    lines.append("    gpu_execution_accelerator: [")
    lines.append(
        f'      {{ name: "tensorrt" parameters {{ key: "precision_mode" value: "{precision}" }} }}'
    )
    lines.append("    ]")
    lines.append("  }")
    lines.append("}\n")

    output_dir = Path(cfg["paths"]["engine_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pbtxt_path = output_dir / "config.pbtxt"
    pbtxt_path.write_text("\n".join(lines))

    print(f"Generated {pbtxt_path}")
    return pbtxt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Triton config.pbtxt from YAML"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    generate_triton_config(args.config)
