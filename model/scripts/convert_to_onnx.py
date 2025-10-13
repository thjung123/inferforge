import argparse
import clip
import torch
import yaml
from pathlib import Path
from wrappers.clip_text_encoder_wrapper import CLIPTextEncoderWrapper


def load_model(model_name: str):
    if model_name == "bert_encoder":
        from transformers import BertModel

        return BertModel.from_pretrained("bert-base-uncased")
    elif model_name == "clip_text_encoder":
        model, _ = clip.load("ViT-B/32", device="cpu")
        return CLIPTextEncoderWrapper(model)
    elif model_name == "clip_image_encoder":
        model, _ = clip.load("ViT-B/32", device="cpu")
        return model.visual
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def convert_to_onnx(cfg_path: str, output_dir: Path = Path(".")) -> Path:
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    model = load_model(model_name).eval()

    # ONNX export paths
    output_path = Path(output_dir or cfg["paths"]["onnx_model"]).parent
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / f"{model_name}.onnx"

    # Example input
    input_cfg = cfg["inputs"][0]
    dummy_shape = tuple(input_cfg["shape"])
    dummy = (
        torch.randint(0, 10, dummy_shape)
        if "int" in input_cfg["dtype"]
        else torch.randn(dummy_shape)
    )

    dynamic_axes = {}
    if cfg.get("dynamic_shapes", {}).get("enabled"):
        for k in cfg["inputs"]:
            dynamic_axes[k["name"]] = (
                {0: "batch", 1: "seq_len"} if len(k["shape"]) > 1 else {0: "batch"}
            )

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=[x["name"] for x in cfg["inputs"]],
        output_names=[x["name"] for x in cfg["outputs"]],
        dynamic_axes=dynamic_axes,
    )

    print(f"[+] Exported {model_name} → {onnx_path}")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--output", help="Output ONNX directory")
    args = parser.parse_args()

    convert_to_onnx(args.config, args.output)
