import argparse
import clip
import torch
import yaml
from pathlib import Path
from wrappers.clip_text_encoder_wrapper import CLIPTextEncoderWrapper


def load_model(model_name: str):
    if model_name == "bert_encoder":
        from transformers import BertModel

        return BertModel.from_pretrained("bert-base-uncased").eval()
    elif model_name == "clip_text_encoder":
        model, _ = clip.load("ViT-B/32", device="cpu")
        return CLIPTextEncoderWrapper(model).eval()
    elif model_name == "clip_image_encoder":
        model, _ = clip.load("ViT-B/32", device="cpu")
        return model.visual.eval()
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_dynamic_axes(model_name: str, cfg: dict) -> dict:
    if not cfg.get("dynamic_shapes", {}).get("enabled"):
        return {}
    if model_name == "clip_text_encoder":
        return {"text": {0: "batch"}, "text_embedding": {0: "batch"}}
    elif model_name == "clip_image_encoder":
        return {"image": {0: "batch"}, "image_embedding": {0: "batch"}}
    elif model_name == "bert_encoder":
        return {
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "token_type_ids": {0: "batch", 1: "seq_len"},
        }
    return {}


def build_dummy_inputs(model_name: str, cfg: dict):
    dummy_inputs = []
    for inp_cfg in cfg["inputs"]:
        shape = [s if s != -1 else 8 for s in inp_cfg["shape"]]
        dtype = inp_cfg["dtype"]
        if "int" in dtype:
            dummy_inputs.append(
                torch.randint(0, 10000, tuple(shape), dtype=torch.int32)
            )
        else:
            dummy_inputs.append(torch.randn(tuple(shape), dtype=torch.float32))
    return tuple(dummy_inputs)


def convert_to_onnx(cfg_path: str, output_dir: Path = Path(".")) -> Path:
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    model = load_model(model_name)
    onnx_path = Path(cfg["paths"]["onnx_model"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_tuple = build_dummy_inputs(model_name, cfg)
    dynamic_axes = get_dynamic_axes(model_name, cfg)
    dynamic_shapes = dynamic_axes if dynamic_axes else None
    torch.onnx.export(
        model,
        dummy_tuple,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=[x["name"] for x in cfg["inputs"]],
        output_names=[x["name"] for x in cfg["outputs"]],
        dynamic_axes=dynamic_shapes,
    )
    print(f"[+] Exported {model_name} → {onnx_path}")
    if dynamic_axes:
        print(f"    Dynamic axes enabled: {dynamic_axes}")
    else:
        print("    Dynamic axes disabled (static batch)")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--output", help="Output ONNX directory")
    args = parser.parse_args()
    convert_to_onnx(args.config, args.output)
