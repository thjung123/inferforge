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


def convert_to_onnx(cfg_path: str, output_dir: Path = Path(".")) -> Path:
    cfg = yaml.safe_load(open(cfg_path))
    model_name = cfg["model_name"]
    model = load_model(model_name)

    # ONNX output path
    output_path = Path(output_dir or cfg["paths"]["onnx_model"]).parent
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = output_path / f"{model_name}.onnx"

    # Build dummy inputs
    dummy_inputs = []
    for inp_cfg in cfg["inputs"]:
        shape = tuple(inp_cfg["shape"])
        dtype = inp_cfg["dtype"]
        name = inp_cfg["name"]

        if "int" in dtype:
            if "token_type" in name:
                dummy_inputs.append(
                    torch.zeros(shape, dtype=torch.int32)
                )  # only 0 or 1
            elif "attention_mask" in name:
                dummy_inputs.append(torch.ones(shape, dtype=torch.int32))  # all 1s
            else:  # e.g. input_ids
                vocab_size = 30522 if "bert" in model_name else 10000
                dummy_inputs.append(
                    torch.randint(0, vocab_size, shape, dtype=torch.int32)
                )
        else:
            dummy_inputs.append(torch.randn(shape))

    dummy_tuple = tuple(dummy_inputs)

    # Dynamic axes config
    dynamic_axes = {}
    if cfg.get("dynamic_shapes", {}).get("enabled"):
        for inp in cfg["inputs"]:
            name = inp["name"]
            shape = inp["shape"]
            dynamic_axes[name] = (
                {0: "batch", 1: "seq_len"} if len(shape) > 1 else {0: "batch"}
            )

        for out in cfg["outputs"]:
            name = out["name"]
            shape = out["shape"]
            dynamic_axes[name] = (
                {0: "batch", 1: "seq_len"} if len(shape) > 1 else {0: "batch"}
            )

    # Export
    torch.onnx.export(
        model,
        dummy_tuple,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[x["name"] for x in cfg["inputs"]],
        output_names=[x["name"] for x in cfg["outputs"]],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
    )

    print(f"[+] Exported {model_name} → {onnx_path}")
    if dynamic_axes:
        print(f"    Dynamic axes enabled: {dynamic_axes}")
    return onnx_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--output", help="Output ONNX directory")
    args = parser.parse_args()

    convert_to_onnx(args.config, args.output)
