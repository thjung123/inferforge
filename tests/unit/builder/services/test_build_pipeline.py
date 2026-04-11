from pathlib import Path

from builder.services.build_pipeline import _build_trtexec_command


def test_trtexec_command_fp16():
    cfg = {
        "model_name": "test",
        "precision": {"default": "fp16"},
        "inputs": [{"name": "input_ids", "shape": [-1, 128]}],
        "paths": {"engine_model_dir": "/tmp/test"},
    }
    onnx_path = Path("/tmp/test.onnx")

    cmd = _build_trtexec_command(cfg, onnx_path)

    assert "trtexec" in cmd
    assert "--fp16" in cmd
    assert f"--onnx={onnx_path}" in cmd
    assert "--verbose" in cmd


def test_trtexec_command_int8():
    cfg = {
        "model_name": "test",
        "precision": {"default": "int8"},
        "inputs": [],
        "paths": {"engine_model_dir": "/tmp/test"},
    }

    cmd = _build_trtexec_command(cfg, Path("/tmp/test.onnx"))

    assert "--int8" in cmd
    assert "--fp16" not in cmd


def test_trtexec_command_dynamic_shapes():
    cfg = {
        "model_name": "test",
        "precision": {"default": "fp16"},
        "inputs": [
            {"name": "input_ids", "shape": [-1, 128]},
            {"name": "mask", "shape": [-1, 128]},
        ],
        "dynamic_shapes": {
            "enabled": True,
            "input_ids": {"min": [1, 1], "opt": [4, 64], "max": [32, 128]},
            "mask": {"min": [1, 1], "opt": [4, 64], "max": [32, 128]},
        },
        "paths": {"engine_model_dir": "/tmp/test"},
    }

    cmd = _build_trtexec_command(cfg, Path("/tmp/test.onnx"))

    min_flag = [c for c in cmd if c.startswith("--minShapes=")]
    opt_flag = [c for c in cmd if c.startswith("--optShapes=")]
    max_flag = [c for c in cmd if c.startswith("--maxShapes=")]

    assert len(min_flag) == 1
    assert len(opt_flag) == 1
    assert len(max_flag) == 1
    assert "input_ids:" in min_flag[0]
    assert "mask:" in min_flag[0]
