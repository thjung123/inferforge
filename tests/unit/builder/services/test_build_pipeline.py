from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

import builder.services.build_pipeline as build_pipeline
from builder.schemas import JobState
from builder.services.build_pipeline import _build_trtexec_command, run_build_pipeline


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


class _FakeSettings:
    model_repository = "/tmp/repo"
    triton_http_url = "http://triton:8000"
    push_to_object_store = False


@pytest.mark.asyncio
async def test_precision_validation_error_fails_closed(monkeypatch):
    preset = {
        "model_type": "bert",
        "source": "src",
        "model_name": "encoder",
        "inputs": [],
        "outputs": [{"name": "logits"}],
    }

    monkeypatch.setattr(build_pipeline, "get_builder_settings", lambda: _FakeSettings())
    monkeypatch.setattr(
        build_pipeline, "_build_single_model", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(build_pipeline, "load_model", AsyncMock(return_value=None))
    monkeypatch.setattr(
        build_pipeline,
        "_validate_precision",
        AsyncMock(side_effect=httpx.ConnectError("triton unreachable")),
    )

    tracker = AsyncMock()

    await run_build_pipeline("job-1", preset, tracker)

    tracker.set_failed.assert_awaited_once()
    statuses = [call.args[1] for call in tracker.update_status.await_args_list]
    assert JobState.READY not in statuses
