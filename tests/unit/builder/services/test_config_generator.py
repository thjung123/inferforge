from pathlib import Path

from builder.services.config_generator import (
    generate_config_pbtxt,
    generate_ensemble_config,
    generate_processor_config,
)


def test_generate_config_pbtxt(tmp_path: Path):
    cfg = {
        "model_name": "test_engine",
        "inputs": [
            {"name": "input_ids", "shape": [-1, 128], "datatype": "INT64"},
        ],
        "outputs": [
            {"name": "output", "shape": [-1, 768], "datatype": "FP32"},
        ],
        "precision": {"default": "fp16"},
        "triton": {"max_batch_size": 64, "instance_count": 2},
    }

    result = generate_config_pbtxt(cfg, tmp_path)

    assert result.exists()
    content = result.read_text()
    assert 'name: "test_engine"' in content
    assert 'backend: "tensorrt_plan"' in content
    assert "max_batch_size: 64" in content
    assert "TYPE_INT64" in content
    assert "TYPE_FP32" in content
    assert "count: 2" in content
    assert 'precision_mode" value: "FP16"' in content


def test_generate_processor_config(tmp_path: Path):
    step = {
        "model_name": "bert_preprocessor",
        "backend": "python",
        "instance_kind": "KIND_CPU",
        "params": {"max_length": 128},
        "inputs": [
            {"name": "TEXTS", "datatype": "STRING", "dims": [-1]},
        ],
        "outputs": [
            {"name": "INPUT_IDS", "datatype": "INT32"},
            {"name": "ATTENTION_MASK", "datatype": "INT32"},
        ],
    }

    result = generate_processor_config(step, tmp_path, max_batch_size=256)

    assert result.exists()
    content = result.read_text()
    assert 'name: "bert_preprocessor"' in content
    assert 'backend: "python"' in content
    assert "max_batch_size: 256" in content
    assert "KIND_CPU" in content
    assert "TYPE_STRING" in content
    # max_length should override dims for non-STRING outputs
    assert "dims: [128]" in content
    assert 'key: "max_length"' in content


def test_processor_config_max_length_skips_string(tmp_path: Path):
    step = {
        "model_name": "test_proc",
        "params": {"max_length": 64},
        "inputs": [{"name": "IN", "datatype": "STRING", "dims": [-1]}],
        "outputs": [
            {"name": "TEXT_OUT", "datatype": "STRING", "dims": [-1]},
            {"name": "IDS", "datatype": "INT32"},
        ],
    }

    result = generate_processor_config(step, tmp_path)
    content = result.read_text()

    # STRING output should keep original dims, not max_length
    assert "TYPE_STRING" in content
    # INT32 should get max_length dims
    assert "dims: [64]" in content


def test_generate_ensemble_config(tmp_path: Path):
    ensemble_cfg = {
        "name": "test_ensemble",
        "max_batch_size": 128,
        "inputs": [
            {"name": "TEXTS", "datatype": "STRING", "dims": [-1]},
        ],
        "outputs": [
            {"name": "emb", "datatype": "FP32", "dims": [768]},
        ],
        "steps": [
            {
                "model_name": "preprocessor",
                "input_map": {"TEXTS": "TEXTS"},
                "output_map": {"IDS": "IDS"},
            },
            {
                "model_name": "encoder",
                "input_map": {"input_ids": "IDS"},
                "output_map": {"output": "emb"},
            },
        ],
    }

    result = generate_ensemble_config(ensemble_cfg, tmp_path)

    assert result.exists()
    content = result.read_text()
    assert 'name: "test_ensemble"' in content
    assert 'platform: "ensemble"' in content
    assert "max_batch_size: 128" in content
    assert "ensemble_scheduling" in content
    assert 'model_name: "preprocessor"' in content
    assert 'model_name: "encoder"' in content
    assert "input_map" in content
    assert "output_map" in content

    # Check version dir created
    assert (tmp_path / "test_ensemble" / "1").is_dir()
