import pytest

from builder.services.preset_loader import list_presets, load_preset


def test_list_presets():
    presets = list_presets()
    assert "bert" in presets
    assert "clip" in presets


def test_load_bert_preset():
    preset = load_preset("bert")
    assert preset["model_type"] == "bert"
    assert "inputs" in preset
    assert "outputs" in preset
    assert "ensemble" in preset


def test_load_clip_preset():
    preset = load_preset("clip")
    assert preset["model_type"] == "clip"
    assert "submodels" in preset
    assert "ensemble" in preset


def test_load_missing_preset():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_preset("nonexistent_model")
