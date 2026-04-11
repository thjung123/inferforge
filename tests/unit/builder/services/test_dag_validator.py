import pytest

from builder.services.dag_validator import DAGValidationError, validate_ensemble_dag


def _make_ensemble(steps, inputs=None, outputs=None):
    return {
        "name": "test_ensemble",
        "inputs": inputs or [{"name": "IN", "datatype": "STRING", "dims": [-1]}],
        "outputs": outputs or [{"name": "OUT", "datatype": "FP32", "dims": [768]}],
        "steps": steps,
    }


def test_valid_dag_passes():
    cfg = _make_ensemble(
        steps=[
            {
                "model_name": "step1",
                "input_map": {"x": "IN"},
                "output_map": {"y": "MID"},
            },
            {
                "model_name": "step2",
                "input_map": {"a": "MID"},
                "output_map": {"b": "OUT"},
            },
        ]
    )
    validate_ensemble_dag(cfg)  # should not raise


def test_missing_input_reference_fails():
    cfg = _make_ensemble(
        steps=[
            {
                "model_name": "step1",
                "input_map": {"x": "NONEXISTENT"},
                "output_map": {"y": "OUT"},
            },
        ]
    )
    with pytest.raises(DAGValidationError, match="NONEXISTENT"):
        validate_ensemble_dag(cfg)


def test_missing_ensemble_output_fails():
    cfg = _make_ensemble(
        outputs=[{"name": "MISSING", "datatype": "FP32", "dims": [768]}],
        steps=[
            {
                "model_name": "step1",
                "input_map": {"x": "IN"},
                "output_map": {"y": "SOMETHING_ELSE"},
            },
        ],
    )
    with pytest.raises(DAGValidationError, match="MISSING"):
        validate_ensemble_dag(cfg)


def test_chained_steps_pass():
    cfg = _make_ensemble(
        steps=[
            {
                "model_name": "a",
                "input_map": {"x": "IN"},
                "output_map": {"y": "T1"},
            },
            {
                "model_name": "b",
                "input_map": {"x": "T1"},
                "output_map": {"y": "T2"},
            },
            {
                "model_name": "c",
                "input_map": {"x": "T2"},
                "output_map": {"y": "OUT"},
            },
        ]
    )
    validate_ensemble_dag(cfg)  # should not raise


def test_real_bert_preset():
    """Validate against actual bert preset ensemble structure."""
    import yaml

    with open("builder/presets/bert.yaml") as f:
        preset = yaml.safe_load(f)

    validate_ensemble_dag(preset["ensemble"])


def test_real_clip_preset():
    """Validate against actual clip preset ensemble structure."""
    import yaml

    with open("builder/presets/clip.yaml") as f:
        preset = yaml.safe_load(f)

    validate_ensemble_dag(preset["ensemble"])
