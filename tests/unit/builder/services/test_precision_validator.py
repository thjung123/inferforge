import numpy as np
import pytest

from builder.services.precision_validator import (
    PrecisionThresholds,
    build_sample_inputs,
    compare_outputs,
    evaluate,
)


def test_identical_outputs_are_perfect():
    a = np.random.default_rng(0).standard_normal((8, 768)).astype(np.float32)
    report = compare_outputs(a, a.copy(), "emb")

    assert report.max_abs_err == 0.0
    assert report.cosine_min == pytest.approx(1.0, abs=1e-6)
    passed, reasons = evaluate(report, PrecisionThresholds())
    assert passed
    assert reasons == []


def test_fp16_scale_perturbation_passes():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((8, 768)).astype(np.float32)
    # ~fp16 rounding scale noise
    b = a + rng.normal(0, 1e-4, a.shape).astype(np.float32)

    report = compare_outputs(a, b, "emb")
    passed, _ = evaluate(
        report, PrecisionThresholds(min_cosine=0.999, max_abs_err=0.05)
    )

    assert passed
    assert report.cosine_min > 0.999
    assert report.err_std >= 0.0


def test_large_deviation_fails():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((8, 768)).astype(np.float32)
    b = rng.standard_normal((8, 768)).astype(np.float32)  # unrelated

    report = compare_outputs(a, b, "emb")
    passed, reasons = evaluate(report, PrecisionThresholds())

    assert not passed
    assert reasons


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_outputs(np.zeros((4, 10)), np.zeros((4, 11)), "emb")


def test_thresholds_from_cfg_overrides_and_defaults():
    cfg = {
        "precision": {
            "validation": {"min_cosine": 0.995, "max_abs_err": 0.1, "num_samples": 4}
        }
    }
    t = PrecisionThresholds.from_cfg(cfg)
    assert (t.min_cosine, t.max_abs_err, t.num_samples) == (0.995, 0.1, 4)

    default = PrecisionThresholds.from_cfg({})
    assert default.min_cosine == 0.999
    assert default.num_samples == 8


def test_build_sample_inputs_dtypes_and_shapes():
    cfg = {
        "inputs": [
            {"name": "input_ids", "shape": [-1, -1], "datatype": "INT32"},
            {"name": "pixel_values", "shape": [-1, 3, 224, 224], "datatype": "FP32"},
        ]
    }
    samples = build_sample_inputs(cfg, num_samples=4)

    assert samples["input_ids"].shape[0] == 4
    assert samples["input_ids"].dtype == np.int64  # ONNX export dtype for ids
    assert samples["pixel_values"].shape == (4, 3, 224, 224)
    assert samples["pixel_values"].dtype == np.float32


def test_build_sample_inputs_is_deterministic():
    cfg = {"inputs": [{"name": "x", "shape": [-1, 8], "datatype": "FP32"}]}
    a = build_sample_inputs(cfg, num_samples=4, seed=7)
    b = build_sample_inputs(cfg, num_samples=4, seed=7)
    np.testing.assert_array_equal(a["x"], b["x"])
