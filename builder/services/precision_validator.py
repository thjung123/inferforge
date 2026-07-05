"""Post-conversion precision validation of the fp16 engine against the fp32 reference."""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("builder")


@dataclass
class PrecisionReport:
    output_name: str
    num_samples: int
    max_abs_err: float
    mean_abs_err: float
    err_std: float
    rel_err_mean: float
    cosine_mean: float
    cosine_min: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PrecisionThresholds:
    min_cosine: float = 0.999
    max_abs_err: float = 0.05
    num_samples: int = 8

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "PrecisionThresholds":
        v = (cfg.get("precision") or {}).get("validation") or {}
        d = cls()
        return cls(
            min_cosine=float(v.get("min_cosine", d.min_cosine)),
            max_abs_err=float(v.get("max_abs_err", d.max_abs_err)),
            num_samples=int(v.get("num_samples", d.num_samples)),
        )


def _cosine_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(an * bn, axis=1)


def compare_outputs(
    reference: np.ndarray, candidate: np.ndarray, output_name: str = "output"
) -> PrecisionReport:
    """Aggregate error statistics between an fp32 reference and a candidate."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if ref.shape != cand.shape:
        raise ValueError(
            f"shape mismatch for '{output_name}': "
            f"reference {ref.shape} vs candidate {cand.shape}"
        )

    n = ref.shape[0]
    ref2 = ref.reshape(n, -1)
    cand2 = cand.reshape(n, -1)

    abs_err = np.abs(ref2 - cand2)
    rel_err = abs_err / (np.abs(ref2) + 1e-9)
    cos = _cosine_per_row(ref2, cand2)

    return PrecisionReport(
        output_name=output_name,
        num_samples=int(n),
        max_abs_err=float(abs_err.max()),
        mean_abs_err=float(abs_err.mean()),
        err_std=float(abs_err.std()),
        rel_err_mean=float(rel_err.mean()),
        cosine_mean=float(cos.mean()),
        cosine_min=float(cos.min()),
    )


def evaluate(
    report: PrecisionReport, thresholds: PrecisionThresholds
) -> tuple[bool, list[str]]:
    """Accept iff worst-case cosine and max error are within tolerance."""
    reasons: list[str] = []
    if report.cosine_min < thresholds.min_cosine:
        reasons.append(
            f"[{report.output_name}] cosine_min {report.cosine_min:.5f} "
            f"< {thresholds.min_cosine}"
        )
    if report.max_abs_err > thresholds.max_abs_err:
        reasons.append(
            f"[{report.output_name}] max_abs_err {report.max_abs_err:.5f} "
            f"> {thresholds.max_abs_err}"
        )
    return (len(reasons) == 0, reasons)


def build_sample_inputs(
    cfg: dict[str, Any], num_samples: int = 8, seed: int = 0
) -> dict[str, np.ndarray]:
    """Deterministic representative inputs for the reference/candidate runs."""
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for inp in cfg["inputs"]:
        name = inp["name"]
        shape = [num_samples] + [(d if d > 0 else 16) for d in inp["shape"][1:]]
        dt = inp["datatype"].upper()
        if dt in ("INT32", "INT64"):
            lname = name.lower()
            if "mask" in lname:
                inputs[name] = rng.integers(0, 2, size=shape, dtype=np.int64)
            elif "type" in lname:
                inputs[name] = np.zeros(shape, dtype=np.int64)
            else:
                inputs[name] = rng.integers(0, 1000, size=shape, dtype=np.int64)
        else:
            inputs[name] = rng.standard_normal(size=shape).astype(np.float32)
    return inputs


def run_onnx_reference(
    onnx_path: Any, sample_inputs: dict[str, np.ndarray], output_names: list[str]
) -> dict[str, np.ndarray]:
    """fp32 reference outputs from the exported ONNX via onnxruntime."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outs = sess.run(list(output_names), dict(sample_inputs))
    return dict(zip(output_names, outs))
