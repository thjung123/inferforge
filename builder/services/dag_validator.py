import logging
from typing import Any

logger = logging.getLogger("builder")


class DAGValidationError(Exception):
    pass


def validate_ensemble_dag(
    ensemble_cfg: dict[str, Any],
    model_outputs: dict[str, set[str]] | None = None,
) -> None:
    """Validate an ensemble's tensor wiring."""
    ensemble_name = ensemble_cfg["name"]
    steps = ensemble_cfg["steps"]

    available: set[str] = {inp["name"] for inp in ensemble_cfg["inputs"]}

    for step in steps:
        model_name = step["model_name"]

        for local_name, tensor_name in step["input_map"].items():
            if tensor_name not in available:
                raise DAGValidationError(
                    f"[{ensemble_name}] step '{model_name}' input_map "
                    f"'{local_name}' references '{tensor_name}', "
                    f"but it is not produced by any previous step. "
                    f"Available: {sorted(available)}"
                )

        if model_outputs is not None and model_name in model_outputs:
            declared = model_outputs[model_name]
            for local_name in step["output_map"]:
                if local_name not in declared:
                    raise DAGValidationError(
                        f"[{ensemble_name}] step '{model_name}' output_map key "
                        f"'{local_name}' is not an output of '{model_name}'. "
                        f"Declared outputs: {sorted(declared)}"
                    )

        for local_name, tensor_name in step["output_map"].items():
            available.add(tensor_name)

    for out in ensemble_cfg["outputs"]:
        if out["name"] not in available:
            raise DAGValidationError(
                f"[{ensemble_name}] ensemble output '{out['name']}' "
                f"is never produced by any step. "
                f"Available: {sorted(available)}"
            )

    logger.info(f"[DAG] {ensemble_name}: validation passed ({len(steps)} steps)")
