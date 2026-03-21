import logging
from typing import Any

logger = logging.getLogger("builder")


class DAGValidationError(Exception):
    pass


def validate_ensemble_dag(ensemble_cfg: dict[str, Any]) -> None:
    ensemble_name = ensemble_cfg["name"]
    steps = ensemble_cfg["steps"]

    # Available tensors start with ensemble-level inputs
    available: set[str] = {inp["name"] for inp in ensemble_cfg["inputs"]}

    for step in steps:
        model_name = step["model_name"]

        # Check every input_map value exists in available tensors
        for local_name, tensor_name in step["input_map"].items():
            if tensor_name not in available:
                raise DAGValidationError(
                    f"[{ensemble_name}] step '{model_name}' input_map "
                    f"'{local_name}' references '{tensor_name}', "
                    f"but it is not produced by any previous step. "
                    f"Available: {sorted(available)}"
                )

        # Register output_map values as available
        for local_name, tensor_name in step["output_map"].items():
            available.add(tensor_name)

    # Check ensemble outputs are all produced
    for out in ensemble_cfg["outputs"]:
        if out["name"] not in available:
            raise DAGValidationError(
                f"[{ensemble_name}] ensemble output '{out['name']}' "
                f"is never produced by any step. "
                f"Available: {sorted(available)}"
            )

    logger.info(f"[DAG] {ensemble_name}: validation passed ({len(steps)} steps)")
