from pathlib import Path
from typing import Any

import yaml

_PRESET_DIR = Path(__file__).resolve().parents[1] / "presets"


def list_presets() -> list[str]:
    return [p.stem for p in _PRESET_DIR.glob("*.yaml")]


def load_preset(model_type: str) -> dict[str, Any]:
    path = _PRESET_DIR / f"{model_type}.yaml"
    if not path.exists():
        available = list_presets()
        raise FileNotFoundError(
            f"Preset '{model_type}' not found. Available: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)
