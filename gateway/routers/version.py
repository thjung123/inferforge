from pathlib import Path

import tomllib
from fastapi import APIRouter

router = APIRouter(redirect_slashes=False)

_pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
with open(_pyproject, "rb") as f:
    APP_VERSION = tomllib.load(f)["project"]["version"]


@router.get("")
@router.get("/")
async def get_version():
    return {
        "version": APP_VERSION,
    }
