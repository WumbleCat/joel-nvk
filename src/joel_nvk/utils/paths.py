"""Absolute paths to the project's well-known directories.

Everything is derived from the package location, so paths stay correct no
matter which directory the process was started from.
"""

from pathlib import Path

# src/joel_nvk/utils/paths.py -> src/joel_nvk/utils -> src/joel_nvk -> src -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def config_dir() -> Path:
    return PROJECT_ROOT / "configs"


def data_dir() -> Path:
    return PROJECT_ROOT / "data"


def outputs_dir() -> Path:
    return PROJECT_ROOT / "outputs"
