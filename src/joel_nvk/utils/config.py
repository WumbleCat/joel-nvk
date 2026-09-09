"""YAML config loading with `default.yaml` as the base layer."""

from pathlib import Path
from typing import Any

import yaml

from joel_nvk.utils.paths import config_dir

DEFAULT_ENV = "default"


def load_config(env: str = DEFAULT_ENV, configs: Path | None = None) -> dict[str, Any]:
    """Load ``configs/<env>.yaml`` merged over ``configs/default.yaml``.

    Args:
        env: Config name without the ``.yaml`` suffix (``dev``, ``production``).
        configs: Override the directory the config files are read from.

    Returns:
        The merged config as a plain dict.
    """
    directory = configs or config_dir()
    merged = _read(directory / f"{DEFAULT_ENV}.yaml")
    if env != DEFAULT_ENV:
        merged = _deep_merge(merged, _read(directory / f"{env}.yaml"))
    return merged


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        current = result.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            result[key] = _deep_merge(current, value)
        else:
            result[key] = value
    return result
