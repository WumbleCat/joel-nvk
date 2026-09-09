"""Fixtures shared by every test module."""

from pathlib import Path

import pytest

from joel_nvk.utils.paths import PROJECT_ROOT


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    return project_root / "configs"


@pytest.fixture
def tmp_outputs(tmp_path: Path) -> Path:
    """An outputs/-shaped directory that is thrown away after the test."""
    for name in ("logs", "figures", "results"):
        (tmp_path / name).mkdir(parents=True)
    return tmp_path
