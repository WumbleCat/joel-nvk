"""PEFT writes a named adapter one directory deeper than the path it was given."""

import pytest

from joel_nvk.models.loading import resolve_adapter_dir


def test_flat_layout_is_returned_as_is(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    assert resolve_adapter_dir(tmp_path) == tmp_path


def test_named_adapter_subdirectory_is_found(tmp_path):
    nested = tmp_path / "math"
    nested.mkdir()
    (nested / "adapter_config.json").write_text("{}")
    assert resolve_adapter_dir(tmp_path) == nested


def test_missing_adapter_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No adapter_config.json"):
        resolve_adapter_dir(tmp_path)


def test_ambiguous_layout_is_an_error(tmp_path):
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "adapter_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="Several adapters"):
        resolve_adapter_dir(tmp_path)
