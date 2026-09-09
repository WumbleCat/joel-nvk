import pytest

from joel_nvk.cli import main


def test_config_command_prints_resolved_config(capsys):
    assert main(["config"]) == 0
    assert "project" in capsys.readouterr().out


def test_version_flag_exits_cleanly():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
