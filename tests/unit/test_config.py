import pytest

from joel_nvk.utils.config import load_config


def test_default_config_loads(configs_dir):
    config = load_config("default", configs=configs_dir)
    assert config["project"]["name"] == "joel-nvk"


def test_env_config_overrides_default(configs_dir):
    default = load_config("default", configs=configs_dir)
    dev = load_config("dev", configs=configs_dir)

    assert dev["logging"]["level"] == "DEBUG"
    # Keys absent from dev.yaml still come from default.yaml.
    assert dev["project"]["seed"] == default["project"]["seed"]


def test_missing_config_raises(configs_dir):
    with pytest.raises(FileNotFoundError):
        load_config("nope", configs=configs_dir)
