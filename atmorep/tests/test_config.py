import atmorep.utils.config as config
import pathlib as pl
import json
import pytest
import typing

def get_sample_legacy_config() -> dict[str, typing.Any]:
    with open(SAMPLE_CONFIG, "r") as fp_config:
        legacy_config_dict = json.load(fp_config)
    
    return legacy_config_dict

SAMPLE_CONFIG = pl.Path(__file__).parent / "model_idwc5e2i3t.json"

IGNORE_LEGACY_OPTIONS = ["file_path", "month"]

@pytest.fixture
def legacy_config_dict():
    legacy_config_dict = get_sample_legacy_config()
    for key in IGNORE_LEGACY_OPTIONS:
        del legacy_config_dict[key]
    
    return legacy_config_dict

@pytest.fixture
def deserialized_config(legacy_config_dict):
    return config.AtmorepConfig.from_dict(legacy_config_dict)

def test_serialization_identity(deserialized_config):
    config_dict = deserialized_config.as_dict()
    redeserialized_config = config.AtmorepConfig.from_dict(config_dict)
    
    assert deserialized_config == redeserialized_config

def test_legacy_compatibility(legacy_config_dict, deserialized_config):
    config_dict = deserialized_config.as_dict()
    
    assert config_dict == legacy_config_dict