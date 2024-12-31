import atmorep.utils.config as config
import atmorep.utils.config_adapter as adapter
import pathlib as pl
import json
import pytest
import typing
import unittest.mock as mock


def get_sample_legacy_config() -> dict[str, typing.Any]:
    with open(SAMPLE_CONFIG, "r") as fp_config:
        legacy_config_dict = json.load(fp_config)

    return legacy_config_dict


def twiddle_option(value, option) -> typing.Any:
    # FIXME doesnt work for None

    def replicate_nested_list_structure(my_list: list[typing.Any]):
        return [
            replicate_nested_list_structure(item) if isinstance(item, list) else None
            for item in my_list
        ]

    if option in ["geo_range_sampling", "n_size"]:
        # special case for structural assumptions for geo_range_sampling
        mock_value = replicate_nested_list_structure(value)
    else:
        # for most options this suffices
        mock_value = mock.create_autospec(value)

    return mock_value


SAMPLE_CONFIG = pl.Path(__file__).parent / "model_idwc5e2i3t.json"

ALL_LEGACY_OPTIONS = list(get_sample_legacy_config().keys())
IGNORE_LEGACY_OPTIONS = ["file_path", "month"]
LEGACY_OPTIONS = [key for key in ALL_LEGACY_OPTIONS if key not in IGNORE_LEGACY_OPTIONS]


@pytest.fixture
def legacy_config_dict():
    legacy_config_dict = get_sample_legacy_config()
    for key in IGNORE_LEGACY_OPTIONS:
        del legacy_config_dict[key]

    return legacy_config_dict

@pytest.fixture
def legacy_config_dict_old_missing_options(legacy_config_dict):
    new_options = [
        "n_size", "num_samples_per_epoch", "num_samples_validate"
    ]
    renamed_options = {
        "num_loader_workers": "loader_num_workers",
        "years_val": "years_test",
        "batch_size": "batch_size_max",
        "batch_size_validation": "batch_size_max"
    }
    for option in new_options:
        try:
            del legacy_config_dict[option]
        except KeyError:
            pass
    
    for new_option, old_option in renamed_options.items():
        value = legacy_config_dict[new_option]
        legacy_config_dict[old_option] = value
    
    return legacy_config_dict


@pytest.fixture
def deserialized_config(legacy_config_dict):
    return config.AtmorepConfig.from_dict(legacy_config_dict)

@pytest.fixture
def config_adapter(legacy_config_dict):
    return adapter.Config.from_dict(legacy_config_dict, user_config=None)


@pytest.fixture
def config_adapter_empty():
    return adapter.Config.init_empty(user_config=None)


def test_serialization_identity(deserialized_config):
    config_dict = deserialized_config.as_dict()
    redeserialized_config = config.AtmorepConfig.from_dict(config_dict)

    assert deserialized_config == redeserialized_config

def test_options_backward_compatible(
    legacy_config_dict_old_missing_options, deserialized_config
):
    redeserialized_config = config.AtmorepConfig.from_dict(
        legacy_config_dict_old_missing_options
    )
    
    assert deserialized_config == redeserialized_config   

@pytest.mark.parametrize("option", LEGACY_OPTIONS)
def test_legacy_compatibility(legacy_config_dict, deserialized_config, option):
    config_dict = deserialized_config.as_dict()

    assert config_dict[option] == legacy_config_dict[option]


@pytest.mark.parametrize("option", LEGACY_OPTIONS)
def test_adapter_read_access(config_adapter, legacy_config_dict, option):
    assert getattr(config_adapter, option) == legacy_config_dict[option]


@pytest.mark.parametrize("option", LEGACY_OPTIONS)
def test_adapter_write_access(config_adapter, option):
    # twiddle option value
    real_value = getattr(config_adapter, option)
    mock_value = twiddle_option(real_value, option)
    setattr(config_adapter, option, mock_value)

    # deserialize to make sure values of new config backend are used
    try:
        config_dict = config_adapter.as_dict()
    except AttributeError:
        assert False

    # make sure option is the same accessed through adapter or serialized dict
    assert config_dict[option] == getattr(config_adapter, option)


@pytest.mark.parametrize("option", LEGACY_OPTIONS)
def test_adapter_add_to_empty(config_adapter_empty, legacy_config_dict, option):
    value = legacy_config_dict[option]
    setattr(config_adapter_empty, option, value)

    assert getattr(config_adapter_empty, option) == value
