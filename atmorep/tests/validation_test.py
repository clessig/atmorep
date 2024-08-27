import pytest
import zarr
import cfgrib 
import xarray as xr
import numpy as np 
import warnings
import os

import atmorep.tests.test_utils as test_utils

# run it with e.g. pytest -s atmorep/tests/validation_test.py --field temperature --model_id ztsut0mr --strategy BERT

N_SAMPLES_MAX = 50
RESAMPLE_LVLS = False

@pytest.fixture(scope="module")
def config():
    return test_utils.ValidationConfig.get()

@pytest.fixture(scope="module")
def target(config):
    return config.get_zarr(
        test_utils.OutputType.target
    )

@pytest.fixture(scope="module")
def prediction(config):
    return config.get_zarr(
        test_utils.OutputType.prediction
    )

@pytest.fixture(scope="module")
def levels(config):
    return config.get_levels(test_utils.OutputType.target)


@pytest.mark.parametrize(
    ("sample", "level"), test_utils.ValidationConfig.get().samples_and_levels()
)
def test_datetime(sample, level, levels, target, config: test_utils.ValidationConfig):
    """
    Check against ERA5 timestamps.
    Loop over all levels individually. 50 random samples for each level.
    """
    level_idx: int = config.data_access.get_level_idx(levels, level)
    
    data, datetime, lats, lons = config.data_access.get_data(
        target,
        config.field,
        sample,
        level_idx,
    )
    year, month = datetime.year, str(datetime.month).zfill(2)

    era5_path = test_utils.ERA5_FNAME.format(
        config.field, level, config.field, year, month, level
    )
    if not os.path.isfile(era5_path):
        warnings.warn(UserWarning((f"Timestamp {datetime} not found in ERA5. Skipping")))
    else:
        era5 = xr.open_dataset(era5_path, engine = "cfgrib")[
            test_utils.FIELD_GRIB_IDX[config.field]
        ].sel(time = datetime, latitude = lats, longitude = lons)

        # assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"
        assert np.isclose(data[0], era5.values[0],rtol=1e-04, atol=1e-07).all(), "Mismatch between ERA5 and AtmoRep Timestamps"


@pytest.mark.parametrize(
    ("sample", "level"), test_utils.ValidationConfig.get().samples_and_levels()
)
def test_coordinates(
    sample, level, levels, target, prediction, config: test_utils.ValidationConfig
):
    """
    Check that coordinates match between target and prediction. 
    Check also that latitude and longitudes are in geographical coordinates
    50 random samples.
    """

    level_idx: int = config.data_access.get_level_idx(levels, level)
    _, datetime_target, lats_target, lons_target = config.data_access.get_data(target,config.field, sample, level_idx)
    _, datetime_pred, lats_pred, lons_pred = config.data_access.get_data(prediction, config.field, sample, level_idx)

    test_utils.test_lats_match(lats_pred, lats_target)
    test_utils.test_lats_in_range(lats_pred)
    test_utils.test_lons_match(lons_pred, lons_target)
    test_utils.test_lons_in_range(lons_pred)
    test_utils.test_datetimes_match(datetime_pred, datetime_target)

@pytest.mark.parametrize(
    ("sample", "level"), test_utils.ValidationConfig.get().samples_and_levels()
)
def test_rmse(
    sample, level, levels, target, prediction, config: test_utils.ValidationConfig
    ):
    """
    Test that for each field the RMSE does not exceed a certain value. 
    50 random samples.
    """

    level_idx: int = config.data_access.get_level_idx(levels, level)
    sample_target, _, _, _ = config.data_access.get_data(target, config.field, sample, level_idx)
    sample_pred, _, _, _ = config.data_access.get_data(prediction, config.field, sample, level_idx)

    assert test_utils.compute_RMSE(sample_target, sample_pred).mean() < test_utils.FIELD_MAX_RMSE[config.field]
