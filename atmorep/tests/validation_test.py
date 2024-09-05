import pytest
import xarray as xr
import numpy as np 
import warnings
import os

import atmorep.tests.test_utils as test_utils

# run it with e.g. pytest -s atmorep/tests/validation_test.py results/idztsut0mr

N_SAMPLES_MAX = 50
RESAMPLE_LVLS = False

MAX_LAT = 90.
MIN_LAT = -90.
MIN_LON = 0.
MAX_LON = 360.

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
class TestValidateOutput:
    @pytest.fixture
    def level_idx(self, levels, level, config: test_utils.ValidationConfig) -> int:
        return config.data_access.get_level_idx(levels, level)
    
    @pytest.fixture
    def target_data(
        self, sample, level_idx, target, config:test_utils.ValidationConfig
    ) -> test_utils.GroupData:
        return config.data_access.get_data(target,config.field, sample, level_idx)

    @pytest.fixture
    def prediction_data(
        self, sample, level_idx, prediction, config:test_utils.ValidationConfig
    ) -> test_utils.GroupData:
        return config.data_access.get_data(prediction,config.field, sample, level_idx)
    
    
    def test_datetime(
        self, target_data, level, config: test_utils.ValidationConfig
    ):
        """
        Check against ERA5 timestamps.
        Loop over all levels individually. 50 random samples for each level.
        """
        year = target_data.datetime.year
        month = str(target_data.datetime.month).zfill(2)

        era5_path = test_utils.ERA5_FILE_TEMPLATE.format(
            config.field, level, config.field, year, month, level
        )
        if not os.path.isfile(era5_path):
            warnings.warn(UserWarning((f"Timestamp {target_data.datetime} not found in ERA5. Skipping")))
        else:
            era5 = xr.open_dataset(era5_path, engine = "cfgrib")[
                test_utils.FIELD_GRIB_IDX[config.field]
            ].sel(
                time = target_data.datetime,
                latitude = target_data.lats,
                longitude = target_data.lons
            )

            # assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"
            assert np.isclose(target_data.data[0], era5.values[0],rtol=1e-04, atol=1e-07).all(), "Mismatch between ERA5 and AtmoRep Timestamps"


    def test_lats_match(self, target_data, prediction_data):
        assert np.all(prediction_data.lats[:] == target_data.lats[:]), "Mismatch between latitudes"


    def test_lats_in_range(self, prediction_data):
        bigger_min = MIN_LAT <= prediction_data.lats[:]
        smaller_max = prediction_data.lats[:] <= MAX_LAT
        assert (bigger_min & smaller_max).all(), f"latitudes outside of between {MIN_LAT} - {MAX_LAT}"


    def test_lons_match(self, prediction_data, target_data):
        assert np.all(prediction_data.lons[:] == target_data.lons[:]), "Mismatch between latitudes"


    def test_lons_in_range(self, prediction_data):
        bigger_min = MIN_LON <= prediction_data.lons[:]
        smaller_max = prediction_data.lons[:] <= MAX_LON
        assert (bigger_min & smaller_max).all(), f"latitudes outside of between {MIN_LON} - {MAX_LON}"


    def test_datetimes_match(self, prediction_data, target_data):
        assert prediction_data.datetime == target_data.datetime, "Mismatch between datetimes"

    def test_rmse(
        self, target_data, prediction_data, config: test_utils.ValidationConfig
    ):
        """
        Test that for each field the RMSE does not exceed a certain value. 
        50 random samples.
        """
        
        assert test_utils.compute_RMSE(
            target_data.data, prediction_data.data
        ).mean() < test_utils.FIELD_MAX_RMSE[config.field]
    
def test_has_expected_timestamps(config: test_utils.ValidationConfig):
    actual = config.get_timestamps_from_data()
    expected = config.timesteps
    is_present = np.isin(expected, actual)
    assert is_present.all(), f"missing expected timestamps: {expected[~is_present]}, given: {actual}"