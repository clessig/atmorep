import datetime
import pytest
import xarray as xr
import numpy as np 
import warnings
import os

import atmorep.tests.test_utils as tu

# run it with e.g. pytest -s atmorep/tests/validation_test.py --result results/idztsut0mr

N_SAMPLES_MAX = 50
RESAMPLE_LVLS = False

MAX_LAT = 90.
MIN_LAT = -90.
MIN_LON = 0.
MAX_LON = 360.

@pytest.fixture(scope="module")
def config():
    return tu.get_config()

@pytest.fixture(scope="module")
def target(config):
    return tu.DataStore.from_config(config, tu.OutputType.target)

@pytest.fixture(scope="module")
def prediction(config):
        return tu.DataStore.from_config(config, tu.OutputType.prediction)


@pytest.fixture(scope="module")
def levels(target):
    return target.levels


@pytest.mark.parametrize(
    ("sample", "level"), tu.get_samples_and_levels()
)
class TestValidateOutput:
    @pytest.fixture
    def target_data(self, sample, level, target: tu.DataStore) -> tu.GroupData:
        return target.get_data(sample, level)

    @pytest.fixture
    def prediction_data(self, sample, level, prediction: tu.DataStore) -> tu.GroupData:
        return prediction.get_data(sample, level)
    
    
    def test_datetime(
        self, target_data, level, config: tu.ValidationConfig
    ):
        """
        Check against ERA5 timestamps.
        Loop over all levels individually. 50 random samples for each level.
        """
        timestamp = target_data.datetimes[0].astype(datetime.datetime)
        year = timestamp.year
        month = str(timestamp.month).zfill(2)

        era5_path = tu.ERA5_FILE_TEMPLATE.format(
            config.field_name, level, config.field_name, year, month, level
        )
        if not os.path.isfile(era5_path):
            warnings.warn(UserWarning((f"Timestamp {target_data.datetimes} not found in ERA5. Skipping")))
        else:
            era5 = xr.open_dataset(era5_path, engine = "cfgrib")[
                tu.FIELD_GRIB_IDX[config.field_name]
            ].sel(
                time = timestamp,
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
        assert (prediction_data.datetimes == target_data.datetimes).all, "Mismatch between datetimes"

    @pytest.mark.skip
    def test_rmse(
        self, target_data, prediction_data, config: tu.ValidationConfig
    ):
        """
        Test that for each field the RMSE does not exceed a certain value. 
        50 random samples.
        """
        
        assert tu.compute_RMSE(
            target_data.data, prediction_data.data
        ).mean() < tu.FIELD_MAX_RMSE[config.field_name]

def test_has_expected_timestamps(config: tu.ValidationConfig, prediction: tu.DataStore):
    actual = prediction.times
    expected = config.timesteps
    is_present = np.isin(expected, actual)

    assert is_present.all(), f"missing expected timestamps: {expected[~is_present]}, given: {actual}"

def test_levels_match(config: tu.ValidationConfig, prediction: tu.DataStore):
    actual = prediction.levels
    expected = np.array(config.field.levels)
    is_present = np.isin(expected, actual)
    
    assert is_present.all(), f"missing expected levels: {expected[~is_present]}, given: {actual}"
    