import pytest
import zarr
import cfgrib 
import xarray as xr
import numpy as np 
import random as rnd
import warnings
import os

import atmorep.tests.test_utils as test_utils

# run it with e.g. pytest -s atmorep/tests/validation_test.py --field temperature --model_id ztsut0mr --strategy BERT

N_SAMPLES_MAX = 50

@pytest.fixture
def field(request):
    return request.config.getoption("field")

@pytest.fixture
def model_id(request):
    return request.config.getoption("model_id")

@pytest.fixture
def epoch(request):
    request.config.getoption("epoch")

@pytest.fixture
def target(model_id, epoch):
    return test_utils.get_group(test_utils.ATMOREP_TARGET, model_id, epoch)

@pytest.fixture
def prediction(model_id, epoch):
    return test_utils.get_group(test_utils.ATMOREP_PRED, model_id, epoch)

@pytest.fixture(autouse = True) 
def data_access(request):
    strategy = request.config.getoption("strategy")
    return test_utils.DataAccess.construct(strategy)

@pytest.fixture(autouse = True) 
def strategy(request):
    return request.config.getoption("strategy")

#TODO: add test for global_forecast vs ERA5

@pytest.mark.gpu
def test_datetime(field, data_access, target):
    """
    Check against ERA5 timestamps.
    Loop over all levels individually. 50 random samples for each level.
    """

    samples = test_utils.get_samples(target, field, N_SAMPLES_MAX)
    levels = data_access.get_levels(target, field)

    for s in samples:
        for level in levels:
            level_idx: int = data_access.get_level_idx(levels, level)
            data, datetime, lats, lons = data_access.get_data(target, field, s, level_idx)
            year, month = datetime.year, str(datetime.month).zfill(2)

            era5_path = test_utils.ERA5_FNAME.format(
                field, level, field, year, month, level
            )
            if not os.path.isfile(era5_path):
                warnings.warn(UserWarning((f"Timestamp {datetime} not found in ERA5. Skipping")))
                continue
            era5 = xr.open_dataset(era5_path, engine = "cfgrib")[test_utils.FIELD_GRIB_IDX[field]].sel(time = datetime, latitude = lats, longitude = lons)

            #assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"
            assert np.isclose(data[0], era5.values[0],rtol=1e-04, atol=1e-07).all(), "Mismatch between ERA5 and AtmoRep Timestamps"

#############################################################################

@pytest.mark.gpu
def test_coordinates(field, data_access, target, prediction):
    """
    Check that coordinates match between target and prediction. 
    Check also that latitude and longitudes are in geographical coordinates
    50 random samples.
    """

    samples = test_utils.get_samples(target, field, N_SAMPLES_MAX)
    levels = data_access.get_levels(target, field)

    for s in samples:
        for level in levels:
            level_idx: int = data_access.get_level_idx(levels, level)
            _, datetime_target, lats_target, lons_target = data_access.get_data(target,field, s, level_idx)
            _, datetime_pred, lats_pred, lons_pred = data_access.get_data(prediction, field, s, level_idx)

            test_utils.test_lats_match(lats_pred, lats_target)
            test_utils.test_lats_in_range(lats_pred)
            test_utils.test_lons_match(lons_pred, lons_target)
            test_utils.test_lons_in_range(lons_pred)
            test_utils.test_datetimes_match(datetime_pred, datetime_target)

#########################################################################

@pytest.mark.gpu
def test_rmse(field, data_access, target, prediction):
    """
    Test that for each field the RMSE does not exceed a certain value. 
    50 random samples.
    """
    
    samples = test_utils.get_samples(target, field, N_SAMPLES_MAX)
    levels = data_access.get_levels(target, field)
    
    for s in samples:
        for level in levels:
            level_idx: int = data_access.get_level_idx(levels, level)
            sample_target, _, _, _ = data_access.get_data(target,field, s, level_idx)
            sample_pred, _, _, _ = data_access.get_data(prediction,field, s, level_idx)

            assert test_utils.compute_RMSE(sample_target, sample_pred).mean() < test_utils.FIELD_MAX_RMSE[field]
