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

@pytest.fixture
def field(request):
    return request.config.getoption("field")

@pytest.fixture
def model_id(request):
    return request.config.getoption("model_id")

@pytest.fixture
def epoch(request):
    request.config.getoption("epoch")

@pytest.fixture(autouse = True) 
def BERT(request):
    strategy = request.config.getoption("strategy")
    return (strategy == 'BERT' or strategy == 'temporal_interpolation')

@pytest.fixture(autouse = True) 
def strategy(request):
    return request.config.getoption("strategy")

#TODO: add test for global_forecast vs ERA5

@pytest.mark.gpu
def test_datetime(field, model_id, BERT, epoch = 0):

    """
    Check against ERA5 timestamps.
    Loop over all levels individually. 50 random samples for each level.
    """

    store = zarr.ZipStore(test_utils.atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    atmorep = zarr.group(store)

    nsamples = min(len(atmorep[field]), 50)
    samples = rnd.sample(range(len(atmorep[field])), nsamples)
    levels = [int(f.split("=")[1]) for f in atmorep[f"{field}/sample=00000"]] if BERT else atmorep[f"{field}/sample=00000"].ml[:]

    get_data = test_utils.get_BERT if BERT else test_utils.get_forecast

    for level in levels:
        #TODO: make it more elegant
        level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]

        for s in samples:
            data, datetime, lats, lons = get_data(atmorep, field, s, level_idx)
            year, month = datetime.year, str(datetime.month).zfill(2)

            era5_path = test_utils.era5_fname().format(field, level, field, year, month, level)
            if not os.path.isfile(era5_path):
                warnings.warn(UserWarning((f"Timestamp {datetime} not found in ERA5. Skipping")))
                continue
            era5 = xr.open_dataset(era5_path, engine = "cfgrib")[grib_index(field)].sel(time = datetime, latitude = lats, longitude = lons)

            #assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"
            assert np.isclose(data[0], era5.values[0],rtol=1e-04, atol=1e-07).all(), "Mismatch between ERA5 and AtmoRep Timestamps"

#############################################################################

@pytest.mark.gpu
def test_coordinates(field, model_id, BERT, epoch = 0):
    """
    Check that coordinates match between target and prediction. 
    Check also that latitude and longitudes are in geographical coordinates
    50 random samples.
    """

    store_t = zarr.ZipStore(test_utils.atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    target = zarr.group(store_t)

    store_p = zarr.ZipStore(test_utils.atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
    pred = zarr.group(store_p)

    nsamples = min(len(target[field]), 50)
    samples = rnd.sample(range(len(target[field])), nsamples)
    levels = [int(f.split("=")[1]) for f in target[f"{field}/sample=00000"]] if BERT else target[f"{field}/sample=00000"].ml[:]

    get_data = test_utils.get_BERT if BERT else test_utils.get_forecast

    for level in levels:
        level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]
        for s in samples:
            _, datetime_target, lats_target, lons_target = get_data(target,field, s, level_idx)
            _, datetime_pred, lats_pred, lons_pred = get_data(pred, field, s, level_idx)

            test_utils.check_lats(lats_pred, lats_target)
            test_utils.check_lons(lons_pred, lons_target)
            test_utils.check_datetimes(datetime_pred, datetime_target)

#########################################################################

@pytest.mark.gpu
def test_rmse(field, model_id, BERT, epoch = 0):
    """
    Test that for each field the RMSE does not exceed a certain value. 
    50 random samples.
    """
    store_t = zarr.ZipStore(test_utils.atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    target = zarr.group(store_t)

    store_p = zarr.ZipStore(test_utils.atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
    pred = zarr.group(store_p)
    
    nsamples = min(len(target[field]), 50)
    samples = rnd.sample(range(len(target[field])), nsamples)
    levels = [int(f.split("=")[1]) for f in target[f"{field}/sample=00000"]] if BERT else target[f"{field}/sample=00000"].ml[:]
    
    get_data = test_utils.get_BERT if BERT else test_utils.get_forecast
    
    for level in levels:
        level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]
        for s in samples:
            sample_target, _, _, _ = get_data(target,field, s, level_idx)
            sample_pred, _, _, _ = get_data(pred,field, s, level_idx)

            assert test_utils.compute_RMSE(sample_target, sample_pred).mean() < test_utils.get_max_RMSE(field)
