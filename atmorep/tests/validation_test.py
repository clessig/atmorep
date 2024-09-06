import pytest
import zarr
import cfgrib 
import xarray as xr
import numpy as np 
import random as rnd
import json
import warnings
import os

from atmorep.tests.test_utils import *

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

# Asma: What is the purpose of this function?
@pytest.fixture(autouse = True) 
def strategy(request):
    return request.config.getoption("strategy")

@pytest.fixture(autouse = True) 
def temporal_interpolation(request): # Asma
    strategy = request.config.getoption("strategy")
    return strategy == 'temporal_interpolation'

#TODO: add test for global_forecast vs ERA5

# def test_datetime(field, model_id, BERT, epoch = 0):

#     """
#     Check against ERA5 timestamps.
#     Loop over all levels individually. 50 random samples for each level.
#     """

#     store = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
#     atmorep = zarr.group(store)

#     nsamples = min(len(atmorep[field]), 50)
#     samples = rnd.sample(range(len(atmorep[field])), nsamples)
#     levels = [int(f.split("=")[1]) for f in atmorep[f"{field}/sample=00000"]] if BERT else atmorep[f"{field}/sample=00000"].ml[:]
   
#     get_data = get_BERT if BERT else get_forecast
   
#     for level in levels:
#         #TODO: make it more elegant
#         level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]

#         for s in samples:
#             data, datetime, lats, lons = get_data(atmorep, field, s, level_idx)
#             year, month = datetime.year, str(datetime.month).zfill(2)

#             era5_path = era5_fname().format(field, level, field, year, month, level)
#             if not os.path.isfile(era5_path):
#                 warnings.warn(UserWarning((f"era5_path = {era5_path} Timestamp {datetime} not found in ERA5. Skipping")))
#                 continue
#             era5 = xr.open_dataset(era5_path, engine = "cfgrib")[grib_index(field)].sel(time = datetime, latitude = lats, longitude = lons)

#             #assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"
#             assert np.isclose(data[0], era5.values[0],rtol=1e-04, atol=1e-07).all(), "Mismatch between ERA5 and AtmoRep Timestamps"

#############################################################################

# def test_coordinates(field, model_id, BERT, epoch = 0):
#     """
#     Check that coordinates match between target and prediction. 
#     Check also that latitude and longitudes are in geographical coordinates
#     50 random samples.
#     """
   
#     store_t = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
#     target = zarr.group(store_t)

#     store_p = zarr.ZipStore(atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
#     pred = zarr.group(store_p)

#     nsamples = min(len(target[field]), 50)
#     samples = rnd.sample(range(len(target[field])), nsamples)
#     levels = [int(f.split("=")[1]) for f in target[f"{field}/sample=00000"]] if BERT else target[f"{field}/sample=00000"].ml[:]
   
#     get_data = get_BERT if BERT else get_forecast

#     for level in levels:
#         level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]
#         for s in samples:
#             _, datetime_target, lats_target, lons_target = get_data(target,field, s, level_idx)
#             _, datetime_pred, lats_pred, lons_pred = get_data(pred, field, s, level_idx)

#             check_lats(lats_pred, lats_target)
#             check_lons(lons_pred, lons_target)
#             check_datetimes(datetime_pred, datetime_target)

#########################################################################

# def test_rmse(field, model_id, BERT, epoch = 0):
#     """
#     Test that for each field the RMSE does not exceed a certain value. 
#     50 random samples.
#     """
#     store_t = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
#     target = zarr.group(store_t)

#     store_p = zarr.ZipStore(atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
#     pred = zarr.group(store_p)
    
#     nsamples = min(len(target[field]), 50)
#     samples = rnd.sample(range(len(target[field])), nsamples)
#     levels = [int(f.split("=")[1]) for f in target[f"{field}/sample=00000"]] if BERT else target[f"{field}/sample=00000"].ml[:]
    
#     get_data = get_BERT if BERT else get_forecast
    
#     for level in levels:
#         level_idx = level if BERT else np.where(levels == level)[0].tolist()[0]
#         for s in samples:
#             sample_target, _, _, _ = get_data(target,field, s, level_idx)
#             sample_pred, _, _, _ = get_data(pred,field, s, level_idx)

#             assert compute_RMSE(sample_target, sample_pred).mean() < get_max_RMSE(field)

##################################################################################
# Asma c:
### Asma: this part was changed
### Asma: will be improved depending on comments
        
def test_idx_time_mask(field, model_id, BERT, epoch = 0):

    """
    Check against ERA5 timestamps, if the right time stamps are being masked
    Loop over all levels individually. 5 random samples for each level.
    """
    if temporal_interpolation:
        store_t = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
        target = zarr.group(store_t)

        store_s = zarr.ZipStore(atmorep_source().format(model_id, model_id, str(epoch).zfill(5)))
        source = zarr.group(store_s)

        nsamples = min(len(target[field]), 5)
        samples = rnd.sample(range(len(target[field])), nsamples)
        levels = [int(f.split("=")[1]) for f in target[f"{field}/sample=00000"]]

        with open(atmorep_json().format(model_id, model_id), 'r') as file_model_json:
            model_json = json.load(file_model_json)
            
        idx_time_mask = model_json['idx_time_mask']
        token_size = 3 # to be improved

        for s in samples:
            datetime_source = get_datetime_source(source,field, s, token_size)
            masked_source_timesteps = datetime_source[idx_time_mask]
            masked_source_timesteps = masked_source_timesteps.flatten()
            for level in levels:
                datetime_target = get_datetime_target(target,field, s, level, token_size)
                datetime_target = datetime_target.flatten()

                for timestep_idx in range(len(datetime_target)):
                    check_masked_timesteps(datetime_target[timestep_idx], masked_source_timesteps[timestep_idx])

    else:
        warnings.warn(UserWarning((f"Strategy is {strategy}")))

    # Asma: TODO
    # def test_idx_masked_data(field, model_id, BERT, epoch = 0):