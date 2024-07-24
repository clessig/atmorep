import pytest
import zarr
import cfgrib 
import xarray as xr
import pandas as pd 
import numpy as np 
import random as rnd
import warnings
import os

from atmorep.tests.test_utils import *


@pytest.fixture
def field(request):
    return request.config.getoption("field")

@pytest.fixture
def model_id(request):
    return request.config.getoption("model_id")

@pytest.fixture 
def epoch(request):
    request.config.getoption("epoch")

##################################################################

#@pytest.mark.parametrize(metafunc): #"field, level, model_id", [("temperature", 105, "ztsut0mr")])
def test_datetime(field, model_id, epoch = 0):

    """
    Check against ERA5 timestamps
    """
    #field, model_id, epoch = get_fixtures(metafunc)

    level = 137
    store = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    atmorep = zarr.group(store)

    #TODO: make it more elegant
    ml_idx = np.where(atmorep[f"{field}/sample=00000"].ml[:] == level)[0].tolist()[0]

    nsamples = min(len(atmorep[field]), 50)
    samples = rnd.sample(range(len(atmorep[field])), nsamples)

    for s in samples:
      
        data = atmorep[f"{field}/sample={s:05d}"].data[ml_idx, 0]
        datetime = pd.Timestamp(atmorep[f"{field}/sample={s:05d}"].datetime[0])
        lats = atmorep[f"{field}/sample={s:05d}"].lat
        lons = atmorep[f"{field}/sample={s:05d}"].lon

        year, month = datetime.year, str(datetime.month).zfill(2)

        era5_path = era5_fname().format(field, level, field, year, month, level)
        if not os.path.isfile(era5_path):
            warnings.warn(UserWarning((f"Timestamp {datetime} not found in ERA5. Skipping")))
            continue
        era5 = xr.open_dataset(era5_path, engine = "cfgrib")[grib_index(field)].sel(time = datetime, latitude = lats, longitude = lons)

        assert (data[0] == era5.values[0]).all(), "Mismatch between ERA5 and AtmoRep Timestamps"

#############################################################################

#@pytest.mark.parametrize("field, model_id", [("temperature", "ztsut0mr")])

def test_coordinates(field, model_id, epoch = 0):
    """
    Check that coordinates match between target and prediction. 
    Check also that latitude and longitudes are in geographical coordinates
    """
    #field, model_id, epoch = get_fixtures(metafunc)
    store_t = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    target = zarr.group(store_t)

    store_p = zarr.ZipStore(atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
    pred = zarr.group(store_p)

    nsamples = min(len(target[field]), 50)
    samples = rnd.sample(range(len(target[field])), nsamples)
    
    for s in samples:
        datetime_target = [pd.Timestamp(i) for i in target[f"{field}/sample={s:05d}"].datetime]
        lats_target = target[f"{field}/sample={s:05d}"].lat
        lons_target = target[f"{field}/sample={s:05d}"].lon

        datetime_pred =  [pd.Timestamp(i) for i in pred[f"{field}/sample={s:05d}"].datetime]
        lats_pred = pred[f"{field}/sample={s:05d}"].lat
        lons_pred = pred[f"{field}/sample={s:05d}"].lon

        check_lats(lats_pred, lats_target)
        check_lons(lons_pred, lons_target)
        check_datetimes(datetime_pred, datetime_target)

#########################################################################

#@pytest.mark.parametrize("field, model_id", [("temperature", "ztsut0mr")])

def test_rmse(field, model_id, epoch = 0):
   # field, model_id, epoch = get_fixtures(metafunc)

    store_t = zarr.ZipStore(atmorep_target().format(model_id, model_id, str(epoch).zfill(5)))
    target = zarr.group(store_t)

    store_p = zarr.ZipStore(atmorep_pred().format(model_id, model_id, str(epoch).zfill(5)))
    pred = zarr.group(store_p)

    nsamples = min(len(target[field]), 50)
    samples = rnd.sample(range(len(target[field])), nsamples)
    
    for s in samples:
        sample_target = target[f"{field}/sample={s:05d}"].data[:]
        sample_pred   = pred[f"{field}/sample={s:05d}"].data[:]

        assert compute_RMSE(sample_target, sample_pred).mean() < get_max_RMSE(field)
