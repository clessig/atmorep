import numpy as np
import pandas as pd 


ERA5_FNAME = r"/gpfs/scratch/ehpc03/data/{}/ml{}/era5_{}_y{}_m{}_ml{}.grib"
ATMOREP_PRED = r"./results/id{}/results_id{}_epoch{}_pred.zarr"
ATMOREP_TARGET = r"./results/id{}/results_id{}_epoch{}_target.zarr"

MAX_LAT = 90.
MIN_LAT = -90.
MAX_LON = 0.
MIN_LON = 360.

FIELD_MAX_RMSE = {
    "temperature": 3,
    "velocity_u": 0.2,  # ????
    "velocity_v": 0.2,  # ????
    "velocity_z": 0.2,  # ????
    "vorticity": 0.2,  # ????
    "divergence": 0.2,  # ????
    "specific_humidity": 0.2,  # ????
    "total_precip": 1,  # ?????
}

FIELD_GRIB_IDX = {
    "velocity_u": "u",
    "temperature": "t",
    "total_precip": "tp",
    "velocity_v": "v",
    "velocity_z": "z",
    "vorticity": "vo",
    "divergence": "d",
    "specific_humidity": "q",
}

##################################################################

def get_BERT(atmorep, field, sample, level):
    atmorep_sample = atmorep[f"{field}/sample={sample:05d}/ml={level:05d}"] 
    data = atmorep_sample.data[0,0] 
    datetime = pd.Timestamp(atmorep_sample.datetime[0,0])
    lats = atmorep_sample.lat[0]
    lons = atmorep_sample.lon[0]
    return data, datetime, lats, lons

def get_forecast(atmorep, field, sample,level_idx):
    atmorep_sample = atmorep[f"{field}/sample={sample:05d}"]
    data = atmorep_sample.data[level_idx, 0]
    datetime = pd.Timestamp(atmorep_sample.datetime[0])
    lats = atmorep_sample.lat
    lons = atmorep_sample.lon
    return data, datetime, lats, lons

######################################

def check_lats(lats_pred, lats_target):
    assert (lats_pred[:] == lats_target[:]).all(), "Mismatch between latitudes"
    assert (lats_pred[:] <= MAX_LAT).all(), f"latitudes are between {np.amin(lats_pred)}- {np.amax(lats_pred)}"
    assert (lats_pred[:] >= MIN_LAT).all(), f"latitudes are between {np.amin(lats_pred)}- {np.amax(lats_pred)}"

def check_lons(lons_pred, lons_target):
    assert (lons_pred[:] == lons_target[:]).all(), "Mismatch between longitudes"
    assert (lons_pred[:] >= MIN_LON).all(), "longitudes are between {np.amin(lons_pred)}- {np.amax(lons_pred)}"
    assert (lons_pred[:] <= MAX_LON).all(), "longitudes are between {np.amin(lons_pred)}- {np.amax(lons_pred)}"

def check_datetimes(datetimes_pred, datetimes_target):
    assert (datetimes_pred == datetimes_target), "Mismatch between datetimes"

######################################

#calculate RMSE
def compute_RMSE(pred, target):
    return np.sqrt(np.mean((pred-target)**2))
