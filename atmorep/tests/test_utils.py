import numpy as np
import pandas as pd 
import zarr


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


def get_group(store_path_template: str, model_id: int, epoch: int) -> zarr.Group:
    store = zarr.ZipStore(
        store_path_template.format(model_id, model_id, str(epoch).zfill(5))
    )
    return zarr.group(store)

def get_levels_BERT(data_store: zarr.Group, field: str):
    return [int(f.split("=")[1]) for f in data_store[f"{field}/sample=00000"]]

def get_levels_forecast(data_store: zarr.Group, field: str):
    return data_store[f"{field}/sample=00000"].ml[:]

def get_data_BERT(data_store: zarr.Group, field: str, sample: int, level: int):
    atmorep_sample = data_store[f"{field}/sample={sample:05d}/ml={level:05d}"] 
    data = atmorep_sample.data[0,0] 
    datetime = pd.Timestamp(atmorep_sample.datetime[0,0])
    lats = atmorep_sample.lat[0]
    lons = atmorep_sample.lon[0]
    return data, datetime, lats, lons

def get_data_forecast(data_store: zarr.Group, field: str, sample: int,level: int):
    atmorep_sample = data_store[f"{field}/sample={sample:05d}"]
    data = atmorep_sample.data[level, 0]
    datetime = pd.Timestamp(atmorep_sample.datetime[0])
    lats = atmorep_sample.lat
    lons = atmorep_sample.lon
    return data, datetime, lats, lons

def get_level_idx_BERT(levels: np.ndarray, level: int) -> int:
    return level

def get_level_idx_forecast(levels: np.ndarray, level: int) -> int:
    return np.where(levels == level)[0].tolist()[0]

######################################


def test_lats_match(lats_pred, lats_target):
    assert (lats_pred[:] == lats_target[:]).all(), "Mismatch between latitudes"


def test_lats_in_range(lats_pred):
    in_range = MIN_LAT <= lats_pred[:] <= MAX_LAT  # TODO: check syntax for np arrays
    assert (in_range).all(), f"latitudes outside of between {MIN_LAT} - {MAX_LAT}"


def test_lons_match(lons_pred, lons_target):
    assert (lons_pred[:] == lons_target[:]).all(), "Mismatch between latitudes"


def test_lons_in_range(lons_pred):
    in_range = MIN_LON <= lons_pred[:] <= MAX_LON  # TODO: check syntax for np arrays
    assert (in_range).all(), f"latitudes outside of between {MIN_LON} - {MAX_LON}"


def test_datetimes_match(datetimes_pred, datetimes_target):
    assert datetimes_pred == datetimes_target, "Mismatch between datetimes"


######################################

#calculate RMSE
def compute_RMSE(pred, target):
    return np.sqrt(np.mean((pred-target)**2))
