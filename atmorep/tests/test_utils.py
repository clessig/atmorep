import abc
from numpy.typing import NDArray
import numpy as np
import pandas as pd 
import zarr
import random as rnd
import itertools as it
import strenum
from collections.abc import Iterable


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

class DataAccess(abc.ABC):
    def __init__(self):
        pass
    
    @classmethod
    def construct(cls, strategy: str):
        match strategy:
            case "BERT":
                return BERT()
            case "temporal_interpolation":
                return BERT()
            case _:
                return Forecast()
            
    @abc.abstractmethod
    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        pass
    
    @abc.abstractmethod
    def get_data(self, data_store: zarr.Group, field: str, sample: int, level):
        pass
    
    @abc.abstractmethod
    def get_level_idx(self, levels: NDArray[np.int64], level: int) -> int:
        pass

class BERT(DataAccess):

    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        iterable = (int(f.split("=")[1]) for f in data_store[f"{field}/sample=00000"])
        return np.fromiter(iterable, int)

    def get_data(self, data_store: zarr.Group, field: str, sample: int, level: int):
        data_sample: NDArray[np.int64] = data_store[
            f"{field}/sample={sample:05d}/ml={level:05d}"
        ] # type: ignore
        data = data_sample.data[0,0] # type: ignore
        datetime = pd.Timestamp(data_sample.datetime[0,0]) # type: ignore
        lats = data_sample.lat[0] # type: ignore
        lons = data_sample.lon[0] # type: ignore
        return data, datetime, lats, lons

    def get_level_idx(self, levels: NDArray[np.int64], level: int) -> int:
        return level


class Forecast(DataAccess):
    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        levels: NDArray[np.int64] = data_store[f"{field}/sample=00000"].ml[:]  # type: ignore (custom metadata)
        print(levels)
        return levels

    def get_data(self,data_store: zarr.Group, field: str, sample: int,level: int):
        # ignore custom metadata attributes of zarr groups
        data_sample = data_store[f"{field}/sample={sample:05d}"]
        data = data_sample.data[level, 0] # type: ignore
        datetime = pd.Timestamp(data_sample.datetime[0]) # type: ignore
        lats = data_sample.lat # type: ignore
        lons = data_sample.lon # type: ignore
        return data, datetime, lats, lons

    def get_level_idx(self, levels: np.ndarray, level: int) -> int:
        return np.where(levels == level)[0].tolist()[0] # multiple indexes per lvl ?



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

# calculate RMSE
def compute_RMSE(pred: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    return np.sqrt(np.mean((pred-target)**2))

def get_samples(data_store: zarr.Group, field: str, n_max: int) -> list[int]:
    nsamples = min(len(data_store[field]), n_max)
    return rnd.sample(range(len(data_store[field])), nsamples)