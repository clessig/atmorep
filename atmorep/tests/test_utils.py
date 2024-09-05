import abc
from collections import namedtuple
import typing
from numpy.typing import NDArray
from typing import Any
import numpy as np
import pandas as pd 
import zarr
import random as rnd
import itertools as it
import strenum
from collections.abc import Iterable
from pathlib import Path
import json
from atmorep.utils.config import Config


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

class OutputType(strenum.StrEnum):
    prediction = "prediction"
    target = "target"

ERA5_PATH_PREFIX_BSC = r"/gpfs/scratch/ehpc03/data/"
ERA5_PATH_PREFIX_JSC = r"/p/data1/slmet/met_data/ecmwf/era5_reduced_level/ml_levels/"

ERA5_FILE_TEMPLATE = ERA5_PATH_PREFIX_JSC + r"{}/ml{}/era5_{}_y{}_m{}_ml{}.grib"

OUTPUT_PATH_TEMPLATE = {
    OutputType.prediction: r"./results/id{}/results_id{}_epoch{}_pred.zarr",
    OutputType.target: r"./results/id{}/results_id{}_epoch{}_target.zarr"
}

##################################################################

GroupData = namedtuple("GroupData", ["data", "datetime", "lats", "lons"])

class DataAccess(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        pass

    @abc.abstractmethod
    def get_data(
        self, data_store: zarr.Group, field: str, sample: int, level
    ) -> GroupData:
        pass

    @abc.abstractmethod
    def get_level_idx(self, levels: NDArray[np.int64], level: int) -> int:
        pass

class BERT(DataAccess):

    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        iterable = (int(f.split("=")[1]) for f in data_store[f"{field}/sample=00000"])
        return np.fromiter(iterable, int)

    def get_data(
        self, data_store: zarr.Group, field: str, sample: int, level: int
    ) -> GroupData:
        data_sample: NDArray[np.int64] = data_store[
            f"{field}/sample={sample:05d}/ml={level:05d}"
        ] # type: ignore
        data = data_sample.data[0,0] # type: ignore
        datetime = pd.Timestamp(data_sample.datetime[0,0]) # type: ignore
        lats = data_sample.lat[0] # type: ignore
        lons = data_sample.lon[0] # type: ignore
        return GroupData(data, datetime, lats, lons)

    def get_level_idx(self, levels: NDArray[np.int64], level: int) -> int:
        return level

class Forecast(DataAccess):
    def get_levels(self, data_store: zarr.Group, field: str) -> NDArray[np.int64]:
        levels: NDArray[np.int64] = data_store[f"{field}/sample=00000"].ml[:]  # type: ignore (custom metadata)
        return levels

    def get_data(
        self,data_store: zarr.Group, field: str, sample: int, level: int
    ) -> GroupData:
        # ignore custom metadata attributes of zarr groups
        data_sample = data_store[f"{field}/sample={sample:05d}"]
        data = data_sample.data[level, 0] # type: ignore
        datetime = pd.Timestamp(data_sample.datetime[0]) # type: ignore
        lats = data_sample.lat # type: ignore
        lons = data_sample.lon # type: ignore
        return GroupData(data, datetime, lats, lons)

    def get_level_idx(self, levels: np.ndarray, level: int) -> int:
        return np.where(levels == level)[0].tolist()[0] # multiple indexes per lvl ?


class ValidationConfig(Config):
    _instance = None
    
    @classmethod
    def from_result(cls, result_dir: Path) -> typing.Self:
        _model_id = result_dir.stem.removeprefix("id")
        
        run_config_path = result_dir / f"model_id{_model_id}.json"
        
        return cls.from_json(run_config_path)
    
    
    @classmethod
    def get(cls) -> "ValidationConfig":
        if cls._instance is None:
            raise RuntimeError("try to get uninitialized Validation Config.")
        return cls._instance
    
    @classmethod
    def set(cls, instance: typing.Self):
        cls._instance = instance
        
    @property
    def model_id(self):
        return self.id
    
    @property
    def strategy(self):
        return self.masking_strategy
    
    @property
    def epoch(self):
        return 0
        
    @property
    def field(self) -> str:
        return self.field_names[0]
    
    @property
    def data_access(self) -> DataAccess:
        match self.strategy:
            case "BERT":
                return BERT()
            case "temporal_interpolation":
                return BERT()
            case _:
                return Forecast()
        
    def get_zarr(self, output_type: OutputType) -> zarr.Group:
        store_path_template = OUTPUT_PATH_TEMPLATE[output_type]
        store = zarr.ZipStore(
            store_path_template.format(self.model_id, self.model_id, str(self.epoch).zfill(5))
        )
        return zarr.group(store)

    def get_levels(
        self, output_type: OutputType = OutputType.target
    ) -> NDArray[np.int64]:
        data_store = self.get_zarr(output_type)
        return self.data_access.get_levels(data_store, self.field)
      
    def get_samples(self, n_max: int) -> list[int]:
        data_store = self.get_zarr(OutputType.target)
        nsamples = min(len(data_store[self.field]), n_max)
        return rnd.sample(range(len(data_store[self.field])), nsamples)
    
    
    def get_timestamps_from_data(self):
        data_store = self.get_zarr(OutputType.prediction)
        n_samples = len(data_store[self.field])
        datetimes = np.empty(
            (
                n_samples,
                self.fields[0].token_size.time*self.forecast_num_tokens
            ),
            dtype="datetime64[h]"
        )
        for sample_key, value in data_store[self.field].groups():
            i = int(sample_key.split("=")[1])
            datetimes[i] = value.datetime
        
        return np.unique(datetimes)
    
    def samples_and_levels(
        self, resample_lvls=False, n_samples_max=50
    ) -> Iterable[tuple[int, int]]:
        levels = self.get_levels()
        if resample_lvls:
            iteration_tuples = (
                (sample, lvl)
                for sample in self.get_samples(n_samples_max)
                for lvl in levels
            )
        else:
            samples = self.get_samples(n_samples_max)
            iteration_items = list(it.product(samples, levels))
        
        return iteration_items


# calculate RMSE
def compute_RMSE(
    pred: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.sqrt(np.mean((pred-target)**2))