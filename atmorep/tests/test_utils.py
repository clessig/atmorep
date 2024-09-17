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
from pathlib import Path
from atmorep.utils.config import Config, FieldConfig
import atmorep.tests.constants as constants


GroupData = namedtuple("GroupData", ["data", "datetimes", "lats", "lons"])

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
    def field_name(self) -> str:
        return list(self.fields.keys())[0]
    
    @property
    def field(self) -> FieldConfig:
        return self.fields[self.field_name]
    
    def get_outpath(self, output: constants.OutputType):
        return constants.OUTPUT_PATH_TEMPLATE[output].format(
            self.model_id,
            self.model_id,
            str(self.epoch).zfill(5)
        )
    
class DataAccess(abc.ABC):
    def __init__(self, config: ValidationConfig):
        self.config = config
    
    @classmethod
    def from_config(cls, config: ValidationConfig):
        match config.strategy:
            case "BERT":
                return BERT(config)
            case "temporal_interpolation":
                return BERT(config)
            case _:
                return Forecast(config)

    def get_times(self, sample: zarr.Group) -> NDArray[np.datetime64]:
        levels = self.get_levels(sample)
        return np.array(self.get_data(sample, levels[0]).datetimes)

    @abc.abstractmethod
    def get_levels(self, sample: zarr.Group) -> NDArray[np.int64]:
        pass

    @abc.abstractmethod
    def get_data(self, sample: zarr.Group, level: int) -> GroupData:
        pass
    
class DataStore:
    def __init__(self, data_access: DataAccess, data_path: Path, field: str,):
        self.data_access = data_access
        
        store = zarr.ZipStore(data_path)
        try:
            self.zarr_samples: zarr.Group = zarr.group(store)[field] # type: ignore
        except:
            groups = list(zarr.group(store).group_keys())
            msg = f"Zarr Store is missing expected group for field {field} in groups: {groups}."
            raise ValueError(msg)
        
        
    @classmethod
    def from_config(cls, config, type: constants.OutputType) -> typing.Self:
        template = constants.OUTPUT_PATH_TEMPLATE[type]
        data_path = Path(
            template.format(config.model_id, config.model_id, config.epoch)
        )
        data_access = DataAccess.from_config(config)
        return cls(data_access, data_path, config.field_name)
    
    @property
    def levels(self) -> NDArray[np.int64]:
        example_sample = self.get_sample(0)
        return self.data_access.get_levels(example_sample)

    
    @property
    def max_lead_time(self) -> int:
        example_sample = self.get_sample(0)
        return len(self.data_access.get_times(example_sample))
    
    @property
    def times(self) -> NDArray[np.datetime64]:
        datetimes_iter = (
            self.data_access.get_times(sample) for _, sample
            in self.zarr_samples.groups()
        )
        datetimes = np.fromiter(
            datetimes_iter,
            dtype=np.dtype(("datetime64[h]", self.max_lead_time)), # allows correct shape
            count=len(self.zarr_samples) # preallocates memory
        )
        
        return np.unique(datetimes)

    def get_data(self, idx: int, level: int) -> GroupData:
        sample = self.get_sample(idx)
        return self.data_access.get_data(sample, level)
    
    def get_sample(self, idx: int) -> zarr.Group:
        try:
            sample: zarr.Group = self.zarr_samples[f"sample={idx:05d}"] # type: ignore
            return sample
        except:
            msg = f"No sample with index {idx:05d} in zarr store."
            raise ValueError(msg)
        
    def get_samples(self, n_max: int) -> list[int]:
        n_samples_avail = len(self.zarr_samples)
        n_samples_drawn = min(n_samples_avail, n_max)
        return rnd.sample(range(n_samples_avail), n_samples_drawn)
    
    def samples_and_levels(
        self, resample_lvls=False, n_samples_max=50
    ) -> list[tuple[int, int]]:
        if resample_lvls:
            iteration_items = [
                (sample, lvl)
                for sample in self.get_samples(n_samples_max)
                for lvl in self.levels
            ]
        else:
            samples = self.get_samples(n_samples_max)
            iteration_items = list(it.product(samples, self.levels))
        
        return iteration_items

class BERT(DataAccess):
    def get_levels(self, sample: zarr.Group) -> NDArray[np.int64]:
        iterable = (int(f.split("=")[1]) for f in sample)
        return np.fromiter(iterable, int)
    
    def get_data(self, sample: zarr.Group, level: int) -> GroupData:
        # level => lvl by key
        data_sample = sample[f"ml={level:05d}"]
        data = data_sample.data[0,0]
        datetime = pd.Timestamp(data_sample.datetime[0,0]) # TODO check
        lats = data_sample.lat[0]
        lons = data_sample.lon[0]
        return GroupData(data, datetime, lats, lons)

class Forecast(DataAccess):
    def get_levels(self, sample: zarr.Group) -> NDArray[np.int64]:
        levels: NDArray[np.int64] = sample.ml[:]  # type: ignore (custom metadata)
        return levels

    def get_data(
        self, sample: zarr.Group, level: int
    ) -> GroupData:
        levels = self.get_levels(sample)
        level_idx = np.where(levels == level)[0][0]

        data = sample.data[level_idx, 0]
        datetimes = np.array(sample.datetime, dtype="datetime64[h]")
        lats = sample.lat
        lons = sample.lon
        return GroupData(data, datetimes, lats, lons)


def compute_RMSE(
    pred: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    return np.sqrt(np.mean((pred-target)**2))

def get_config():
    return ValidationConfig.get()

def set_config(config: ValidationConfig):
    return ValidationConfig.set(config)

def get_samples_and_levels():
    return DataStore.from_config(
        get_config(), constants.OutputType.prediction
    ).samples_and_levels()