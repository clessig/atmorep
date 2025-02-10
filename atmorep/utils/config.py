from collections.abc import Sequence
import json
from numbers import Number
from pathlib import Path
import typing
import dataclasses
import datetime as dt
import numpy as np


@dataclasses.dataclass
class TimeLatLon:
    time: int
    lat: int
    lon: int
    
    def __mul__(self, other: typing.Any):
        other = self._as_time_lat_lon(other)
            
        return TimeLatLon(
            self.time * other.time,
            self.lat * other.lat,
            self.lon * other.lon
        )
    
    @classmethod
    def _as_time_lat_lon(cls, other: typing.Any):
        match other:
            case TimeLatLon():
                return other
            case int() | float():
                return TimeLatLon(int(other), int(other), int(other))
            case (int(), int(), int()):
                return TimeLatLon(*other)
            case _:            
                msg = f"{other} of type {type(other)} cannot be interpreted as TimeLatLon."
                raise ValueError(msg)

@dataclasses.dataclass
class FieldConfig:
    name: str
    normalization: str
    levels: list[int]
    token_size: TimeLatLon
    patch_size: TimeLatLon
    
    @classmethod
    def from_list(cls, field: list[typing.Any]) -> typing.Self: # 3.11 feature
        return cls(
            field[0],
            field[-1],
            [int(lvl) for lvl in field[2]],
            TimeLatLon(*field[4]),
            TimeLatLon(*field[3])
        )
    
    @property
    def patch_length(self) -> TimeLatLon:
        return self.patch_size * self.token_size
    
    def get_max_lead_time(self, n_forecast_tokens: int) -> int:
        return self.token_size.time * n_forecast_tokens
    
@dataclasses.dataclass
class Config:
    id: str
    fields: dict[str, FieldConfig]
    dates: list[dt.datetime]
    forecast_num_tokens: int
    masking_strategy: str
    data_source: Path

    @classmethod
    def from_json(cls, config_path: Path) -> typing.Self: # 3.11 feature
        with open(config_path, "r") as config_file:
            config_dict = json.load(config_file)
            
        return cls(
            config_dict["wandb_id"],
            cls._get_fields(config_dict["fields"]),
            [dt.datetime(*date) for date in config_dict["dates"]],
            int(config_dict["forecast_num_tokens"]),
            config_dict["BERT_strategy"],
            Path(config_dict["file_path"])
        )
            
    def as_json(self, config_path: Path):
        pass # TODO

    @property
    def is_global(self) -> bool:
        return self.masking_strategy in ["global_forecast"]
    
    @property
    def max_lead_time(self) -> int:
        return max(
            field.get_max_lead_time(self.forecast_num_tokens)
            for name, field in self.fields.items()
        )
    
    @property
    def timesteps(self):
        timesteps = []
        
        for date in self.dates:
            for lead_time in range(self.max_lead_time):
                timesteps.append(date - dt.timedelta(hours=lead_time))
        
        return np.array(timesteps, dtype="datetime64[h]")
                
    
    @staticmethod
    def _get_fields(fields: list[list[typing.Any]]) -> dict[str, FieldConfig]:
        field_list = [FieldConfig.from_list(field) for field in fields]
        return {field.name: field for field in field_list}
    