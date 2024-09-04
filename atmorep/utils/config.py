from collections import namedtuple
import json
from pathlib import Path
import typing
from atmorep import config
import dataclasses
import datetime


TimeLatLon = namedtuple("TimeLatLon", ["time", "lat", "lon"])

@dataclasses.dataclass
class PatchConfig:
    pass

@dataclasses.dataclass
class FieldConfig:
    name: str
    normalization: str
    levels: list[int]
    tokensize: TimeLatLon[int]
    patchsize: TimeLatLon[int]
    
    @classmethod
    def from_list(cls, field: list[typing.Any]) -> typing.Self: # 3.11 feature
        print(field)
        return cls(
            field[0],
            field[-1],
            [int(lvl) for lvl in field[2]],
            TimeLatLon(*field[4]),
            TimeLatLon(*field[3])
        )
    

@dataclasses.dataclass
class Config:
    id: str
    fields: list[FieldConfig]
    dates: list[datetime.datetime]
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
            [datetime.datetime(*date) for date in config_dict["dates"]],
            int(config_dict["forecast_num_tokens"]),
            config_dict["BERT_strategy"],
            Path(config_dict["file_path"])
        )
            
    def as_json(self, config_path: Path):
        pass
    
    @property
    def field_names(self) -> list[str]:
        return [field.name for field in self.fields]
    
    @staticmethod
    def _get_fields(fields: list[list[typing.Any]]):
        return [FieldConfig.from_list(field) for field in fields]