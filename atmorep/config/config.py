import json
from pathlib import Path
import dataclasses
import typing

PLATFORM_CONFIG_PATH = Path(__file__).resolve().parent

# doesnt work for shared environments
ATMOREP_PROJECT_DIR = PLATFORM_CONFIG_PATH.parent.parent

@dataclasses.dataclass
class UserConfig:
    models: Path
    results: Path
    output: Path
    
    @classmethod
    def from_path(cls, output_dir: Path) -> typing.Self:
        return cls(
            output_dir / "models",
            output_dir / "results",
            output_dir / "output",
        )

def get_known_platforms() -> list[str]:
    return [config_file.stem for config_file in PLATFORM_CONFIG_PATH.iterdir()]

@dataclasses.dataclass
class HPC_Platform:
    input_data: Path
    pretained_models: Path
    
    @classmethod
    def get_platform(cls, platform: str) -> typing.Self:
        known_platforms = get_known_platforms()
        platform_config_file = PLATFORM_CONFIG_PATH / f"{platform}.json"
        
        try:
            with open(platform_config_file, "r") as fp:
                platform_config = json.load(fp)
        except FileNotFoundError as e:
            msg = f"computing platform: {platform_config_file} is unknown, should be one of : {', '.join(known_platforms)}"
            raise ValueError(msg)
        
        platform = cls(**platform_config)
        return platform

    
# compatibiltiy facade

_platform = "jsc"
# platform = "bsc"
# platform = "atos"

_my_platform = HPC_Platform.get_platform(_platform)

path_data = _my_platform.input_data / 'era5_y1979_2021_res025_chunk8.zarr/'
path_models = _my_platform.pretained_models