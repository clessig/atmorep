import strenum

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
    OutputType.prediction: r"./results/id{}/results_id{}_epoch{:05d}_pred.zarr",
    OutputType.target: r"./results/id{}/results_id{}_epoch{:05d}_target.zarr"
}