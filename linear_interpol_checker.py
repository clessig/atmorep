import zarr
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime
import copy
import os

from atmorep.utils_analysis.metrics import Scores, calc_scores_item
from atmorep.utils_analysis.read_atmorep_data import HandleAtmoRepData
from atmorep.datasets.data_writer import write_item
from atmorep.utils.utils import unique_unsorted

if __name__ == '__main__':

    #######################################################################################################################

    def extract_wandb_id(f_path, run_id):
        """Extract the wandb_id from a single file."""
        file_path_x = f_path.format(run_id)
        with open(file_path_x, "r") as file:
            for line in file:
                if "wandb_id" in line:
                    try:
                        # Split the line and handle the prefix like '0:'
                        parts = line.split(":")
                        if "wandb_id" in parts[-2].strip():  # Check if the second-to-last part is the key
                            return parts[-1].strip()  # Return the final part as the value
                    except (ValueError, SyntaxError):
                        pass
        return None

    def compute_RMSE(pred, target):
        return np.sqrt(np.mean((pred-target)**2))

    ########################################################################################################################

    run_ids = ["10721397", "10721398", "10721399", "10721400", "10721401"]
    file_path = "./logs/output_eval_{}.out"

    # wandb_id_list = ["xtf7l144"] # checker combined 2 date
    wandb_id_list = [extract_wandb_id(file_path, run_id) for run_id in run_ids]    

    ########################################################################################################################
    input_dir = "./results/"
    field = "temperature"
    levels = [96, 105, 114, 123, 137]

    epoch = 0
    rmse_dict = {
        "96": 0, 
        "105": 0, 
        "114": 0, 
        "123": 0, 
        "137": 0
    }
    ########################################################################################################################
    print("#################### checker ####################")
    for model_id in wandb_id_list:
        print(f"model_id = {model_id}")
        ar_data = HandleAtmoRepData(model_id, input_dir)
        da_source  = ar_data.read_data(field, "source", ml = 96) # Asma, five levels are loaded no matter the level specified here
        interpol_per_ml = []
        for idx_level, level in enumerate(levels): # to be re-adjusted
            print(f"level = {level}")
            da_target_dc  = ar_data.read_data(field, "target", ml = level)
            interpol_per_neighbourhood = []
            rmse_val = 0
            n_samples = 0
            #################################################################################################################
            datetime_all = [da_source[isample].datetime.values for isample in range(len(da_source))]
            datetime_all = np.unique(datetime_all)
            time_frame = len(datetime_all)

            ds_o = xr.Dataset( coords={'datetime': datetime_all, 
                                        'lat' : np.linspace( -90., 90., num=180*4+1, endpoint=True), 
                                        'lon' : np.linspace( 0., 360., num=360*4, endpoint=False) }
                                        )

            ds_o['before_interpol'] = (['datetime', 'lat', 'lon'], np.zeros( (time_frame, 721, 1440)))
            ds_o['interpol'] = (['datetime', 'lat', 'lon'], np.zeros( (time_frame, 721, 1440)))
            #################################################################################################################
            print("interpolation and rmse computation ..")
            rmse_interpol = 0
            n_tokens_tot = 0
            for isample in range(len(da_target_dc)):
                datetime_target = da_target_dc[isample].datetime.values
                lat_target = da_target_dc[isample].lat.values
                lon_target = da_target_dc[isample].lon.values
                # selecting the neighborhood
                interpol   = da_source[isample][idx_level].sel(
                    datetime = np.unique(datetime_target),
                    lat = np.unique(lat_target),
                    lon = unique_unsorted(lon_target)
                )
                # applying the mask
                for itoken in da_target_dc[isample].itoken.values:
                    datetime_target_t = da_target_dc[isample][itoken].datetime.values
                    lat_target_t = da_target_dc[isample][itoken].lat.values
                    lon_target_t = da_target_dc[isample][itoken].lon.values
                    interpol.loc[dict(datetime=datetime_target_t, lat=lat_target_t, lon=lon_target_t)] = np.nan
                    ds_o['before_interpol'].loc[ dict( datetime=datetime_target_t, 
                                    lat=lat_target_t, lon=lon_target_t) ] = copy.deepcopy(interpol.loc[dict(datetime=datetime_target_t, lat=lat_target_t, lon=lon_target_t)])
                to_be_interpol2 = interpol.sel(
                    datetime=np.unique(da_target_dc[isample].datetime.values),
                    lon=np.unique(da_target_dc[isample].lon.values)
                )
                got_interpol2 = to_be_interpol2.interpolate_na(dim="lat", method="linear")
                got_extrapol2 = got_interpol2.ffill(dim="lat")
                got_extrapol2 = got_extrapol2.bfill(dim="lat")
                ds_o['interpol'].loc[ dict( datetime=np.unique(da_target_dc[isample].datetime.values), 
                                            lon=np.unique(da_target_dc[isample].lon.values), lat=got_interpol2.lat)] = got_extrapol2
                for itoken in da_target_dc[isample].itoken.values:
                    datetime_target_t = da_target_dc[isample][itoken].datetime.values
                    lat_target_t = da_target_dc[isample][itoken].lat.values
                    lon_target_t = da_target_dc[isample][itoken].lon.values
                    interpol_tok = ds_o['interpol'].sel(lat = lat_target_t,lon = lon_target_t, datetime=datetime_target_t)
                    rmse_interpol+=compute_RMSE(interpol_tok, da_target_dc[isample][itoken].data)
                    n_tokens_tot+=1
            rmse_interpol/=n_tokens_tot
            print(f"rmse_interpol = {rmse_interpol.values}")
            rmse_dict[str(level)]+=rmse_interpol.values
    print("------------------------------------------------------------------------------------")
    for key in rmse_dict.keys():
        rmse_dict[key]/=len(wandb_id_list)
        print(f"level {key} rmse: {rmse_dict[key]}")