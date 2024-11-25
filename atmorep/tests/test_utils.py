import numpy as np
import pandas as pd 

def era5_fname():
    return "/gpfs/scratch/ehpc03/data/{}/ml{}/era5_{}_y{}_m{}_ml{}.grib"

def atmorep_pred():
    return "./results/id{}/results_id{}_epoch{}_pred.zarr"

def atmorep_target():
    return "./results/id{}/results_id{}_epoch{}_target.zarr"

def atmorep_source():
    return "./results/id{}/results_id{}_epoch{}_source.zarr"

def atmorep_json():
    return "./results/id{}/model_id{}.json"

def grib_index(field):
    grib_idxs = {"velocity_u": "u",
                 "temperature": "t", 
                 "total_precip": "tp", 
                 "velocity_v": "v", 
                 "velocity_z": "z", 
                 "vorticity" : "vo", 
                 "divergence" : "d", 
                 "specific_humidity": "q"}

    return grib_idxs[field]

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

def get_datetime(atmorep, field, sample, BERT=None, level=None):
    if BERT:
        atmorep_sample = atmorep[f"{field}/sample={sample:05d}/ml={level:05d}"] 
        return np.unique(atmorep_sample.datetime[:])
    else:
        atmorep_sample = atmorep[f"{field}/sample={sample:05d}"] 
        return atmorep_sample.datetime[:]

######################################

def check_lats(lats_pred, lats_target):
    assert (lats_pred[:] == lats_target[:]).all(), "Mismatch between latitudes"
    assert (lats_pred[:] <= 90.).all(), f"latitudes are between {np.amin(lats_pred)}- {np.amax(lats_pred)}"
    assert (lats_pred[:] >= -90.).all(), f"latitudes are between {np.amin(lats_pred)}- {np.amax(lats_pred)}"

def check_lons(lons_pred, lons_target):
    assert (lons_pred[:] == lons_target[:]).all(), "Mismatch between longitudes"
    assert (lons_pred[:] >= 0.).all(), "longitudes are between {np.amin(lons_pred)}- {np.amax(lons_pred)}"
    assert (lons_pred[:] <= 360.).all(), "longitudes are between {np.amin(lons_pred)}- {np.amax(lons_pred)}"

def check_datetimes(datetimes_pred, datetimes_target):
    assert (datetimes_pred == datetimes_target), "Mismatch between datetimes"

######################################

#calculate RMSE
def compute_RMSE(pred, target):
    return np.sqrt(np.mean((pred-target)**2))


def get_max_RMSE(field):
    #TODO: optimize thresholds
    values = {"temperature" : 3, 
              "velocity_u" : 0.2, #????
              "velocity_v": 0.2,  #????
              "velocity_z": 0.2,  #????
              "vorticity" : 0.2,    #????
              "divergence": 0.2,    #????
              "specific_humidity": 0.2,  #????
              "total_precip": 1, #?????
            }
    
    return values[field]