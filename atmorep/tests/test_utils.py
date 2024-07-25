import numpy as np

def era5_fname():
    return "/gpfs/scratch/ehpc03/data/{}/ml{}/era5_{}_y{}_m{}_ml{}.grib"

def atmorep_pred():
    return "./results/id{}/results_id{}_epoch{}_pred.zarr"

def atmorep_target():
    return "./results/id{}/results_id{}_epoch{}_target.zarr"

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
    values = {"temperature" : 1.8, 
              "velocity_u" : 0.005, #????
              "velocity_v": 0.005,  #????
              "velocity_z": 0.005,  #????
              "vorticity" : 0.2,    #????
              "divergence": 0.2,    #????
              "specific_humidity": 0.7,  #????
              "total_precip": 9999, #?????
            }
    
    return values[field]