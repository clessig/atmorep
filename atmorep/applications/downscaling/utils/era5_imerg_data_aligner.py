import zarr
import glob
import numpy as np
import pandas as pd

import xarray as xr


def get_lat_lon_range(zarr_group,lat_keyname, lon_keyname):
    lats = zarr_group[lat_keyname][:]
    lons = zarr_group[lon_keyname][:]
    return (min(lats),max(lats)),(min(lons),max(lons))

def time_range(timeseries_frame,strf_format):
    timeseries_frame_strformatted = timeseries_frame['time'].dt.strftime(strf_format)

    return timeseries_frame_strformatted.iloc[[0,-1]].values

def find_time_indices_range_in_era5(era5_group,timerange,strf_format):
    time_series = pd.DataFrame(
                        {'time' : pd.to_datetime(era5_group['time'][:])})['time'].dt.strftime(strf_format)
    
    intial_index = time_series[time_series==timerange[0]].index[0]
    last_index = time_series[time_series==timerange[1]].index[0]

    return (int(intial_index),int(last_index+1))

def lat_lon_indices_range_in_era5(era5_group,lat_range,lon_range):
    lats = era5_group['lats'][:]
    lons = era5_group['lons'][:]
    lons = np.where(lons>180, lons-360,lons)
    
    lat_min_index = np.where(lats>=lat_range[0])[0][0]
    lat_max_index = np.where(lats<=lat_range[1])[0][-1]

    if lon_range[0] < 0.0:
        lon_min_index = np.where((lons>=lon_range[0]) & (lons < 0))[0][0]
    else:
        lon_min_index = np.wehre((lons>=lon_range[0]) & (lons > 0))[0][0]

    if lon_range[1] < 0.0:
        lon_max_index = np.where((lons<=lon_range[1]) & (lons < 0))[0][-1]
    else:
        lon_max_index = np.where((lons<=lon_range[1]) * (lons > 0))[0][-1]

    return ((int(lat_min_index),int(lat_max_index)),
             (int(lon_min_index),int(lon_max_index)))


def save_precipitaion_nc_era5(era5_group,lat_range,lon_range,time_range):
    
    lon_list = None

    if lon_range > 1440 - 108 and lon_range < 1440:
        lon_list = np.arange(lon_range,1440)
        lon_list = np.concatenate((lon_list,np.arange(0,(lon_range+108)%1440)))

    if lon_range > 1440:
        lon_range = lon_range%1440
    
    if lon_list is None:
        era5_array = era5_group['data_sfc'][
            time_range:time_range+3,
            0,
            lat_range:lat_range+54,
            lon_range:lon_range+108]*1e3

        era_5_subset_dataset = xr.Dataset(
                            {'total_precip' : (
                                ("time","lats","lons"),era5_array)},
                            coords={
                                "time": era5_group['time'][time_range: time_range+3],
                                "lats" : era5_group['lats'][lat_range: lat_range+54],
                                "lons" : era5_group['lons'][lon_range: lon_range+108]
                            },)


    else:
        era5_array = era5_group['data_sfc'][
                    time_range:time_range+3,
                    0,
                    lat_range:lat_range+54,
                    lon_list]*1e3

        era_5_subset_dataset = xr.Dataset(
                        {'total_precip' : (
                            ("time","lats","lons"),era5_array)},
                        coords={
                            "time": era5_group['time'][time_range: time_range+3],
                            "lats" : era5_group['lats'][lat_range: lat_range+54],
                            "lons" : era5_group['lons'][lon_list]
                        },)

    file_name = f"../nc_files_new/era_5_{time_range-era5_timeframe_range[0]}_{lat_range}_{lon_range}.nc"
    print(file_name)
    era_5_subset_dataset.to_netcdf(file_name)

def save_precipitaion_nc_imerg(imerg_group,lat_range,lon_range,time_range,era5_group):
    imerg_lats = imerg_group['lats'][:]
    imerg_lons = imerg_group['lons'][:]

    #lat_min_index = int(np.where(imerg_lats<=(era5_group['lats'])[lat_range])[0][-1])
    #lat_max_index = int(np.where(imerg_lats>=(era5_group['lats'])[lat_range+54])[0][0])

    #lon_min_index = int(np.where(imerg_lons<=era5_group['lons'][lon_range])[0][-1])
    #lon_max_index = int(np.where(imerg_lons>=era5_group['lons'][lon_range+108])[0][0])

    time_range = int(time_range)    

    #print(type(time_range))
    #print(type(lat_min_index),type(lat_max_index))
    #print(type(lon_min_index),type(lon_max_index))
    
    imerg_array = imerg_group['data_sfc'][
                time_range:time_range+3,
                0,
                lat_range*3:(lat_range+54)*3,
                lon_range*3:(lon_range+108)*3]

               # lat_min_index:lat_max_index,
               # lon_min_index:lon_max_index]

    imerg_subset_dataset = xr.Dataset(
             {'total_precip': (
                ("time","lats","lons"), imerg_array)},
            
             coords = {
                "time" : imerg_group['time'][time_range:time_range+3],
                "lats" : imerg_lats[lat_range*3:(lat_range+54)*3],
                "lons" : imerg_lons[lon_range*3:(lon_range+108)*3]
             })

    file_name = f"../nc_files_new/imerg_{time_range}_{lat_range}_{lon_range}.nc"
    print(file_name)
    imerg_subset_dataset.to_netcdf(file_name)

def save_era5_whole(era5_group,time_range):
    
    era5_whole = era5_group['data_sfc'][
        time_range:time_range+3,
        0,
        :,
        :]*1e3

    era_5_subset_dataset = xr.Dataset(
                    {'total_precip' : (
                        ("time","lats","lons"),era5_whole)},
                    coords={
                        "time": era5_group['time'][time_range: time_range+3],
                        "lats" : era5_group['lats'][:],
                        "lons" : era5_group['lons'][:]
                    },)

    file_name = f"../nc_files_new/era_5_{time_range-era5_timeframe_range[0]}_whole.nc"
    print(file_name)
    era_5_subset_dataset.to_netcdf(file_name)

def lat_lon_time_range(global_zarr_filepath, local_zarr_filepath):

    global_group = zarr.group(global_zarr_filepath)
    local_group = zarr.group(local_zarr_filepath)

    local_lat_range, local_lon_range = get_lat_lon_range(local_group,'lats','lons')
    
    local_timeseries = local_group['time'][:]

    local_df = pd.DataFrame(
                    {'time' : pd.to_datetime(local_timeseries)}).sort_values('time')
    
    timeframe_range_local = time_range(local_df,"%Y-%m-%d-%H-%M-%S")

    global_timeframe_range = find_time_indices_range_in_era5(
        global_group,
        timeframe_range_local,
        "%Y-%m-%d-%H-%M-%S")

    assert((global_timeframe_range[1]-global_timeframe_range[0]) == len(local_timeseries))

    lat_indices_range, lon_indices_range = lat_lon_indices_range_in_era5(
                    global_group,
                    local_lat_range,
                    local_lon_range)

    return {'lats' : lat_indices_range,
            'lons' : lon_indices_range,
            'time' : global_timeframe_range}

if __name__ == "__main__":

    era5_zarr_file =  "/p/scratch/atmo-rep/data/era5_1deg/months/era5_y1979_2021_res025_chunk8.zarr"
    imerg_zarr_file = "/p/scratch/atmo-rep/data/imerg/imerg_regridded/imerg_regrid_y2003_2021_res083_chunk8.zarr"
    old_imerg_zarr_file =  "/p/scratch/atmo-rep/data/imerg/imerg_regridded/imerg_regrid_y2003_2021_res008_chunk8.zarr"
    
    era5_group = zarr.group(era5_zarr_file)
    imerg_group = zarr.group(imerg_zarr_file)
    old_imerg_group = zarr.group(old_imerg_zarr_file)

    old_imerg_lat_range, old_imerg_lon_range = get_lat_lon_range(old_imerg_group,'lats','lons')

    print("old_imerg_lat",old_imerg_lat_range)
    print("old_imerg_lon",old_imerg_lon_range)
    
    imerg_lat_range, imerg_lon_range = get_lat_lon_range(imerg_group,'lats','lons')

    print("imerg_lat",imerg_lat_range)
    print("imerg_lon",imerg_lon_range)

    era5_lat_range, era5_lon_range = get_lat_lon_range(era5_group,'lats','lons')
    
    print("era5_lat",era5_lat_range)
    print("era5_lon",era5_lon_range)

    imerg_timeseries = imerg_group['time'][:]

    imerg_df = pd.DataFrame(
                    {'time' : pd.to_datetime(imerg_timeseries)}).sort_values('time')
    
    timeframe_range_imerg = time_range(imerg_df,"%Y-%m-%d-%H-%M-%S")

    print(timeframe_range_imerg)

    era5_timeframe_range = find_time_indices_range_in_era5(era5_group,timeframe_range_imerg,"%Y-%m-%d-%H-%M-%S")

    assert((era5_timeframe_range[1]-era5_timeframe_range[0]) == len(imerg_timeseries))

    print('assert_passed')

    lat_indices_range, lon_indices_range = lat_lon_indices_range_in_era5(era5_group,imerg_lat_range,imerg_lon_range)

    print("lat_indices_range",lat_indices_range)
    print("lon_indices_range",lon_indices_range)

    print(era5_group['lats'][[lat_indices_range[0],lat_indices_range[1]]])
    print(era5_group['lons'][[lon_indices_range[0],lon_indices_range[1]]])

    time_list = np.random.choice(np.arange(0,len(imerg_timeseries-3)), size=(20,), replace=False)
    era5_time_list = np.arange(era5_timeframe_range[0],era5_timeframe_range[1])[time_list] 

    print(time_list)
    print(era5_time_list-era5_timeframe_range[0])
    print(era5_time_list)
    print(era5_group['time'][era5_time_list])

    era5_lats = np.arange(lat_indices_range[0],lat_indices_range[1]-54)
    
    if lon_indices_range[1] > lon_indices_range[0] :
        era5_lons = np.arange(lon_indices_range[0],lon_indices_range[1]-108)
    else:
        era5_lons = np.arange(lon_indices_range[0], lon_indices_range[1] + len(era5_group['lons'][:])-108)

    lats = np.random.choice( np.arange(0, lat_indices_range[1]-54-lat_indices_range[0]), size=(20,), replace=False)
    
    if lon_indices_range[1] < lon_indices_range[0]:
        lons = np.random.choice( np.arange(0, lon_indices_range[1]+1440-108-lon_indices_range[0]), size=(20,),replace=False)
    else:
        lons = np.random.choice( np.arange(0, lon_indices_range[1]-108-lon_indices_range[0]), size=(20,),replace=False)

    era5_lats_list = era5_lats[lats]
    era5_lons_list = era5_lons[lons]

    for i in range(20):
        save_precipitaion_nc_era5(era5_group,era5_lats_list[i],era5_lons_list[i],era5_time_list[i])
        save_precipitaion_nc_imerg(imerg_group,lats[i],lons[i],time_list[i],era5_group)
        save_era5_whole(era5_group,era5_time_list[i])
