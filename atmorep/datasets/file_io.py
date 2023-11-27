####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description :
#
#  license     :
#
####################################################################################################

import torch
import numpy as np
import xarray as xr

####################################################################################################

def netcdf_file_loader(fname, field, time_padding = [0,0,1], days_in_month = 0, static=False) :
  
  ds = xr.open_dataset(fname, engine='netcdf4')[field]

  if not static:
    # TODO: test that only time_padding[0] *or* time_padding[1] != 0
    if time_padding[0] != 0 :
      ds = ds[-time_padding[0]*time_padding[2] : ]
    elif time_padding[1] != 0 :
      ds = ds[ : time_padding[1]*time_padding[2]]
    if time_padding[2] > 1 :
      ds = ds[::time_padding[2]]

  x = torch.from_numpy(np.array( ds, dtype=np.float32))
  ds.close()
  
  return x

####################################################################################################

def grib_file_loader(fname, field, time_padding = [0,0,1], days_in_month = 0, static=False) :
  
  ds = xr.open_dataset(fname, engine='cfgrib',
                              backend_kwargs={'time_dims':('valid_time','indexing_time')})[field]

  # work-around for bug in download where for every month 31 days have been downloaded
  if days_in_month > 0 :
    ds = ds[:24*days_in_month]

  if not static:
    # TODO: test that only time_padding[0] *or* time_padding[1] != 0
    if time_padding[0] != 0 :
      ds = ds[-time_padding[0]*time_padding[2] : ]
    elif time_padding[1] != 0 :
      ds = ds[ : time_padding[1]*time_padding[2]]
    if time_padding[2] > 1 :
      ds = ds[::time_padding[2]]

  x = torch.from_numpy(np.array(ds, dtype=np.float32))
  ds.close()
  
  # assume grib files are clean and NaNs are introduced through handling of "missing values"
  if np.isnan( x).any() :
    x_shape = x.shape
    x = x.flatten()
    x[np.argwhere( np.isnan( x))] = 9999.
    x = np.reshape( x, x_shape)
  
  return x

####################################################################################################

def bin_file_loader( fname, field, time_padding = [0,0,1], static=False, file_shape = (-1, 721, 1440)) :

  ds = np.fromfile(fname,  dtype=np.float32) 
  print("INFO:: reshaping binary file into {}".format(file_shape))   
  if not static: 
    ds = np.reshape(ds, file_shape)
    ds = ds[time_padding[0]:(ds.shape[0] - time_padding[0])]
  else:
    ds = np.reshape(ds, file_shape[1:])
  x = torch.from_numpy(ds)
  return x
