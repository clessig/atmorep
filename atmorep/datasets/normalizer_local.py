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

import code
import numpy as np
import xarray as xr

import atmorep.config.config as config

class NormalizerLocal() :

  def __init__(self, field_info, vlevel, file_shape, data_type = 'era5', level_type = 'ml') :

    if 'cosmo_rea6' == data_type :
      path = ''
      fname_base = './data/{}/normalization/{}/normalization_mean_var_{}_y{}_m{:02d}_{}{}.bin'
      self.year_base = 1995
      self.year_last = 2017
      lat_min, lat_max = 90. - 70.25, 90. - 27.5
      lon_min, lon_max = -12.5, 37.0
      is_global = False
      res = 0.25 / 4
    else :
      path = str(config.path_data)
      fname_base = './data/{}/normalization/{}/normalization_mean_var_{}_y{}_m{:02d}_{}{}.bin'
      self.year_base = config.year_base
      self.year_last = config.year_last
      lat_min, lat_max = 0., 180.
      lon_min, lon_max = 0., 360.
      res = 0.25
      is_global = True

    self.corr_data = [ ]
    for year in range( self.year_base, self.year_last+1) :
      for month in range( 1, 12+1) :
        corr_fname = fname_base.format( data_type, field_info[0], field_info[0],
                                        year, month, level_type, vlevel)
        ns_lat = int( (lat_max-lat_min) / res + 1)  # (1 if is_global else 0) )
        ns_lon = int( (lon_max-lon_min) / res + (0 if is_global else 1) )
        x = np.fromfile( corr_fname, dtype=np.float32).reshape( (file_shape[1], file_shape[2], 2))
        x = xr.DataArray( x, [ ('lat', np.linspace( lat_min, lat_max, num=ns_lat, endpoint=True)),
                                ('lon', np.linspace( lon_min, lon_max, num=ns_lon, endpoint=False)),
                               ('data', ['mean', 'var']) ])
        self.corr_data.append( x)

  def normalize( self, year, month, data, coords) :

    corr_data_ym = self.corr_data[ (year - self.year_base) * 12 + (month-1) ]
    mean = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='mean').values
    var = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='var').values
    if (var == 0.).all() :
      print( f'var == 0 :: ym : {year} / {month}')
      assert False

    if len(data.shape) > 2 :
      for i in range( data.shape[0]) :
        data[i] = (data[i] - mean) / var
    else :
      data = (data - mean) / var

    return data

  def denormalize( self, year, month, data, coords) :

    corr_data_ym = self.corr_data[ (year - self.year_base) * 12 + (month-1) ]
    mean = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='mean').values
    var = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='var').values

    if len(data.shape) > 2 :
      for i in range( data.shape[0]) :
        data[i] = (data[i] * var) + mean
    else :
      data = (data * var) + mean

    return data

  