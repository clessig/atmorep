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

    fname_base = './data/{}/normalization/{}/normalization_mean_var_{}_y{}_m{:02d}_{}{}.bin'
    self.year_base = config.datasets[data_type]['extent'][0][0]
    self.year_last = config.datasets[data_type]['extent'][0][1]
    lat_min, lat_max = config.datasets[data_type]['extent'][1]
    lat_min, lat_max = 90. - lat_min, 90. - lat_max
    lat_min, lat_max = (lat_min, lat_max) if lat_min < lat_max else (lat_max, lat_min)
    lon_min, lon_max = config.datasets[data_type]['extent'][2]
    res = config.datasets[data_type]['resolution'][1]
    is_global = config.datasets[data_type]['is_global']

    self.corr_data = [ ]
    for year in range( self.year_base, self.year_last+1) :
      for month in range( 1, 12+1) :
        corr_fname = fname_base.format( data_type, field_info[0], field_info[0],
                                        year, month, level_type, vlevel)
        ns_lat = int( (lat_max-lat_min) / res + 1)
        ns_lon = int( (lon_max-lon_min) / res + (0 if is_global else 1) )
        # TODO: remove file_shape (ns_lat, ns_lon contains same information)
        x = np.fromfile( corr_fname, dtype=np.float32).reshape( (file_shape[1], file_shape[2], 2))
        # TODO, TODO, TODO: remove once recomputed
        if 'cerra' == data_type :
          x[:,:,0] = 340.
          x[:,:,1] = 600.
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
    #print(data.mean(), data.std())
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

  