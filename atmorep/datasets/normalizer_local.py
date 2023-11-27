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

    fname_base = '{}/normalization/{}/normalization_mean_var_{}_y{}_m{:02d}_{}{}.bin'

    self.corr_data = [ ]
    for year in range( config.year_base, config.year_last+1) :
      for month in range( 1, 12+1) :
        corr_fname = fname_base.format( str(config.path_data), field_info[0], field_info[0],
                                        year, month, level_type, vlevel)
        x = np.fromfile( corr_fname, dtype=np.float32).reshape( (file_shape[1], file_shape[2], 2))
        x = xr.DataArray( x, [ ('lat', np.linspace( 0., 180., num=180*4+1, endpoint=True)),
                                ('lon', np.linspace( 0., 360., num=360*4, endpoint=False)),
                               ('data', ['mean', 'var']) ])
        self.corr_data.append( x)

  def normalize( self, year, month, data, coords) :

    corr_data_ym = self.corr_data[ (year - config.year_base) * 12 + month ]
    mean = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='mean').values
    var = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='var').values

    if len(data.shape) > 2 :
      for i in range( data.shape[0]) :
        data[i] = (data[i] - mean) / var
    else :
      data = (data - mean) / var

    return data

  def denormalize( self, year, month, data, coords) :

    corr_data_ym = self.corr_data[ (year - config.year_base) * 12 + month ]
    mean = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='mean').values
    var = corr_data_ym.sel( lat=coords[0], lon=coords[1], data='var').values

    if len(data.shape) > 2 :
      for i in range( data.shape[0]) :
        data[i] = (data[i] * var) + mean
    else :
      data = (data * var) + mean

    return data

  