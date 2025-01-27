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
import atmorep.config.config as config
from atmorep.utils.utils import log_transform, invert_log_transform

######################################################
#                   Normalize                        #
######################################################

class Normalizer(object):

  def __init__(self, norm, type, eps = 1e-6, year_base = 1979, log_tr = False):
    self.type = type
    self.year_base = year_base
    self.log_tr = log_tr
    self.norm = norm
    self.eps = eps

  def get_norm(self, lat_ran, lon_ran):
    if self.type != 'global':  
      if lat_ran[0] < lat_ran[-1] and lon_ran[0] < lon_ran[-1]:
        lat_max, lat_min = max(lat_ran), min(lat_ran)
        lon_max, lon_min = max(lon_ran), min(lon_ran)
        # print(lat_min, lat_max, lon_min, lon_max)
        return self.norm[:,:,lat_min:lat_max+1,lon_min:lon_max+1]
      else:
        return self.norm[ ... , lat_ran[:,np.newaxis], lon_ran[np.newaxis,:]]
    else:
      return self.norm

  def normalize(self, data, dates, lat_idxs, lon_idxs) : 
    # breakpoint()
    norm_sel = self.get_norm(lat_idxs, lon_idxs)
    corr_data = np.array([norm_sel[12*(dt.year-self.year_base) + dt.month-1] for dt in dates])
    mean, var = corr_data[:, 0], corr_data[:, 1]
    if (var == 0.).all() :
      print( f'Warning: var == 0') 
      assert False

    if self.log_tr:
      data = log_transform(data, eps = self.eps)
  
    if self.type == 'global': 
      return self.normalize_global(data, mean, var)
    else:
      return self.normalize_local( data, mean, var)
  
  ######################################################
  def normalize_local(self, data, mean, var) :
    # breakpoint()
    data = (data - mean) / var
    return data

  ######################################################
  def normalize_global(self, data, mean, var) :
    for i in range( data.shape[0]) :
      data[i] = (data[i] - mean[i]) / var[i]
    return data


  ######################################################
  #                  Denormalize                       #
  ######################################################
  def denormalize(self, data, dates, lat_idxs, lon_idxs): 
    norm_sel = self.get_norm(lat_idxs, lon_idxs)
    corr_data = np.array([norm_sel[12*(dt.year-self.year_base) + dt.month-1] for dt in dates])
    mean, var = corr_data[:, 0], corr_data[:, 1]
    if self.type == 'global': #len(norm.shape) > 2 :
      data = self.denormalize_global(data, mean, var)
    else:
      data = self.denormalize_local(data, mean, var)  
  
    if self.log_tr:
      data = invert_log_transform(data, eps = self.eps)
  
    return data

  ######################################################

  def denormalize_local(self, data, mean, var) :
    if len(data.shape) > 3: #ensemble
      for i in range( data.shape[0]) :
        data[i] = (data[i] * var) + mean
    else:
        data = (data * var) + mean
    return data

  ######################################################

  def denormalize_global(self, data, mean, var) :
    if len(data.shape) > 3: #ensemble
      data = data.swapaxes(0,1)
      for i in range( data.shape[0]) :
        data[i] = ((data[i] * var[i]) + mean[i])
      data = data.swapaxes(0,1)
    else:
      for i in range( data.shape[0]) :
        data[i] = (data[i] * var[i]) + mean[i]
    
    return data