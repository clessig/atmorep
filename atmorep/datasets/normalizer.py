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

from atmorep.utils.logger import logger

######################################################
#                   Normalize                        #
######################################################

def normalize( data, norm, dates, year_base = 1979) :
  corr_data = np.array([norm[12*(dt.year-year_base) + dt.month-1] for dt in dates])
  mean, var = corr_data[:, 0], corr_data[:, 1]
  if (var == 0.).all() :
    logger.info('Warning: var == 0')
    assert False
  if len(norm.shape) > 2 : #global norm
    return normalize_local(data, mean, var)
  else:
    return normalize_global( data, mean, var)
  
######################################################
def normalize_local( data, mean, var) :
  data = (data - mean) / var
  return data

######################################################
def normalize_global( data, mean, var) :
  for i in range( data.shape[0]) :
    data[i] = (data[i] - mean[i]) / var[i]
  return data


######################################################
#                  Denormalize                       #
######################################################
def denormalize(data, norm, dates, year_base = 1979) :
  corr_data = np.array([norm[12*(dt.year-year_base) + dt.month-1] for dt in dates])
  mean, var = corr_data[:, 0], corr_data[:, 1]
  if len(norm.shape) > 2 :
    return denormalize_local(data, mean, var)
  else:
    return denormalize_global(data, mean, var)  

######################################################

def denormalize_local(data, mean, var) :
  if len(data.shape) > 3: #ensemble
    for i in range( data.shape[0]) :
      data[i] = (data[i] * var) + mean
  else:
      data = (data * var) + mean
  return data

######################################################

def denormalize_global(data, mean, var) :
  if len(data.shape) > 3: #ensemble
    data = data.swapaxes(0,1)
    for i in range( data.shape[0]) :
      data[i] = ((data[i] * var[i]) + mean[i])
    data = data.swapaxes(0,1)
  else:
    for i in range( data.shape[0]) :
      data[i] = (data[i] * var[i]) + mean[i]
    
  return data