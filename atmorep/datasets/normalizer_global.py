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

import numpy as np

import atmorep.config.config as config
import pdb

class NormalizerGlobal() :

  def __init__(self, field_info, vlevel, file_shape, data_type = 'era5', level_type = 'ml') :
   
    # TODO: use path from config and pathlib.Path()
    fname_base = '{}/{}/normalization/{}/global_normalization_mean_var_{}_{}{}.bin'
    
    fn = field_info[0]
    corr_fname = fname_base.format( str(config.path_data), data_type, fn, fn, level_type, vlevel)
    self.corr_data = np.fromfile(corr_fname, dtype=np.float32).reshape( (-1, 4))
 

  def normalize( self, year, month, data, coords = None) :
    #breakpoint()
    corr_data_ym = self.corr_data[ np.where(np.logical_and(self.corr_data[:,0] == float(year),
                                            self.corr_data[:,1] == float(month))) , 2:].flatten()
    #breakpoint()                                        
    data_temp = (data - corr_data_ym[0]) / corr_data_ym[1]
    # print(data_temp.mean(), data_temp.std())
    return (data - corr_data_ym[0]) / corr_data_ym[1]

  def denormalize( self, year, month, data, coords = None) :
    corr_data_ym = self.corr_data[ np.where(np.logical_and(self.corr_data[:,0] == float(year),
                                            self.corr_data[:,1] == float(month))) , 2:].flatten()
    data_temp = (data * corr_data_ym[1]) + corr_data_ym[0]
    #print("after denorm", data_temp.mean(), data_temp.std())                   
    return (data * corr_data_ym[1]) + corr_data_ym[0]
  