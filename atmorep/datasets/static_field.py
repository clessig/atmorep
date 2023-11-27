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
import pathlib
import numpy as np
import math
import os, sys
import time
import itertools

import atmorep.utils.utils as utils
from atmorep.utils.utils import shape_to_str
from atmorep.utils.utils import days_until_month_in_year
from atmorep.utils.utils import days_in_month
from atmorep.utils.utils import tokenize

from atmorep.datasets.data_loader import DataLoader


class StaticField() : 
    
  ###################################################
  def __init__( self, file_path, field_info, batch_size, data_type = 'reanalysis',
                file_shape = (-1, 720, 1440), file_geo_range = [[90.,-90.], [0.,360.]],
                num_tokens = [3, 9, 9], token_size = [1, 9, 9], 
                smoothing = 0, file_format = 'grib', corr_type = 'global') :
    '''
      Data set for single dynamic field at a single vertical level
    '''

    self.field_info  = field_info
    self.file_path   = file_path
    self.file_shape  = file_shape
    self.file_format = file_format
    self.smoothing   = smoothing
    self.corr_type   = corr_type

    # # work internally with mathematical latitude coordinates in [0,180]
    # self.is_global = np.abs(file_geo_range[0][0])==90. and file_geo_range[1][0]==0. \
    #                   and np.abs(file_geo_range[0][0])==90. and file_geo_range[1][1]==360.
    # self.file_geo_range = [ -np.array(file_geo_range[0]) + 90. , file_geo_range[1] ]
    # self.file_geo_range[0] = np.flip( self.file_geo_range[0]) \
    #           if self.file_geo_range[0][0] > self.file_geo_range[0][1] else self.file_geo_range[0]

    # work internally with mathematical latitude coordinates in [0,180]
    self.file_geo_range = [ -np.array(file_geo_range[0]) + 90. , np.array(file_geo_range[1]) ]
    # enforce that georange is North to South
    self.geo_range_flipped = False
    if self.file_geo_range[0][0] > self.file_geo_range[0][1] : 
      self.file_geo_range[0] = np.flip( self.file_geo_range[0])
      self.geo_range_flipped = True
      print( 'Flipped georange')
    print( '{} :: geo_range : {}'.format( field_info[0], self.file_geo_range) )
    self.is_global = 0. == self.file_geo_range[0][0] and 0. == self.file_geo_range[1][0] \
                      and 180. == self.file_geo_range[0][1] and 360. == self.file_geo_range[1][1]
    print( '{} :: is_global : {}'.format( field_info[0], self.is_global) )

    self.batch_size = batch_size
    self.num_tokens = torch.tensor( num_tokens, dtype=torch.int)
    rem1 = (num_tokens[1]*token_size[1]) % 2
    rem2 = (num_tokens[2]*token_size[2]) % 2
    t1 = num_tokens[1]*token_size[1]
    t2 = num_tokens[2]*token_size[2]
    self.grid_delta = [ [int((t1+rem1)/2), int(t1/2)], [int((t2+rem2)/2), int(t2/2)] ]
    assert( num_tokens[1] < file_shape[1])
    assert( num_tokens[2] < file_shape[2])
    self.tok_size = token_size
    #assert( file_shape[1] % token_size[1] == 0)
    #assert( file_shape[2] % token_size[2] == 0)

    # resolution
    # TODO: non-uniform resolution in latitude and longitude
    self.res = (file_geo_range[1][1] - file_geo_range[1][0])
    self.res /= file_shape[2] if self.is_global else (file_shape[2]-1)

    self.data_field = None

    self.loader = DataLoader( self.file_path, self.file_shape, data_type,
                              file_format = self.file_format, 
                              smoothing = self.smoothing )

  ###################################################
  def load_data( self, years_months, idxs_perm, batch_size = None) :

    self.idxs_perm = idxs_perm
    loader = self.loader
    
    if batch_size : 
      self.batch_size = batch_size

    # load data
    self.data_field = loader.get_static_field( self.field_info[0], [-1, -1])
    
    # # corrections:
    self.correction_field = loader.get_correction_static_field( self.field_info[0], self.corr_type )
   
    mean = self.correction_field[0]
    std = self.correction_field[1]
  
    self.data_field = (self.data_field - mean) / std
  
    if self.geo_range_flipped :
      self.data_field = torch.flip( self.data_field, [0])

    # # basics statistics
    # print( 'INFO:: data stats {} : {} / {}'.format( self.field_info[0], 
    #                                                 self.data_field.mean(), 
    #                                                 self.data_field.std()) )

  ###################################################
  def set_data( self, date_pos ) :
    '''
      date_pos = np.array( [ [year, month, day, hour, lat, lon], ...]  )
        - lat \in [-90,90] = [90N, 90S]
        - (year,month) pairs should be a limited number since all data for these is loaded
    '''

    # extract required years and months
    years_months_all = np.array( [ [it[0], it[1]] for it in date_pos ], dtype=np.int64)
    self.years_months = list( zip( np.unique(years_months_all[:,0]), 
                                   np.unique( years_months_all[:,1] )))

    # load data and corrections
    self.load_data()

    # generate all the data
    self.idxs_perm = np.zeros( (date_pos.shape[0], 4), dtype=np.int64)
    for idx, item in enumerate( date_pos) :

      assert item[2] >= 1 and item[2] <= 31
      assert item[3] >= 0 and item[3] < int(24 / self.time_sampling)
      assert item[4] >= -90. and item[4] <= 90.

      # find year 
      for i_ym, ym in enumerate( self.years_months) :
        if ym[0] == item[0] and ym[1] == item[1] :
          break

      it = (item[2] - 1.) * 24. + item[3] + self.tok_size[0]
      idx_lat = int( (item[4] + 90.) * 720. / 180.)
      idx_lon = int( (item[5] % 360) * 1440. / 360.)

      self.idxs_perm[idx] = np.array( [i_ym, it, idx_lat, idx_lon], dtype=np.int64)

  ###############################################
  def __getitem__( self, bidx) :

    tn = self.grid_delta
    num_tokens = self.num_tokens
    tok_size = self.tok_size
    geor = self.file_geo_range

    idx = bidx * self.batch_size

    # physical fields
    patch_s = [nt*ts for nt,ts in zip(self.num_tokens,self.tok_size)] 
    x = torch.zeros( self.batch_size, 1, patch_s[1], patch_s[2] )
    cids = torch.zeros( self.batch_size, num_tokens.prod(), 8)

    # 721 etc have grid points at the beginning and end which leads to incorrect results in places
    file_shape = np.array(self.file_shape)
    file_shape = file_shape-1 if not self.is_global else np.array(self.file_shape)-np.array([0,1,0])

    # for all items in batch
    for jj in range( self.batch_size) :

      # perform a deep copy to not overwrite cid for other fields
      cid = np.array( self.idxs_perm[idx][1:]).copy()

      # map to grid coordinates (first map to normalized [0,1] coords and then to grid coords)
      cid[2] = np.mod( cid[2], 360.) if self.is_global else cid[2]
      assert cid[1] >= geor[0][0] and cid[1] <= geor[0][1], 'invalid latitude for geo_range' 
      cid[1] = ( (cid[1] - geor[0][0]) / (geor[0][1] - geor[0][0]) ) * file_shape[1]
      cid[2] = ( ((cid[2]) - geor[1][0]) / (geor[1][1] - geor[1][0]) ) * file_shape[2]
      assert cid[1] >= 0 and cid[1] < self.file_shape[1]
      assert cid[2] >= 0 and cid[2] < self.file_shape[2]

      # alignment when parent field has different resolution than this field
      cid = np.round( cid).astype( np.int64)

      # periodic boundary conditions around equator
      ran_lon = np.array( list( range( cid[2]-tn[1][0], cid[2]+tn[1][1])))
      if self.is_global :
        ran_lon = np.mod( ran_lon, self.file_shape[2])
      else :
        # sanity check for indices for files with local window
        # this should be controlled by georange_sampling for sampling
        assert any( ran_lon >= 0) or any( ran_lon < self.file_shape[2])

      ran_lat = np.array( list( range( cid[1]-tn[0][0], cid[1]+tn[0][1])))
      assert any( ran_lat >= 0) or any( ran_lat < self.file_shape[1])

      # current data
      x[jj,0] = np.take( np.take( self.data_field, ran_lat, 0), ran_lon, 1)

      # set per token information
      lats = ran_lat[int(tok_size[1]/2)::tok_size[1]] * self.res + self.file_geo_range[0][0]
      lons = ran_lon[int(tok_size[2]/2)::tok_size[2]] * self.res + self.file_geo_range[1][0]
      stencil = torch.tensor(list(itertools.product(lats,lons)))
      cids[jj,:,4:6] = stencil
      cids[jj,:,7] = self.res

      idx += 1

    return (x, cids)

  ###################################################
  def __len__(self):
    return int(self.idxs_perm.shape[0] / self.batch_size)
