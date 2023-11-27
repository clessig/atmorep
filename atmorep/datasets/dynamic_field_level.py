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
import gc
import code
# code.interact(local=locals())

import atmorep.utils.utils as utils
from atmorep.utils.utils import shape_to_str
from atmorep.utils.utils import days_until_month_in_year
from atmorep.utils.utils import days_in_month

from atmorep.datasets.data_loader import DataLoader
from atmorep.datasets.normalizer_global import NormalizerGlobal
from atmorep.datasets.normalizer_local import NormalizerLocal

class DynamicFieldLevel() : 
    
  ###################################################
  def __init__( self, file_path, years_data, field_info,
                batch_size, data_type = 'era5',
                file_shape = [-1, 721, 1440], file_geo_range = [[-90.,90.], [0.,360.]],
                num_tokens = [3, 9, 9], token_size = [1, 9, 9], 
                level_type = 'pl', vl = 975, time_sampling = 1,
                smoothing = 0, file_format = 'grib', corr_type = 'local', 
                log_transform_data = False ) :
    '''
      Data set for single dynamic field at a single vertical level
    '''

    self.years_data = years_data
    self.field_info   = field_info
    self.file_path    = file_path
    self.file_shape   = file_shape
    self.file_format  = file_format
    self.level_type   = level_type
    self.vl = vl
    self.time_sampling = time_sampling
    self.smoothing    = smoothing
    self.corr_type    = corr_type
    self.log_transform_data = log_transform_data

    self.years_months = []

    # work internally with mathematical latitude coordinates in [0,180]
    self.file_geo_range = [ -np.array(file_geo_range[0]) + 90. , np.array(file_geo_range[1]) ]
    # enforce that georange is North to South
    self.geo_range_flipped = False
    if self.file_geo_range[0][0] > self.file_geo_range[0][1] : 
      self.file_geo_range[0] = np.flip( self.file_geo_range[0])
      self.geo_range_flipped = True
    self.is_global = 0. == self.file_geo_range[0][0] and 0. == self.file_geo_range[1][0] \
                      and 180. == self.file_geo_range[0][1] and 360. == self.file_geo_range[1][1]

    # resolution
    # TODO: non-uniform resolution in latitude and longitude
    self.res = (file_geo_range[1][1] - file_geo_range[1][0])
    self.res /= file_shape[2] if self.is_global else (file_shape[2]-1)
    
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

    self.data_field = None

    if self.corr_type == 'global' :
      self.normalizer = NormalizerGlobal( field_info, vl, self.file_shape, data_type)
    else :
      self.normalizer = NormalizerLocal( field_info, vl, self.file_shape, data_type)

    self.loader = DataLoader( self.file_path, self.file_shape, data_type,
                              file_format = self.file_format, level_type = self.level_type, 
                              smoothing = self.smoothing, log_transform=self.log_transform_data)

  ###################################################
  def load_data( self, years_months, idxs_perm, batch_size = None) :

    self.idxs_perm = idxs_perm.copy()

    # nothing to be loaded
    if  set(years_months) in set(self.years_months):
      return

    self.years_months = years_months

    if batch_size : 
      self.batch_size = batch_size
    loader = self.loader

    self.files_offset_days = []
    for year, month in self.years_months :
      self.files_offset_days.append( days_until_month_in_year( year, month) )

    # load data
    # self.data_field is a list of lists of torch tensors
    # [i] : year/month
    # [i][j] : field per year/month
    # [i][j] : len_data_per_month x num_tokens_lat x num_tokens_lon x token_size x token_size
    # this ensures coherence in the data access
    del self.data_field
    gc.collect()
    self.data_field = loader.get_single_field( self.years_months, self.field_info[0], 
                                               self.level_type, self.vl, [-1, -1], 
                                               [self.num_tokens[0] * self.tok_size[0], 0, 
                                               self.time_sampling])

    # apply normalization and log-transform for each year-month data
    for j in range( len(self.data_field) ) :

      if self.corr_type == 'local' :
        coords = [ np.linspace( 0., 180., num=180*4+1, endpoint=True), 
                   np.linspace( 0., 360., num=360*4, endpoint=False) ]
      else :
        coords = None

      (year, month) = self.years_months[j]
      self.data_field[j] = self.normalizer.normalize( year, month, self.data_field[j], coords)

      # basics statistics
      print( 'INFO:: data stats {} : {} / {}'.format( self.field_info[0], 
                                                      self.data_field[j].mean(), 
                                                      self.data_field[j].std()) )
    
  ###############################################
  def __getitem__( self, bidx) :

    tn = self.grid_delta
    num_tokens = self.num_tokens
    tok_size = self.tok_size
    tnt = self.num_tokens[0] * self.tok_size[0]
    cat = torch.cat
    geor = self.file_geo_range

    idx = bidx * self.batch_size

    # physical fields
    patch_s = [nt*ts for nt,ts in zip(self.num_tokens,self.tok_size)] 
    x = torch.zeros( self.batch_size, patch_s[0], patch_s[1], patch_s[2] )
    cids = torch.zeros( self.batch_size, num_tokens.prod(), 8)

    # offset from previous month to be able to sample all time slices in current one
    offset_t = int(num_tokens[0] * tok_size[0])
    # 721 etc have grid points at the beginning and end which leads to incorrect results in places
    file_shape = np.array(self.file_shape)
    file_shape = file_shape-1 if not self.is_global else np.array(self.file_shape)-np.array([0,1,0])

    # for all items in batch
    for jj in range( self.batch_size) :

      i_ym = int(self.idxs_perm[idx][0])
      # perform a deep copy to not overwrite cid for other fields
      cid = np.array( self.idxs_perm[idx][1:]).copy()
      cid_orig = cid.copy()

      # map to grid coordinates (first map to normalized [0,1] coords and then to grid coords)
      cid[2] = np.mod( cid[2], 360.) if self.is_global else cid[2]
      assert cid[1] >= geor[0][0] and cid[1] <= geor[0][1], 'invalid latitude for geo_range' 
      cid[1] = ( (cid[1] - geor[0][0]) / (geor[0][1] - geor[0][0]) ) * file_shape[1]
      cid[2] = ( ((cid[2]) - geor[1][0]) / (geor[1][1] - geor[1][0]) ) * file_shape[2]
      assert cid[1] >= 0 and cid[1] < self.file_shape[1]
      assert cid[2] >= 0 and cid[2] < self.file_shape[2]

      # alignment when parent field has different resolution than this field
      cid = np.round( cid).astype( np.int64)

      ran_t = list( range( cid[0]-tnt+1 + offset_t, cid[0]+1 + offset_t))
      if any(np.array(ran_t) >= self.data_field[i_ym].shape[0]) :
        print( '{} : {} :: {}'.format( self.field_info[0], self.years_months[i_ym], ran_t ))

      # periodic boundary conditions around equator
      ran_lon = np.array( list( range( cid[2]-tn[1][0], cid[2]+tn[1][1])))
      if self.is_global :
        ran_lon = np.mod( ran_lon, self.file_shape[2])
      else :
        # sanity check for indices for files with local window
        # this should be controlled by georange_sampling for sampling
        assert all( ran_lon >= 0) and all( ran_lon < self.file_shape[2])

      ran_lat = np.array( list( range( cid[1]-tn[0][0], cid[1]+tn[0][1])))
      assert all( ran_lat >= 0) and all( ran_lat < self.file_shape[1])
      
      # current data
      # if self.geo_range_flipped : 
      #   print( '{} : {} / {}'.format( self.field_info[0], ran_lat, ran_lon) )
      if np.max(ran_t) >= self.data_field[i_ym].shape[0] :
        print( 'WARNING: {} : {} :: {}'.format( self.field_info[0], ran_t, self.years_months[i_ym]) )
      x[jj] = np.take( np.take( self.data_field[i_ym][ran_t], ran_lat, 1), ran_lon, 2)

      # set per token information
      assert self.time_sampling == 1
      ran_tt = np.flip( np.arange( cid[0], cid[0]-tnt, -tok_size[0]))
      years = self.years_months[i_ym][0] * np.ones( ran_tt.shape)
      days_in_year = self.files_offset_days[i_ym] + (ran_tt / 24.)
      # wrap year around
      mask = days_in_year < 0
      years[ mask ] -= 1
      days_in_year[ mask ] += 365
      hours = np.mod( ran_tt, 24)
      lats = ran_lat[int(tok_size[1]/2)::tok_size[1]] * self.res + self.file_geo_range[0][0]
      lons = ran_lon[int(tok_size[2]/2)::tok_size[2]] * self.res + self.file_geo_range[1][0]
      stencil = torch.tensor(list(itertools.product(lats,lons)))
      tstencil = torch.tensor( [ [y, d, h, self.vl] for y,d,h in zip( years, days_in_year, hours)],
                                  dtype=torch.float)
      txlist = list( itertools.product( tstencil, stencil))
      cids[jj,:,:6] = torch.cat( [torch.cat(tx).unsqueeze(0) for tx in txlist], 0)
      cids[jj,:,6] = self.vl
      cids[jj,:,7] = self.res

      idx += 1

    return (x, cids) 

  ###################################################
  def __len__(self):
    return int(self.idxs_perm.shape[0] / self.batch_size)
