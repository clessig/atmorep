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
import math
import itertools
import code
# code.interact(local=locals())

from atmorep.datasets.dynamic_field_level import DynamicFieldLevel
from atmorep.datasets.static_field import StaticField

from atmorep.utils.utils import days_until_month_in_year
from atmorep.utils.utils import days_in_month

import atmorep.config.config as config

class MultifieldDataSampler( torch.utils.data.IterableDataset):
    
  ###################################################
  def __init__( self, file_path, years_data, fields, batch_size, 
                num_t_samples, num_patches_per_t, num_load, pre_batch, 
                rng_seed = None, file_shape = (-1, 721, 1440),
                level_type = 'ml', time_sampling = 1, 
                smoothing = 0, file_format = 'grib', month = None, lat_sampling_weighted = True,
                geo_range = [[-90.,90.], [0.,360.]], 
                fields_targets = [], pre_batch_targets = None
              ) :
    '''
      Data set for single dynamic field at an arbitrary number of vertical levels
    '''
    super( MultifieldDataSampler).__init__()

    self.fields = fields
    self.batch_size = batch_size

    self.pre_batch = pre_batch

    self.years_data = years_data
    self.time_sampling = time_sampling
    self.month        = month
    self.range_lat    = 90. - np.array( geo_range[0])
    self.range_lon    = np.array( geo_range[1])
    self.geo_range    = geo_range

    # order North to South
    self.range_lat = np.flip(self.range_lat) if self.range_lat[1] < self.range_lat[0] \
                                             else self.range_lat

    # prepare range_lat and range_lon for sampling
    self.is_global = 0 == self.range_lat[0] and self.range_lon[0] == 0.  \
                          and 180. == self.range_lat[1] and 360. == self.range_lon[1]
    
    # TODO: this assumes file_shape is set correctly and not just per field and it defines a 
    # reference grid, likely has to be the coarsest
    self.res = 360. / file_shape[2]
    
    # avoid wrap around at poles
    pole_offset = np.ceil(fields[0][3][1] * fields[0][4][1] / 2) * self.res
    self.range_lat[0] = pole_offset if self.range_lat[0] < pole_offset else self.range_lat[0]
    self.range_lat[1] =180.-pole_offset if 180.-self.range_lat[1]<pole_offset else self.range_lat[1]

    self.lat_sampling_weighted = lat_sampling_weighted

    self.level_type = level_type
    self.smoothing = smoothing

    self.file_path    = config.path_data
    self.file_shape   = file_shape
    self.file_format  = file_format
    self.num_load = num_load
    self.num_patches_per_t = int(num_patches_per_t)
    self.num_t_samples = int(num_t_samples)

    self.fields_targets = fields_targets
    self.pre_batch_targets = pre_batch_targets

    # convert to mathematical latitude and ensure North -> South ordering
    # shrink so that cookie cutting based on sampling does not exceed domain if it is not global
    if not self.is_global :
      # TODO: check that field data is consistent and covers the same spatial domain 
      # TODO: code below assumes that fields[0] is global
      # TODO: code below does not handle anisotropic grids
      finfo = self.fields[0]
      # ensure that delta is a multiple of the coarse grid resolution
      ngrid1 = finfo[3][1] * finfo[4][1]
      ngrid2 = finfo[3][2] * finfo[4][2]
      delta1 = 0.5 * self.res * (ngrid1-1 if ngrid1 % 2==0 else ngrid1+1)
      delta2 = 0.5 * self.res * (ngrid2-1 if ngrid2 % 2==0 else ngrid2+1)
      self.range_lat += np.array([delta1, -delta1])
      self.range_lon += np.array([delta2, -delta2])

    # ensure all data loaders use same rng_seed and hence generate consistent data
    if not rng_seed :
      rng_seed = np.random.randint( 0, 100000, 1)[0]
    self.rng = np.random.default_rng( rng_seed)

    # create (source) fields
    self.datasets = self.create_loaders( fields)

    # create (target) fields 
    self.datasets_targets = self.create_loaders( fields_targets)

  ###################################################
  def create_loaders( self, fields ) :

    datasets = []
    for field_idx, field_info in enumerate(fields) :

      datasets.append( [])

      # extract field info
      (vls, num_tokens, token_size) = field_info[2:5]

      if len(field_info) > 6 :
        corr_type = field_info[6]
      else:
        corr_type = 'global'

      smoothing = self.smoothing
      log_transform_data = False
      if len(field_info) > 7 :
        (data_type, file_shape, file_geo_range, file_format) = field_info[7][:4]
        if len( field_info[7]) > 6 :
          smoothing = field_info[7][6]
          print( '{} : smoothing = {}'.format( field_info[0], smoothing) )
        if len( field_info[7]) > 7 :
          log_transform_data = field_info[7][7]
          print( '{} : log_transform_data = {}'.format( field_info[0], log_transform_data) )
      else :
        data_type = 'era5'
        file_format = self.file_format
        file_shape = self.file_shape
        file_geo_range = [[90.,-90.], [0.,360.]]

      # static fields
      if 0 == field_info[1][0] :
        datasets[-1].append( StaticField( self.file_path, field_info, self.batch_size, data_type,
                                          file_shape, file_geo_range,
                                          num_tokens, token_size, smoothing, file_format, corr_type) )
                                         
      # dynamic fields
      elif 1 == field_info[1][0] :
        for vlevel in vls :
          datasets[-1].append( DynamicFieldLevel( self.file_path, self.years_data, field_info,
                                                  self.batch_size, data_type,
                                                  file_shape, file_geo_range,
                                                  num_tokens, token_size,
                                                  self.level_type, vlevel, self.time_sampling, 
                                                  smoothing, file_format, corr_type, 
                                                  log_transform_data ) )
      
      else :
          assert False

    return datasets 

  ###################################################
  def shuffle( self) :

    # ensure that different parallel loaders create independent random shuffles
    delta = torch.randint( 0, 100000, (1,)).item()
    self.rng.bit_generator.advance( delta)

    self.idxs_perm = np.zeros( (0, 4), dtype=np.int64)

    # latitude, first map to mathematical lat coords in [0,180.], then to [0,pi] then
    # to z-value in [-1,1]
    if self.lat_sampling_weighted :
      lat_r = np.cos( self.range_lat/180. * np.pi)
    else :
      lat_r = self.range_lat

    # 1.00001 is a fudge factor since np.round(*.5) leads to flooring instead of proper up-rounding
    res_inv = 1.0 / self.res * 1.00001

    # loop over individual data year-month items 
    for i_ym in range( len(self.years_months)) :
    
      ym = self.years_months[i_ym]
      
      # ensure a constant size of work load of data loader independent of the month length 
      # factor of 128 is a fudge parameter to ensure that mod-ing leads to sufficiently 
      # random wrap-around (with 1 instead of 128 there is clustering on the first days)
      hours_in_day = int( 24 / self.time_sampling)
      time_slices = 128 * 31 * hours_in_day
      time_slices_i_ym = hours_in_day * days_in_month( ym[0], ym[1])
      idxs_perm_temp = np.mod(self.rng.permutation(time_slices), time_slices_i_ym)
      # fixed number of time samples independent of length of month
      idxs_perm_temp = idxs_perm_temp[:self.num_t_samples]
      idxs_perm = np.zeros( (self.num_patches_per_t *idxs_perm_temp.shape[0],4) )

      # split up into file index and local index
      idx = 0
      for it in idxs_perm_temp :
        
        idx_patches = self.rng.random( (self.num_patches_per_t, 2) )
        # for jj in idx_patches :
        for jj in idx_patches :
          # area consistent sampling on the sphere (with less patches close to the pole)
          # see https://graphics.stanford.edu/courses/cs448-97-fall/notes.html , Lecture 7
          # for area preserving sampling of the sphere
          # py \in [0,180], px \in [0,360] (possibly with negative values for lon)
          if self.lat_sampling_weighted :
            py = ((np.arccos(lat_r[0] + (lat_r[1]-lat_r[0]) * jj[0]) / np.pi) * 180.)
          else :
            py = (lat_r[0] + (lat_r[1]-lat_r[0]) * jj[0])
          px = jj[1] * (self.range_lon[1] - self.range_lon[0]) + self.range_lon[0]

          # align with grid
          py = self.res * np.round( py * res_inv)
          px = self.res * np.round( px * res_inv)

          idxs_perm[idx] = np.array( [i_ym, it, py, px])
          idx = idx + 1

      self.idxs_perm = np.concatenate( (self.idxs_perm, idxs_perm[:idx]))

    # shuffle again to avoid clustering of patches by loop over idx_patches above
    self.idxs_perm = self.idxs_perm[self.rng.permutation(self.idxs_perm.shape[0])]
    self.idxs_perm = self.idxs_perm[self.rng.permutation(self.idxs_perm.shape[0])]
    # restrict to multiples of batch size
    lenbatch = int(math.floor(self.idxs_perm.shape[0] / self.batch_size)) * self.batch_size
    self.idxs_perm = self.idxs_perm[:lenbatch]
    # # DEBUG
    # print( 'self.idxs_perm.shape = {}'.format(self.idxs_perm.shape ))
    # rank = torch.distributed.get_rank()
    # fname = 'idxs_perm_rank{}_{}.dat'.format( rank, shape_to_str( self.idxs_perm.shape))
    # self.idxs_perm.tofile( fname)

  ###################################################
  def set_full_time_range( self) :

    self.idxs_perm = np.zeros( (0, 4), dtype=np.int64)

    # latitude, first map to mathematical lat coords in [0,180.], then to [0,pi] then
    # to z-value in [-1,1]
    if self.lat_sampling_weighted :
      lat_r = np.cos( self.range_lat/180. * np.pi)
    else :
      lat_r = self.range_lat

    # 1.00001 is a fudge factor since np.round(*.5) leads to flooring instead of proper up-rounding
    res_inv = 1.0 / self.res * 1.00001

    # loop over individual data year-month items 
    for i_ym in range( len(self.years_months)) :

      ym = self.years_months[i_ym]

      hours_in_day = int( 24 / self.time_sampling)
      idxs_perm_temp = np.arange( hours_in_day * days_in_month( ym[0], ym[1]))
      idxs_perm = np.zeros( (self.num_patches_per_t *idxs_perm_temp.shape[0],4) )

      # split up into file index and local index
      idx = 0
      for it in idxs_perm_temp :

        idx_patches = self.rng.random( (self.num_patches_per_t, 2) )
        for jj in idx_patches :
          # area consistent sampling on the sphere (with less patches close to the pole)
          # see https://graphics.stanford.edu/courses/cs448-97-fall/notes.html , Lecture 7
          # for area preserving sampling of the sphere
          # py \in [0,180], px \in [0,360] (possibly with negative values for lon)
          if self.lat_sampling_weighted :
            py = ((np.arccos(lat_r[0] + (lat_r[1]-lat_r[0]) * jj[0]) / np.pi) * 180.)
          else :
            py = (lat_r[0] + (lat_r[1]-lat_r[0]) * jj[0])
          px = jj[1] * (self.range_lon[1] - self.range_lon[0]) + self.range_lon[0]

          # align with grid
          py = self.res * np.round( py * res_inv)
          px = self.res * np.round( px * res_inv)

          idxs_perm[idx] = np.array( [i_ym, it, py, px])
          idx = idx + 1

      self.idxs_perm = np.concatenate( (self.idxs_perm, idxs_perm[:idx]))

    # shuffle again to avoid clustering of patches by loop over idx_patches above
    self.idxs_perm = self.idxs_perm[self.rng.permutation(self.idxs_perm.shape[0])]
    # restrict to multiples of batch size
    lenbatch = int(math.floor(self.idxs_perm.shape[0] / self.batch_size)) * self.batch_size
    self.idxs_perm = self.idxs_perm[:lenbatch]

    # # DEBUG
    # print( 'self.idxs_perm.shape = {}'.format(self.idxs_perm.shape ))
    # fname = 'idxs_perm_{}_{}.dat'.format( self.epoch_counter, shape_to_str( self.idxs_perm.shape))
    # self.idxs_perm.tofile( fname)

  ###################################################
  def load_data( self, batch_size = None) :

    years_data = self.years_data
    
    # ensure proper separation of different random samplers
    delta = torch.randint( 0, 1000, (1,)).item()
    self.rng.bit_generator.advance( delta)

    # select num_load random months and years 
    perms = np.concatenate( [self.rng.permutation( np.arange(len(years_data))) for i in range(64)])
    perms = perms[:self.num_load]
    if self.month : 
      self.years_months = [ (years_data[iyear], self.month) for iyear in perms]
    else : 
      # stratified sampling of month to ensure proper distribution, needs to be adapted for 
      # number of parallel workers not being divisible by 4
      # rank, ms = torch.distributed.get_rank() % 4, 3
      # perms_m = np.concatenate( [self.rng.permutation( np.arange( rank*ms+1, (rank+1)*ms+1))
                                                                              # for i in range(16)])
      perms_m = np.concatenate( [self.rng.permutation( np.arange( 1, 12+1)) for i in range(16)])
      self.years_months = [ ( years_data[iyear], perms_m[i]) for i,iyear in enumerate(perms)]

    # generate random permutations passed to the loaders for individual files 
    # to ensure consistent processing
    self.shuffle()

    # perform actual loading of data
 
    for ds_field in self.datasets :
      for ds in ds_field :
        ds.load_data( self.years_months, self.idxs_perm, batch_size)

    for ds_field in self.datasets_targets :
      for ds in ds_field :
        ds.load_data( self.years_months, self.idxs_perm, batch_size)

  ###################################################
  def set_data( self, times_pos, batch_size = None) :
    '''
      times_pos = np.array( [ [year, month, day, hour, lat, lon], ...]  )
        - lat \in [90,-90] = [90N, 90S]
        - lon \in [0,360]
        - (year,month) pairs should be a limited number since all data for these is loaded
    '''

    # extract required years and months
    years_months_all = np.array( [ [it[0], it[1]] for it in times_pos ], dtype=np.int64)
    self.years_months = list( zip( np.unique(years_months_all[:,0]),
                                   np.unique( years_months_all[:,1] )))

    # generate all the data
    self.idxs_perm = np.zeros( (len(times_pos), 4))
    for idx, item in enumerate( times_pos) :

      assert item[2] >= 1 and item[2] <= 31
      assert item[3] >= 0 and item[3] < int(24 / self.time_sampling)
      assert item[4] >= -90. and item[4] <= 90.

      # find year
      for i_ym, ym in enumerate( self.years_months) :
        if ym[0] == item[0] and ym[1] == item[1] :
          break

      # last term: correct for window from last file that is loaded
      it = (item[2] - 1) * (24./self.time_sampling) + item[3]
      # it = item[2] * (24./self.time_sampling) + item[3]
      idx_lat = item[4]
      idx_lon = item[5]

      # work with mathematical lat coordinates from here on
      self.idxs_perm[idx] = np.array( [i_ym, it, 90. - idx_lat, idx_lon])

    for ds_field in self.datasets :
      for ds in ds_field :
        ds.load_data( self.years_months, self.idxs_perm, batch_size)

    for ds_field in self.datasets_targets :
      for ds in ds_field :
        ds.load_data( self.years_months, self.idxs_perm, batch_size)

  ###################################################
  def set_global( self, times, batch_size = None, token_overlap = [0, 0]) :
    ''' generate patch/token positions for global grid '''

    token_overlap = torch.tensor( token_overlap).to(torch.int64)

    # assumed that sanity checking that field data is consistent has been done 
    ifield = 0
    field = self.fields[ifield]

    res = self.res
    side_len = torch.tensor( [field[3][1] * field[4][1] * res, field[3][2] * field[4][2] * res] )
    overlap = torch.tensor( [token_overlap[0]*field[4][1]*res, token_overlap[1]*field[4][2]*res] )
    side_len_2 = side_len / 2.
    assert all( overlap <= side_len_2), 'token_overlap too large for #tokens, reduce if possible'

    # generate tiles
    times_pos = []
    for ctime in times :

      lat = side_len_2[0].item()
      num_tiles_lat = 0
      while (lat + side_len_2[0].item()) < 180. :
        num_tiles_lat += 1
        lon = side_len_2[1].item() - overlap[1].item()/2.
        num_tiles_lon = 0
        while (lon - side_len_2[1]) < 360. :
          times_pos += [[*ctime, -lat + 90., np.mod(lon,360.) ]]
          lon += side_len[1].item() - overlap[1].item()
          num_tiles_lon += 1
        lat += side_len[0].item() - overlap[0].item()

      # add one additional row if no perfect tiling (sphere is toric in longitude so no special
      # handling necessary but not in latitude)
      # the added row is such that it goes exaclty down to the South pole and the offset North-wards
      # is computed based on this
      lat -= side_len[0] - overlap[0]
      if lat - side_len_2[0] < 180. :
        num_tiles_lat += 1
        lat = 180. - side_len_2[0].item() + res
        lon = side_len_2[1].item() - overlap[1].item()/2.
        while (lon - side_len_2[1]) < 360. :
          times_pos += [[*ctime, -lat + 90., np.mod(lon,360.) ]]
          lon += side_len[1].item() - overlap[1].item()

    # adjust batch size if necessary so that the evaluations split up across batches of equal size
    batch_size = num_tiles_lon

    print( 'Number of batches per global forecast: {}'.format( num_tiles_lat) )

    self.set_data( times_pos, batch_size)

  ###################################################
  def set_location( self, pos, years, months, num_t_samples_per_month, batch_size = None) :
    ''' random time sampling for fixed location '''

    times_pos = []
    for i_ym, ym in enumerate(itertools.product( years, months )) :

      # ensure a constant size of work load of data loader independent of the month length 
      # factor of 128 is a fudge parameter to ensure that mod-ing leads to sufficiently 
      # random wrap-around (with 1 instead of 128 there is clustering on the first days)
      hours_in_day = int( 24 / self.time_sampling)
      d_i_m = days_in_month( ym[0], ym[1]) 
      perms = self.rng.permutation( num_t_samples_per_month * d_i_m)
      # ensure that days start at 1
      perms = np.mod( perms[ : num_t_samples_per_month], (d_i_m-1) ) + 1
      rhs = self.rng.integers(low=0, high=hours_in_day, size=num_t_samples_per_month )

      for rh, perm in zip( rhs, perms) :
        times_pos += [[ ym[0], ym[1], perm, rh, pos[0], pos[1]] ]

    # adjust batch size if necessary so that the evaluations split up across batches of equal size
    while 0 != (len(times_pos) % batch_size) :
      batch_size -= 1
    assert batch_size >= 1

    self.set_data( times_pos, batch_size)

  ###################################################
  def __iter__(self):

    iter_start, iter_end = self.worker_workset()

    for bidx in range( iter_start, iter_end) :

      sources = []
      for ds_field in self.datasets : 
        sources.append( [ds_level[bidx] for ds_level in ds_field])
      # perform batch pre-processing, e.g. BERT-type masking
      if self.pre_batch :
        sources = self.pre_batch( sources)

      targets = []
      for ds_field in self.datasets_targets :
        targets.append( [ds_level[bidx] for ds_level in ds_field])
      # perform batch pre-processing, e.g. BERT-type masking
      if self.pre_batch_targets :
        targets = self.pre_batch_targets( targets)

      yield (sources,targets)

  ###################################################
  def __len__(self):
      return len(self.datasets[0][0])

  ###################################################
  def worker_workset( self) :

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None: 
      iter_start = 0
      iter_end = len(self.datasets[0][0])

    else:  
      # split workload
      temp = len(self.datasets[0][0])
      per_worker = int( np.floor( temp / float(worker_info.num_workers) ) )
      worker_id = worker_info.id
      iter_start = int(worker_id * per_worker)
      iter_end = int(iter_start + per_worker)
      if worker_info.id+1 == worker_info.num_workers :
        iter_end = int(temp)

    return iter_start, iter_end
 
