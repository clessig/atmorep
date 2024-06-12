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
import zarr
import pandas as pd
from datetime import datetime
import time
import os

# from atmorep.datasets.normalizer_global import NormalizerGlobal
# from atmorep.datasets.normalizer_local import NormalizerLocal
from atmorep.datasets.normalizer import normalize
from atmorep.utils.utils import tokenize

class MultifieldDataSampler( torch.utils.data.IterableDataset):
    
  ###################################################
  def __init__( self, file_path, fields, years, batch_size, pre_batch, n_size,
                num_samples, with_shuffle = False, time_sampling = 1, with_source_idxs = False,
                fields_targets = None, pre_batch_targets = None ) :
    '''
      Data set for single dynamic field at an arbitrary number of vertical levels

      nsize : neighborhood in (tsteps, deg_lat, deg_lon)
    '''
    super( MultifieldDataSampler).__init__()

    self.fields = fields
    self.batch_size = batch_size
    self.n_size = n_size
    self.num_samples = num_samples
    self.with_source_idxs = with_source_idxs
    self.with_shuffle = with_shuffle
    self.pre_batch = pre_batch
    
    assert os.path.exists(file_path), f"File path {file_path} does not exist"
    self.ds = zarr.open( file_path)

    self.ds_global = self.ds.attrs['is_global']

    self.lats = np.array( self.ds['lats'])
    self.lons = np.array( self.ds['lons'])
    
    sh = self.ds['data'].shape
    st = self.ds['time'].shape
    self.ds_len = st[0] 
    print( f'self.ds[\'data\'] : {sh} :: {st}')
    print( f'self.lats : {self.lats.shape}', flush=True)
    print( f'self.lons : {self.lons.shape}', flush=True)
    self.fields_idxs = []

    self.time_sampling = time_sampling
    self.range_lat = np.array( self.lats[ [0,-1] ])
    self.range_lon = np.array( self.lons[ [0,-1] ])
    self.res = np.array(self.ds.attrs['res'])
    self.year_base = self.ds['time'][0].astype(datetime).year

    # ensure neighborhood does not exceed domain (either at pole or for finite domains)
    self.range_lat += np.array([n_size[1] / 2., -n_size[1] / 2.])
    # lon: no change for periodic case
    if self.ds_global < 1.:
      self.range_lon += np.array([n_size[2]/2., -n_size[2]/2.])
    # data normalizers
    self.normalizers = []
    for ifield, field_info in enumerate(fields) :
      corr_type = 'global' if len(field_info) <= 6 else field_info[6]
      nf_name = 'global_norm' if corr_type == 'global' else 'norm'
      self.normalizers.append( [] )
      for vl in field_info[2]: 
        if vl == 0:
          field_idx = self.ds.attrs['fields_sfc'].index( field_info[0])
          self.normalizers[ifield] += [self.ds[f'normalization/{nf_name}_sfc'].oindex[ :, :, field_idx]] 
        else:
          vl_idx = self.ds.attrs['levels'].index(vl)
          field_idx = self.ds.attrs['fields'].index( field_info[0])
          self.normalizers[ifield] += [self.ds[f'normalization/{nf_name}'].oindex[ :, :, field_idx, vl_idx]] 
    # extract indices for selected years
    self.times = pd.DatetimeIndex( self.ds['time'])
    idxs_years = self.times.year == years[0]
    for year in years[1:] :
      idxs_years = np.logical_or( idxs_years, self.times.year == year)
    self.idxs_years = np.where( idxs_years)[0]
    # logging.getLogger('atmorep').info( f'Dataset size for years {years}: {len(self.idxs_years)}.')
    print( f'Dataset size for years {years}: {len(self.idxs_years)}.', flush=True)

  ###################################################
  def shuffle( self) :

    worker_info = torch.utils.data.get_worker_info()
    rng_seed = None
    if worker_info is not None :
      rng_seed = int(time.time()) // (worker_info.id+1) + worker_info.id

    rng = np.random.default_rng( rng_seed)
    self.idxs_perm_t = rng.permutation( self.idxs_years)[ : self.num_samples // self.batch_size]
    
    lats = rng.random(self.num_samples) * (self.range_lat[1] - self.range_lat[0]) +self.range_lat[0]
    lons = rng.random(self.num_samples) * (self.range_lon[1] - self.range_lon[0]) +self.range_lon[0]

    # align with grid
    res_inv = 1.0 / self.res * 1.00001
    lats = self.res[0] * np.round( lats * res_inv[0])
    lons = self.res[1] * np.round( lons * res_inv[1])

    self.idxs_perm = np.stack( [lats, lons], axis=1)

  ###################################################
  def __iter__(self):
    if self.with_shuffle :
      self.shuffle()

    lats, lons = self.lats, self.lons
    ts, n_size = self.time_sampling, self.n_size
    ns_2 = np.array(self.n_size) / 2.
    res = self.res

    iter_start, iter_end = self.worker_workset()
   
    for bidx in range( iter_start, iter_end) :
      sources, token_infos = [[] for _ in self.fields], [[] for _ in self.fields]
      sources_infos, source_idxs = [], []
  
      i_bidx = self.idxs_perm_t[bidx]
      idxs_t = list(np.arange( i_bidx - n_size[0]*ts, i_bidx, ts, dtype=np.int64))
      data_tt_sfc = self.ds['data_sfc'].oindex[idxs_t]
      data_tt = self.ds['data'].oindex[idxs_t]
      for sidx in range(self.batch_size) :
        # i_bidx = self.idxs_perm_t[bidx]
        # idxs_t = list(np.arange( i_bidx - n_size[0]*ts, i_bidx, ts, dtype=np.int64))

        idx = self.idxs_perm[bidx*self.batch_size+sidx]
        # slight asymetry with offset by res/2 is required to match desired token count
        lat_ran = np.where(np.logical_and(lats>idx[0]-ns_2[1]-res[0]/2.,lats<idx[0]+ns_2[1]))[0]
        # handle periodicity of lon
        assert not ((idx[1]-ns_2[2]) < 0. and (idx[1]+ns_2[2]) > 360.)
        il, ir = (idx[1]-ns_2[2]-res[1]/2., idx[1]+ns_2[2])
        if il < 0. :
          lon_ran = np.concatenate( [np.where( lons > il+360)[0], np.where(lons < ir)[0]], 0)
        elif ir > 360. :
          lon_ran = np.concatenate( [np.where( lons > il)[0], np.where(lons < ir-360)[0]], 0)
        else : 
          lon_ran = np.where(np.logical_and( lons > il, lons < ir))[0]
        
        sources_infos += [ [ self.ds['time'][ idxs_t ].astype(datetime), 
                             self.lats[lat_ran], self.lons[lon_ran], self.res ] ]

        if self.with_source_idxs :
          source_idxs += [ (idxs_t, lat_ran, lon_ran) ]

        # extract data
        for ifield, field_info in enumerate(self.fields):  
          source_lvl, tok_info_lvl  = [], []
          tok_size  = field_info[4]
          corr_type = 'global' if len(field_info) <= 6 else field_info[6]
        
          for ilevel, vl in enumerate(field_info[2]):
            if vl == 0 : #surface level
              field_idx = self.ds.attrs['fields_sfc'].index( field_info[0])
              data_t = data_tt_sfc[ :, field_idx ]
            else :
              field_idx = self.ds.attrs['fields'].index( field_info[0])
              vl_idx = self.ds.attrs['levels'].index(vl)
              data_t = data_tt[ :, field_idx, vl_idx ]
          
            source_data, tok_info = [], []
            # extract data, normalize and tokenize
            cdata = np.take( np.take( data_t, lat_ran, -2), lon_ran, -1)
                
            normalizer = self.normalizers[ifield][ilevel]
            if corr_type != 'global':   
              normalizer = np.take( np.take( normalizer, lat_ran, -2), lon_ran, -1) 
            cdata = normalize(cdata, normalizer, sources_infos[-1][0], year_base = self.year_base)
            source_data = tokenize( torch.from_numpy( cdata), tok_size )    
            # token_infos uses center of the token: *last* datetime and center in space
            dates = self.ds['time'][ idxs_t ].astype(datetime)
            cdates = dates[tok_size[0]-1::tok_size[0]]
            dates = [(d.year, d.timetuple().tm_yday-1, d.hour) for d in cdates] #-1 is to start days from 0
            lats_sidx = self.lats[lat_ran][ tok_size[1]//2 :: tok_size[1] ]
            lons_sidx = self.lons[lon_ran][ tok_size[2]//2 :: tok_size[2] ]
            # tensor product for token_infos
            tok_info += [[[[[ year, day, hour, vl, lat, lon, vl, self.res[0]] for lon in lons_sidx]
                                                                              for lat in lats_sidx]
                                                                  for (year, day, hour) in dates]]

            source_lvl += [ source_data ]
            tok_info_lvl += [ torch.tensor(tok_info, dtype=torch.float32).flatten( 1, -2)]      
          sources[ifield] += [ torch.stack(source_lvl, 0) ]
          token_infos[ifield] += [ torch.stack(tok_info_lvl, 0) ]
      
      # concatenate batches   
      sources = [torch.stack(sources_field).transpose(1,0) for sources_field in sources]
      token_infos = [torch.stack(tis_field).transpose(1,0) for tis_field in token_infos]
      sources = self.pre_batch( sources, token_infos )
      # TODO: implement (only required when prediction target comes from different data stream)
      targets, target_info = None, None
      target_idxs = None
      yield ( sources, targets, (source_idxs, sources_infos), (target_idxs, target_info))

  ###################################################
  def set_data( self, times_pos, batch_size = None) :
    '''
      times_pos = np.array( [ [year, month, day, hour, lat, lon], ...]  )
        - lat \in [90,-90] = [90N, 90S]
        - lon \in [0,360]
        - (year,month) pairs should be a limited number since all data for these is loaded
    '''
    # generate all the data
    self.idxs_perm = np.zeros( (len(times_pos), 2))
    self.idxs_perm_t = []
    self.num_samples = len(times_pos)
    for idx, item in enumerate( times_pos) :

      assert item[2] >= 1 and item[2] <= 31
      assert item[3] >= 0 and item[3] < int(24 / self.time_sampling)
      assert item[4] >= -90. and item[4] <= 90.

      tstamp = pd.to_datetime( f'{item[0]}-{item[1]}-{item[2]}-{item[3]}', format='%Y-%m-%d-%H')
      
      self.idxs_perm_t += [ np.where( self.times == tstamp)[0]+1 ] #The +1 assures that tsamp is included in the range

      # work with mathematical lat coordinates from here on
      self.idxs_perm[idx] = np.array( [90. - item[4], item[5]])
   
    self.idxs_perm_t = np.array(self.idxs_perm_t).squeeze()

  ###################################################
  def set_global( self, times, batch_size = None, token_overlap = [0, 0]) :
    ''' generate patch/token positions for global grid '''
    token_overlap = np.array( token_overlap).astype(np.int64)

    # assumed that sanity checking that field data is consistent has been done 
    ifield = 0
    field = self.fields[ifield]

    res = self.res
    side_len = np.array( [field[3][1] * field[4][1]*res[0], field[3][2] * field[4][2]*res[1]] )
    overlap = np.array([token_overlap[0]*field[4][1]*res[0],token_overlap[1]*field[4][2]*res[1]])
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
        lat = 180. - side_len_2[0].item() + res[0]
        lon = side_len_2[1].item() - overlap[1].item()/2.
        while (lon - side_len_2[1]) < 360. :
          times_pos += [[*ctime, -lat + 90., np.mod(lon,360.) ]]
          lon += side_len[1].item() - overlap[1].item()

    # adjust batch size if necessary so that the evaluations split up across batches of equal size
    batch_size = num_tiles_lon
 
    print( 'Number of batches per global forecast: {}'.format( num_tiles_lat) )

    self.set_data( times_pos, batch_size)

  ###################################################
  def __len__(self):
      return self.num_samples // self.batch_size

  ###################################################
  def worker_workset( self) :

    worker_info = torch.utils.data.get_worker_info()

    if worker_info is None: 
      iter_start = 0
      iter_end = self.num_samples
   
    else:  
      # split workload
      per_worker = len(self) // worker_info.num_workers
      worker_id = worker_info.id
      iter_start = int(worker_id * per_worker)
      iter_end = int(iter_start + per_worker)
      if worker_info.id+1 == worker_info.num_workers :
        iter_end = len(self)

    return iter_start, iter_end

