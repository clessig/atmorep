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
import zarr
import pandas as pd
import pdb
import code
from atmorep.utils.utils import days_until_month_in_year
from atmorep.utils.utils import days_in_month
from datetime import datetime

import atmorep.config.config as config

from atmorep.datasets.normalizer_global import NormalizerGlobal
from atmorep.datasets.normalizer_local import NormalizerLocal
from atmorep.utils.utils import tokenize


class MultifieldDataSampler( torch.utils.data.IterableDataset):
    
  ###################################################
  def __init__( self, fields, years, batch_size, pre_batch, n_size, num_samples_per_epoch,
                rng_seed = None, time_sampling = 1, with_source_idxs = False,
                fields_targets = None, pre_batch_targets = None ) :
    '''
      Data set for single dynamic field at an arbitrary number of vertical levels

      nsize : neighborhood in (tsteps, deg_lat, deg_lon)
    '''
    super( MultifieldDataSampler).__init__()

    self.fields = fields
    self.batch_size = batch_size
    self.n_size = n_size
    self.num_samples = num_samples_per_epoch
    self.with_source_idxs = with_source_idxs

    self.pre_batch = pre_batch
    
    # create (source) fields
    # config.path_data
    fname_source = '/p/scratch/atmo-rep/era5_res0025_1979.zarr'
    fname_source = '/p/scratch/atmo-rep/era5_res0025_2021.zarr'
    fname_source = '/p/scratch/atmo-rep/data/era5_1deg/era5_res0025_2021_final.zarr'
    # fname_source = '/p/scratch/atmo-rep/era5_res0100_2021_t5.zarr'
    self.ds = zarr.open( fname_source)
    self.ds_global = self.ds.attrs['is_global']
    self.ds_len = self.ds['data'].shape[0]

    # sanity checking
    # assert self.ds['data'].shape[0] == self.ds['time'].shape[0]
    # assert self.ds_len >= num_samples_per_epoch

    self.lats = np.array( self.ds['lats'])
    self.lons = np.array( self.ds['lons'])

    sh = self.ds['data'].shape
    st = self.ds['time'].shape
    print( f'self.ds[\'data\'] : {sh} :: {st}')
    print( f'self.lats : {self.lats.shape}', flush=True)
    print( f'self.lons : {self.lons.shape}', flush=True)
    self.fields_idxs = []

    # for f in fields:
    #  self.fields_idxs = [self.ds.attrs['fields'].index( f[0]) if f[0] in self.ds.attrs['fields'] 
    #   self.fields_idxs = np.array( [self.ds.attrs['fields'].index( f[0]) for f in fields]) 
    # self.levels_idxs = np.array( [self.ds.attrs['levels'].index( ll) for ll in levels])
    # self.fields_idxs = [0, 1, 2]
    # self.levels_idxs = [0, 1]
   # self.levels = levels #[123, 137]  # self.ds['levels']

    # TODO
    # # create (target) fields 
    # self.datasets_targets = self.create_loaders( fields_targets)
    # self.fields_targets = fields_targets
    # self.pre_batch_targets = pre_batch_targets

    self.time_sampling = time_sampling
    self.range_lat = np.array( self.lats[ [0,-1] ])
    self.range_lon = np.array( self.lons[ [0,-1] ])

    self.res = np.zeros( 2)
    self.res[0] = self.ds.attrs['resol'][0] #(self.range_lat[1]-self.range_lat[0]) / (self.ds['data'].shape[-2]-1)
    self.res[1] = self.ds.attrs['resol'][1] #(self.range_lon[1]-self.range_lon[0]) / (self.ds['data'].shape[-1]-1)
    
    # ensure neighborhood does not exceed domain (either at pole or for finite domains)
    self.range_lat += np.array([n_size[1] / 2., -n_size[1] / 2.])
    # lon: no change for periodic case
    if self.ds_global < 1.:
      self.range_lon += np.array([n_size[2]/2., -n_size[2]/2.])

    # ensure all data loaders use same rng_seed and hence generate consistent data
    if not rng_seed :
      rng_seed = np.random.randint( 0, 100000, 1)[0]
    self.rng = np.random.default_rng( rng_seed)

    # data normalizers
    self.normalizers = []
    for _, field_info in enumerate(fields) :
      self.normalizers.append( [])
      corr_type = 'global' if len(field_info) <= 6 else field_info[6]
      ner = NormalizerGlobal if corr_type == 'global' else NormalizerLocal
      for vl in field_info[2]: 
        data_type = 'data_sfc' if vl == 0 else 'data' #surface field
        self.normalizers[-1] += [ ner( field_info, vl, 
                                  np.array(self.ds[data_type].shape)[[0,-2,-1]]) ]
    # extract indices for selected years
    self.times = pd.DatetimeIndex( self.ds['time'])
    # idxs = np.zeros( self.ds['time'].shape[0], dtype=np.bool_)
    # self.idxs_years = np.array( [])
    # for year in years :
    #   idxs = np.where( (self.times >= f'{year}-1-1') & (self.times <= f'{year}-12-31'))[0]
    #   assert idxs.shape[0] > 0, f'Requested year is not in dataset {fname_source}. Aborting.'
    #   self.idxs_years = np.append( self.idxs_years, idxs[::self.time_sampling])
    # TODO, TODO, TODO:
    self.idxs_years = np.arange( self.ds_len)

  ###################################################
  def shuffle( self) :

    rng = self.rng
    self.idxs_perm_t = rng.permutation( self.idxs_years)[:(self.num_samples // self.batch_size)]

    lats = rng.random(self.num_samples) * (self.range_lat[1] - self.range_lat[0]) +self.range_lat[0]
    lons = rng.random(self.num_samples) * (self.range_lon[1] - self.range_lon[0]) +self.range_lon[0]

    # align with grid
    res_inv = 1.0 / self.res * 1.00001
    lats = self.res[0] * np.round( lats * res_inv[0])
    lons = self.res[1] * np.round( lons * res_inv[1])

    self.idxs_perm = np.stack( [lats, lons], axis=1)

  ###################################################
  def __iter__(self):

    # TODO: if we keep this then we should remove the rng_seed argument for the constuctor
    self.rng = np.random.default_rng()
    self.shuffle()

    lats, lons = self.lats, self.lons
    #fields_idxs, levels_idxs = self.fields_idxs, self.levels_idxs
    ts, n_size = self.time_sampling, self.n_size
    ns_2 = np.array(self.n_size) / 2.
    res = self.res

    iter_start, iter_end = self.worker_workset()

    for bidx in range( iter_start, iter_end) :
     
      idx = self.idxs_perm_t[bidx]
      idxs_t = list(np.arange( idx-n_size[0]*ts, idx, ts, dtype=np.int64))
      data_t = []

      for _, field_info in enumerate(self.fields) :
        data_lvl = []
        for vl in field_info[2]:
          if vl == 0: #surface level
            field_idx = self.ds.attrs['fields_sfc'].index( field_info[0])
            data_lvl += [self.ds['data_sfc'].oindex[ idxs_t, field_idx]]
          else:
            field_idx = self.ds.attrs['fields'].index( field_info[0])
            vl_idx = self.ds.attrs['levels'].index(vl)
            data_lvl += [self.ds['data'].oindex[ idxs_t, field_idx, vl_idx]]
        data_t += [data_lvl]
      
      sources, sources_infos, source_idxs, token_infos = [], [], [], []
      lat_ran, lon_ran = [], []
    
      for sidx in range(self.batch_size) :

        idx = self.idxs_perm[bidx*self.batch_size+sidx]
        # slight assymetry with offset by res/2 is required to match desired token count
        lat_ran += [np.where(np.logical_and(lats > idx[0]-ns_2[1]-res[0]/2.,lats < idx[0]+ns_2[1]))[0]]
        # handle periodicity of lon
        assert not ((idx[1]-ns_2[2]) < 0. and (idx[1]+ns_2[2]) > 360.)
        il, ir = (idx[1]-ns_2[2]-res[1]/2., idx[1]+ns_2[2])
        if il < 0. :
          lon_ran += [np.concatenate( [np.where( lons > il+360)[0], np.where(lons < ir)[0]], 0)]
        elif ir > 360. :
          lon_ran += [np.concatenate( [np.where( lons > il)[0], np.where(lons < ir-360)[0]], 0)]
        else : 
          lon_ran += [np.where(np.logical_and( lons > il, lons < ir))[0]]
        
        sources_infos += [ [ self.ds['time'][ idxs_t ].astype(datetime), 
                           self.lats[lat_ran][-1], self.lons[lon_ran][-1], self.res ] ]

        if self.with_source_idxs :
          source_idxs += [ (idxs_t, lat_ran[-1], lon_ran[-1]) ]
       
      # extract data
      # TODO: temporal window can span multiple months
      year, month = self.times[ idxs_t[-1] ].year, self.times[ idxs_t[-1] ].month
      for ifield, field_info in enumerate(self.fields):

        source_lvl, source_info_lvl, tok_info_lvl  = [], [], []
        tok_size = field_info[4]
        for ilevel, vl in enumerate(field_info[2]): 
         
          nf = self.normalizers[ifield][ilevel].normalize
          source_data, tok_info = [], []
         
          for sidx in range(self.batch_size) :
            #normalize and tokenize           
            source_data += [ tokenize( torch.from_numpy(nf( year, month, np.take( np.take( data_t[ifield][ilevel], 
                                        lat_ran[sidx], -2), lon_ran[sidx], -1), (lat_ran[sidx], lon_ran[sidx]))), tok_size ) ]
          
            dates = self.ds['time'][ idxs_t ].astype(datetime)
             #store only center of the token: 
             #in time we store the *last* datetime in the token, not the center
            dates = [(d.year, d.timetuple().tm_yday, d.hour) for d in dates][tok_size[0]-1::tok_size[0]]
            lats_sidx = self.lats[lat_ran[sidx]][int(tok_size[1]/2)::tok_size[1]]
            lons_sidx = self.lons[lon_ran[sidx]][int(tok_size[2]/2)::tok_size[2]]
            # info_data += [[[[[ year, day, hour, vl, 
            #                   lat, lon, vl, self.res[0]] for lon in lons] for lat in lats] for (year, day, hour) in dates]] #zip(years, days, hours)]]
                       
            tok_info += [[[[[ year, day, hour, vl, 
                              lat, lon, vl, self.res[0]] for lon in lons_sidx] for lat in lats_sidx] for (year, day, hour) in dates]] 

          #level
          source_lvl += [torch.stack(source_data, dim = 0)]
          # source_info_lvl += [info_data]
          tok_info_lvl += [tok_info]

        #field
        sources += [torch.stack(source_lvl, dim = 0)] #torch.Size([3, 16, 12, 6, 12, 3, 9, 9])
        # sources_infos += [torch.Tensor(np.array(source_info_lvl))] # torch.Size([3, 16, 36, 54, 108, 8])
        #token_infos += [torch.Tensor(np.array(tok_info_lvl))] # torch.Size([3, 16, 12, 6, 12, 8])
        # extract batch info. level info stored in cf.fields. not stored here.
        
        token_infos += [torch.Tensor(np.array(tok_info_lvl)).reshape(len(tok_info_lvl), len(tok_info_lvl[0]), -1, 8)] #torch.Size([3, 16, 864, 8])
      
      sources = self.pre_batch(sources,  
                                token_infos )   

      # TODO: implement targets
      targets, target_info = None, None
      target_idxs = None
      #this already goes back to trainer.py. 
      #source_info needed to remove log_validate in trainer.py
      yield ( sources, targets, (source_idxs, sources_infos), (target_idxs, target_info))

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

