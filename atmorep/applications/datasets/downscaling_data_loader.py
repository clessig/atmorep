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
import code

# from atmorep.datasets.normalizer_global import NormalizerGlobal
# from atmorep.datasets.normalizer_local import NormalizerLocal
from atmorep.datasets.normalizer import normalize
from atmorep.utils.utils import tokenize, get_weights
from atmorep.applications.downscaling.utils.era5_imerg_data_aligner import lat_lon_time_range
from atmorep.utils.logger import logger


class MultifieldDownscalingSampler( torch.utils.data.IterableDataset):
    
    def __init__(self, input_file_path, target_file_path, input_fields, target_fields, years, batch_size, n_size,
                 num_samples, downscaling_ratio, with_shuffle=False, with_source_idxs = True, with_target_idxs=True) :
        '''
          Iterable torch dataset for ERA5 to IMERG downscaling task loading data at an arbitrary number of vertical levels
    
          input_file_path: path to input ERA5 zarr storage
          target_file_path: path to input IMERG zarr storage
          input_fields: configuration list for input data fields
          target_fields: configuration list for input data fields
          years: year or list of years from which data should be sampled
          batch_size: number of samples per batch
          nsize : neighborhood in (tsteps, nlat_ERA5, nlon_ERA5) -> different to MultifieldDataSampler!!!
          num_samples: total number of samples to draw
          downscaling_ratio: downscaling factor between ERA5 input and IMERG target data
          with_shuffle: flag for shuffling samples
          with_source_idxs:
          with_target_idxs:   
        '''       
        super(MultifieldDownscalingSampler, self).__init__()

        self.input_fields = input_fields
        self.target_fields = target_fields
        self.n_size = n_size
        self.num_samples = num_samples
        self.with_source_idxs = with_source_idxs
        self.with_target_idxs = with_target_idxs
        self.batch_size = batch_size
        self.with_shuffle = with_shuffle
        self.downscaling_ratio = downscaling_ratio

        if not os.path.exists(input_file_path):
            FileNotFoundError(f"Input zarr store {input_file_path} does not exist.")

        if not os.path.exists(input_file_path):
            FileNotFoundError(f"Target zarr store {target_file_path} does not exist.")

        # open zarr stores
        self.era5_ds = zarr.group(input_file_path)
        self.imerg_ds = zarr.group(target_file_path)

        # get coordinate information from ERA5 and IMERG data
        self.era5_res = self.era5_ds.attrs['res']
        self.imerg_res = self.imerg_ds.attrs['res']

        self.era5_times = self.era5_ds["time"]
        self.era5_lats = np.array(self.era5_ds['lats'])
        self.era5_lons = np.array(self.era5_ds['lons'])

        self.imerg_times = self.imerg_ds["time"]
        self.imerg_lats = np.array(self.imerg_ds['lats'])
        self.imerg_lons = np.array(self.imerg_ds['lons'])
        self.imerg_lons = np.where(self.imerg_lons < 0, self.imerg_lons + 360, self.imerg_lons)

        self.era5_num_lats = self.era5_ds['lats'].shape[0]
        self.era5_num_lons = self.era5_ds['lons'].shape[0]

        # spatial shift of first IMERG data point w.r.t. first ERA5 data point
        # Note: IMERG grid must already be centered around ERA5 grid accordingly during preprocessing
        self.dx_shift = (self.downscaling_ratio - 1)/2.*np.array(self.imerg_res)

        # Note: Neighborhood sampling is based on ERA5 indices which serve as anchor points
        # Parameters for centered spatial slicing
        # In case that the number of grid points is even, the center (anchor) point for the local neighborhood is shifted west/south
        # Example nor n_size=4: 
        #  |           |   |           |
        #  x   o   x   x   x   o   x   x 
        #  |           |   |           |
        # The leftmost and rightmost neighborhoods incl. their boundaries | and anchor points o are shown. 
        nsize_lat_half = int((self.n_size[1] - 1 )/2)
        self.nsize_lat = np.asarray([nsize_lat_half, nsize_lat_half])
        if self.n_size[1] % 2 == 0:
            self.nsize_lat[1] += 1

        nsize_lon_half = int((self.n_size[2] - 1 )/2)
        self.nsize_lon = np.asarray([nsize_lon_half, nsize_lon_half])
        if self.n_size[2] % 2 == 0:
            self.nsize_lon[1] += 1

        # ensure that sampling is restricted to spatio-temporal index ranges where ERA5 and IMERG data are available
        _, tidx_era5, _ = np.intersect1d(self.era5_times, self.imerg_times, return_indices=True)
        _, latidx_era5, _ = np.intersect1d(self.era5_lats, self.imerg_lats, return_indices=True)
        _, lonidx_era5, _ = np.intersect1d(self.era5_lons, self.imerg_lons, return_indices=True)
        
        # handle situation where the contigious domain cross the zero meridian
        split_idx = np.where(np.diff(lonidx_era5) != 1)[0] + 1    # find indices where the difference between consecutive elements is not 1
        
        # Add the start and end indices to form ranges
        idx_segs = np.split(lonidx_era5, split_idx)
        contiguous_segs = [(idx_seg[0], idx_seg[-1]) for idx_seg in idx_segs]

        # save ERA5 anchor point indices for sampling
        self.range_ilat = np.array([min(latidx_era5) + self.nsize_lat[0], max(latidx_era5) - self.nsize_lat[1]])
        self.anchor_ilat = np.arange(*self.range_ilat)
        
        if len(contiguous_segs) == 2:
            range_ilon = [min(contiguous_segs[1]), max(contiguous_segs[0])]
        elif len(contiguous_segs) == 1:
            range_ilon = [min(lonidx_era5), max(lonidx_era5)]
        else:
            raise ValueError(f"More than two splitting indices identified along longitude dimension. Ensure that IMERG-domain is contigious.")
        

        self.range_ilon = np.array([range_ilon[0] + self.nsize_lon[0], range_ilon[1] - self.nsize_lon[1]])
        if self.range_ilon[0] > self.range_ilon[1]:
            self.anchor_ilon = np.arange(self.range_ilon[0], self.range_ilon[1] + self.era5_num_lons)
            self.anchor_ilon = self.anchor_ilon%self.era5_num_lons
        else:
            self.anchor_ilon = np.arange(*self.range_ilon)       
        
        # Note: sample time-index is not centered, but placed at the end of the time sequence
        # add n_size[0] to avoid sampling from time steps at whch no IMERG data is available
        self.range_itime = np.array([min(tidx_era5) + self.n_size[0], max(tidx_era5)])
        
        self.year_base = self.era5_ds['time'][self.range_itime[0]].astype(datetime).year

        self.input_normalizers = []

        for ifield, field_info in enumerate(input_fields):
            corr_type = 'global' if len(field_info) <=6 else field_info[6]
            nf_name = 'global_norm' if corr_type == 'global' else 'norm'
            self.input_normalizers.append([])
            for vl in field_info[2]: 
                if vl == 0:
                    field_idx = self.era5_ds.attrs['fields_sfc'].index( field_info[0])
                    n_name = f'normalization/{nf_name}_sfc'
                    self.input_normalizers[ifield] += [self.era5_ds[n_name].oindex[ :, :, field_idx]] 
                else:
                    vl_idx = self.era5_ds.attrs['levels'].index(vl)
                    field_idx = self.era5_ds.attrs['fields'].index( field_info[0])
                    n_name = f'normalization/{nf_name}'
                    self.input_normalizers[ifield] += [self.era5_ds[n_name].oindex[ :, :, field_idx, vl_idx] ]


        self.target_normalizers = []

        for ifield, field_info in enumerate(target_fields):
            corr_type = 'global' if len(field_info) <=6 else field_info[6]
            nf_name = 'global_norm' if corr_type == 'global' else 'norm'
            self.target_normalizers.append([])
            for vl in field_info[2]: 
                if vl == 0:
                    field_idx = self.imerg_ds.attrs['fields_sfc'].index( field_info[0])
                    n_name = f'normalization/{nf_name}_sfc'
                    self.target_normalizers[ifield] += [self.imerg_ds[n_name].oindex[ :, :, field_idx]] 
                else:
                    vl_idx = self.target_ds.attrs['levels'].index(vl)
                    field_idx = self.imerg_ds.attrs['fields'].index( field_info[0])
                    n_name = f'normalization/{nf_name}'
                    self.target_normalizers[ifield] += [self.imerg_ds[n_name].oindex[ :, :, field_idx, vl_idx] ]

        # filter data that is not part of the desired period
        years_input_file = np.asarray([str(year) for year in self.era5_times[self.range_itime[0]:self.range_itime[1]].astype('datetime64[Y]')])
        valid_years_str = [str(year) for year in range(years[0], (years[1] if len(years)>1 else years[0]) + 1)]
        mask_year = np.isin(years_input_file, valid_years_str)

        #self.valid_time_indices = np.where(logical_array)[0]
        self.anchor_itime = np.arange(*self.range_itime)[mask_year]
        self.num_samples = min(self.num_samples, self.anchor_itime.shape[0])
            

    def shuffle(self):

        worker_info = torch.utils.data.get_worker_info()
        rng_seed = None

        if worker_info is not None:
            rng_seed = int(time.time()) // (worker_info.id+1) + worker_info.id

        rng = np.random.default_rng(rng_seed)

        # get random time index
        self.idx_perm_t = rng.choice(self.anchor_itime, self.num_samples // self.batch_size, replace=False)

        idx_lat, idx_lon = rng.choice(self.anchor_ilat, self.num_samples, replace=True), \
                           rng.choice(self.anchor_ilon, self.num_samples, replace=True)

        self.idx_perm_s = np.stack([idx_lat, idx_lon], axis=1)        

    def __iter__(self):
        
        if self.with_shuffle:
            self.shuffle()

        lats, lons = self.era5_lats, self.era5_lons
        n_size = self.n_size
        ts = 1                             # To-Do: add as parsing parameter!!!

        iter_start , iter_end = self.worker_workset()

        for bidx in range( iter_start, iter_end):

            sources, token_infos = [[] for _ in self.input_fields], [[] for _ in self.input_fields]
            targets, target_token_infos = [[] for _ in self.target_fields], [[] for _ in self.target_fields],
            sources_infos, source_idxs = [], []
            target_infos, target_idxs = [], []

            # get matching time index for IMERG dataset
            i_bidx_era5 = self.idx_perm_t[bidx]
            i_bidx_imerg = np.where(self.era5_times[i_bidx_era5] == self.imerg_times)[0][0]

            # get list of time indices for slicing
            idxs_t_era5 = list(np.arange( i_bidx_era5 - n_size[0]*ts, i_bidx_era5, ts, dtype=np.int64))
            idxs_t_imerg = list(np.arange( i_bidx_imerg - n_size[0]*ts, i_bidx_imerg, ts, dtype=np.int64))

            data_t = self.era5_ds['time'][idxs_t_era5[0]:idxs_t_era5[1]].astype(datetime)
            
            # extract data for time steps at hand
            data_era5_tt_sfc = self.era5_ds['data_sfc'][idxs_t_era5]
            data_era5_tt = self.era5_ds['data'][idxs_t_era5]

            data_imerg_tt_sfc = self.imerg_ds['data_sfc'][idxs_t_imerg]

            for sidx in range(self.batch_size):

                idx_era5 = self.idx_perm_s[bidx*self.batch_size+sidx]

                ilat_range_era5 = np.arange(idx_era5[0] - self.nsize_lat[0], idx_era5[0] + self.nsize_lat[1] + 1)
                # handle periodicity in zonal (longitude) direction
                ilon_range_era5 = np.arange(idx_era5[1] - self.nsize_lon[0], idx_era5[1] + self.nsize_lon[1] + 1)
                
                ilon_range_era5 = np.where(ilon_range_era5 < 0, ilon_range_era5 + self.era5_num_lons, ilon_range_era5)
                ilon_range_era5 = ilon_range_era5 % self.era5_num_lons

                # get corresponding spatial sampling indices for IMERG dataset
                ilat_imerg0 = np.where(np.isclose(self.era5_lats[ilat_range_era5[0]] - self.dx_shift[0], self.imerg_lats))[0]
                ilon_imerg0 = np.where(np.isclose((self.era5_lons[ilon_range_era5[0]] - self.dx_shift[1]
                                        if self.era5_lons[ilon_range_era5[0]] - self.dx_shift[1] >= 0.0
                                        else self.era5_lons[ilon_range_era5[0]] - self.dx_shift[1] + 360.0), self.imerg_lons))[0]

                assert len(ilat_imerg0) == 1, f"Could not find required IMERG latitude grid point for first ERA5 grid point at {self.era5_lats[ilat_range_era5[0]]} deg"
                assert len(ilon_imerg0) == 1, f"Could not find required IMERG longitude grid point for first ERA5 grid point at {self.era5_lons[ilon_range_era5[0]]} deg"

                ilat_imerg0, ilon_imerg0 = ilat_imerg0[0], ilon_imerg0[0]
                ilat_range_imerg = np.arange(ilat_imerg0, ilat_imerg0 + self.n_size[1]*self.downscaling_ratio)
                ilon_range_imerg = np.arange(ilon_imerg0, ilon_imerg0 + self.n_size[2]*self.downscaling_ratio)

                # start data retrieval        
                sources_infos += [ [ self.era5_ds['time'][ idxs_t_era5 ].astype(datetime), 
                                     self.era5_lats[ilat_range_era5], self.era5_lons[ilon_range_era5], self.era5_res ] ]
                target_infos += [ [ self.imerg_ds['time'][ idxs_t_imerg ].astype(datetime), 
                                     self.imerg_lats[ilat_range_imerg], self.imerg_lons[ilon_range_imerg], self.imerg_res ] ]
        
                if self.with_source_idxs :
                   source_idxs += [ (idxs_t_era5, ilat_range_era5, ilon_range_era5) ]

                if self.with_target_idxs:
                   target_idxs += [ (idxs_t_imerg, ilat_range_imerg, ilon_range_imerg) ]
        
                # extract input ERA5 data
                for ifield, field_info in enumerate(self.input_fields):  
                  source_lvl, tok_info_lvl  = [], []
                  tok_size  = field_info[4]
                  num_tokens = field_info[3]
                  corr_type = 'global' if len(field_info) <= 6 else field_info[6]
                
                  for ilevel, vl in enumerate(field_info[2]):
                    if vl == 0 : #surface level
                      field_idx = self.era5_ds.attrs['fields_sfc'].index( field_info[0])
                      data_era5_t = data_era5_tt_sfc[ :, field_idx ]
                    else :
                      field_idx = self.era5_ds.attrs['fields'].index( field_info[0])
                      vl_idx = self.era5_ds.attrs['levels'].index(vl)
                      data_era5_t = data_era5_tt[ :, field_idx, vl_idx ]
                  
                    source_data, tok_info = [], []
                    # extract data, normalize and tokenize
                    cdata = data_era5_t[ ... , ilat_range_era5[:, np.newaxis], ilon_range_era5[np.newaxis, :]]

                    # NOTE: uncomment/activate the following after testing
                    normalizer = self.input_normalizers[ifield][ilevel]
                    if corr_type != 'global':
                        normalizer = normalizer[ ... , ilat_range_era5[:,np.newaxis], ilon_range_era5[np.newaxis,:]]
 
                    cdata = normalize(cdata, normalizer, sources_infos[-1][0], year_base = self.year_base)
                    
                    source_data = tokenize( torch.from_numpy( cdata), tok_size )  
                    
                    # token_infos uses center of the token: *last* datetime and center in space
                    dates = self.era5_ds['time'][ idxs_t_era5 ].astype(datetime)
                    cdates = dates[tok_size[0]-1::tok_size[0]]
                    # use -1 is to start days from 0
                    dates = [(d.year, d.timetuple().tm_yday-1, d.hour) for d in cdates] 
                    lats_sidx = self.era5_lats[ilat_range_era5][ tok_size[1]//2 :: tok_size[1] ]
                    lons_sidx = self.era5_lons[ilon_range_era5][ tok_size[2]//2 :: tok_size[2] ]
                    # tensor product for token_infos
                    tok_info += [[[[[ year, day, hour, vl, lat, lon, vl, self.era5_res[0]] for lon in lons_sidx]
                                                                                      for lat in lats_sidx]
                                                                          for (year, day, hour) in dates]]
        
                    source_lvl += [ source_data]
                    tok_info_lvl += [ torch.tensor(tok_info, dtype=torch.float32).flatten( 1, -2)]
                      
                  sources[ifield] += [ torch.stack(source_lvl, 0) ]
                  token_infos[ifield] += [ torch.stack(tok_info_lvl, 0) ]

                for ifield, field_info in enumerate(self.target_fields):  
                  target_lvl, tok_info_lvl  = [], []
                  target_tok_size  = field_info[4]
                  target_num_tokens = field_info[3]
                  corr_type = 'global' if len(field_info) <= 6 else field_info[6]
                  
                  for ilevel, vl in enumerate(field_info[2]):
                    if vl == 0 : #surface level
                        field_idx = self.imerg_ds.attrs['fields_sfc'].index( field_info[0])
                        data_t_imerg = data_imerg_tt_sfc[:,field_idx]
                    else :
                        raise ValueError(f"No multi-level data available in IMERG-dataset. Please check the target_field-configuration.")
                  
                    target_data, tok_info = [], []
                    # extract data, normalize and tokenize
                    cdata = data_t_imerg[ ... , ilat_range_imerg[:,np.newaxis], ilon_range_imerg[np.newaxis,:]]

                    # NOTE: uncomment the following after testing
                    normalizer = self.target_normalizers[ifield][ilevel]
                    if corr_type != 'global': 
                        if ilat_range_imerg[0] < ilat_range_imerg[-1] and ilon_range_imerg[0] < ilon_range_imerg[-1]:
                            lat_max, lat_min = max(ilat_range_imerg), min(ilat_range_imerg)
                            lon_max, lon_min = max(ilon_range_imerg), min(ilon_range_imerg)
                            normalizer = normalizer[:,:,lat_min:lat_max+1,lon_min:lon_max+1]
                        else:
                            normalizer = normalizer[ ... , ilat_range_imerg[:,np.newaxis], ilon_range_imerg[np.newaxis,:]]
                    cdata = normalize(cdata, normalizer, sources_infos[-1][0], year_base = self.year_base)
                    
                    target_data = tokenize(torch.from_numpy(cdata), target_tok_size ) 

                      
                    # token_infos uses center of the token: *last* datetime and center in space
                    dates = self.imerg_ds['time'][ idxs_t_imerg ].astype(datetime)
                    cdates = dates[tok_size[0]-1::tok_size[0]]
                    # use -1 is to start days from 0
                    dates = [(d.year, d.timetuple().tm_yday-1, d.hour) for d in cdates] 
                    lats_sidx = self.imerg_lats[ilat_range_imerg][ target_tok_size[1]//2 :: target_tok_size[1] ]
                    lons_sidx = self.imerg_lons[ilon_range_imerg][ target_tok_size[2]//2 :: target_tok_size[2] ]
                    # tensor product for token_infos
                    tok_info += [[[[[ year, day, hour, vl, lat, lon, vl, self.imerg_res[0]] for lon in lons_sidx]
                                                                                            for lat in lats_sidx]
                                                                          for (year, day, hour) in dates]]
        
                    target_lvl += [ target_data]
                    tok_info_lvl += [ torch.tensor(tok_info, dtype=torch.float32).flatten( 1, -2)]      
                  targets[ifield] += [ torch.stack(target_lvl, 0) ]
                  target_token_infos[ifield] += [ torch.stack(tok_info_lvl, 0) ]

            sources = [torch.stack(sources_field) for sources_field in sources]
            token_infos = [torch.stack(tis_field) for tis_field in token_infos]

            targets = [torch.stack(targets_field) for targets_field in targets]
            
            yield ((sources, token_infos), (source_idxs, sources_infos), targets, (target_idxs,target_infos))    
  
    def __len__(self):
        return self.num_samples // self.batch_size


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
