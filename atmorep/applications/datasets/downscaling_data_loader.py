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
    
    def __init__(self, input_file_path, target_file_path, input_fields,
        target_fields, years, batch_size, n_size, num_samples, downscaling_ratio,
        with_shuffle=False, with_source_idxs = False, with_target_idxs=False) :
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
      
            assert os.path.exists(input_file_path), f"Input file path {input_file_path} does not exist"
            assert os.path.exists(target_file_path), f"Target file path {target_file_path} does not exist"

            self.era5_ds = zarr.group(input_file_path)
            self.imerg_ds = zarr.group(target_file_path)

            self.era5_res = self.era5_ds.attrs['res']
            self.imerg_res = self.imerg_ds.attrs['res']

            self.era5_lats = np.array(self.era5_ds['lats'])
            self.era5_lons = np.array(self.era5_ds['lons'])

            self.imerg_lats = np.array(self.imerg_ds['lats'])
            self.imerg_lons = np.array(self.imerg_ds['lons'])

            self.era5_num_lats = self.era5_ds['lats'].shape[0]
            self.era5_num_lons = self.era5_ds['lons'].shape[0]
            
            self.range_lat = np.array([0,-1])
            self.range_lon = np.array([0,-1])
            self.range_time = np.array([0,-1])

            if self.era5_ds.attrs['is_global']:

                self.global_indices_range = lat_lon_time_range(
                    input_file_path,
                    target_file_path
                )
                
                lat_range = self.global_indices_range['lats']
                lon_range = self.global_indices_range['lons']
                time_range = self.global_indices_range['time']

                self.range_lat = np.array([lat_range[0], lat_range[1]])
                self.range_lon = np.array([lon_range[0], lon_range[1]])

                self.range_time = np.array([
                    time_range[0],
                    time_range[1]])
            
            self.year_base = self.era5_ds['time'][self.range_time[0]].astype(datetime).year

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


            years_input_file = np.asarray(
                pd.DataFrame(
                    {'time' : pd.to_datetime(self.era5_ds['time'][self.range_time[0]:self.range_time[1]])})['time'].dt.strftime("%Y"))

            logical_array = np.where(years_input_file == str(years[0]), True, False)

            if len(years) > 1:
                for year in range(years[0],years[1])[1:] :
                    logical_array = np.logical_or(logical_array,
                         np.where(years_input_file == str(year), True, False))
        
            self.valid_time_indices = np.where(logical_array)[0]

            self.num_samples = min( self.num_samples, self.valid_time_indices.shape[0] - n_size[0])
            

    def shuffle(self):

        worker_info = torch.utils.data.get_worker_info()
        rng_seed = None

        if worker_info is not None:
            rng_seed = int(time.time()) // (worker_info.id+1) + worker_info.id

        rng = np.random.default_rng( rng_seed)
        self.idx_perm_era5_t = rng.permutation( 
            self.valid_time_indices[self.n_size[0]:])[: self.num_samples // self.batch_size]

        self.idx_perm_imerg_t = np.arange(self.range_time[1]-self.range_time[0])[self.idx_perm_era5_t]

        era5_lats = np.arange(self.range_lat[0],self.range_lat[1]-self.n_size[1])
        
        if self.range_lon[1] > self.range_lon[0]:
            era5_lons = np.arange(self.range_lon[0], self.range_lon[1]-self.n_size[2])
        else:
            era5_lons = np.arange(self.range_lon[0], self.range_lon[1]+self.era5_num_lons-self.n_size[2])

        imerg_lats_based_on_era5 = np.random.choice( np.arange(0, self.range_lat[1]-self.n_size[1]-self.range_lat[0]), size=(self.num_samples,), replace=False)
        
        if self.range_lon[1] < self.range_lon[0]:
            imerg_lons_based_on_era5 = np.random.choice( np.arange(0, self.range_lon[1]+self.era5_num_lons-self.n_size[2]-self.range_lon[0]), size=(self.num_samples,),replace=False)
        else:
            imerg_lons_based_on_era5 = np.random.choice( np.arange(0, self.range_lon[1]-self.n_size[2]-self.range_lon[0]), size=(self.num_samples,),replace=False)

        era5_selected_lats = era5_lats[imerg_lats_based_on_era5]
        era5_selected_lons = era5_lons[imerg_lons_based_on_era5]

        self.idxs_perm_era5 = np.stack( [era5_selected_lats,era5_selected_lons], axis=1)
        self.idxs_perm_imerg = np.stack( [imerg_lats_based_on_era5, imerg_lons_based_on_era5], axis=1)

    def __iter__(self):
        
        if self.with_shuffle:
            self.shuffle()
        
        lats, lons = self.era5_lats, self.era5_lons
        n_size = self.n_size
        res = self.era5_res

        iter_start , iter_end = self.worker_workset()

        for bidx in range( iter_start, iter_end):

            sources, token_infos = [[] for _ in self.input_fields], [[] for _ in self.input_fields]
            targets, target_token_infos = [[] for _ in self.target_fields], [[] for _ in self.target_fields],
            sources_infos, source_idxs = [], []
            target_infos, target_idxs = [], []

            i_bidx = self.idx_perm_era5_t[bidx]
            i_bidx_imerg = self.idx_perm_imerg_t[bidx]

            idxs_t_era5 = [i_bidx - n_size[0], i_bidx]
            idxs_t_imerg = [i_bidx_imerg - n_size[0], i_bidx_imerg]

            data_t = self.era5_ds['time'][idxs_t_era5[0]:idxs_t_era5[1]].astype(datetime)

            data_era5_tt_sfc = self.era5_ds['data_sfc'][idxs_t_era5[0]:idxs_t_era5[1]]
            data_era5_tt = self.era5_ds['data'][idxs_t_era5[0]:idxs_t_era5[1]]

            data_imerg_tt_sfc = self.imerg_ds['data_sfc'][idxs_t_imerg[0]:idxs_t_imerg[1]]

            for sidx in range(self.batch_size):

                idx_era5 = self.idxs_perm_era5[bidx*self.batch_size+sidx]

                idx_lat_era5 = [idx_era5[0], idx_era5[0] + n_size[1]]
                idx_lon_era5 = [idx_era5[1], idx_era5[1] + n_size[2]]

                lon_list = None

                if idx_lon_era5[0] > self.era5_num_lons - n_size[2] and idx_lon_era5[0] < self.era5_num_lons:
                    lon_list = np.arange(idx_lon_era5[0], self.era5_num_lons)
                    lon_list = np.concatenate((lon_list,np.arange(0,idx_lon_era5[1]%self.era5_num_lons)))

                if idx_lon_era5[0] > self.era5_num_lons:
                    idx_lon_era5[0] = idx_lon_era5[0]%self.era5_num_lons
                    idx_lon_era5[1] = idx_lon_era5[1]%self.era5_num_lons
            
                sources_infos += [ [ self.era5_ds['time'][idxs_t_era5[0]:idxs_t_era5[1]].astype(datetime),
                                     self.era5_ds['lats'][idx_lat_era5[0]:idx_lat_era5[1]],
                                     self.era5_ds['lons'][np.arange(idx_lon_era5[0],idx_lon_era5[1]) if lon_list is None else lon_list],
                                     self.era5_res ] ]
                
                if self.with_source_idxs :
                    source_idxs += [ (np.arange(idxs_t_era5[0],idxs_t_era5[1]),
                                      np.arange(idx_lat_era5[0],idx_lat_era5[1]),
                                      np.arange(idx_lon_era5[0],idx_lon_era5[1]) if lon_list is None else np.array(lon_list) ) ]
                

                idx_imerg = self.idxs_perm_imerg[bidx*self.batch_size+sidx]

                idx_lat_imerg = [idx_imerg[0]*self.downscaling_ratio, (idx_imerg[0] + n_size[1])*self.downscaling_ratio]
                idx_lon_imerg = [idx_imerg[1]*self.downscaling_ratio, (idx_imerg[1] + n_size[2])*self.downscaling_ratio]
                target_infos += [ [ self.imerg_ds['time'][idxs_t_imerg[0]:idxs_t_imerg[1]].astype(datetime),
                                    self.imerg_ds['lats'][idx_lat_imerg[0]:idx_lat_imerg[1]],
                                    self.imerg_ds['lons'][idx_lon_imerg[0]:idx_lon_imerg[1]],
                                    self.imerg_res ] ]

                if self.with_target_idxs :
                    target_idxs += [ (np.arange(idxs_t_imerg[0],idxs_t_imerg[1]),
                                      np.arange(idx_lat_imerg[0],idx_lat_imerg[1]),
                                      np.arange(idx_lon_imerg[0],idx_lon_imerg[1]) ) ]
                                      

                for ifield, field_info in enumerate(self.input_fields):
                    source_lvl, tok_info_lvl = [], []
                    tok_size = field_info[4]
                    num_tokens = field_info[3]

                    corr_type = 'global' if len(field_info) <=6 else field_info[6]

                    for ilevel, vl in enumerate(field_info[2]):
                        if vl == 0:
                            field_idx = self.era5_ds.attrs['fields_sfc'].index( field_info[0])
                            data_t = data_era5_tt_sfc[:,field_idx]
                        else:
                            field_idx = self.era5_ds.attrs['fields'].index( field_info[0])
                            vl_idx = self.era5_ds.attrs['levels'].index( vl)
                            data_t = data_era5_tt[:,field_idx, vl_idx]
                        
                        source_data, tok_info = [], []
                        
                        cdata = data_t[ :, 
                                 idx_lat_era5[0]:idx_lat_era5[1], 
                                 idx_lon_era5[0]:idx_lon_era5[1]] 

                        
                        normalizer = self.input_normalizers[ifield][ilevel]
                        
                        if corr_type != 'global':
                            normalizer = normalizer[ : , : , 
                                   idx_lat_era5[0]:idx_lat_era5[1] ,
                                   idx_lon_era5[0]:idx_lon_era5[1]]
                        
                        cdata = normalize(cdata, normalizer, sources_infos[-1][0], year_base = self.year_base)

                        source_data = tokenize( torch.from_numpy( cdata), tok_size)

                        dates = self.era5_ds['time'][ idxs_t_era5[0]:idxs_t_era5[1]].astype(datetime)
                        cdates = dates[tok_size[0]-1::tok_size[0]]

                        dates = [(d.year, d.timetuple().tm_yday-1, d.hour) for d in cdates]
                        lats_sidx = self.era5_ds['lats'][idx_lat_era5[0]: idx_lat_era5[1]][tok_size[1]//2 :: tok_size[1] ]
                        if lon_list is None:
                            lons_sidx = self.era5_ds['lons'][idx_lon_era5[0]: idx_lon_era5[1]][tok_size[2]//2 :: tok_size[2] ]
                        else:
                            lons_idx = self.era5_ds['lons'][lon_list][tok_size[2]//2 :: tok_size[2]]

                        tok_info += [[[[[ year, day, hour, vl, lat, lon, vl, self.input_res[0]] for lon in lons_sidx]
                                                                              for lat in lats_sidx]
                                                                  for (year, day, hour) in dates]]                                                       
                        source_lvl += [ source_data ]
                        
                        tok_info_lvl += [torch.tensor( tok_info, dtype=torch.float32).flatten( 1,-2)]

                    sources[ifield] += [ torch.stack(source_lvl, 0) ]
                    token_infos[ifield] += [ torch.stack(tok_info_lvl, 0)]
                
                for ifield, field_info in enumerte(self.target_fields):
                    target_lvl, target_tok_info_lvl = [], []
                    target_tok_size = field_info[4]
                    target_num_tokens = field_info[3]

                    for ilevel, vl in enumerate(field_info[2]):
                        if vl == 0:
                            field_idx = self.imerg_ds.attrs['fields_sfc'].index( field_info[0])
                            data_t_imerg = data_imerg_tt_sfc[:,field_idx]
                        else:
                            field_idx = self.imerg_ds.attrs['fields'].index( field_info[0])
                            vl_idx = self.imerg_ds.attrs['levels'].index( vl)
                            data_t_imerg = data_imerg_tt[:,field_idx, vl_idx]

                        target_data, target_tok_info = [], []

                        cdata_imerg = data_t_imerg[ :, 
                                        idx_lat_imerg[0]:idx_lat_imerg[1],
                                        idx_lon_imerg[0]:idx_lon_imerg[1] ]
                        
                        normalizer = self.target_normalizers[ifield][ilevel]
                        
                        if corr_type != 'global':
                            normalizer = normalizer[ : , : , 
                                   idx_lat_era5[0]:idx_lat_era5[1] ,
                                   idx_lon_era5[0]:idx_lon_era5[1]]

                        cdata_imerg = normalize(cdata_imerg, normalizer, target_infos[-1][0], year_base = self.year_base)
                        
                        target_data = tokenize( torch.from_numpy( cdata_imerg), target_tok_size)
                        ##need to fill based on how the positional encoding needs to be done for output_latent_arrays
                        target_lvl += [ target_data ]
                        target_tok_info_lvl += [np.array([0])]

                    targets[ifield] += [torch.stack(target_lvl, 0)]
                    target_token_infos[ifield] += [ torch.stack(target_tok_info_lvl, 0)]

            sources = [torch.stack(sources_field).transpose(1,0) for sources_field in sources]
            token_infos = [torch.stack(tis_field).transpose(1,0) for tis_field in token_infos]

            targets = [torch.stack(targets_field).transpose(1,0) for targets_field in targets]
            target_token_infos = [torch.stack(target_tis_field).transpose(1,0) for target_tis_field in target_token_infos]

            logger.info("len(sources)", len(sources))
            logger.info("len(token_infos)", len(token_infos))

            logger.info("len(sources[0])", sources[0].shape)
            logger.info("len(token_infos[0])", token_infos[0].shape)

            logger.info("len(targets)",len(targets))
            logger.info("len(target_token_infos)", len(target_token_infos))

            logger.info("len(targets[0])", targets[0].shape)
            logger.info("len(target_token_infos[0]", target_token_infos[0].shape)
            
            yield ((sources, token_infos), (source_idxs, sources_infos), (target, target_token_infos))
  
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
