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
from atmorep.applications.downscaling.utils.data_quality_check import lat_lon_time_range
from atmorep.utils.logger import logger


class MultifieldDownscalingSampler( torch.utils.data.IterableDataset):
    
    def __init__(self, input_file_path, target_file_path, input_fields,
        target_fields, years, batch_size, n_size, num_samples, with_shuffle=False,
        with_source_idxs = False) :
            super(MultifieldDownscalingSampler, self).__init__()

            self.input_fields = input_fields
            self.target_fields = target_fields
            self.n_size = n_size
            self.num_samples = num_samples
            self.with_source_idxs = with_source_idxs
            self.batch_size = batch_size
            self.with_shuffle = with_shuffle
    

            assert os.path.exists(input_file_path), f"Input file path {input_file_path} does not exist"
            assert os.path.exists(target_file_path), f"Target file path {target_file_path} does not exist"

            self.input_ds = zarr.group(input_file_path)
            self.target_ds = zarr.group(target_file_path)

            self.input_res = self.input_ds.attrs['res']
            self.target_res = self.target_ds.attrs['res']

            self.lats = np.array(self.input_ds['lats'])
            self.lons = np.array(self.input_ds['lons'])
            
            self.range_lat = np.array([0,-1])
            self.range_lon = np.array([0,-1])
            self.range_time = np.array([0,-1])

            if self.input_ds.attrs['is_global']:

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
            
            self.year_base = self.input_ds['time'][self.range_time[0]].astype(datetime).year
            self.range_lat += np.array([0,-n_size[1]])
            self.range_lon +=  np.array([0,-n_size[2]])
            #self.range_time += np.array([0,-n_size[0]])

            self.input_normalizers = []

            for ifield, field_info in enumerate(input_fields):
                corr_type = 'global' if len(field_info) <=6 else field_info[6]
                nf_name = 'global_norm' if corr_type == 'global' else 'norm'
                self.input_normalizers.append([])
                for vl in field_info[2]: 
                    if vl == 0:
                        field_idx = self.input_ds.attrs['fields_sfc'].index( field_info[0])
                        n_name = f'normalization/{nf_name}_sfc'
                        self.input_normalizers[ifield] += [self.input_ds[n_name].oindex[ :, :, field_idx]] 
                    else:
                        vl_idx = self.input_ds.attrs['levels'].index(vl)
                        field_idx = self.input_ds.attrs['fields'].index( field_info[0])
                        n_name = f'normalization/{nf_name}'
                        self.input_normalizers[ifield] += [self.input_ds[n_name].oindex[ :, :, field_idx, vl_idx] ]


            self.target_normalizers = []

            for ifield, field_info in enumerate(target_fields):
                corr_type = 'global' if len(field_info) <=6 else field_info[6]
                nf_name = 'global_norm' if corr_type == 'global' else 'norm'
                self.target_normalizers.append([])
                for vl in field_info[2]: 
                    if vl == 0:
                        field_idx = self.target_ds.attrs['fields_sfc'].index( field_info[0])
                        n_name = f'normalization/{nf_name}_sfc'
                        self.target_normalizers[ifield] += [self.target_ds[n_name].oindex[ :, :, field_idx]] 
                    else:
                        vl_idx = self.target_ds.attrs['levels'].index(vl)
                        field_idx = self.target_ds.attrs['fields'].index( field_info[0])
                        n_name = f'normalization/{nf_name}'
                        self.target_normalizers[ifield] += [self.target_ds[n_name].oindex[ :, :, field_idx, vl_idx] ]


            years_input_file = np.asarray(
                pd.DataFrame(
                    {'time' : pd.to_datetime(self.input_ds['time'][self.range_time[0]:self.range_time[1]])})['time'].dt.strftime("%Y"))

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
        self.idx_perm_t = rng.permutation( 
            self.valid_time_indices[self.n_size[0]:])[: self.num_samples // self.batch_size]
        
        #lats = np.random.choice( self.input_ds['lats'][
        #                    self.range_lat[0]:self.range_lat[1]], size=(self.num_samples,))

        lats = np.random.choice( np.arange(self.range_lat[0],self.range_lat[1]), size=(self.num_samples,))

        #lons = np.random.choice( self.input_ds['lons'][
        #                    self.range_lon[0]:self.range_lon[1]], size=(self.num_samples,))

        lons = np.random.choice( np.arange(self.range_lon[0], self.range_lon[1]), size=(self.num_samples,))

        self.idxs_perm = np.stack( [lats,lons], axis=1)

    def __iter__(self):
        
        if self.with_shuffle:
            self.shuffle()
        
        lats, lons = self.lats, self.lons
        n_size = self.n_size
        res = self.input_res

        iter_start , iter_end = self.worker_workset()

        for bidx in range( iter_start, iter_end):

            sources, token_infos = [[] for _ in self.input_fields], [[] for _ in self.input_fields]
            sources_infos, source_idxs = [], []

            i_bidx = self.idx_perm_t[bidx]

            idxs_t = [i_bidx - n_size[0], i_bidx]

            for sidx in range(self.batch_size):

                idx = self.idxs_perm[bidx*self.batch_size+sidx]

                idxs_lat = [idx[0], idx[0] + n_size[1]]
                idxs_lon = [idx[1], idx[1] + n_size[2]]
            
                sources_infos += [ [ self.input_ds['time'][idxs_t[0]:idxs_t[1]].astype(datetime),
                                     self.input_ds['lats'][idxs_lat[0]:idxs_lat[1]],
                                     self.input_ds['lons'][idxs_lon[0]:idxs_lon[1]],
                                     self.input_res ] ]

                if self.with_source_idxs :
                    source_idxs += [ (np.arange(idxs_t[0],idxs_t[1]),
                                      np.arange(idxs_lat[0],idxs_lat[1]),
                                      np.arange(idxs_lon[0],idxs_lon[1]) ) ]

                for ifield, field_info in enumerate(self.input_fields):
                    source_lvl, tok_info_lvl = [], []
                    tok_size = field_info[4]
                    num_tokens = field_info[3]

                    corr_type = 'global' if len(field_info) <=6 else field_info[6]

                    for ilevel, vl in enumerate(field_info[2]):
                        if vl == 0:
                            field_idx = self.input_ds.attrs['fields_sfc'].index( field_info[0])
                            data_t = self.input_ds['data_sfc'][idxs_t[0]:idxs_t[1],field_idx]
                        else:
                            field_idx = self.input_ds.attrs['fields'].index( field_info[0])
                            vl_idx = self.input_ds.attrs['levels'].index( vl)
                            data_t = self.input_ds['data'][idxs_t[0]:idxs_t[1],field_idx, vl_idx]
                        
                        source_data, tok_info = [], []
                        
                        cdata = data_t[ :, 
                                 idxs_lat[0]:idxs_lat[1], 
                                 idxs_lon[0]:idxs_lon[1]] 

                        
                        normalizer = self.input_normalizers[ifield][ilevel]
                        
                        if corr_type != 'global':
                            normalizer = normalizer[ : , : , 
                                   idxs_lat[0]:idxs_lat[1] ,
                                   idxs_lon[0]:idxs_lon[1]]
                        
                        cdata = normalize(cdata, normalizer, sources_infos[-1][0], year_base = self.year_base)

                        source_data = tokenize( torch.from_numpy( cdata), tok_size)

                        dates = self.input_ds['time'][ idxs_t[0]:idxs_t[1]].astype(datetime)
                        cdates = dates[tok_size[0]-1::tok_size[0]]

                        dates = [(d.year, d.timetuple().tm_yday-1, d.hour) for d in cdates]
                        lats_sidx = self.input_ds['lats'][idxs_lat[0]: idxs_lat[1]][tok_size[1]//2 :: tok_size[1] ]
                        lons_sidx = self.input_ds['lons'][idxs_lon[0]: idxs_lon[1]][tok_size[2]//2 :: tok_size[2] ]

                        tok_info += [[[[[ year, day, hour, vl, lat, lon, vl, self.input_res[0]] for lon in lons_sidx]
                                                                              for lat in lats_sidx]
                                                                  for (year, day, hour) in dates]]                                                       
                        source_lvl += [ source_data ]
                        
                        tok_info_lvl += [torch.tensor( tok_info, dtype=torch.float32).flatten( 1,-2)]

                    sources[ifield] += [ torch.stack(source_lvl, 0) ]
                    token_infos[ifield] += [ torch.stack(tok_info_lvl, 0)]
                
            sources = [torch.stack(sources_field).transpose(1,0) for sources_field in sources]
            token_infos = [torch.stack(tis_field).transpose(1,0) for tis_field in token_infos]

            logger.info("len(sources)", len(sources))
            logger.info("len(token_infos)", len(token_infos))

            logger.info("len(sources[0])", sources[0].shape)
            logger.info("len(token_infos[0])", token_infos[0].shape)
            
            yield ((sources, token_infos), (source_idxs, sources_infos))

        
    
    def __len__(self):
        return self.num_samples // self.batch_size


    def find_field_indices(self,zarr_group,file_path,fields):

        idx_dict  = {}
        
        field_attrs = zarr_group.attrs['fields']
        surface_attrs = zarr.group.attrs['fields_sfc']
        for fidx,field in enumerate(fields):
            if field[0] in field_attrs:
                idx_dict[field[0]] = ('data',field_attrs.index(field[0]))
            elif field[0] in surface_attrs:
                idx_dict[field[0]] = ('data_sfc',surface_attrs.index(field[0]))
            else:
                assert False, f"{field[0]} in not there in zarr_file {file_path}"
        
        return idx_dict



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
