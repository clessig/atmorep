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
import xarray as xr
import zarr
import atmorep.config.config as config
from atmorep.datasets.data_writer import write_item as write_item

def write_downscale( model_id, epoch, batch_idx, source_levels, target_levels,
                        sources, targets, preds, ensembles, source_coords, target_coords,
                        zarr_store_type = 'ZipStore') :
    ''' 
      sources : num_fields x [field name , data]
      targets :
      preds, ensemble share coords with targets
    '''
    sources_coords = [[ *coord_field ] for coord_field in source_coords]
    targets_coords = [[ *coord_field ] for coord_field in target_coords]
    fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'

    zarr_store = getattr( zarr, zarr_store_type)

    store_source = zarr_store( fname.format( 'source'))
    exp_source = zarr.group(store=store_source)

    for fidx, field in enumerate(sources):
        ds_field = exp_source.require_group( f'{field[0]}')
        batch_size = field[1].shape[0]
        for bidx in range( batch_size):
            sample = batch_idx* batch_size + bidx
            write_item(ds_field, sample, field[1][bidx], source_levels[fidx], sources_coords[fidx][bidx])
    store_source.close()

    store_target = zarr_store( fname.format( 'target'))
    exp_target = zarr.group(store=store_target)
    for fidx, field in enumerate(targets):
        ds_field = exp_target.require_group( f'{field[0]}')
        batch_size = field[1].shape[0]
        for bidx in range(batch_size):
            sample = batch_idx * batch_size + bidx
            write_item(ds_field, sample, field[1][bidx], target_levels[fidx], targets_coords[fidx][bidx])
    store_target.close()

    store_pred = zarr_store( fname.format( 'pred'))
    exp_pred = zarr.group(store=store_pred)
    for fidx,field in enumerate(preds) :
        ds_field = exp_pred.require_group( f'{field[0]}')
        batch_size = field[1].shape[0]
        for bidx in range(batch_size):
            sample = batch_idx * batch_size + bidx
            write_item(ds_field, sample, field[1][bidx], target_levels[fidx], targets_coords[fidx][bidx])
    store_pred.close()


    store_ens = zarr_store( fname.format( 'ens'))
    exp_ens = zarr.group(store=store_ens)
    for fidx, field in enumerate(ensembles):
        ds_field = exp_ens.require_group( f'{field[0]}')
        batch_size = field[1].shape[0]
        for bidx in range(field[1].shape[0]):
            sample = batch_idx * batch_size + bidx
            write_item( ds_field, sample, field[1][bidx], target_levels[fidx], targets_coords[fidx][bidx])
    store_ens.close()

    





