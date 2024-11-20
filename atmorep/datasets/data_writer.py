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

def write_item(ds_field, name_idx, data, levels, coords, name  = 'sample' ):
  ds_batch_item = ds_field.create_group( f'{name}={name_idx:05d}' )
  ds_batch_item.create_dataset( 'data', data=data)
  ds_batch_item.create_dataset( 'ml', data=levels)
  ds_batch_item.create_dataset( 'datetime', data=coords[0].astype('datetime64[ns]'))
  ds_batch_item.create_dataset( 'lat', data=np.array(coords[1]).astype(np.float32))
  ds_batch_item.create_dataset( 'lon', data=np.array(coords[2]).astype(np.float32))
  return ds_batch_item

####################################################################################################
# Abstracted helper functions to reduce dependency

def process_zarr_store(store_type, batch_idx,  fname, group_func, *args):
  '''Handle Zarr store creation and operation.'''
  zarr_store = getattr(zarr, store_type)(fname)
  group_func(zarr_store, batch_idx, *args)
  zarr_store.close()

def write_group(store, batch_idx, fields, levels, coords):
  '''Generic function to write a Zarr group'''
  exp_group = zarr.group(store=store)
  for fidx, field in enumerate(fields):
    ds_field = exp_group.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range(batch_size):
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels, coords[fidx][bidx])

def write_nested_group(store, batch_idx, fields, levels, coords):
  '''Generic function for writing nested datasets (e.g., targets, preds)'''
  exp_group = zarr.group(store=store)
  for fidx, field in enumerate(fields):
    if not field[1]:  # Skip if no data
      continue
    batch_size = len(field[1][0])
    ds_field = exp_group.require_group( f'{field[0]}')

    for bidx in range(batch_size):
      sample = batch_idx * batch_size + bidx
      ds_field_sample = ds_field.create_group(f'sample={sample:05d}')
      for vidx in range(len(levels[fidx])) :
        write_item(ds_field_sample, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], coords[fidx][bidx][vidx], name = 'ml' )

def write_attention_group(store, batch_idx, attn, levels, coords):
  '''Generic function for writing attention data into Zarr groups.'''
  exp_attn = zarr.group(store=store)
  for fidx, atts_f in enumerate(attn) :
      ds_field = exp_attn.require_group( f'{atts_f[0]}')
      ds_field_b = ds_field.require_group( f'batch={batch_idx:05d}')
      for lidx, atts_f_l in enumerate(atts_f[1]) :  # layer in the network
        ds_f_l = ds_field_b.require_group( f'layer={lidx:05d}')
        ds_f_l.create_dataset( 'ml', data=levels[fidx])
        ds_f_l.create_dataset( 'datetime', data=coords[0][fidx])
        ds_f_l.create_dataset( 'lat', data=coords[1][fidx])
        ds_f_l.create_dataset( 'lon', data=coords[2][fidx])
        ds_f_l_h = ds_f_l.require_group('heads')
        for hidx, atts_f_l_head in enumerate(atts_f_l) :  # number of attention head
          if atts_f_l_head != None :
            ds_f_l_h.create_dataset(f'{hidx}', data=atts_f_l_head.numpy() )

####################################################################################################
def write_forecast( model_id, epoch, batch_idx, levels, sources, 
                    targets, preds, ensembles, coords,  
                    zarr_store_type = 'ZipStore' ) :
  ''' 
    Refactored Write_Forecast
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''
  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'
  
  sources_coords = [[c[:3] for c in coord_field ] for coord_field in coords]
  targets_coords = [[[c[-1], c[1], c[2]] for c in coord_field ] for coord_field in coords]
  
  # Process each dataset type
  for dataset_type, fields, field_levels, field_coords in zip(
    ['source', 'target', 'pred', 'ens'],
    [sources, targets, preds, ensembles],
    [levels] * 4,
    [sources_coords, targets_coords, targets_coords, targets_coords]
  ):
    if not fields:  # Skip if fields are empty
      continue
    process_zarr_store(
      zarr_store_type,
      batch_idx,
      fname.format(dataset_type),
      write_group,
      fields,
      field_levels,
      field_coords
    )

####################################################################################################
def write_BERT( model_id, epoch, batch_idx, levels, sources, 
                targets, preds, ensembles, coords, 
                zarr_store_type = 'ZipStore' ) : 
                                            
  '''
    Refactored Write_BERT
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''
  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'

  sources_coords = [[c[:3] for c in coord_field ] for coord_field in coords]
  targets_coords = [[c[3:] for c in coord_field ] for coord_field in coords]

  # Process each dataset type
  for dataset_type, fields, field_coords, write_func in zip(
    ['source', 'target', 'pred', 'ens'],
    [sources, targets, preds, ensembles],
    [sources_coords, targets_coords, targets_coords, targets_coords],
    [write_group, write_nested_group, write_nested_group, write_nested_group]  # Function selector
  ):
    if not fields:  # Skip if fields are empty
      continue
    process_zarr_store(
      zarr_store_type,
      batch_idx,
      fname.format(dataset_type),
      write_func,  
      fields,
      levels,
      field_coords
    )

####################################################################################################
def write_attention(model_id, epoch, batch_idx, levels, attn, coords, zarr_store_type = 'ZipStore' ) :

  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'
  # Process the attention data
  process_zarr_store(
    zarr_store_type,
    batch_idx,
    fname.format(fname),
    write_attention_group, 
    attn, 
    levels, 
    coords)
