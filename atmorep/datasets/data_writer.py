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
import code
import datetime
import atmorep.config.config as config

def write_item(ds_field, name_idx, data, levels, coords, name  = 'sample' ):
  ds_batch_item = ds_field.create_group( f'{name}={name_idx:05d}' )
  ds_batch_item.create_dataset( 'data', data=data)
  ds_batch_item.create_dataset( 'ml', data=levels)
  ds_batch_item.create_dataset( 'datetime', data=coords[0].astype(np.datetime64))
  ds_batch_item.create_dataset( 'lat', data=coords[1].astype(np.float32))
  ds_batch_item.create_dataset( 'lon', data=coords[2].astype(np.float32))
  return ds_batch_item

####################################################################################################
def write_forecast( model_id, epoch, batch_idx, levels, sources, sources_coords,
                                                targets, targets_coords,
                                                preds, ensembles,
                                                zarr_store_type = 'ZipStore' ) :
  ''' 
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''

  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'

  zarr_store = getattr( zarr, zarr_store_type)

  store_source = zarr_store( fname.format( 'source'))
  exp_source = zarr.group(store=store_source)
 
  for fidx, field in enumerate(sources) :
    ds_field = exp_source.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      ds_batch_item = write_item(ds_field, sample, field[1][bidx], levels, sources_coords[fidx][bidx]) #[t[bidx] for t in sources_coords] )
      # ds_batch_item = ds_field.create_group( f'sample={sample:05d}' )
      # ds_batch_item.create_dataset( 'data', data=field[1][bidx])
      # ds_batch_item.create_dataset( 'ml', data=levels)
      # ds_batch_item.create_dataset( 'datetime', data=sources_coords[0][bidx])
      # ds_batch_item.create_dataset( 'lat', data=sources_coords[1][bidx])
      # ds_batch_item.create_dataset( 'lon', data=sources_coords[2][bidx])
  store_source.close()

  store_target = zarr_store( fname.format( 'target'))
  exp_target = zarr.group(store=store_target)
  for fidx, field in enumerate(targets) :
    ds_field = exp_target.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      ds_batch_item = write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx]) #[t[bidx] for t in targets_coords] )
      # ds_batch_item = ds_field.create_group( f'sample={sample:05d}' )
      # ds_batch_item.create_dataset( 'data', data=field[1][bidx])
      # ds_batch_item.create_dataset( 'ml', data=levels)
      # ds_batch_item.create_dataset( 'datetime', data=targets_coords[0][bidx])
      # ds_batch_item.create_dataset( 'lat', data=targets_coords[1][bidx])
      # ds_batch_item.create_dataset( 'lon', data=targets_coords[2][bidx])
  store_target.close()

  store_pred = zarr_store( fname.format( 'pred'))
  exp_pred = zarr.group(store=store_pred)
  for fidx, field in enumerate(preds) :
    ds_field = exp_pred.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      ds_batch_item = write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx]) #[t[bidx] for t in targets_coords] )
      # ds_batch_item = ds_field.create_group( f'sample={sample:05d}' )
      # ds_batch_item.create_dataset( 'data', data=field[1][bidx])
      # ds_batch_item.create_dataset( 'ml', data=levels)
      # ds_batch_item.create_dataset( 'datetime', data=targets_coords[0][bidx])
      # ds_batch_item.create_dataset( 'lat', data=targets_coords[1][bidx])
      # ds_batch_item.create_dataset( 'lon', data=targets_coords[2][bidx])
  store_pred.close()

  store_ens = zarr_store( fname.format( 'ens'))
  exp_ens = zarr.group(store=store_ens)
  for fidx, field in enumerate(ensembles) :
    ds_field = exp_ens.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      ds_batch_item = write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx]) # [t[bidx] for t in targets_coords] )
      # ds_batch_item = ds_field.create_group( f'sample={sample:05d}' )
      # ds_batch_item.create_dataset( 'data', data=field[1][bidx])
      # ds_batch_item.create_dataset( 'ml', data=levels)
      # ds_batch_item.create_dataset( 'datetime', data=targets_coords[0][bidx])
      # ds_batch_item.create_dataset( 'lat', data=targets_coords[1][bidx])
      # ds_batch_item.create_dataset( 'lon', data=targets_coords[2][bidx])
  store_ens.close()

####################################################################################################
def write_BERT( model_id, epoch, batch_idx, levels, sources, #sources_coords,
                                            targets, # targets_coords,
                                            preds, ensembles, coords, 
                                            zarr_store_type = 'ZipStore' ) :
  '''
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''

  # breakpoint()
  sources_coords = [[c[:3] for c in coord_field ] for coord_field in coords]
  targets_coords = [[c[3:] for c in coord_field ] for coord_field in coords]
  
  # fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch}.zarr'
  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'

  zarr_store = getattr( zarr, zarr_store_type)

  store_source = zarr_store( fname.format( 'source'))
  exp_source = zarr.group(store=store_source)
  for fidx, field in enumerate(sources) :
    ds_field = exp_source.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      ds_batch_item = write_item(ds_field, sample, field[1][bidx], levels[fidx], sources_coords[fidx][bidx] )
      # ds_batch_item = ds_field.create_group( f'sample={sample:05d}' )
      # ds_batch_item.create_dataset( 'data', data=field[1][bidx])
      # ds_batch_item.create_dataset( 'ml', data=levels[fidx])
      # ds_batch_item.create_dataset( 'datetime', data=sources_coords[fidx][bidx][0])
      # ds_batch_item.create_dataset( 'lat', data=sources_coords[fidx][bidx][1])
      # ds_batch_item.create_dataset( 'lon', data=sources_coords[fidx][bidx][2])
  store_source.close()

  store_target = zarr_store( fname.format( 'target'))
  exp_target = zarr.group(store=store_target)
  for fidx, field in enumerate(targets) :
    if 0 == len(field[1]) :  # skip fields that were not predicted
        continue
    batch_size = len(field[1][0])
    ds_field = exp_target.require_group( f'{field[0]}')
    for bidx in range( len(field[1][0])) :
      sample = batch_idx * batch_size + bidx
      ds_target_b = ds_field.create_group( f'sample={sample:05d}')
      for vidx in range(len(levels[fidx])) :
        ds_target_b_l = write_item(ds_target_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], targets_coords[fidx][bidx][vidx], name = 'ml' )
        # ds_target_b_l = ds_target_b.require_group( f'ml={levels[fidx][vidx]}')
        # ds_target_b_l.create_dataset( 'data', data=field[1][vidx][bidx])
        # ds_target_b_l.create_dataset( 'ml', data=levels[fidx][vidx])
        # ds_target_b_l.create_dataset( 'datetime', data=targets_coords[fidx][bidx][0][vidx])
        # ds_target_b_l.create_dataset( 'lat', data=targets_coords[fidx][bidx][1][vidx])
        # ds_target_b_l.create_dataset( 'lon', data=targets_coords[fidx][bidx][2][vidx])
  store_target.close()

  store_pred = zarr_store( fname.format( 'pred'))
  exp_pred = zarr.group(store=store_pred)
  for fidx, field in enumerate(preds) :
    if 0 == len(field[1]) :  # skip fields that were not predicted
      continue
    batch_size = len(field[1][0])
    ds_pred = exp_pred.require_group( f'{field[0]}')
    for bidx in range( len(field[1][0])) :
      sample = batch_idx * batch_size + bidx
      ds_pred_b = ds_pred.create_group( f'sample={sample:05d}')
      for vidx in range(len(levels[fidx])) :
        ds_pred_b_l = write_item(ds_pred_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], 
                                  targets_coords[fidx][bidx][vidx], name = 'ml' )
        # ds_pred_b_l = ds_pred_b.create_group( f'ml={levels[fidx][vidx]}')
        # ds_pred_b_l.create_dataset( 'data', data=field[1][vidx][bidx])
        # ds_pred_b_l.create_dataset( 'ml', data=levels[fidx][vidx])
        # ds_pred_b_l.create_dataset( 'datetime', data=targets_coords[fidx][bidx][0][vidx])
        # ds_pred_b_l.create_dataset( 'lat', data=targets_coords[fidx][bidx][1][vidx])
        # ds_pred_b_l.create_dataset( 'lon', data=targets_coords[fidx][bidx][2][vidx])
  store_pred.close()

  store_ens = zarr_store( fname.format( 'ens'))
  exp_ens = zarr.group(store=store_ens)
  for fidx, field in enumerate(ensembles) :
    if 0 == len(field[1]) :  # skip fields that were not predicted
      continue
    batch_size = len(field[1][0])
    ds_ens = exp_ens.require_group( f'{field[0]}')
    for bidx in range( len(field[1][0])) :
      sample = batch_idx * batch_size + bidx
      ds_ens_b = ds_ens.create_group( f'sample={sample:05d}')
      for vidx in range(len(levels[fidx])) :
        ds_ens_b_l = write_item(ds_ens_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], 
                                targets_coords[fidx][bidx][vidx], name = 'ml' )

        # ds_ens_b_l = ds_ens_b.create_group( f'ml={levels[fidx][vidx]}')
        # ds_ens_b_l.create_dataset( 'data', data=field[1][vidx][bidx])
        # ds_ens_b_l.create_dataset( 'ml', data=levels[fidx][vidx])
        # ds_ens_b_l.create_dataset( 'datetime', data=targets_coords[fidx][bidx][0][vidx])
        # ds_ens_b_l.create_dataset( 'lat', data=targets_coords[fidx][bidx][1][vidx])
        # ds_ens_b_l.create_dataset( 'lon', data=targets_coords[fidx][bidx][2][vidx])
  store_ens.close()

####################################################################################################
def write_attention(model_id, epoch, batch_idx, levels, attn, attn_coords, zarr_store_type = 'ZipStore' ) :

  fname =  f'{config.path_results}/id{model_id}/results_id{model_id}_epoch{epoch:05d}' + '_{}.zarr'
  zarr_store = getattr( zarr, zarr_store_type)

  store_attn = zarr_store( fname.format( 'attention'))
  exp_attn = zarr.group(store=store_attn)

  for fidx, atts_f in enumerate(attn) :
    ds_field = exp_attn.require_group( f'{atts_f[0]}')
    ds_field_b = ds_field.require_group( f'batch={batch_idx:05d}')
    for lidx, atts_f_l in enumerate(atts_f[1]) :  # layer in the network
      ds_f_l = ds_field_b.require_group( f'layer={lidx:05d}')
      ds_f_l.create_dataset( 'ml', data=levels[fidx])
      ds_f_l.create_dataset( 'datetime', data=attn_coords[0][fidx])
      ds_f_l.create_dataset( 'lat', data=attn_coords[1][fidx])
      ds_f_l.create_dataset( 'lon', data=attn_coords[2][fidx])
      ds_f_l_h = ds_f_l.require_group('heads')
      for hidx, atts_f_l_head in enumerate(atts_f_l) :  # number of attention head
        if atts_f_l_head != None :
          ds_f_l_h.create_dataset(f'{hidx}', data=atts_f_l_head.numpy() )
  store_attn.close()
