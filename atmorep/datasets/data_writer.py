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
def write_forecast(user_config: config.UserConfig, model_id, epoch, batch_idx, levels, sources, targets, preds, ensembles, coords, zarr_store_type = 'ZipStore' ) :
  ''' 
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''
  sources_coords = [[c[:3] for c in coord_field ] for coord_field in coords]
  targets_coords = [[[c[-1], c[1], c[2]] for c in coord_field ] for coord_field in coords]
  fname_template = f"results_id{model_id}_epoch{epoch:05d}"+r"_{}.zarr"
  dirname = user_config.results / f"id{model_id}"

  zarr_store = getattr( zarr, zarr_store_type)

  store_source = zarr_store( dirname / fname_template.format( 'source'))
  exp_source = zarr.group(store=store_source)
 
  for fidx, field in enumerate(sources) :
    ds_field = exp_source.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels, sources_coords[fidx][bidx])
  store_source.close()

  store_target = zarr_store( dirname / fname_template.format( 'target'))
  exp_target = zarr.group(store=store_target)
  for fidx, field in enumerate(targets) :
    ds_field = exp_target.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx])
  store_target.close()

  store_pred = zarr_store( dirname / fname_template.format( 'pred'))
  exp_pred = zarr.group(store=store_pred)
  for fidx, field in enumerate(preds) :
    ds_field = exp_pred.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx]) 
  store_pred.close()

  store_ens = zarr_store( dirname / fname_template.format( 'ens'))
  exp_ens = zarr.group(store=store_ens)
  for fidx, field in enumerate(ensembles) :
    ds_field = exp_ens.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels, targets_coords[fidx][bidx])
  store_ens.close()

####################################################################################################
def write_BERT(user_config: config.UserConfig, model_id, epoch, batch_idx, levels, sources, 
                targets, preds, ensembles, coords, 
                zarr_store_type = 'ZipStore' ) : 
                                            
  '''
    sources : num_fields x [field name , data]
    targets :
    preds, ensemble share coords with targets
  '''

  sources_coords = [[c[:3] for c in coord_field ] for coord_field in coords]
  targets_coords = [[c[3:] for c in coord_field ] for coord_field in coords]
  
  fname_template = f"results_id{model_id}_epoch{epoch:05d}"+r"_{}.zarr"
  dirname = user_config.results / f"id{model_id}"

  zarr_store = getattr( zarr, zarr_store_type)

  store_source = zarr_store(dirname / fname_template.format( 'source'))
  exp_source = zarr.group(store=store_source)
  for fidx, field in enumerate(sources) :
    ds_field = exp_source.require_group( f'{field[0]}')
    batch_size = field[1].shape[0]
    for bidx in range( field[1].shape[0]) :
      sample = batch_idx * batch_size + bidx
      write_item(ds_field, sample, field[1][bidx], levels[fidx], sources_coords[fidx][bidx] )
  store_source.close()

  store_target = zarr_store( dirname / fname_template.format( 'target'))
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
        write_item(ds_target_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], targets_coords[fidx][bidx][vidx], name = 'ml' )
  store_target.close()

  store_pred = zarr_store( dirname / fname_template.format( 'pred'))
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
        write_item(ds_pred_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], 
                                  targets_coords[fidx][bidx][vidx], name = 'ml' )
  store_pred.close()

  store_ens = zarr_store( dirname / fname_template.format( 'ens'))
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
        write_item(ds_ens_b, levels[fidx][vidx], field[1][vidx][bidx], levels[fidx][vidx], 
                                targets_coords[fidx][bidx][vidx], name = 'ml' )
  store_ens.close()

####################################################################################################
def write_attention(user_config: config.UserConfig, model_id, epoch, batch_idx, levels, attn, coords, zarr_store_type = 'ZipStore' ) :

  fname = user_config.results / f"id{model_id}" / f"results_id{model_id}_epoch{epoch:05d}_attention.zarr"
  
  zarr_store = getattr( zarr, zarr_store_type)

  store_attn = zarr_store(fname)
  exp_attn = zarr.group(store=store_attn)

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
  store_attn.close()
