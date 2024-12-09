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
from functools import partial
import code

####################################################################################################
def prepare_batch_BERT_multifield( cf, rngs, fields, BERT_strategy, fields_data, fields_infos) :
  
  fields_tokens_masked_idx_list = [[] for _ in fields_data]
  fields_targets = [[] for _ in fields_data]
  sources = [[] for _ in fields_data]
  token_infos = [[] for _ in fields_data]

  if not BERT_strategy :
    BERT_strategy = cf.BERT_strategy

  if BERT_strategy == 'BERT' :
    bert_f = prepare_batch_BERT_field
  elif BERT_strategy == 'global_forecast' :
    bert_f = prepare_batch_BERT_forecast_field
  elif BERT_strategy == 'forecast' :
    bert_f = prepare_batch_BERT_forecast_field
  elif BERT_strategy == 'temporal_interpolation' :
    bert_f = prepare_batch_BERT_temporal_field
  elif BERT_strategy == 'data_compression' :
    bert_f = prepare_batch_BERT_data_compression_field # prepare_batch_BERT_forecast_field
  else :
    assert False

  rng_idx = 1
  for ifield, (field, infos) in enumerate(zip(fields_data, fields_infos)) :
    for ilevel, (field_data, token_info) in enumerate(zip(field, infos)) :
     
      # no masking for static fields or if masking rate = 0
      if fields[ifield][1][0] > 0 and fields[ifield][5][0] > 0. :

        ret = bert_f( cf, ifield, field_data, token_info, rngs[rng_idx])
        (field_data, token_info, target, tokens_masked_idx_list) = ret
        
        if target is not None :
          fields_targets[ifield].append( target)
        fields_tokens_masked_idx_list[ifield].append( tokens_masked_idx_list)

      rng_idx += 1

      sources[ifield].append( field_data.unsqueeze(1) )
      token_infos[ifield].append( token_info )

    # merge along vertical level
    sources[ifield] = torch.cat( sources[ifield], 1)
    token_infos[ifield] = torch.cat( token_infos[ifield], 1)
    # merge along vertical level, for target we have level, batch, ... ordering 
    fields_targets[ifield] = torch.cat( fields_targets[ifield],0) \
                                if len(fields_targets[ifield]) > 0 else fields_targets[ifield]

  return (sources, token_infos, fields_targets, fields_tokens_masked_idx_list)

####################################################################################################
def prepare_batch_BERT_field( cf, ifield, source, token_info, rng) :

  # shortcuts
  mr = partial( torch.nn.functional.interpolate, mode='trilinear')
  fl = torch.flatten
  tr = torch.transpose
  sq = torch.squeeze
  usq = torch.unsqueeze 
  cnt_nz = torch.count_nonzero

  # collapse token dimensions 
  source_shape0 = source.shape
  source = torch.flatten( torch.flatten( source, 1, 3), 2, 4)
  
  # select random token in the selected space-time cube to be masked/deleted
  BERT_frac = cf.fields[ifield][5][0]
  BERT_frac_mask = cf.fields[ifield][5][1]
  BERT_frac_rndm = cf.fields[ifield][5][2]
  BERT_frac_mr = cf.fields[ifield][5][3]
  BERT_mr_max = 2
  token_size = cf.fields[ifield][4]
  batch_dim = source.shape[0]
  num_tokens = source.shape[1] 
 
  masking_ratios = rng.random( batch_dim) * BERT_frac
  # number of tokens masked per batch entry
  nums_masked = np.ceil( num_tokens * masking_ratios).astype(np.int64)
  tokens_masked_idx_list = [ torch.tensor(rng.permutation(num_tokens)[:nms]) for nms in nums_masked]
  
  # linear indices for masking
  tokens_masked_idx_list = [tokens_masked_idx_list[i] + num_tokens * i for i in range(batch_dim)]
  idx = torch.cat( tokens_masked_idx_list)

  # flatten along first two dimension to simplify linear indexing (which then requires an
  # easily computable row offset)
  source_shape = source.shape
  source = torch.flatten( source, 0, 1)
 
  # keep masked tokens for loss computation
  target = source[idx].clone()

  # climatological mean of normalized data
  global_mean = 0. * torch.mean(source, 0)
  global_std  = torch.std(source, 0)

  # Conditional masking (all are sampled independently so the fractions are only amortized)
  # BERT_frac_mask = fraction masked       (80%)
  # BERT_frac_rndm = fraction noisy tokens (10%)
  # face_mr_BERT   = fraction of multi-res "masked" tokens (downsampled instead of fully masked)
  # remainder      = untouched             (10%)
  # 
  # conditional idx for tokens to be masked with mean or random values
  idx_mask_cond = torch.tensor( rng.random( idx.shape[0])) < BERT_frac_mask
  idx_rndm_cond = torch.tensor( rng.random( idx.shape[0])) < BERT_frac_rndm
  idx_mr_cond   = torch.tensor( rng.random( idx.shape[0])) < BERT_frac_mr
  
  # set mask 
  source[ idx[idx_mask_cond] ] = global_mean
  
  # set noisy tokens (noise is field independent)
  # TODO: fudge factor of 0.1
  cnt_nz_rndm = cnt_nz(idx_rndm_cond)
  dim_embed = source.shape[1]
  rnd_w = tr( torch.tensor( rng.random( cnt_nz_rndm, dtype=np.float32)).repeat((dim_embed,1)), 0,1 )
  source[ idx[idx_rndm_cond] ] = rnd_w * global_mean + (1.-rnd_w) * source[ idx[idx_rndm_cond] ] + \
            0.1 * global_std * torch.randn( (cnt_nz_rndm, source.shape[1]))
  
  # randomly coarsened tokens
  if BERT_frac_mr > 0. and idx_mr_cond.shape[0] > 0 and idx_mr_cond.any().item() :
    # version with uniform multi-res downsampling per batch for computational efficiency 
    # (in particular on GPU)
    ts = torch.tensor( [token_size[1], token_size[2]], dtype=torch.int)
    mrs = ((rng.random() * (ts - BERT_mr_max*ts)) + BERT_mr_max*ts).int()
    mrs = (token_size[0], mrs[0], mrs[1])
    ts = token_size 
    # interpolate to smaller size and then interpolate up -> coarsened version with same size
    # unsqueeze(usq()) is required since channel dimension is expected
    temp = mr( mr( usq( source[ idx[idx_mr_cond] ].reshape( (-1,ts[0],ts[1],ts[2])), 1), mrs), ts) 
    source[ idx[idx_mr_cond] ] = sq( fl( temp, -3, -1))

  # recover batch dimension which was flattend for easier indexing and also token dimensions
  source = torch.reshape( torch.reshape( source, source_shape), source_shape0)
  
  return (source, token_info, target, tokens_masked_idx_list)

####################################################################################################
def prepare_batch_BERT_forecast_field( cf, ifield, source, token_info, rng) :
 
  nt = cf.forecast_num_tokens
  num_tokens = source.shape[-6:-3]
  num_tokens_space = num_tokens[1] * num_tokens[2] 
  idxs = (num_tokens[0]-nt) * num_tokens_space + torch.arange(nt * num_tokens_space)
  
  # collapse token dimensions 
  source_shape0 = source.shape
  source = torch.flatten( torch.flatten( source, 1, 3), 2, 4)

  # linear indices for masking
  num_tokens = source.shape[1]
  tokens_masked_idx_list = [idxs + num_tokens * i for i in range( source.shape[0] )]
  idx = torch.cat( tokens_masked_idx_list)

  source_shape = source.shape
  # flatten along first two dimension to simplify linear indexing (which then requires an
  # easily computable row offset)
  source = torch.flatten( source, 0, 1)
 
  # keep masked tokens for loss computation
  target = source[idx].clone()

  # masking
  global_mean = 0. * torch.mean(source, 0)
  source[ idx ] = global_mean
 
  # recover batch dimension which was flattend for easier indexing
  source = torch.reshape( torch.reshape( source, source_shape), source_shape0)
 
  return (source, token_info, target, tokens_masked_idx_list)

####################################################################################################
def prepare_batch_BERT_temporal_field( cf, ifield, source, token_info, rng) :
 
  num_tokens = source.shape[-6:-3]
  num_tokens_space = num_tokens[1] * num_tokens[2]
  
  #backward compatibility: mask only middle token
  if not hasattr( cf, 'idx_time_mask'):
    idx_time_mask = int( np.floor(num_tokens[0] / 2.)) 
    idxs = idx_time_mask * num_tokens_space + torch.arange(num_tokens_space)
  else: #list of idx_time_mask
    idxs = torch.concat([i*num_tokens_space + torch.arange(num_tokens_space) for i in cf.idx_time_mask])
  
  # collapse token dimensions 
  source_shape0 = source.shape
  source = torch.flatten( torch.flatten( source, 1, 3), 2, 4)

  # linear indices for masking
  num_tokens = source.shape[1]
  idx = torch.cat( [idxs + num_tokens * i for i in range( source.shape[0] )] )
  tokens_masked_idx_list = [idxs + num_tokens * i for i in range( source.shape[0] )]
  source_shape = source.shape
  # flatten along first two dimension to simplify linear indexing (which then requires an
  # easily computable row offset)
  source = torch.flatten( source, 0, 1)
 
  # keep masked tokens for loss computation
  target = source[idx].clone()

  # masking
  global_mean = 0. * torch.mean(source, 0)
  source[ idx ] = global_mean
 
  # recover batch dimension which was flattend for easier indexing
  source = torch.reshape( torch.reshape( source, source_shape), source_shape0)
 
  return (source, token_info, target, tokens_masked_idx_list)

 ########################################### Asma date = 19.11.2024 ########################################################
def prepare_batch_BERT_data_compression_field( cf, ifield, source, token_info, rng) :
  
  idxs = torch.empty(0, dtype=torch.int64) # new, Asma
  
  vlevel = int(token_info[0,0,0,6]) # new, develop appropriate, Asma
  target = torch.empty(0, dtype=torch.float32) # new, Asma
  tokens_masked_idx_list = [torch.empty(0, dtype=torch.int64)] # Asma changed develop # new, Asma

  if cf.experiment_type != "unmask_checker" and cf.experiment_type !=  "unmask_checker_combined":
    nt = cf.forecast_num_tokens # new, Asma
    num_tokens = source.shape[-6:-3] # new, Asma
    num_tokens_space = num_tokens[1] * num_tokens[2]  # new, Asma
  else:
    batch_dim = source.shape[0] 
    num_tokens = source.shape[1]

  if vlevel in cf.to_mask: 
    match cf.experiment_type:
      case 'unmask_first':
        idxs = (num_tokens[0]-nt) * num_tokens_space + torch.arange(nt * num_tokens_space)
      case 'unmask_last':
        idxs = torch.arange(nt * num_tokens_space) 
      case 'unmask_middle':
        nt_middle = int(nt/2)
        idxs = torch.cat((torch.arange(0, nt_middle * num_tokens_space), torch.arange((nt_middle+1) * num_tokens_space, (nt+1) * num_tokens_space)), 0)
      case 'unmask_first_last':
        idxs = (num_tokens[0]-(nt+1)) * num_tokens_space + torch.arange(nt * num_tokens_space)
      case 'unmask_first_last_middle':
        nt_middle = int(nt/2)
        idxs_first_half =  num_tokens_space + torch.arange(nt_middle * num_tokens_space) # shift by 1 from the start
        idxs_second_half = torch.arange((nt_middle+2) * num_tokens_space, (nt+2) * num_tokens_space)        
        idxs = torch.cat((idxs_first_half, idxs_second_half), 0)
      case 'unmask_checker':
        pattern = torch.cat([ torch.tensor([1,3,4,6]) + i for i in range(0, num_tokens, 8)])
        idxs = torch.cat([pattern for nms in range(batch_dim)])
      case 'unmask_checker_combined':
        pattern = torch.cat([ torch.tensor([1,3,4,6, 8, 9, 10, 11, 12, 13, 14, 15]) + i for i in range(0, num_tokens, 16)])
        idxs = torch.cat([pattern for nms in range(batch_dim)])
      case _:
        # stands for masking the whole level
        idxs = torch.arange(nt * num_tokens_space)

    # collapse token dimensions 
    source_shape0 = source.shape
    source = torch.flatten( torch.flatten( source, 1, 3), 2, 4)

    num_tokens = source.shape[1]

    tokens_masked_idx_list = [idxs + num_tokens * i for i in range( source.shape[0] )] 
    idx = torch.cat(tokens_masked_idx_list) # new, Asma, should be merged after

    # flatten along first two dimension to simplify linear indexing (which then requires an
    # easily computable row offset)
    source_shape = source.shape
    source = torch.flatten( source, 0, 1)
    
    target = source[idx].clone()
    # set mask 
    source[ idx ] = 0.

    # recover batch dimension which was flattend for easier indexing and also token dimensions
    source = torch.reshape( torch.reshape( source, source_shape), source_shape0)
    
  return (source, token_info, target, tokens_masked_idx_list)
  
########################################### End of Asma changes ########################################################