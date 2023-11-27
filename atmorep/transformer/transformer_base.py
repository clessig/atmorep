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
import code

from atmorep.utils.utils import identity


####################################################################################################
def checkpoint_wrapper( cmodule, *kwargs) :
  if cmodule.training :
    return torch.utils.checkpoint.checkpoint( cmodule, *kwargs, use_reentrant=False)
  else :
    return cmodule(*kwargs)

####################################################################################################
def positional_encoding_harmonic_absolute( x, token_infos) :
  '''space time harmonic positional encoding'''

  dim_embed = x.shape[-1]
  dev = x.get_device()
  
  idxs_t1 = token_infos[:,:,1]   # day of the year
  idxs_t2 = token_infos[:,:,2]   # hour of the day
  idxs_lat = token_infos[:,:,4]  # lat
  idxs_lon = token_infos[:,:,5]  # lon

  pe = torch.zeros( token_infos.shape[0], token_infos.shape[1], dim_embed, device=dev)
  xs = (2. * np.pi *  torch.arange( 0, dim_embed, 2, device=dev) / dim_embed)
  pe[:,:,0::2] = torch.cat( [(0.5 * torch.sin( torch.outer( 8 * idxs_lat[i], xs) ) 
            + torch.sin( torch.outer( idxs_t1[i], xs) )).unsqueeze(0) for i in range(token_infos.shape[0]) ], 0)
  pe[:,:,1::2] = torch.cat( [(0.5 * torch.cos( torch.outer( 8 * idxs_lon[i], xs) ) 
            + torch.sin( torch.outer( idxs_t2[i], xs) )).unsqueeze(0) for i in range(token_infos.shape[0]) ], 0)
  x += pe.reshape( x.shape)

  return x

####################################################################################################
def positional_encoding_harmonic( x, num_levels, num_tokens, with_cls = False) :
  '''space time harmonic positional encoding'''

  dim_embed = x.shape[-1]
  dev = x.get_device()
  
  # num_tokens = x.shape[-3:-1]
  # len_token_seq = num_levels * np.prod(num_tokens)
  # pe = torch.zeros( len_token_seq, dim_embed, device=dev)
  # position = torch.arange( 0, len_token_seq).unsqueeze(1)
  # div = torch.exp(torch.arange( 0, dim_embed, 2) * -(math.log(1000) / dim_embed))

  # pe[:, 0::2] = torch.sin(position * div)
  # pe[:, 1::2] = torch.cos(position * div)
  # pe = pe.unsqueeze(0)

  # x += pe.reshape( x[0].shape )


  idx = torch.arange( np.prod( x.shape[1:-1]), device=dev)
  num_tokens_t_lat_lon = np.prod( num_tokens)
  num_tokens_lat_lon = num_tokens[1] * num_tokens[2]
  idxs_v = (idx / num_tokens_t_lat_lon).int()
  # idxs_v = num_tokens_t_lat_lon
  temp = torch.remainder( idx, num_tokens_t_lat_lon)
  idxs_t = (temp / num_tokens_lat_lon).int()
  temp = torch.remainder( idx, num_tokens_lat_lon)
  idxs_lat = (temp / num_tokens[1]).int()
  idxs_lon = torch.remainder( temp, num_tokens[2])

  pe = torch.zeros( idx.shape[0], dim_embed, device=dev)
  xs = (2. * np.pi *  torch.arange( 0, dim_embed, 2, device=dev) / dim_embed)
  pe[:, 0::2] = 0.5 * torch.sin( torch.outer( 8 * idxs_lat, xs) ) \
                  + torch.sin( torch.outer( idxs_t, xs) )
  pe[:, 1::2] = 0.5 * torch.cos( torch.outer( 8 * idxs_lon, xs) ) \
                  + torch.cos( torch.outer( idxs_v , xs) )
  if with_cls :
    x[:,1:] += pe.reshape( x[0,1:].shape)
  else :
    x += pe.reshape( x[0].shape)

  return x

####################################################################################################
class MLP(torch.nn.Module):

  def __init__(self, dim_embed, num_layers = 2, with_lnorm = True, dim_embed_out = None, 
                     nonlin = torch.nn.GELU(), dim_internal_factor = 2, dropout_rate = 0.,
                     grad_checkpointing = False, with_residual = True) :
    """
    Multi-layer perceptron

    dim_embed   : embedding dimension
    num_layers  : number of layers
    nonlin      : nonlinearity
    dim_internal_factor : factor for number of hidden dimension relative to input / output
    """
    super(MLP, self).__init__()

    if not dim_embed_out :
      dim_embed_out = dim_embed

    self.with_residual = with_residual

    dim_internal = int( dim_embed * dim_internal_factor)
    if with_lnorm : 
      self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)
    else :
      self.lnorm = torch.nn.Identity()

    self.blocks = torch.nn.ModuleList()
    self.blocks.append( torch.nn.Linear( dim_embed, dim_internal))
    self.blocks.append( nonlin)
    self.blocks.append( torch.nn.Dropout( p = dropout_rate))
    
    for _ in range( num_layers-2) :
      self.blocks.append( torch.nn.Linear( dim_internal, dim_internal))
      self.blocks.append( nonlin)
      self.blocks.append( torch.nn.Dropout( p = dropout_rate))
    
    self.blocks.append( torch.nn.Linear( dim_internal, dim_embed_out))
    self.blocks.append( nonlin)

    if dim_embed == dim_embed_out :
      self.proj_residual = torch.nn.Identity()
    else :
      self.proj_residual = torch.nn.Linear( dim_embed, dim_embed_out)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  def forward( self, x, y = None) :
    
    x_in = x
    x = self.lnorm( x)
    
    for block in self.blocks:
      x = self.checkpoint( block, x)

    if self.with_residual :
      x += x_in

    return x

####################################################################################################
def prepare_token_info( cf, token_info) :

  if len( token_info.shape) > 3 :
    token_info = torch.flatten( torch.flatten( token_info, -3, -2), 1, 2)
  token_info[:,:,0] /= 2100.   # absolute year
  token_info[:,:,1] /= 365.    # day in year
  token_info[:,:,2] /= 24.     # hour in day
  token_info[:,:,3] /= 1000.   # level
  token_info[:,:,4] /= cf.file_shape[1]
  token_info[:,:,5] /= cf.file_shape[2]
  token_info *= 0.01

  return token_info

####################################################################################################
def prepare_token( xin, embed, embed_token_info, with_cls = True) :

  (token_seq, token_info) = xin
  num_tokens = token_seq.shape[-6:-3]
  num_levels = token_seq.shape[1]

  # embedding, flatten along token dimension and spatial dimensions
  token_seq_embed = embed( torch.flatten( torch.flatten( token_seq, -3, -1), -3, -2) )
  
  # add auxiliary, global token information
  token_info = embed_token_info( token_info).to( token_seq_embed.device, non_blocking=True )
  # token_info = prepare_token_info( cf, token_info)
  token_info = token_info.reshape([-1] + list(token_seq_embed.shape[1:-1])+[token_info.shape[-1]])
  token_seq_embed = torch.cat( [token_seq_embed, token_info], -1)

  # class token
  if with_cls :
    # initialize to zero (mean of data)
    tts = token_seq_embed.shape
    cls_token =  torch.zeros( (tts[0], 1, tts[2]), device=token_seq_embed.device)
  
  # add positional encoding
  token_seq_embed = positional_encoding_harmonic( token_seq_embed, num_levels, num_tokens)

  # add class token after positional encoding
  if with_cls :
    token_seq_embed = torch.cat( [ cls_token, token_seq_embed ], 1)

  return token_seq_embed
