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

from atmorep.transformer.transformer_base import MLP, prepare_token
from atmorep.transformer.transformer_attention import MultiSelfAttentionHead
from atmorep.transformer.transformer_attention import MultiInterAttentionHead
from atmorep.transformer.axial_attention import MultiFieldAxialAttention


class TransformerDownscaler(torch.nn.Module) :

  def __init__(self, cf, field_idx, with_embed = True):
    ''' Functionally identical to TransformerEncoder; duplicated for simpler parameter handling '''
    
    super(TransformerDownscaler, self).__init__()

    self.cf = cf
    self.field_idx = field_idx
    self.with_embed = with_embed

  ###################################
  def create( self) :

    cf = self.cf
    with_ln = cf.with_layernorm

    self.fields_index = {}
    for ifield, field_info in enumerate(cf.fields_targets) :
      self.fields_index[ field_info[0] ] = ifield 

    field_info = cf.fields_targets[self.field_idx]
    
    # learnable linear embedding
    if self.with_embed :
      net_dim_input = np.prod(field_info[4]) 
      self.embed = torch.nn.Linear( net_dim_input, field_info[1][1]- cf.size_token_info_net)

    # num_heads_coupling
    dor = 0.1 # # cf.dropout_rate
    self.heads = torch.nn.ModuleList()
    self.mlps = torch.nn.ModuleList()
    for il in range( cf.downscaling_num_layers) :

      nhc = cf.coupling_num_heads_per_field * len( field_info[1][2])
      nhs = cf.downscaling_num_heads - nhc
      # number of tokens
      n_toks = torch.tensor( field_info[3], dtype=torch.int64)
      s_tok_info = cf.size_token_info_net
        
      dims_embed = [ field_info[1][1] + s_tok_info ]
      vl_num_tokens = [len(field_info[2])] + field_info[3]
      for field_coupled in field_info[1][2] : 
        if 'axial' in cf.downscaling_att_type :
          finfo_other = cf.fields_targets[ self.fields_index[field_coupled] ]
          dims_embed.append( finfo_other[1][1] + s_tok_info )
          vl_num_tokens.append( [len(finfo_other[2])] + finfo_other[3] )
        else : 
          for _ in range(cf.coupling_num_heads_per_field) :
            finfo_other = cf.fields_targets[ self.fields_index[field_coupled] ]
            dims_embed.append( finfo_other[1][1] + s_tok_info)
            vl_num_tokens.append( [len(finfo_other[2])] + finfo_other[3] )

      # attention heads
      if 'dense' == cf.downscaling_att_type :
        head = MultiInterAttentionHead( nhs, nhc, dims_embed, with_ln, dor)
      elif 'axial' in cf.downscaling_att_type :
        par = True if 'parallel' in cf.downscaling_att_type else False
        head = MultiFieldAxialAttention( [3,2,1], dims_embed, nhs, nhc, par)
      else :
        assert False, 'Unsupported attention type: ' + cf.decoder_att_type
      self.heads.append( head)
      # feature space mapping sub-block
      self.mlps.append( MLP( dims_embed[0], cf.downscaling_num_mlp_layers, 
                                            with_ln, dropout_rate=dor))

    return self

  ###################################
  def forward(self, xin):
    ''' '''
    assert False

  