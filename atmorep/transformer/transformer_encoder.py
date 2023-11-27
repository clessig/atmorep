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

from atmorep.transformer.transformer_base import MLP
from atmorep.transformer.transformer_attention import MultiInterAttentionHead
from atmorep.transformer.axial_attention import MultiFieldAxialAttention


class TransformerEncoder(torch.nn.Module) :

  def __init__(self, cf, field_idx, with_embed = True):
    '''  '''
    
    super(TransformerEncoder, self).__init__()

    self.cf = cf
    self.field_idx = field_idx
    self.with_embed = with_embed

  ###################################
  def create( self) :

    cf = self.cf
    with_ln = cf.with_layernorm

    self.fields_index = {}
    for ifield, field_info in enumerate(cf.fields) :
      self.fields_index[ field_info[0] ] = ifield 

    field_info = cf.fields[self.field_idx]
    
    # learnable linear embedding
    if self.with_embed :
      net_dim_input = np.prod(field_info[4]) 
      self.embed = torch.nn.Linear( net_dim_input, field_info[1][1]- cf.size_token_info_net)

    # num_heads_coupling
    dor = cf.dropout_rate
    self.heads = torch.nn.ModuleList()
    self.mlps = torch.nn.ModuleList()
    for il in range( cf.encoder_num_layers) :

      nhc = cf.coupling_num_heads_per_field * len( field_info[1][2])
      # nhs = cf.encoder_num_heads - nhc
      nhs = cf.encoder_num_heads
      # number of tokens
      n_toks = torch.tensor( field_info[3], dtype=torch.int64)
        
      dims_embed = [ field_info[1][1] ]
      vl_num_tokens = [len(field_info[2])] + field_info[3]
      for field_coupled in field_info[1][2] : 
        if 'axial' in cf.encoder_att_type :
          finfo_other = cf.fields[ self.fields_index[field_coupled] ]
          dims_embed.append( finfo_other[1][1] )
          vl_num_tokens.append( [len(finfo_other[2])] + finfo_other[3] )
        else : 
          for _ in range(cf.coupling_num_heads_per_field) :
            finfo_other = cf.fields[ self.fields_index[field_coupled] ]
            dims_embed.append( finfo_other[1][1] )
            vl_num_tokens.append( [len(finfo_other[2])] + finfo_other[3] )

      # attention heads
      if 'dense' == cf.encoder_att_type :
        head = MultiInterAttentionHead( nhs, nhc, dims_embed, with_ln, dor, cf.with_qk_lnorm, 
                                        cf.grad_checkpointing, with_attention=cf.attention )
      elif 'axial' in cf.encoder_att_type :
        par = True if 'parallel' in cf.encoder_att_type else False
        head = MultiFieldAxialAttention( [3,2,1], dims_embed, nhs, nhc, par, dor)
      else :
        assert False, 'Unsupported attention type: ' + cf.decoder_att_type
      self.heads.append( head)
      # feature space mapping sub-block
      self.mlps.append( MLP( dims_embed[0], cf.encoder_num_mlp_layers, with_ln, dropout_rate=dor,
                             grad_checkpointing = cf.grad_checkpointing))

    return self

  ###################################
  def forward(self, xin):
    ''' '''
    assert False

  