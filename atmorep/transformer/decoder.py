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
from atmorep.transformer.transformer_attention import MultiSelfAttentionHead, MultiCrossAttentionHead


class Decoder(torch.nn.Module) :

  ###################################
  def __init__(self, num_layers, dim_embed = 2048, 
                     num_heads = 8, num_mlp_layers = 2,
                     self_att = False, cross_att_ratio = 0.5 ):
    '''
    Vaswani transformer corresponds to self_att = True and cross_att_ratio = 1.
    '''
    super( Decoder, self).__init__()
    
    self.num_layers = num_layers
    self.dim_embed = dim_embed 

    self.len_block = 2
    if self_att :
      self.len_block = 3

    num_heads_other = int(num_heads * cross_att_ratio)
    num_heads_self = num_heads - num_heads_other

    self.blocks = torch.nn.ModuleList()
    for _ in range( self.num_layers) :
      # attention sub-block
      if self_att :
        self.blocks.append( MultiSelfAttentionHead( dim_embed, num_heads_self))
      # cross attention between encoder and decoder
      self.blocks.append( MultiCrossAttentionHead( dim_embed, num_heads_self, num_heads_other))
      # feature space mapping sub-block
      self.blocks.append( MLP( dim_embed, num_mlp_layers))

  ###################################
  def forward(self, token_seq_embed, encoder_out):
    '''Evaluate decoder'''

    for il in range(self.num_layers) : 
      token_seq_embed = self.blocks[2*il]( token_seq_embed, encoder_out[il] )
      token_seq_embed = self.blocks[2*il+1]( token_seq_embed, encoder_out[il] )

    return token_seq_embed

  ###################################
  def get_attention( self, xin, iblock) :
    ''' 
    Get attention and projected values from specific layer and her head
    '''

    assert False

    # # embedding
    # token_seq_embed = prepare_token( self, xin, self.size_token_info)

    # # attention heads + feature space mappings
    # for idx in range(2*iblock) : 
    #   token_seq_embed = self.blocks[idx]( token_seq_embed)
    # atts, vsh = self.blocks[2*iblock].get_attention( token_seq_embed)

    # return (atts, vsh)
