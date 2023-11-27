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

class Transformer(torch.nn.Module) :

  def __init__(self, num_layers, dim_input, dim_embed = 2048, num_heads = 8, num_mlp_layers = 2,
                     with_lin_embed = True, with_dropout = False,
                     size_token_info = 6):
    '''
    '''
    
    super(Transformer, self).__init__()
    self.num_layers = num_layers

    self.size_token_info = size_token_info

    # learnable linear embedding
    if with_lin_embed :
      self.token_embed = torch.nn.Linear( dim_input, dim_embed - size_token_info)
    else :
      self.token_embed = torch.nn.Identity()

    self.blocks = torch.nn.ModuleList()
    for _ in range( num_layers) :
      # attention sub-block
      self.blocks.append( MultiSelfAttentionHead( dim_embed, num_heads))
      # dropout
      if with_dropout :
        self.blocks.append( torch.nn.Dropout( p=0.05) )
      # feature space mapping sub-block
      self.blocks.append( MLP( dim_embed, num_mlp_layers))

  ###################################
  def forward(self, xin):
    '''Evaluate transformer encoder'''
    
    token_seq_embed = prepare_token( self, xin)

    # attention heads + feature space mappings
    for block in self.blocks : 
      token_seq_embed = block( token_seq_embed)

    return token_seq_embed

  ###################################
  def forward_retain(self, xin):
    '''Evaluate and retain intermediate representation'''
    
    token_seq_embed = prepare_token( self, xin)
    tokens_layers = [token_seq_embed]

    # attention heads + feature space mappings
    for il in range(self.num_layers) : 
      token_seq_embed = self.blocks[2*il]( token_seq_embed)
      token_seq_embed = self.blocks[2*il+1]( token_seq_embed)
      tokens_layers.append( token_seq_embed)

    return tokens_layers

  ###################################
  def get_attention( self, xin, iblock) :
    ''' 
    Get attention and projected values from specific layer and her head
    '''

    # embedding
    token_seq_embed = prepare_token( self, xin)

    # attention heads + feature space mappings
    for idx in range(2*iblock) : 
      token_seq_embed = self.blocks[idx]( token_seq_embed)
    atts, vsh = self.blocks[2*iblock].get_attention( token_seq_embed)

    return (atts, vsh)
