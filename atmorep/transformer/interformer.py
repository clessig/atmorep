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


class Interformer(torch.nn.Module) :

  def __init__(self, num_layers, dim_input, dims_embed,
                      num_heads_self = 2, num_heads_coupling = 2, num_mlp_layers = 2,
                      num_tokens = [], size_token_info = 6):
    '''
    
    '''
    
    super(Interformer, self).__init__()
    self.num_layers = num_layers
    self.num_tokens = num_tokens
    self.size_token_info = size_token_info
    
    # learnable linear embedding
    self.token_embed = torch.nn.Linear( dim_input, dims_embed[0] - size_token_info)

    self.blocks = torch.nn.ModuleList()
    for _ in range( num_layers) :
      # attention sub-block
      self.blocks.append( MultiInterAttentionHead( dims_embed, num_heads_self, num_heads_coupling, 
                                                   num_tokens ))
      # feature space mapping sub-block
      self.blocks.append( MLP( dims_embed[0], num_mlp_layers))

  def forward(self, xin):
    # never called directly but only through multiformer
    assert False