####################################################################################################
#
#  Copyright (C) 2024
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

from atmorep.transformer.mlp import MLP
from atmorep.transformer.transformer_attention import MultiSelfAttentionHead


class Transformer(torch.nn.Module) :

  def __init__(self, num_layers, dim_embed, num_heads = 8, num_mlp_layers = 2,
                     dropout_rate = 0., with_qk_lnorm=True):
    '''
    '''
    
    super(Transformer, self).__init__()

    self.num_layers = num_layers

    self.blocks = torch.nn.ModuleList()
    for _ in range( num_layers) :
      self.blocks.append(MultiSelfAttentionHead( dim_embed, num_heads, dropout_rate, with_qk_lnorm))
      self.blocks.append(MLP( dim_embed, num_mlp_layers, dropout_rate=dropout_rate))

  def forward(self, x):
    '''Evaluate transformer encoder'''

    # attention heads + feature space mappings
    for block in self.blocks : 
      x = block( x)

    return x
