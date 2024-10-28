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

from atmorep.transformer.mlp import MLP
from atmorep.transformer.transformer_attention import MultiSelfAttentionHead,MultiCrossAttentionHead
from atmorep.transformer.axial_attention import MultiFieldAxialAttention
from atmorep.utils.utils import identity
from atmorep.transformer.transformer_base import checkpoint_wrapper

class TransformerDecoder(torch.nn.Module) :

  ###################################
  def __init__(self, cf, field_info ):
    '''
    Vaswani transformer corresponds to self_att = True and cross_att_ratio = 1. *and* encoder_out 
    passed to forward is the output of the encoder (duplicated to match the number of layers)
    '''
    super( TransformerDecoder, self).__init__()
    
    self.cf = cf
    self.num_layers = cf.decoder_num_layers
    self.dim_embed = field_info[1][1]

    # TODO: split up create() for consistency

    num_heads = cf.decoder_num_heads
    num_mlp_layers = cf.decoder_num_mlp_layers 
    self_att = cf.decoder_self_att
    cross_att_ratio = cf.decoder_cross_att_ratio 

    num_heads_other = int(num_heads * cross_att_ratio)
    num_heads_self = num_heads - num_heads_other

    dim_embed = self.dim_embed

    # first layers, potentially with U-Net type coupling
    self.blocks = torch.nn.ModuleList()
    for il in range( min( cf.encoder_num_layers, cf.decoder_num_layers) ) :

      # self attention sub-block (as in original Vaswani)
      if self_att :
        self.blocks.append( MultiSelfAttentionHead( dim_embed, num_heads, cf.dropout_rate, 
                                                    cf.decoder_att_type, cf.with_qk_lnorm) )
      # cross attention between encoder and decoder
      if 'dense' == cf.decoder_att_type :
        self.blocks.append( MultiCrossAttentionHead( dim_embed, num_heads_self, num_heads_other, 
                                                  cf.dropout_rate, cf.with_qk_lnorm, cf.attention))
      elif 'axial' in cf.decoder_att_type :
        par = True if 'parallel' in cf.encoder_att_type else False
        self.blocks.append( MultiFieldAxialAttention( [3,2,1], [dim_embed,dim_embed], 
                                          num_heads_self, num_heads_other, par, cf.dropout_rate) )
      else :
        assert False, 'Unsupported attention type: ' + cf.decoder_att_type 
      # feature space mapping sub-block
      self.blocks.append( MLP( dim_embed, num_mlp_layers, cf.with_layernorm, 
                               dropout_rate = cf.dropout_rate, 
                               grad_checkpointing = cf.grad_checkpointing) )

    # remaining strictly non-coupled layers (if decoder is deeper than the encoder)
    dim_embed = self.dim_embed
    for il in range( cf.encoder_num_layers, cf.decoder_num_layers) :
      self.blocks.append( MultiSelfAttentionHead( dim_embed, num_heads, cf.dropout_rate, 
                                                  cf.decoder_att_type, cf.with_qk_lnorm))
      self.blocks.append( MLP( dim_embed, num_mlp_layers, cf.with_layernorm,
                               grad_checkpointing = cf.grad_checkpointing ))

    self.checkpoint = identity
    if cf.grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  ###################################
  def device( self):
    return next(self.parameters()).device

  ###################################
  def forward(self, x):
    '''Evaluate decoder'''

    dev = self.device()

    (decoder_in, encoder_out) = x
    encoder_out.reverse()
    
    token_seq_embed = decoder_in.to( dev, non_blocking=True)

    atts = []
    car = self.cf.decoder_cross_att_rate
    for il in range(self.num_layers) : 
      token_seq_embed, att = self.checkpoint( self.blocks[2*il], token_seq_embed, 
                                            encoder_out[int(car*il)].to(dev, non_blocking=True) )
      token_seq_embed = self.checkpoint( self.blocks[2*il+1], token_seq_embed, 
                                            encoder_out[int(car*il)].to(dev, non_blocking=True) )
      atts += [ att ]

    return token_seq_embed, atts

  ###################################
  def get_attention( self, xin, iblock) :
    ''' 
    Get attention and projected values from specific layer and her head
    '''

    #    assert False
    print("inside get_attention in transformer_decoder.py")
    # embedding
    token_seq_embed = decoder_in.to( dev, non_blocking=True)
    car = self.cf.decoder_cross_att_rate
    for il in range(self.num_layers) :
      token_seq_embed = self.checkpoint( self.blocks[2*il], token_seq_embed, 
                                            encoder_out[int(car*il)].to(dev, non_blocking=True) )
      
      atts = self.blocks[2*il].get_attention( token_seq_embed )

    return atts #(atts, vsh)
