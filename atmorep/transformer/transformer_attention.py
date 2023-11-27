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
from enum import Enum
import code

from atmorep.transformer.axial_attention import AxialAttention
from atmorep.utils.utils import identity
from atmorep.transformer.transformer_base import checkpoint_wrapper

####################################################################################################
class CouplingAttentionMode( Enum) :
  indeterminate = 0
  q_coupling = 1
  kv_coupling = 2


####################################################################################################

class AttentionHead(torch.nn.Module):

  def __init__(self, proj_dims, proj_dims_qs = -1, with_qk_lnorm = False, with_attention=False) :
    '''Attention head'''

    super(AttentionHead, self).__init__()

    if proj_dims_qs == -1 :
      proj_dims_qs = proj_dims[0]

    self.proj_qs = torch.nn.Linear( proj_dims_qs, proj_dims[1], bias = False)
    self.proj_ks = torch.nn.Linear( proj_dims[0], proj_dims[1], bias = False)
    self.proj_vs = torch.nn.Linear( proj_dims[0], proj_dims[1], bias = False)
    
    self.softmax = torch.nn.Softmax(dim=-1)

    if with_qk_lnorm :
      self.lnorm_qs = torch.nn.LayerNorm( proj_dims[1], elementwise_affine=False)
      self.lnorm_ks = torch.nn.LayerNorm( proj_dims[1], elementwise_affine=False)
    else :
      self.lnorm_qs = torch.nn.Identity()
      self.lnorm_ks = torch.nn.Identity()

    self.forward = self.forward_attention if with_attention else self.forward_evaluate

  def attention( self, qs, ks) :
    '''Compute attention'''
    return torch.matmul( qs, torch.transpose( ks, -2, -1))

  def forward_evaluate( self, xs_q, xs_k_v = None) :
    '''Evaluate attention head'''

    xs_k_v = xs_q if None == xs_k_v else xs_k_v

    out_shape = xs_q.shape
    qs = self.lnorm_qs( self.proj_qs( torch.flatten( xs_q, 1, -2) ))
    ks = self.lnorm_ks( self.proj_ks( torch.flatten( xs_k_v, 1, -2) ))
    vs = self.proj_vs( torch.flatten( xs_k_v, 1, -2) )
    # normalization increases interpretability since otherwise the scaling of the values 
    # interacts with the attention values
    # torch.nn.functional.normalize( vs, dim=-1)

    scaling = 1. / torch.sqrt( torch.tensor(qs.shape[2]))
    vsp = torch.matmul( self.softmax( scaling * self.attention( qs, ks)),  vs)
    return (vsp.reshape( [-1] + list(out_shape[1:-1]) + [vsp.shape[-1]]), None)

  def forward_attention( self, xs_q, xs_k_v = None) :
    '''Evaluate attention head and also return attention'''

    xs_k_v = xs_q if None == xs_k_v else xs_k_v

    out_shape = xs_q.shape
    kv_shape = xs_k_v.shape
    qs = self.lnorm_qs( self.proj_qs( torch.flatten( xs_q, 1, -2) ))
    ks = self.lnorm_ks( self.proj_ks( torch.flatten( xs_k_v, 1, -2) ))
    vs = self.proj_vs( torch.flatten( xs_k_v, 1, -2) )
    # normalization increases interpretability since otherwise the scaling of the values 
    # interacts with the attention values
    # torch.nn.functional.normalize( vs, dim=-1)

    scaling = 1. / torch.sqrt( torch.tensor(qs.shape[2]))
    att = self.attention( qs, ks)
    vsp = torch.matmul( self.softmax( scaling * att),  vs)
    return ( vsp.reshape( [-1] + list(out_shape[1:-1]) + [vsp.shape[-1]]), 
             att.reshape( [-1] + list(out_shape[1:-1]) + list(kv_shape[1:-1])).detach().cpu() )

####################################################################################################

class MultiSelfAttentionHead(torch.nn.Module):

  def __init__(self, dim_embed, num_heads, dropout_rate = 0., att_type = 'dense', 
                     with_qk_lnorm = False, grad_checkpointing = False, with_attention = False ) :
    
    super(MultiSelfAttentionHead, self).__init__()

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = int(dim_embed / num_heads)

    self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)

    self.heads = torch.nn.ModuleList()
    if 'dense' == att_type :
      for n in range( num_heads) :
        self.heads.append( AttentionHead( [dim_embed, self.dim_head_proj], 
                                      with_qk_lnorm= with_qk_lnorm, with_attention=with_attention))
    elif 'axial' in att_type :
      self.heads.append( AxialAttention( dim = dim_embed, dim_index = -1, heads = num_heads,
                                         num_dimensions = 3) )
    else :
      assert False, 'Unsuppored attention type.'

    # proj_out is done is axial attention head so do not repeat it
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False) \
                                                if att_type == 'dense' else torch.nn.Identity()
    self.dropout = torch.nn.Dropout( p=dropout_rate)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  def forward( self, x, y = None) :

    x_in = x
    x = self.lnorm( x)

    outs, atts = [], []
    for head in self.heads :
      y, att = self.checkpoint( head, x)
      outs.append( y)
      atts.append( y)
    outs = torch.cat( outs, -1)

    outs = self.dropout( self.checkpoint( self.proj_out, outs) )

    return x_in + outs, atts

####################################################################################################

class MultiCrossAttentionHead(torch.nn.Module):

  def __init__(self, dim_embed, num_heads, num_heads_other, dropout_rate = 0., with_qk_lnorm =False,
                     grad_checkpointing = False, with_attention=False):
    super(MultiCrossAttentionHead, self).__init__()

    self.num_heads = num_heads
    self.num_heads_other = num_heads_other

    assert 0 == dim_embed % (num_heads + num_heads_other)
    self.dim_head_proj = int( dim_embed / (num_heads + num_heads_other) )

    self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)
    if num_heads_other > 0 :
      self.lnorm_other = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)
    else : 
      self.lnorm_other = torch.nn.Identity()

    # self attention heads
    self.heads = torch.nn.ModuleList()
    for n in range( num_heads) :
      self.heads.append( AttentionHead( [dim_embed, self.dim_head_proj],
                                      with_qk_lnorm = with_qk_lnorm, with_attention=with_attention))
    
    # cross attention heads
    self.heads_other = torch.nn.ModuleList()
    for n in range( num_heads_other) :
      self.heads_other.append( AttentionHead( [dim_embed, self.dim_head_proj],
                                      with_qk_lnorm = with_qk_lnorm, with_attention=with_attention))

    # proj_out is done is axial attention head so do not repeat it
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  def forward( self, x, x_other) :
    
    x_in = x
    x = self.lnorm( x)
    x_other = self.lnorm_other( x_other)

    # output tensor where output of heads is linearly concatenated
    outs, atts = [], []

    # self attention
    for head in self.heads :
      y, att = self.checkpoint( head, x)
      outs.append( y)
      atts.append( att) 

    # cross attention
    for head in self.heads_other :
      y, att = self.checkpoint( head, x, x_other)
      outs.append( y)
      atts.append( att)

    outs = torch.cat( outs, -1)
    outs = self.dropout( self.checkpoint( self.proj_out, outs) )

    return x_in + outs, atts

####################################################################################################

class MultiInterAttentionHead(torch.nn.Module):

  #####################################
  def __init__( self, num_heads_self, num_heads_coupling, dims_embed, with_lnorm = True, 
                      dropout_rate = 0., with_qk_lnorm = False, grad_checkpointing = False,
                      with_attention=False) :
    '''Multi-head attention with multiple interacting fields coupled through attention.'''

    super(MultiInterAttentionHead, self).__init__()

    self.num_fields = len(dims_embed)

    # self.coupling_mode = coupling_mode
    # assert 0 == (dims_embed[0] % (num_heads_self + num_heads_coupling))
    # self.dim_head_proj = int(dims_embed[0] / (num_heads_self + num_heads_coupling))
    self.dim_head_proj = int(dims_embed[0] / num_heads_self)

    # layer norms for all fields
    self.lnorms = torch.nn.ModuleList()
    for ifield in range( self.num_fields) :
      if with_lnorm :
        self.lnorms.append( torch.nn.LayerNorm( dims_embed[ifield], elementwise_affine=False))
      else : 
        self.lnorms.append( torch.nn.Identity())

    # self attention heads
    self.heads_self = torch.nn.ModuleList()
    for n in range( num_heads_self) :
      self.heads_self.append( AttentionHead( [dims_embed[0], self.dim_head_proj],
                                      dims_embed[0], with_qk_lnorm, with_attention=with_attention ))
    
    # coupling attention heads
    self.heads_coupling = torch.nn.ModuleList()
    for ifield in range( num_heads_coupling) :
      arg1 = [dims_embed[ifield+1], self.dim_head_proj]
      self.heads_coupling.append( AttentionHead( arg1, dims_embed[0], with_qk_lnorm,
                                                 with_attention=with_attention ))

    # self.proj_out = torch.nn.Linear( dims_embed[0], dims_embed[0], bias = False)
    self.proj_out = torch.nn.Linear( self.dim_head_proj * (num_heads_self + num_heads_coupling), dims_embed[0], bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  #####################################
  def forward( self, *args) :
    '''Evaluate block'''

    x_in = args[0]

    # layer norm for each field
    fields_lnormed = []
    for ifield, field in enumerate( args) :
      fields_lnormed.append( self.lnorms[ifield](field) )
    
    # output tensor where output of heads is linearly concatenated
    outs, atts = [], []

    # self attention
    for head in self.heads_self :
      y, att = self.checkpoint( head, fields_lnormed[0], fields_lnormed[0])
      outs.append( y)
      atts.append( att)
      
    # inter attention
    for ifield, head in enumerate(self.heads_coupling) :
      y, att = self.checkpoint( head, fields_lnormed[0], fields_lnormed[ifield+1])
      outs.append( y)
      atts.append( att)

    outs = torch.cat( outs, -1)
    outs = self.dropout( self.checkpoint( self.proj_out, outs) ) 

    return x_in + outs, atts


