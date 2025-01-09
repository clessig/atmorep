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
from enum import Enum

from atmorep.utils.utils import identity
from atmorep.transformer.transformer_base import checkpoint_wrapper

####################################################################################################
class CouplingAttentionMode( Enum) :
  indeterminate = 0
  q_coupling = 1
  kv_coupling = 2


class MultiSelfAttentionHead(torch.nn.Module):

  #########################################
  def __init__(self, dim_embed, num_heads, dropout_rate=0., with_qk_lnorm=True, with_flash=True) :
    
    super(MultiSelfAttentionHead, self).__init__()

    self.num_heads = num_heads
    self.with_flash = with_flash

    assert 0 == dim_embed % num_heads
    self.dim_head_proj = int(dim_embed / num_heads)
    self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)
    self.proj_heads = torch.nn.Linear( dim_embed, num_heads*3*self.dim_head_proj, bias = False)
    self.proj_out   = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout    = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    lnorm = torch.nn.LayerNorm if with_qk_lnorm else torch.nn.Identity
    self.ln_q = lnorm( self.dim_head_proj, elementwise_affine=False)
    self.ln_k = lnorm( self.dim_head_proj, elementwise_affine=False)
    
    # with_flash = False
    if with_flash :
      self.att = torch.nn.functional.scaled_dot_product_attention
    else :
      self.att = self.attention
      self.softmax = torch.nn.Softmax(dim=-1)

  #########################################
  def forward( self, x) :

    split, tr = torch.tensor_split, torch.transpose
    
    x_in = x
    x = self.lnorm( x)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    s = [ *x.shape[:-1], self.num_heads, -1]
    qs, ks, vs = split( self.proj_heads( x).reshape(s).transpose( 2, 1), 3, dim=-1)
    qs, ks = self.ln_q( qs), self.ln_k( ks)
    
    # correct ordering of tensors with seq dimension second but last is critical
    with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
      outs = self.att( qs, ks, vs).transpose( 2, 1)
      
    return x_in + self.dropout( self.proj_out( outs.flatten( -2, -1)) )

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)
      
  #########################################
  def score( self, q, k) :
    return torch.matmul( q, torch.transpose( k, -2, -1))

####################################################################################################

class MultiCrossAttentionHead(torch.nn.Module):

  def __init__(self, dim_embed, num_heads, num_heads_other, dropout_rate = 0., with_qk_lnorm =False,
                     grad_checkpointing = False, with_attention=False, with_flash=True):
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

    self.proj_heads = torch.nn.Linear( dim_embed, num_heads*3*self.dim_head_proj, bias = False)
    
    self.proj_heads_o_q = torch.nn.Linear(dim_embed, num_heads_other*self.dim_head_proj, bias=False)
    self.proj_heads_o_kv= torch.nn.Linear(dim_embed,num_heads_other*2*self.dim_head_proj,bias=False)
    
    self.ln_q = torch.nn.LayerNorm( self.dim_head_proj)
    self.ln_k = torch.nn.LayerNorm( self.dim_head_proj)

    # proj_out is done is axial attention head so do not repeat it
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate)

    if with_flash :
      self.att = torch.nn.functional.scaled_dot_product_attention
    else :
      self.att = self.attention
    self.softmax = torch.nn.Softmax(dim=-1)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  def forward( self, x, x_other) :
    
    x_in = x
    x, x_other = self.lnorm( x), self.lnorm_other( x_other)

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    x = x.flatten( 1, -2)
    s = [ *x.shape[:-1], self.num_heads, -1]
    qs, ks, vs = torch.tensor_split( self.proj_heads( x).reshape(s).transpose( 2, 1), 3, dim=-1)
    qs, ks = self.ln_q( qs), self.ln_k( ks)
    
    s = [ *x.shape[:-1], self.num_heads_other, -1]
    qs_o = self.proj_heads_o_q( x).reshape(s).transpose( 2, 1)
    x_o = x_other.flatten( 1, -2)
    s = [ *x_o.shape[:-1], self.num_heads_other, -1]
    ks_o, vs_o = torch.tensor_split( self.proj_heads_o_kv(x_o).reshape(s).transpose( 2, 1),2,dim=-1)
    qs_o, ks_o = self.ln_q( qs_o), self.ln_k( ks_o)

    # correct ordering of tensors with seq dimension second but last is critical
    with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
      s = list(x_in.shape)
      s[-1] = -1
      outs_self = self.att( qs, ks, vs).transpose( 2, 1).flatten( -2, -1).reshape(s)
      outs_other = self.att( qs_o, ks_o, vs_o).transpose( 2, 1).flatten( -2, -1).reshape(s)
      outs = torch.cat( [outs_self, outs_other], -1)
    
    outs = self.dropout( self.checkpoint( self.proj_out, outs) )
    atts = []
    return x_in + outs, atts

  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)

####################################################################################################

class MultiInterAttentionHead(torch.nn.Module):

  #####################################
  def __init__( self, num_heads_self, num_fields_other, num_heads_coupling_per_field, dims_embed,
                      with_lnorm = True, dropout_rate = 0., with_qk_lnorm = False, 
                      grad_checkpointing = False, with_attention=False, with_flash=True) :
    '''Multi-head attention with multiple interacting fields coupled through attention.'''

    super(MultiInterAttentionHead, self).__init__()

    self.num_heads_self = num_heads_self
    self.num_heads_coupling_per_field = num_heads_coupling_per_field
    self.num_fields = len(dims_embed)

    self.dim_head_proj = int(dims_embed[0] / num_heads_self)

    # layer norms for all fields
    self.lnorms = torch.nn.ModuleList()
    ln = torch.nn.LayerNorm if with_lnorm else torch.nn.Identity
    for ifield in range( self.num_fields) :
      self.lnorms.append( ln( dims_embed[ifield], elementwise_affine=False))

    # self-attention
    nnc = num_fields_other * num_heads_coupling_per_field
    self.proj_out = torch.nn.Linear( self.dim_head_proj * (num_heads_self + nnc), 
                                     dims_embed[0], bias = False)
    self.dropout = torch.nn.Dropout( p=dropout_rate) if dropout_rate > 0. else torch.nn.Identity()

    nhs = num_heads_self
    self.proj_heads = torch.nn.Linear( dims_embed[0], nhs*3*self.dim_head_proj, bias = False)
    
    # cross-attention
    nfo = num_fields_other
    if nfo > 0:
      nhc_dim = num_heads_coupling_per_field * self.dim_head_proj
      self.proj_heads_other = torch.nn.ModuleList()
      # queries from primary source/target field
      self.proj_heads_other.append( torch.nn.Linear( dims_embed[0], nhc_dim*nfo, bias=False))
      # keys, values for other fields
      for i in range(nfo) :
        self.proj_heads_other.append( torch.nn.Linear( dims_embed[i+1], 2*nhc_dim, bias=False))
    
    ln = torch.nn.LayerNorm if with_qk_lnorm else torch.nn.Identity
    self.ln_qk = torch.nn.ModuleList([ln( self.dim_head_proj, elementwise_affine=False),  ln( self.dim_head_proj, elementwise_affine=False)])
    
    self.ln_k_other = torch.nn.ModuleList([ln(self.dim_head_proj,elementwise_affine=False) for _ in range(nfo)])
    
    if with_flash :
      self.att = torch.nn.functional.scaled_dot_product_attention
    else :
      self.att = self.attention
    self.softmax = torch.nn.Softmax(dim=-1)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  #####################################
  def forward( self, *args) :
    '''Evaluate block'''
    
    x_in, atts = args[0], []

    # layer norm for each field
    fields_lnormed = []
    for ifield, field in enumerate( args) :
      fields_lnormed.append( self.lnorms[ifield](field) )

    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    # collapse three space and time dimensions for dense space-time attention
    #proj_heads: torch.Size([16, 3, 128, 2048])
    field_proj = self.proj_heads( fields_lnormed[0].flatten(1,-2))
    s = [ *field_proj.shape[:-1], self.num_heads_self, -1 ]
    qs, ks, vs = torch.tensor_split( field_proj.reshape(s).transpose(-3,-2), 3, dim=-1)

    qs, ks = self.ln_qk[0]( qs), self.ln_qk[1]( ks)
    if len(fields_lnormed) > 1 :

      field_proj = self.proj_heads_other[0]( fields_lnormed[0].flatten(1,-2))
      s = [ *field_proj.shape[:-1], len(fields_lnormed)-1, self.num_heads_coupling_per_field, -1 ]
      qs_other = field_proj.reshape(s).permute( [-3, 0, -2, 1, -1])
    
      ofields_projs = []
      for i,f in enumerate(fields_lnormed[1:]) :
        f_proj = self.proj_heads_other[i+1](f.flatten(1,-2)) 
        s = [ *f_proj.shape[:-1], self.num_heads_coupling_per_field, -1 ]
        ks_o, vs_o = torch.tensor_split( f_proj.reshape(s).transpose(-3,-2), 2, dim=-1)
        ofields_projs += [ (self.ln_k_other[i]( ks_o), vs_o) ]

    # correct ordering of tensors with seq dimension second but last is critical
    with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
      
      # self-attention
      s = list(fields_lnormed[0].shape)
      outs = self.att( qs, ks, vs).transpose( -3, -2).flatten( -2, -1).reshape(s)
      
      # cross-attention
      if len(fields_lnormed) > 1 :
        s[-1] = -1
        outs_other = [self.att( q, k, v).transpose( -3, -2).flatten( -2, -1).reshape(s)
                                                    for (q,(k,v)) in zip(qs_other,ofields_projs)] 
        outs = torch.cat( [outs, *outs_other], -1)

    # code.interact( local=locals())
    outs = self.dropout( self.proj_out( outs))

    return x_in + outs, atts

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    return torch.matmul( self.softmax( scaling * self.score( q, k)), v)
      
  #########################################
  def score( self, q, k) :
    # code.interact( local=locals())
    return torch.matmul( q, torch.transpose( k, -2, -1))
