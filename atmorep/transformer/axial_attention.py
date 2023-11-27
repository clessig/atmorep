####################################################################################################
#
#  Copyright (C) 2022
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration / lucidrains@github.com
# 
#  description : Based on:
# https://github.com/lucidrains/axial-attention/blob/master/axial_attention/axial_attention.py
#
#  license     :
#
####################################################################################################

# 
import torch
from torch import nn
from operator import itemgetter
import code
# code.interact(local=locals())

####################################################################################################

# helper functions

def exists(val):
    return val is not None

def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))

def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)

# calculates the permutation to bring the input tensor to something attend-able
# also calculates the inverse permutation to bring the tensor back to its original shape

def calculate_permutations(num_dimensions, emb_dim, axial_dims = None) :
    
  total_dimensions = num_dimensions + 2
  emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
  if not axial_dims :
    axial_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

  permutations = []

  for axial_dim in axial_dims:
    last_two_dims = [axial_dim, emb_dim]
    dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
    permutation = [*dims_rest, *last_two_dims]
    permutations.append(permutation)
      
  return permutations

####################################################################################################

# helper classes

class PermuteToFrom(nn.Module):
    
  def __init__(self, permutation, fn):
      
    super().__init__()
    self.fn = fn
    _, inv_permutation = sort_and_return_indices(permutation)
    self.permutation = permutation
    self.inv_permutation = inv_permutation

  def forward(self, x, kv = None, **kwargs):

    if None == kv :

      axial = x.permute(*self.permutation).contiguous()

      shape = axial.shape
      *_, t, d = shape

      # merge all but axial dimension
      axial = axial.reshape(-1, t, d)

      # attention
      axial = self.fn(axial, **kwargs)

      # restore to original shape and permutation
      axial = axial.reshape(*shape)
      axial = axial.permute(*self.inv_permutation).contiguous()
      
    else :
                    
      axial = x.permute(*self.permutation)
      axial_kv = kv.permute(*self.permutation)

      shape = list(axial.shape)
      shape_kv = axial_kv.shape
      *_, t, d = shape
      *_, t_kv, d_kv = shape_kv

      # adjust token number along non-axial dimensions for efficient axial attention
      # physically consistent by inserting the missing tokens for a lower res version
      # axial dimension does not have to be matched; repeat_interleave is noop if ratio = 1
      ratio1, ratio2 = shape[1] / shape_kv[1], shape[2] / shape_kv[2]
      if int(ratio1) != 1 :
        axial_kv = torch.repeat_interleave( axial_kv, max(1,int(ratio1)), dim=1 ).contiguous()
        axial    = torch.repeat_interleave( axial, max(1,int(1./ratio1)), dim=1 ).contiguous()
      if int(ratio2) != 1 :
        axial_kv = torch.repeat_interleave( axial_kv, max(1,int(ratio2)), dim=2 ).contiguous()
        axial    = torch.repeat_interleave( axial, max(1,int(1./ratio2)), dim=2 ).contiguous()
      assert axial.shape[1:3] == axial_kv.shape[1:3], 'axial attn requires matching token numbers'


      # merge all but axial dimension
      axial_shape_perm = list(axial.shape)
      axial = axial.reshape(-1, t, d)
      axial_kv = axial_kv.reshape(-1, t_kv, d_kv)

      # attention
      axial = self.fn( axial, axial_kv, **kwargs)

      # embedding dimension does not match input since not un-embedded in head
      axial_shape_perm[-1] = -1
      shape[-1] = -1
      
      # restore original, permuted shape modulo embedding dimension
      axial = axial.reshape( *axial_shape_perm)
      
      # recover original token numbers for q if necessary by taking mean over inserted 
      if shape[1] < shape_kv[1] :
        axial = axial.reshape( shape[:2] + [int(1./ratio1)] + shape[2:] ).mean( dim=2)
      if shape[2] < shape_kv[2] :
        axial = axial.reshape( shape[:3] + [int(1./ratio2)] + shape[3:] ).mean( dim=3)

      # restore to original shape modulo embedding dimension
      axial = axial.reshape( *shape)
      axial = axial.permute( *self.inv_permutation).contiguous()

    return axial

####################################################################################################
# self attention

class SelfAttention(nn.Module):

  #############################################
  def __init__(self, dim, heads, dim_heads = None):
      
    super().__init__()
    
    self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
    dim_hidden = self.dim_heads * heads

    self.heads = heads
    self.to_q = nn.Linear(dim, dim_hidden, bias = False)
    self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias = False)
    self.to_out = nn.Linear(dim_hidden, dim)

  #############################################
  def forward(self, x, kv = None):
      
    kv = x if kv is None else kv
    q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

    b, t, d, h, e = *q.shape, self.heads, self.dim_heads

    merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
    q, k, v = map(merge_heads, (q, k, v))

    dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
    dots = dots.softmax(dim=-1)
    out = torch.einsum('bij,bje->bie', dots, v)

    out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
    out = self.to_out(out)

    return out

####################################################################################################
# axial attention class

class AxialAttention( nn.Module):

  #############################################
  def __init__(self, dim, num_dimensions = 2, heads = 8, dim_heads = None, dim_index = -1, 
                        sum_axial_out = True):

    assert (dim % heads) == 0, 'hidden dimension must be divisible by number of heads'
    super().__init__()
    
    self.dim = dim
    self.total_dimensions = num_dimensions + 2
    self.dim_index = dim_index if dim_index > 0 else (dim_index + self.total_dimensions)

    attentions = []
    for permutation in calculate_permutations(num_dimensions, dim_index):
      attentions.append(PermuteToFrom(permutation, SelfAttention(dim, heads,dim_heads)))

    self.axial_attentions = nn.ModuleList(attentions)
    self.sum_axial_out = sum_axial_out

  #############################################
  def forward(self, x, kv = None):

    assert len(x.shape) == self.total_dimensions, 'input does not have correct number of dimensions'
    assert x.shape[self.dim_index] == self.dim, 'input does not have correct input dimension'

    if self.sum_axial_out:
      return sum(map(lambda axial_attn: axial_attn(x, kv), self.axial_attentions))

    out = x
    for axial_attn in self.axial_attentions:
      out = axial_attn(out)

    return out

####################################################################################################

class CrossAttention(nn.Module):

  #############################################
  def __init__(self, dims_embed, num_heads, dim_heads):
    
    super().__init__()
    
    assert 2 == len(dims_embed)
    self.dim_heads = dim_heads
    dim_hidden = self.dim_heads * num_heads

    self.num_heads = num_heads
    self.to_q = nn.Linear( dims_embed[0], dim_hidden, bias = False) 
    self.to_kv = nn.Linear( dims_embed[1], 2 * dim_hidden, bias = False) 
    
  #############################################
  def forward(self, x, kv):
    
    # per head projection
    q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))
    
    # extract require shape / dimension parameters
    b, t, d, h, e = *q.shape, self.num_heads, self.dim_heads

    # reshape and transpose so that axial dimension is second but last
    mh = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
    q, k, v = map( mh, (q, k, v))

    # compute attention 
    dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
    dots = dots.softmax(dim=-1)
    out = torch.einsum('bij,bje->bie', dots, v)

    # recover input shape
    out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
   
    return out

####################################################################################################

class MultiFieldAxialAttention( nn.Module):

  #############################################
  def __init__(self, att_dims, dims_embed, num_heads_self, num_heads_cross, 
                     sum_axial_out = False, dropout_rate = 0.0):

    super().__init__()

    assert 0 == (dims_embed[0] % (num_heads_self + num_heads_cross))
    dim_embed_heads = int(dims_embed[0] / (num_heads_self + num_heads_cross))

    self.num_fields = len(dims_embed)
    self.dims_embed = dims_embed
    self.att_dims = att_dims
    num_dimensions = 3   # vertical, time, space (folded together) 
    idx_embed = -1

    self.sum_axial_out = sum_axial_out

    self.lnorms = torch.nn.ModuleList()
    for ifield in range( self.num_fields) :
      self.lnorms.append( nn.LayerNorm( dims_embed[ifield], elementwise_affine=False) )
    
    # for all axes, create self and cross attention heads
    self.axial_attentions = nn.ModuleList( [nn.ModuleList() for _ in att_dims] )
    for ip, permutation in enumerate(calculate_permutations( num_dimensions, idx_embed, att_dims)) :
      
      # self-attention 
      self.axial_attentions[ip].append( PermuteToFrom( permutation, CrossAttention( [dims_embed[0],dims_embed[0]], num_heads_self, dim_embed_heads)))
      
      # cross attention
      for n in range( 1, len(dims_embed)) :
        hs = PermuteToFrom( permutation, CrossAttention( [dims_embed[0], dims_embed[n]], 
                                                          num_heads_cross, dim_embed_heads))
        self.axial_attentions[ip].append( hs)

    dim_hidden = dim_embed_heads * (num_heads_self + (num_heads_cross * (self.num_fields-1)) )
    to_outs = [nn.Linear( dim_hidden, dims_embed[0]) for _ in att_dims]
    self.to_outs = torch.nn.ModuleList( to_outs)

    self.dropout = torch.nn.Dropout( p=dropout_rate)

  #############################################
  def forward(self, *args):

    pass_through = args[0]
    fields_lnormed = [self.lnorms[ifield](field) for ifield, field in enumerate( args)]
    
    out = []
    for i_ax, axial_attns in enumerate( self.axial_attentions):
      outs_axis = []
      for i, attn_heads in enumerate(axial_attns) :
        outs_axis.append( attn_heads( fields_lnormed[0], fields_lnormed[i]) )
      ax_out = self.to_outs[i_ax]( torch.cat( outs_axis, -1) )
      # parallel processing with sum_axial_out and then summation or serial otherwise
      if self.sum_axial_out :
        out.append( ax_out)
      else : 
        fields_lnormed[0] = ax_out 
    
    if self.sum_axial_out :
      out = sum( out)
    else :
      out = fields_lnormed[0]

    out = self.dropout( out)

    return pass_through + out
