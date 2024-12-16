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
from atmorep.transformer.transformer_attention import MultiSelfAttentionHead, MultiCrossAttentionHead
from atmorep.transformer.axial_attention import MultiFieldAxialAttention
from atmorep.utils.utils import identity
from atmorep.transformer.transformer_base import checkpoint_wrapper

import numpy as np
from atmorep.utils.logger import logger


class PerceiverCrossAttentionHead(torch.nn.Module):

  def __init__(self, dim_embed, coupled_emb_dims, dropout_rate = 0., with_qk_lnorm =False,
                     grad_checkpointing = False, with_attention=False, with_flash=False):
    super(PerceiverCrossAttentionHead, self).__init__()

    self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)

    self.coupled_lnorms = torch.nn.ModuleList()

    logger.info("coupled_emb_dims",coupled_emb_dims)
    if coupled_emb_dims is not None:
        for coupled_emb_dim in coupled_emb_dims:
            self.coupled_lnorms.append( torch.nn.LayerNorm( coupled_emb_dim, elementwise_affine=False))  

    self.proj_latent = torch.nn.Linear( dim_embed, dim_embed, bias = True)

    self.proj_kv= torch.nn.Linear(dim_embed,dim_embed*2,bias=True)

    self.proj_kv_coupled = torch.nn.ModuleList()

    if coupled_emb_dims is not None:
        for coupled_emb_dim in coupled_emb_dims:
            self.proj_kv_coupled.append( torch.nn.Linear( coupled_emb_dim, dim_embed*2, bias=True))
    
    self.ln_q = torch.nn.LayerNorm( dim_embed)
    self.ln_k = torch.nn.LayerNorm( dim_embed)

    self.ln_k_coupled = torch.nn.ModuleList()
    
    if coupled_emb_dims is not None:
        for _ in coupled_emb_dims:
            self.ln_k_coupled.append(torch.nn.LayerNorm( dim_embed))

    # proj_out is done is axial attention head so do not repeat it
    self.proj_out = torch.nn.Linear( dim_embed, dim_embed, bias = True)
    self.dropout = torch.nn.Dropout( p=dropout_rate)

    if with_flash :
      self.att = torch.nn.functional.scaled_dot_product_attention
    else :
      self.att = self.attention
    self.softmax = torch.nn.Softmax(dim=-1)

    self.checkpoint = identity
    if grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

  def forward( self, latent_queries, x, x_couples) :
    
    b = x.shape[0]
    x = self.lnorm( x)
    # project onto heads and q,k,v and ensure these are 4D tensors as required for flash attention
    x = x.flatten( 1, -2)
    ks, vs = torch.tensor_split( self.proj_kv( x), 2, dim=-1)
    coupled_ks, coupled_vs = [], []
    for x_couple_idx,x_couple in enumerate(x_couples):
        x_couples[x_couple_idx] = self.coupled_lnorms[x_couple_idx](x_couple).flatten( 1,-2)
        coupled_k, coupled_v = torch.tensor_split(
                self.proj_kv_coupled[x_couple_idx](x_couples[x_couple_idx]), 2, dim=-1)
        coupled_ks.append(coupled_k)
        coupled_vs.append(coupled_v)

    
    qs = self.proj_latent(latent_queries).repeat([b,1,1])

    qs, ks = self.ln_q( qs), self.ln_k( ks)

    for ln_idx,ln_k_couple in enumerate(self.ln_k_coupled):
        coupled_ks[ln_idx] = ln_k_couple(coupled_ks[ln_idx])
    
    if len(coupled_ks) > 0:
        ks = torch.cat([ks, *coupled_ks],dim=1) 
        vs = torch.cat([vs, *coupled_vs],dim=1)

    # correct ordering of tensors with seq dimension second but last is critical
    #with torch.nn.attention.sdpa_kernel( torch.nn.attention.SDPBackend.FLASH_ATTENTION) :
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION, torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]) :
      s = list(x.shape)
      s[-1] = -1
      outs = self.att( qs, ks, vs) 
    
    outs = self.dropout( self.checkpoint( self.proj_out, outs) )
    atts = []
    return outs, atts

  #########################################
  def attention( self, q, k, v) :
    scaling = 1. / torch.sqrt( torch.tensor(q.shape[-1]))
    att_output = torch.matmul( self.softmax( scaling * self.score( q, k)), v)
    return att_output
  
  #########################################
  def score( self, q, k) :
    # code.interact( local=locals())
    return torch.matmul( q, torch.transpose( k, -2, -1))
####################################################################################################


class PerceiverOutputProjection(torch.nn.Module):
    def __init__(self, input_emb, output_emb):
        super(PerceiverOutputProjection, self).__init__()
        self.input_emb = input_emb
        self.output_emb = output_emb
        self.proj = torch.nn.Linear(input_emb, output_emb)

    def forward(self, x):
        x = self.proj( x)
        return x

class Perceiver(torch.nn.Module) :
    def __init__( self, cf, field_info, coupled_embed_dims) :
        super(Perceiver, self).__init__()
        self.cf = cf
        self.num_layers = cf.perceiver_num_layers
        self.dim_embed = field_info[1][1]
         
        self.coupled_embed_dims = ( coupled_embed_dims
                                    if len(coupled_embed_dims) > 0
                                    else None)

        self.num_heads = cf.perceiver_num_heads
        self.num_mlp_layers = cf.perceiver_num_mlp_layers

        self.field_info = field_info

    def create( self):

        dim_embed = self.dim_embed
        coupled_emb_dims = self.coupled_embed_dims
        cf = self.cf
        field_info = self.field_info

        self.latent_arrays = torch.nn.Parameter(torch.empty(cf.num_latent_queries, dim_embed))

        if cf.init_scale > 0.0:
            with torch.no_grad():
                self.latent_arrays.normal_(0.0,cf.init_scale)

        self.cross_attn = PerceiverCrossAttentionHead(dim_embed, coupled_emb_dims, cf.dropout_rate, with_qk_lnorm = cf.with_qk_lnorm, grad_checkpointing = cf.grad_checkpointing)


        self.blocks = torch.nn.ModuleList()

        for il in range(cf.perceiver_num_layers):
            self.blocks.append( MultiSelfAttentionHead( dim_embed, self.num_heads, cf.dropout_rate,
                                            cf.with_qk_lnorm))

        self.output_proj = PerceiverOutputProjection( dim_embed, cf.perceiver_output_emb)

        output_shape = [len(field_info[2]), field_info[3][0], field_info[3][1], field_info[3][2]]
        self.output_latents = torch.nn.Parameter(torch.empty(np.prod(output_shape),cf.perceiver_output_emb))

        if cf.init_scale > 0.0:
            with torch.no_grad():
                self.output_latents.normal_(0.0,cf.init_scale)

        self.output_cross_attn = PerceiverCrossAttentionHead(cf.perceiver_output_emb, [], cf.dropout_rate, with_qk_lnorm=cf.with_qk_lnorm, grad_checkpointing=cf.grad_checkpointing)

        return self

    def forward( self, x, x_couples):
        
        x, atts =  self.cross_attn(self.latent_arrays, x, x_couples)
        for idx,block in enumerate(self.blocks):
            x = block(x)
        x = self.output_proj(x)
        x ,atts = self.output_cross_attn(self.output_latents, x, [])

        return x










