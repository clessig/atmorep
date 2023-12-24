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
import collections
import math
import numpy as np
import copy
import code
# code.interact(local=locals())

import atmorep.utils.utils as utils

from atmorep.transformer.transformer_base import prepare_token
from atmorep.transformer.transformer_base import MLP

from atmorep.transformer.transformer_downscaler import TransformerDownscaler
from atmorep.transformer.tail_ensemble import TailEnsemble

from atmorep.transformer.transformer_base import positional_encoding_harmonic
from atmorep.transformer.transformer_base import positional_encoding_harmonic_absolute

from atmorep.core.atmorep_model import AtmoRep

####################################################################################################
class AtmoRepDownscaling( AtmoRep) :

  def __init__(self, cf) :
    '''Constructor'''
 
    super( AtmoRepDownscaling, self).__init__( cf)

  ###################################################
  @staticmethod
  def load_create( cf, devices) :
  
    net = AtmoRepDownscaling( cf)
    net = super( AtmoRepDownscaling, net).create( devices)

    # net = AtmoRep.load( cf.base_model_id, devices, cf, cf.base_epoch)
    # cast base class to AtmoRepDownscaling
    # net.__class__ = AtmoRepDownscaling

    # disable optimization as specified
    if not cf.downscaling_optimize_encoder :
      net.embed_token_info.requires_grad = False
      net.encoders.requires_grad = False

    # disable optimization as specified
    if not cf.downscaling_optimize_decoder :
      net.decoders.requires_grad = False

    # use cf of loaded model during loading but then overwrite with the current one
    net.create( devices)

    return net
  
  ###################################################
  # @staticmethod
  def load( self, model_id, devices, cf = None, epoch = -2) :
    '''Load network from checkpoint'''

    if not cf : 
      cf = utils.Config()
      cf.load_json( model_id)

    self.load_state_dict( torch.load( utils.get_model_filename( self, model_id, epoch) ) )

    print( 'Loaded model {} at epoch {}'.format( model_id, epoch))

  ###################################################
  def save( self, epoch = -2) :
    '''Save network '''

    # save entire network
    torch.save( self.state_dict(), utils.get_model_filename( self, self.cf.wandb_id, epoch) )

    # TODO: save individual downscaling parts

  ###################################################
  def create( self, devices) :
    '''Create network'''

    cf = self.cf
    # self.devices = devices
    self.downscaler_fields_coupling_idx = []

    self.fields_index = {}
    for ifield, field_info in enumerate(cf.fields_targets) :
      self.fields_index[ field_info[0] ] = ifield 
    
    # downscaling specific net

    self.field_pred_idxs = []
    for ifield, field_info in enumerate(cf.fields_targets) :
      if field_info[7][5] > 0. :  # only fields with non-zero weight
        self.field_pred_idxs.append( ifield)

    # sanity checking input (code should be generalized that it can handle general case)
    for ifield, field_info in enumerate(cf.fields_targets) :
      if len(field_info[7][4]) > 0 :
        assert len(field_info[7][4]) == 1, 'Only one parent field currently supported.'
        assert field_info[7][4] != cf.fields[ifield][0], 'Indices of corresponding fields in' \
                                                + 'cf.fields and cf.fields_targets have to match.'

    self.downscaler_embed_token_info = torch.nn.Linear( cf.size_token_info, cf.size_token_info_net)
    if len(cf.fields_targets[0][1]) > 4 :
      # load if field[0] is loaded
      assert 0
      name = self.__class__.__name__ + '_embed_token_info'
      mloaded = torch.load( get_model_filename( name, cf.fields[0][1][4][0], cf.fields[0][1][4][1]))
      self.embed_token_info.load_state_dict( mloaded)
      print( 'Loaded embed_token_info from id = {}.'.format( cf.fields[0][1][4][0] ) )
    else :
      # no convergence without proper initalization
      torch.nn.init.constant_( self.downscaler_embed_token_info.weight, 0.01)
      self.downscaler_embed_token_info.bias.data.fill_(0.01)

    # todo, todo, todo: for all fields: list of different downscalers that can also 
    # adapt dimension for the case of multiple parents fields
    s_tok_info = cf.size_token_info_net
    self.pre_downscalers = torch.nn.ModuleList()
    self.downscaler = torch.nn.ModuleList()

    for field_idx, field_info in enumerate(cf.fields_targets) : 

      # if no parent field is specified then it will be "injected" before the downscaling network
      if len(field_info[7][4]) > 0 :
        # look up embedding dimension of parent field
        parent_field = field_info[7][4]  
        for field_info_p in cf.fields :
          if field_info_p[0] == parent_field[0] :
            dim_embed_p = field_info_p[1][1]
            break
        dim_in = dim_embed_p + s_tok_info 
        dim_out = field_info[1][1] + s_tok_info
        self.pre_downscalers.append( MLP( dim_in, dim_embed_out=dim_out, with_residual=False))
      else : 
        # mapping network here effectively acts as embedding network
        dim_in = np.prod(field_info[4]) + s_tok_info
        dim_out = field_info[1][1] + s_tok_info
        self.pre_downscalers.append( MLP( dim_in, dim_embed_out=dim_out, with_residual=False))
      
      self.downscaler.append( TransformerDownscaler( cf, field_idx, False).create())

      # indices of coupled fields for efficient access in forward
      self.downscaler_fields_coupling_idx.append( [field_idx])
      for field_coupled in field_info[1][2] : 
        if 'axial' in cf.encoder_att_type :
          self.downscaler_fields_coupling_idx[field_idx].append( self.fields_index[field_coupled] )
        else :
          for _ in range(cf.coupling_num_heads_per_field) :
            self.downscaler_fields_coupling_idx[field_idx].append(self.fields_index[field_coupled])

    s_tok_info = cf.size_token_info_net
    self.downscaler_tails = torch.nn.ModuleList()
    for ifield, field_info in enumerate(cf.fields_targets) :
      nps = np.prod(field_info[4])
      self.downscaler_tails.append( TailEnsemble( cf, field_info[1][1]+s_tok_info, 
                                                  nps, cf.downscaling_net_tail_num_nets ).create())

    # # initialize to small weights
    # utils.init_weights_uniform( self.downscaler, 0.075)
    # utils.init_weights_uniform( self.downscaler_tails, 0.05)

    # set devices

    for field_idx, field_info in enumerate(cf.fields_targets) :
      # field_info = cf.fields[ self.field_pred_idxs[field_idx] ]
      device = self.devices[0]
      if len(field_info[1]) > 3 :
        device = self.devices[ field_info[1][3] ]
      self.downscaler[field_idx].to(device)
      self.downscaler_tails[field_idx].to(device)
      self.pre_downscalers[field_idx].to(device)
      # self.downscaler_embed_token_info.to( device)
    self.downscaler_embed_token_info.to( self.devices[-1])

  ###################################################
  def forward( self, xin) :
    '''Evaluate network'''

    # (xin_core, xin_down, xin_down_mask) = xin
    xin_core = xin

    # embedding
    cf = self.cf
    emb_net_ti = self.embed_token_info
    fields_embed = [prepare_token( field_data, emb_net, emb_net_ti, cf.with_cls ) 
                          for fidx,(field_data,emb_net) in enumerate(zip( xin_core, self.embeds))]
    
    # encoder
    embeds_layers = [[] for i in range(len(fields_embed))]
    for ib in range(self.cf.encoder_num_layers) :
      fields_embed, _ = self.forward_encoder_block( ib, fields_embed) 
      [embeds_layers[idx].append( field) for idx, field in enumerate(fields_embed)]
      # [embeds_layers[idx].append( fields_embed[i]) for idx,i in enumerate(self.field_pred_idxs)]
        
    # encoder-decoder coupling / token transformations
    (decoders_in, embeds_layers) = self.encoder_to_decoder( embeds_layers)

    fields_embed = []
    for idx, _ in enumerate( decoders_in) :
    
      # TODO, TODO, TODO: handle generically
      # decoder
      # if 2 == idx :
      # if 0 == idx :
      if 0 == idx or 2 == idx :
        field_embed, _ = self.decoders[idx]( (decoders_in[idx], embeds_layers[idx]) )
      else :
        field_embed = None

      field_embed = self.decoder_to_downscaler( idx, field_embed)
      fields_embed.append( field_embed)

    # # "inject" fields that have no parent field and are passed directly to downscaling part
    # if len( self.fields_downscaling_injected) > 0 :
    #   fields_embed.append( *self.fields_downscaling_injected)

    # embed and concatenate token infos
    embed_token_info = self.downscaler_embed_token_info
    fields_embed_temp = []
    fi = 0
    for _, (field_embed,token_info) in enumerate(fields_embed) : 
      if field_embed == None :
        continue
      field_embed = field_embed[:,-1].unsqueeze(1) # use only last / lowest level
      # add auxiliary, global token information
      token_info = embed_token_info( token_info)
      token_info = token_info.reshape([-1] + list(field_embed.shape[1:-1])+[token_info.shape[-1]])
      field_embed = torch.cat( [field_embed, token_info], -1)
      # positional encoding for downscaler
      field_embed = positional_encoding_harmonic( field_embed, len(cf.fields_targets[fi][2]), 
                                                                cf.fields_targets[fi][3])
      fields_embed_temp.append( self.pre_downscalers[fi]( field_embed) )
      fi += 1
    fields_embed = fields_embed_temp

    #  evaluate downscaler
    for ib in range(cf.downscaling_num_layers) :
      fields_embed = self.forward_downscaler_block( ib, fields_embed)

    #  tail networks
    preds = []
    for idx, _ in enumerate(self.field_pred_idxs) :
      tail_in = self.decoder_to_tail( idx, fields_embed[idx] )
      pred = self.checkpoint( self.downscaler_tails[idx], tail_in)
      preds.append( pred)

    return preds, None

  ###################################################
  def forward_downscaler_block( self, iblock, fields_embed) :
    ''' evaluate one block (attention and mlp) '''

    # double buffer for commutation-invariant result (w.r.t evaluation order of transformers)
    fields_embed_cur = []

    # attention heads
    for ifield in range( len(fields_embed)) :
      dev = fields_embed[ifield].device
      fields_in = [fields_embed[i].to(dev) for i in self.downscaler_fields_coupling_idx[ifield]]
      # unpack list in argument for checkpointing
      y, _ = self.checkpoint( self.downscaler[ifield].heads[iblock], *fields_in)
      fields_embed_cur.append( y )

    # MLPs 
    for ifield in range( len(fields_embed)) :
      fields_embed_cur[ifield] = self.checkpoint( self.downscaler[ifield].mlps[iblock],
                                                  fields_embed_cur[ifield] )

    return fields_embed_cur
