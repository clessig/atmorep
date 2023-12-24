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

import os
import torch
import torchinfo
import numpy as np
import code
# code.interact(local=locals())

import os
import pathlib
import datetime
import time
import math
from typing import TypeVar

import functools

import wandb

from atmorep.core.trainer import Trainer_Base
from atmorep.core.atmorep_model import AtmoRep
from atmorep.core.atmorep_model import AtmoRepData

from atmorep.applications.downscaling.atmorep_downscaling import AtmoRepDownscaling

from atmorep.training.bert import prepare_batch_BERT_multifield
from atmorep.transformer.transformer_base import positional_encoding_harmonic

import atmorep.utils.utils as utils
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import shape_to_str
from atmorep.utils.utils import get_model_filename
from atmorep.utils.utils import relMSELoss
from atmorep.utils.utils import init_torch
from atmorep.utils.utils import Gaussian
from atmorep.utils.utils import CRPS
from atmorep.utils.utils import NetMode
from atmorep.utils.utils import tokenize

####################################################################################################
class Trainer_Downscaling( Trainer_Base) :

  ###################################################
  def __init__( self, cf_downscaling, devices) :

    cf = utils.Config()
    cf.load_json( cf_downscaling.base_model_id)
    # overwrite all info that is common and add new one
    for key, value in cf_downscaling.get_self_dict().items() :
      cf.__dict__[key] = value
    # save merged config
    if cf.with_wandb and 0 == cf.hvd_rank :
      cf.write_json( wandb)
      cf.print()

    Trainer_Base.__init__( self, cf, devices)

    p = torch.randint( low=0, high=100000, size=(1,))[0].item()
    self.rngs = [np.random.default_rng(i*p) for i in range( len(cf.fields) * 8  )]
    self.rngs_targets = [np.random.default_rng(i*p) for i in range( len(cf.fields) * 8  )]
    self.pre_batch = functools.partial( prepare_batch_BERT_multifield, self.cf, self.rngs, 
                                                                       self.cf.fields, 'identity' )
    # self.pre_batch_targets = functools.partial( prepare_batch_BERT_multifield, self.cf, 
    #                                           self.rngs_targets, self.cf.fields_targets, 'BERT' )
    self.pre_batch_targets = None

    self.mode_test = False
    self.rng = np.random.default_rng()

    self.save_test_once = True

  ###################################################
  def create( self) :
    
    assert False, 'not implemented, in particular proper initalization of AtmoRep'

  ###################################################
  def load_create( self) :

    net = AtmoRepDownscaling.load_create( self.cf, self.devices)
    self.model = AtmoRepData( net)

    self.model.create( self.pre_batch, self.devices, False, self.pre_batch_targets )

    # overwrite fields predictions with fields_targets
    cf = self.cf

    fields_prediction = []
    self.fields_prediction_idx = []
    self.loss_weights = torch.zeros( len(cf.fields_targets) )
    for ifield, field in enumerate(cf.fields_targets) :
      if field[7][5] > 0. : 
        self.loss_weights[ifield] = field[7][5]
        fields_prediction.append( [field[0], field[7][5] ])
        self.fields_prediction_idx.append( ifield)
    # update 
    cf.fields_prediction = fields_prediction

    # TODO: pass the properly to model / net
    self.model.net.encoder_to_decoder = self.encoder_to_decoder
    self.model.net.decoder_to_tail = self.decoder_to_tail
    self.model.net.decoder_to_downscaler = self.decoder_to_downscaler

    return self

  ###################################################
  @classmethod
  def load( Typename, cf, model_id, epoch, devices) :
    
    trainer = Typename( cf, devices).load_create()
    trainer.model.net.load( model_id, devices, cf, epoch)

    print( 'Loaded model id = {} at epoch = {}.'.format( model_id, epoch) )

    return trainer

  ###################################################
  def prepare_batch( self, xin) :
    '''Move data to device and some additional final preprocessing before model eval'''

    cf = self.cf
    devs = self.devices

    # unpack loader output
    # xin[0] since BERT does not have targets
    (sources, token_infos, _, _, _) = xin[0]

    # network input
    dev = self.device_in
    batch_data_core = [ ( sources[i].to( devs[ cf.fields[i][1][3] ], non_blocking=True), 
        token_infos[i].to( self.devices[0], non_blocking=True) ) for i in range(len(sources)) ]

    # target
    dev = self.device_out
    self.targets = []
    self.targets_token_infos = []
    fields_injected = []
    # for all fields_target
    for ifield, target in enumerate( xin[1] ) :
      tok_size = self.cf.fields_targets[ifield][4]
      temp = []
      temp2 = []

      # process vertical levels
      for target_vl, target_token_info_vl in target :

        # TODO: all/most of this should be moved to pre_batch_targets 
        # item is field data and token_info
        target_vl_tok = tokenize( target_vl, tok_size).unsqueeze(1).to( dev, non_blocking=True)
        shape = [-1] + list(target_vl_tok.shape[1:-3]) + [self.cf.size_token_info]
        target_token_info_vl = target_token_info_vl.reshape( shape)

        # select single time step as downscaling target: currently middle one
        # TODO: should be specifiable parameter
        tstep = 3 if target_vl_tok.shape[2] > 2 else 0   # static fields need tstep = 0
        target_vl_tok = target_vl_tok[:,:,tstep].unsqueeze(2)
        
        temp.append( target_vl_tok )
        temp2.append( target_token_info_vl[:,:,tstep].unsqueeze(2) )

      # merge vertical levels
      target = torch.cat( temp, 1).flatten( -3, -1).flatten( 1, 4)
      target_token_infos = torch.cat( temp2, 1).flatten( 1, -2) 
      # targets: all fields with loss weight > 0.
      if self.cf.fields_targets[ifield][7][5] > 0. :
        self.targets.append( target.flatten( 0, 1))
        self.targets_token_infos.append( target_token_infos)
      # no parent field is specified -> injected (TODO: cleaner mechanism for this)
      if len(self.cf.fields_targets[ifield][7][4]) == 0 :
        fields_injected.append( (target.to(dev).unsqueeze(1).unsqueeze(1), 
                                 target_token_infos.to(dev)) )
    
    # move to model for later injection
    self.model.net.fields_downscaling_injected = fields_injected

    return batch_data_core

  ###################################################
  def encoder_to_decoder( self, embeds_layers) :
    return ([embeds_layers[i][-1] for i in range(len(embeds_layers))] , embeds_layers )

  ###################################################
  def decoder_to_downscaler( self, idx, token_seq) :

    # parent_field = self.cf.fields_targets[idx][6][4]
    # # TODO, handle multiple partent fields
    # found = False
    # for field_info_parent in self.cf.fields : 
    #   if parent_field == field_info_parent[0] :
    #     break
    # if not found :
    #   return (None, None)
    field_info_parent = None
    parent_field = self.cf.fields[idx][0]
    for didx, field_info in enumerate(self.cf.fields_targets) :
      if field_info[7][4] == [parent_field] :
        field_info_parent = self.cf.fields[idx]
        break
    if not field_info_parent :
      return (None, None)

    # recover individual space dimensions
    num_toks_space = field_info_parent[3][1:]
    token_seq = token_seq.reshape( list(token_seq.shape[:3]) + num_toks_space + [-1])

    # select time step (middle one)
    token_seq = token_seq[:,:,6].unsqueeze(2)
    
    # if not self.mode_test :

    #   token_seq_shape = token_seq.shape
    #   token_seq = token_seq.flatten( 0, -2)
      
    #   perm = self.rng.permutation( token_seq.shape[0])
    #   token_seq[ perm[ int( 0.1 * perm.shape[0]) ] , : ] = 0.
      
    #   token_seq = token_seq.reshape( token_seq_shape)

    token_seq = token_seq.flatten( -3, -2)

    dev = self.devices[ self.cf.fields_targets[0][1][3] ]
    return ( token_seq.to( dev, non_blocking=True), 
             self.targets_token_infos[didx].to( dev, non_blocking=True) )
    # return (token_seq, self.targets_token_infos[idx].to( token_seq.device, non_blocking=True))

  ###################################################
  def decoder_to_tail( self, idx_pred, pred) :
    '''Positional encoding of masked tokens for tail network evaluation'''

    field_idx = self.fields_prediction_idx[idx_pred]
    dev = pred.device # self.devices[ self.cf.fields[field_idx][1][3] ]

    pred = pred.flatten( 0, 3)
    num_tokens = self.cf.fields_targets[field_idx][3]
    idx = torch.arange( np.prod( pred.shape[1:-1]), device=dev)

    # compute space time indices of all tokens
    num_tokens_space = num_tokens[1] * num_tokens[2]
    # remove offset introduced by linearization
    target_idxs_t = (idx / num_tokens_space).int()
    temp = torch.remainder( idx, num_tokens_space)
    target_idxs_x = (temp / num_tokens[1]).int()
    target_idxs_y = torch.remainder( temp, num_tokens[2])

    # apply harmonic positional encoding
    dim_embed = pred.shape[-1]
    pe = torch.zeros( target_idxs_x.shape[0], dim_embed, device=dev)
    xs = (2. * np.pi / dim_embed) * torch.arange( 0, dim_embed, 2, device=dev) 
    pe[:, 0::2] = 0.5 * torch.sin( torch.outer( 8 * target_idxs_x, xs) ) \
                    + torch.sin( torch.outer( target_idxs_t, xs) )
    pe[:, 1::2] = 0.5 * torch.cos( torch.outer( 8 * target_idxs_y, xs) ) \
                    + torch.cos( torch.outer( target_idxs_t, xs) )

    pred += pe

    return pred

  ###################################################
  def log_validate( self, epoch, bidx, log_sources, log_preds) :
    '''Logging of validation set'''

    cf = self.cf
    if not hasattr( cf, 'wandb_id') :
      return

    s2s = shape_to_str

    fname_base = './results/id{}/test_id{}_rank{}_epoch{:0>5}_batch{:0>5}'
    fname_base = fname_base.format( cf.wandb_id, cf.wandb_id, cf.hvd_rank, epoch, bidx) 
    fname_base += '_{}_{}_{}.dat'
  
    # save source: remains identical so just save ones
    (sources, token_infos, _, _, tokens_masked_idx_list) = log_sources

    for fidx, field_info in enumerate(cf.fields) : 

      fn = field_info[0]

      fname = fname_base.format( 'source', fn, s2s(sources[fidx].shape))
      sources[fidx].cpu().detach().numpy().tofile( fname)

      fname = fname_base.format( 'token_infos', fn, s2s(token_infos[fidx].shape))
      token_infos[fidx].cpu().detach().numpy().tofile( fname)

    # TODO: handle generic fields_target that are not sorted by fields
    for fidx, field_info in enumerate(cf.fields_prediction) :

      fn = field_info[0]

      fname = fname_base.format( 'preds', fn, s2s(log_preds[fidx][0].shape))
      log_preds[fidx][0].cpu().detach().numpy().tofile( fname)

      fname = fname_base.format( 'target', fn, s2s(self.targets[fidx].shape))
      self.targets[fidx].cpu().detach().numpy().tofile( fname)

      ttinfos = self.targets_token_infos[fidx]
      fname = fname_base.format( 'target_token_infos', fn, s2s(ttinfos.shape))
      ttinfos.cpu().detach().numpy().tofile( fname)

      # tmi = self.tokens_masked_idx[fidx]
      # fname = fname_base.format( 'targets_tokens_masked_idx', fn, s2s(tmi.shape))
      # tmi.cpu().detach().numpy().tofile( fname)

      # fname = fname_base.format( 'ensembles', fn, s2s( log_preds[fidx][2].shape))
      # log_preds[fidx][2].cpu().detach().numpy().tofile( fname )

  # ##################################################
  # def log_validate( self, epoch, batch_idx, log_sources, log_preds) :
  #   '''Logging for BERT_strategy=BERT.'''

  #   cf = self.cf
  #   detok = utils.detokenize

  #   # save source: remains identical so just save ones
  #   (sources, token_infos, targets, tokens_masked_idx, tokens_masked_idx_list) = log_sources

  #   sources_out, targets_out, preds_out, ensembles_out = [ ], [ ], [ ], [ ]
  #   sources_dates_out, sources_lats_out, sources_lons_out = [ ], [ ], [ ]
  #   targets_dates_out, targets_lats_out, targets_lons_out = [ ], [ ], [ ]

  #   for fidx, field_info in enumerate(cf.fields) : 

  #     # reconstruct coordinates
  #     is_predicted = fidx in self.fields_prediction_idx
  #     num_levels = len(field_info[2])
  #     num_tokens = field_info[3]
  #     token_size = field_info[4]
  #     lat_d_h, lon_d_h = int(np.floor(token_size[1]/2.)), int(np.floor(token_size[2]/2.))
  #     tinfos = token_infos[fidx].reshape( [-1, num_levels, *num_tokens, cf.size_token_info])
  #     res = tinfos[0,0,0,0,0][-1].item()
  #     batch_size = tinfos.shape[0]

  #     sources_b = detok( sources[fidx].numpy())

  #     if is_predicted :
  #       # split according to levels
  #       lens_levels = [t.shape[0] for t in tokens_masked_idx[fidx]]
  #       targets_b = torch.split( targets[fidx], lens_levels)
  #       preds_mu_b = torch.split( log_preds[fidx][0], lens_levels)
  #       preds_ens_b = torch.split( log_preds[fidx][2], lens_levels)
  #       # split according to batch
  #       lens_batches = [ [bv.shape[0] for bv in b] for b in tokens_masked_idx_list[fidx] ]
  #       targets_b = [torch.split( targets_b[vidx], lens) for vidx,lens in enumerate(lens_batches)]
  #       preds_mu_b = [torch.split(preds_mu_b[vidx], lens) for vidx,lens in enumerate(lens_batches)]
  #       preds_ens_b =[torch.split(preds_ens_b[vidx],lens) for vidx,lens in enumerate(lens_batches)]
  #       # recover token shape
  #       targets_b = [[targets_b[vidx][bidx].reshape([-1, *token_size]) 
  #                                                                   for bidx in range(batch_size)]
  #                                                                   for vidx in range(num_levels)]
  #       preds_mu_b = [[preds_mu_b[vidx][bidx].reshape([-1, *token_size]) 
  #                                                                   for bidx in range(batch_size)]
  #                                                                   for vidx in range(num_levels)]
  #       preds_ens_b = [[preds_ens_b[vidx][bidx].reshape( [-1, cf.net_tail_num_nets, *token_size])
  #                                                                    for bidx in range(batch_size)]
  #                                                                    for vidx in range(num_levels)]

  #     # for all batch items
  #     coords_b = []
  #     for bidx, tinfo in enumerate(tinfos) :

  #       # use first vertical levels since a column is considered
  #       lats = np.arange(tinfo[0,0,0,0,4]-lat_d_h*res, tinfo[0,0,-1,0,4]+lat_d_h*res+0.001,res)
  #       if tinfo[0,0,0,-1,5] < tinfo[0,0,0,0,5] :
  #         lons = np.remainder( np.arange( tinfo[0,0,0,0,5] - lon_d_h*res, 
  #                                       360. + tinfo[0,0,0,-1,5] + lon_d_h*res + 0.001, res), 360.)
  #       else :
  #         lons = np.arange(tinfo[0,0,0,0,5]-lon_d_h*res, tinfo[0,0,0,-1,5]+lon_d_h*res+0.001,res)
  #       lons = np.remainder( lons, 360.)

  #       # time stamp in token_infos is at start time so needs to be advanced by token_size[0]-1
  #       s = utils.token_info_to_time( tinfo[0,0,0,0,:3] ) - pd.Timedelta(hours=token_size[0]-1)
  #       e = utils.token_info_to_time( tinfo[0,-1,0,0,:3] )
  #       dates = pd.date_range( start=s, end=e, freq='h')

  #       # target etc are aliasing targets_b which simplifies bookkeeping below
  #       if is_predicted :
  #         target = [targets_b[vidx][bidx] for vidx in range(num_levels)]
  #         pred_mu = [preds_mu_b[vidx][bidx] for vidx in range(num_levels)]
  #         pred_ens = [preds_ens_b[vidx][bidx] for vidx in range(num_levels)]

  #       dates_masked_l, lats_masked_l, lons_masked_l = [], [], []
  #       for vidx, _ in enumerate(field_info[2]) :

  #         normalizer = self.model.normalizer( fidx, vidx)
  #         y, m = dates[0].year, dates[0].month
  #         sources_b[bidx,vidx] = normalizer.denormalize( y, m, sources_b[bidx,vidx], [lats, lons])

  #         if is_predicted :

  #           # TODO: make sure normalizer_local / normalizer_global is used in data_loader
  #           idx = tokens_masked_idx_list[fidx][vidx][bidx]
  #           tinfo_masked = tinfos[bidx,vidx].flatten( 0,2)
  #           tinfo_masked = tinfo_masked[idx]
  #           lad, lod = lat_d_h*res, lon_d_h*res
  #           lats_masked, lons_masked, dates_masked = [], [], []
  #           for t in tinfo_masked :

  #             lats_masked.append( np.expand_dims( np.arange(t[4]-lad, t[4]+lad+0.001,res), 0))
  #             lons_masked.append( np.expand_dims( np.arange(t[5]-lod, t[5]+lod+0.001,res), 0))

  #             r = pd.date_range( start=utils.token_info_to_time(t), periods=token_size[0], freq='h')
  #             dates_masked.append( np.expand_dims(r.to_pydatetime().astype( 'datetime64[s]'), 0) )

  #           lats_masked = np.concatenate( lats_masked, 0)
  #           lons_masked = np.remainder( np.concatenate( lons_masked, 0), 360.)
  #           dates_masked = np.concatenate( dates_masked, 0)

  #           for ii,(t,p,e,la,lo) in enumerate(zip( target[vidx], pred_mu[vidx], pred_ens[vidx],
  #                                                   lats_masked, lons_masked)) :
  #             targets_b[vidx][bidx][ii] = normalizer.denormalize( y, m, t, [la, lo])
  #             preds_mu_b[vidx][bidx][ii]  = normalizer.denormalize( y, m, p, [la, lo])
  #             preds_ens_b[vidx][bidx][ii] = normalizer.denormalize( y, m, e, [la, lo])

  #           dates_masked_l += [ dates_masked ]
  #           lats_masked_l += [ [90.-lat for lat in lats_masked] ]
  #           lons_masked_l += [ lons_masked ]

  #       dates = dates.to_pydatetime().astype( 'datetime64[s]')

  #       coords_b += [ [dates, 90.-lats, lons, dates_masked_l, lats_masked_l, lons_masked_l] ]

  #     fn = field_info[0]
  #     sources_out.append( [fn, sources_b])
  #     if is_predicted :
  #       targets_out.append([fn, [[t.numpy(force=True) for t in t_v] for t_v in targets_b]])
  #       preds_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_mu_b]])
  #       ensembles_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_ens_b]])
  #     else :
  #       targets_out.append( [fn, []])
  #       preds_out.append( [fn, []])
  #       ensembles_out.append( [fn, []])

  #     sources_dates_out.append( [c[0] for c in coords_b])
  #     sources_lats_out.append( [c[1] for c in coords_b])
  #     sources_lons_out.append( [c[2] for c in coords_b])
  #     if is_predicted :
  #       targets_dates_out.append( [c[3] for c in coords_b])
  #       targets_lats_out.append( [c[4] for c in coords_b])
  #       targets_lons_out.append( [c[5] for c in coords_b])
  #     else :
  #       targets_dates_out.append( [ ])
  #       targets_lats_out.append( [ ])
  #       targets_lons_out.append( [ ])

  #   levels = [[np.array(l) for l in field[2]] for field in cf.fields]
  #   write_BERT( cf.wandb_id, epoch, batch_idx,
  #                            levels, sources_out,
  #                            [sources_dates_out, sources_lats_out, sources_lons_out],
  #                            targets_out, [targets_dates_out, targets_lats_out, targets_lons_out],
  #                            preds_out, ensembles_out )

  # def log_attention( self, epoch, bidx, log) : 
  #   '''Hook for logging: output attention maps.'''
  #   cf = self.cf

  #   attention, token_infos = log
  #   attn_dates_out, attn_lats_out, attn_lons_out = [ ], [ ], [ ]
  #   attn_out = []
  #   for fidx, field_info in enumerate(cf.fields) : 
  #     # reconstruct coordinates
  #     is_predicted = fidx in self.fields_prediction_idx
  #     num_levels = len(field_info[2])
  #     num_tokens = field_info[3]
  #     token_size = field_info[4]
  #     lat_d_h, lon_d_h = int(np.floor(token_size[1]/2.)), int(np.floor(token_size[2]/2.))
  #     tinfos = token_infos[fidx].reshape( [-1, num_levels, *num_tokens, cf.size_token_info])
  #     coords_b = []

  #     for tinfo in tinfos :
  #       # use first vertical levels since a column is considered
  #       res = tinfo[0,0,0,0,-1]
  #       lats = np.arange(tinfo[0,0,0,0,4]-lat_d_h*res, tinfo[0,0,-1,0,4]+lat_d_h*res+0.001,res*token_size[1])
  #       if tinfo[0,0,0,-1,5] < tinfo[0,0,0,0,5] :
  #         lons = np.remainder( np.arange( tinfo[0,0,0,0,5] - lon_d_h*res, 
  #                                         360. + tinfo[0,0,0,-1,5] + lon_d_h*res + 0.001, res*token_size[2]), 360.)
  #       else :
  #         lons = np.arange(tinfo[0,0,0,0,5]-lon_d_h*res, tinfo[0,0,0,-1,5]+lon_d_h*res+0.001,res*token_size[2])
  #       lons = np.remainder( lons, 360.)

  #       dates = np.array([(utils.token_info_to_time(tinfo[0,t,0,0,:3])) for t in range(tinfo.shape[1])], dtype='datetime64[s]')
  #       coords_b += [ [dates, lats, lons] ]

  #     if is_predicted:
  #       attn_out.append([field_info[0], attention[fidx]])
  #       attn_dates_out.append([c[0] for c in coords_b])
  #       attn_lats_out.append( [c[1] for c in coords_b])
  #       attn_lons_out.append( [c[2] for c in coords_b])
  #     else:
  #       attn_dates_out.append( [] )
  #       attn_lats_out.append( [] )
  #       attn_lons_out.append( [] )
        
  #   levels = [[np.array(l) for l in field[2]] for field in cf.fields]
  #   write_attention(cf.wandb_id, epoch,
  #                   bidx, levels, attn_out, [attn_dates_out,attn_lons_out,attn_lons_out])