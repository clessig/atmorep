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
import time

from pathlib import Path
import os
import datetime
import functools

import wandb
# import horovod.torch as hvd
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.utils.data.distributed

import atmorep.config.config as config

from atmorep.core.atmorep_model import AtmoRep
from atmorep.core.atmorep_model import AtmoRepData
from atmorep.training.bert import prepare_batch_BERT_multifield
from atmorep.transformer.transformer_base import positional_encoding_harmonic

import atmorep.utils.token_infos_transformations as token_infos_transformations

from atmorep.utils.utils import Gaussian, CRPS, kernel_crps, weighted_mse, NetMode, tokenize, detokenize
from atmorep.utils.logger import logger
from atmorep.datasets.data_writer import write_forecast, write_BERT, write_attention
from atmorep.datasets.normalizer import denormalize

####################################################################################################
class Trainer_Base() :

  def __init__( self, cf, devices ) :

    self.cf = cf
    self.devices = devices
    self.device_in = devices[0]
    self.device_out = devices[-1]

    self.fields_prediction_idx = []
    self.loss_weights = torch.zeros( len(cf.fields_prediction) )
    for ifield, field in enumerate(cf.fields_prediction) :
      self.loss_weights[ifield] = self.cf.fields_prediction[ifield][1]
      for idx, field_info in enumerate(cf.fields) :
        if field_info[0] == field[0] :
          self.fields_prediction_idx.append( idx)
          break
    self.loss_weights = self.loss_weights.to( self.device_out)

    self.MSELoss = torch.nn.MSELoss()

    # transformation for token infos
    if hasattr( cf, 'token_infos_transformation') :
      self.tok_infos_trans = getattr( token_infos_transformations, cf.token_infos_transformation)
    else :
      self.tok_infos_trans = getattr( token_infos_transformations, 'identity')

    if 0 == cf.par_rank :
      directory = Path( config.path_results, 'id{}'.format( cf.wandb_id))
      if not os.path.exists(directory):
        os.makedirs( directory)
      directory = Path( config.path_models, 'id{}'.format( cf.wandb_id))
      if not os.path.exists(directory):
        os.makedirs( directory)

  ###################################################
  def create( self, load_embeds=True) :
    net = AtmoRep( self.cf) 
    self.model = AtmoRepData( net)

    self.model.create( self.pre_batch, self.devices, load_embeds)
    
    # TODO: pass the properly to model / net
    self.model.net.encoder_to_decoder = self.encoder_to_decoder
    self.model.net.decoder_to_tail = self.decoder_to_tail
    return self

  ###################################################
  @classmethod
  def load( Typename, cf, model_id, epoch, devices) :
    trainer = Typename( cf, devices).create( load_embeds=False)
    trainer.model.net = trainer.model.net.load( model_id, devices, cf, epoch)

    # TODO: pass the properly to model / net
    trainer.model.net.encoder_to_decoder = trainer.encoder_to_decoder
    trainer.model.net.decoder_to_tail = trainer.decoder_to_tail

    epoch_print = epoch if epoch> -2 else ''
    logger.info('Loaded model id = {} at epoch = {}.', model_id, epoch_print)
    return trainer

  ###################################################
  def save( self, epoch) :
    self.model.net.save( epoch)

  ###################################################
  def get_learn_rates( self) :

    cf = self.cf
    size_padding = 5
    learn_rates = np.zeros( cf.num_epochs + size_padding) 
  
    learn_rates[:cf.lr_start_epochs] = np.linspace( cf.lr_start, cf.lr_max, num = cf.lr_start_epochs)
    lr = learn_rates[cf.lr_start_epochs-1]
    ic = 0
    for epoch in range( cf.lr_start_epochs, cf.num_epochs + size_padding) :
      lr = max( lr / cf.lr_decay_rate, cf.lr_min)
      learn_rates[epoch] = lr 
      if ic > 9999 :  # sanity check
        assert "Maximum number of epochs exceeded."

    return learn_rates

  ###################################################
  def run( self, epoch = -1) :

    cf = self.cf
    model = self.model

    learn_rates = self.get_learn_rates()
    
    if cf.with_ddp :
      self.model_ddp = torch.nn.parallel.DistributedDataParallel( model, static_graph=True)
      if not cf.optimizer_zero :
        self.optimizer = torch.optim.AdamW( self.model_ddp.parameters(), lr=cf.lr_start,
                                            weight_decay=cf.weight_decay)
      else :
        self.optimizer = ZeroRedundancyOptimizer(self.model_ddp.parameters(),
                                                              optimizer_class=torch.optim.AdamW,
                                                              lr=cf.lr_start )
    else :
      self.optimizer = torch.optim.AdamW( self.model.parameters(), lr=cf.lr_start,
                                          weight_decay=cf.weight_decay)
    
    self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cf.with_mixed_precision)

    if 0 == cf.par_rank :
      logger.debug("{}", self.model.net)
      model_parameters = filter(lambda p: p.requires_grad, self.model_ddp.parameters())
      num_params = sum([np.prod(p.size()) for p in model_parameters])
      logger.info('Number of trainable parameters: {:,}', num_params)
  
    if cf.test_initial :
      cur_test_loss = self.validate( epoch, cf.BERT_strategy).cpu().numpy()
      test_loss = np.array( [cur_test_loss])
    else :
      # generic value based on data normalization
      test_loss = np.array( [1.0]) 
    epoch += 1
   
    if cf.profile :
      lr = learn_rates[epoch]
      for g in self.optimizer.param_groups:
        g['lr'] = lr
      self.profile()

    # training loop
    while True :

      if epoch >= cf.num_epochs :
        break
        
      lr = learn_rates[epoch]
      for g in self.optimizer.param_groups:
        g['lr'] = lr

      tstr = datetime.datetime.now().strftime("%H:%M:%S")
      logger.info('{} : {} :: batch_size = {}, lr = {}',
                  epoch, tstr,cf.batch_size, lr)

      self.train( epoch)

      if cf.with_wandb and 0 == cf.par_rank :
        self.save( epoch)

      cur_test_loss = self.validate( epoch, cf.BERT_strategy).cpu().numpy()
      # self.validate( epoch, 'forecast')

      # save model 
      if cur_test_loss < test_loss.min() :
        self.save( -2)
      test_loss = np.append( test_loss, [cur_test_loss])

      epoch += 1

    tstr = datetime.datetime.now().strftime("%H:%M:%S")
    logger.info('Finished training at {} with test loss = {}.',
                tstr, test_loss[-1])

    # save final network
    if cf.with_wandb and 0 == cf.par_rank :
      self.save( -2)
    
  ###################################################
  def train( self, epoch):

    model = self.model
    cf = self.cf
    
    model.mode( NetMode.train)
    self.optimizer.zero_grad()

    loss_total = [[] for i in range(len(cf.losses)) ]
    std_dev_total = [[] for i in range(len(self.fields_prediction_idx)) ]
    mse_loss_total = []
    grad_loss_total = []
    ctr = 0

    self.optimizer.zero_grad()
    time_start = time.time()
    
    for batch_idx in range( model.len( NetMode.train)) :
      
      batch_data = self.model.next()
      _, _, _, tmksd_list, weight_list = batch_data[0]
      with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cf.with_mixed_precision):
        batch_data = self.prepare_batch( batch_data)
        preds, _ = self.model_ddp( batch_data)
        #breakpoint()
        loss, mse_loss, losses = self.loss( preds, batch_idx, tmksd_list, weight_list)
      
      self.grad_scaler.scale(loss).backward()
      self.grad_scaler.step(self.optimizer)
      self.grad_scaler.update()

      self.optimizer.zero_grad()

      [loss_total[idx].append( losses[key]) for idx, key in enumerate(losses)]
      mse_loss_total.append( mse_loss.detach().cpu() )
      grad_loss_total.append( loss.detach().cpu() )
      [std_dev_total[idx].append( pred[1].detach().cpu()) for idx, pred in enumerate(preds)]

      # logging

      if int((batch_idx * cf.batch_size) / 8) > ctr :
        
        # wandb logging
        if cf.with_wandb and (0 == cf.par_rank) :
          loss_dict = { "training loss": torch.mean( torch.tensor( mse_loss_total)), 
                        "gradient loss": torch.mean( torch.tensor( grad_loss_total)) }
          # log individual loss terms for individual fields
          for idx, cur_loss in enumerate(loss_total) :
            loss_name = self.cf.losses[idx]
            lt = torch.tensor(cur_loss)
            for i, field in enumerate(cf.fields_prediction) :
              idx_name = loss_name + ', ' + field[0]
              idx_std_name = 'stddev, ' + field[0]
              loss_dict[idx_name] = torch.mean( lt[:,i]).cpu().detach()
              loss_dict[idx_std_name] = torch.mean(torch.cat(std_dev_total[i],0)).cpu().detach()
          wandb.log( loss_dict )
      
          # console output
          samples_sec = cf.batch_size / (time.time() - time_start)
          template_str = 'epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:1.5f} : {:1.5f} :: {:1.5f} ({:2.2f} s/sec)'
          logger.info(template_str, epoch, batch_idx, model.len( NetMode.train),
                      100. * batch_idx/model.len(NetMode.train),
                      torch.mean(torch.tensor(grad_loss_total)),
                      torch.mean(torch.tensor(mse_loss_total)),
                      torch.mean(preds[0][1]), samples_sec)
    
          # save model (use -2 as epoch to indicate latest, stored without epoch specification)
          self.save( -2)

        # reset
        loss_total = [[] for i in range(len(cf.losses)) ]
        mse_loss_total = []
        grad_loss_total = []
        std_dev_total = [[] for i in range(len(self.fields_prediction_idx)) ]
        
        ctr += 1
        time_start = time.time()

    # save gradients
    if cf.save_grads and cf.with_wandb and (0 == cf.par_rank) :
    
      dir_name = './grads/id{}'.format( cf.wandb_id)
      if not os.path.exists(dir_name):
        os.makedirs(dir_name)

      rmsprop_ws = []
      for k in range( len(self.optimizer.state_dict()['state']) ) : 
        rmsprop_ws.append(self.optimizer.state_dict()['state'][k]['exp_avg_sq'].mean().unsqueeze(0))
      rmsprop_ws = torch.cat( rmsprop_ws)
      fname = '{}/{}_epoch{}_rmsprop.npy'.format( dir_name, cf.wandb_id, epoch)
      np.save( fname, rmsprop_ws.cpu().detach().numpy() ) 
     
      idx = 0
      for name, param in self.model.named_parameters():
        if param.requires_grad : 
          fname = '{}/{}_epoch{}_{:05d}_{}_grad.npy'.format( dir_name, cf.wandb_id, epoch, idx,name)
          np.save( fname, param.grad.cpu().detach().numpy() ) 
          idx += 1

    # clean memory
    self.optimizer.zero_grad()
    del batch_data, loss, loss_total, mse_loss_total, grad_loss_total, std_dev_total
      
  ###################################################
  def profile( self):

    model = self.model
    cf = self.cf

    model.mode( NetMode.train)
    self.optimizer.zero_grad()

    # See https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    # for details on how to load and analyse report
    # https://pytorch.org/blog/trace-analysis-for-masses/

    # do for all par_ranks to avoid that they run out of sync
    logger.info('---------------------------------')
    logger.info('Profiling:')
    pname = './logs/profile_par_rank' + str(cf.par_rank) + '_' + cf.wandb_id + '/profile'
    with torch.profiler.profile( activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA],
                      schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                      on_trace_ready=torch.profiler.tensorboard_trace_handler(pname),
                      profile_memory=True, record_shapes=True, with_stack=True) as prof:
      for batch_idx in range( 2 * (1+1+3) ) :

        batch_data = self.model.next()

        with torch.autocast(device_type='cuda',dtype=torch.float16, enabled=cf.with_mixed_precision):
          batch_data = self.prepare_batch( batch_data)
          preds, _ = self.model_ddp( batch_data)
          loss, mse_loss, losses = self.loss( preds, batch_idx)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        self.optimizer.zero_grad()

        prof.step()

    logger.info('Profiling finished.')
    logger.info('---------------------------------')

  ###################################################
  def validate( self, epoch, BERT_test_strategy = 'BERT'):

    cf = self.cf
    BERT_strategy_train = cf.BERT_strategy
    cf.BERT_strategy = BERT_test_strategy
    self.model.mode( NetMode.test)
    total_loss = 0.
    total_losses = torch.zeros( len(self.fields_prediction_idx) )
    test_len = 0

    self.mode_test = True
      
    # run test set evaluation
    with torch.no_grad() : 
      for it in range( self.model.len( NetMode.test)) :
        batch_data = self.model.next()
        if cf.par_rank < cf.log_test_num_ranks :
          # keep on cpu since it will otherwise clog up GPU memory
          (sources, _ , targets, tmis_list, _) = batch_data[0]
          log_sources = ( [source.detach().clone().cpu() for source in sources ],
                          [target.detach().clone().cpu() for target in targets ],
                           tmis_list)
        
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cf.with_mixed_precision):
          batch_data = self.prepare_batch( batch_data)
          preds, atts = self.model( batch_data)
        loss = torch.tensor( 0.)
        ifield = 0
        for pred, idx in zip( preds, self.fields_prediction_idx) :
          
          target = self.targets[idx]
          # hook for custom test loss
          self.test_loss( pred, target)
          # base line loss
          cur_loss = self.MSELoss( pred[0], target = target ).cpu().item()
    
          loss += cur_loss 
          total_losses[ifield] += cur_loss
          ifield += 1
  
        total_loss += loss
        test_len += 1
        
        # store detailed results on current test set for book keeping
        if cf.par_rank < cf.log_test_num_ranks :
          log_preds = [[p.detach().clone().cpu() for p in pred] for pred in preds]
          self.log_validate( epoch, it, log_sources, log_preds)
          if cf.attention:
            self.log_attention( epoch, it, atts)
                             
    # average over all nodes
    total_loss /= test_len * len(self.cf.fields_prediction)
    total_losses /= test_len

    if cf.with_ddp :
      total_loss_cuda = total_loss.cuda()
      total_losses_cuda = total_losses.cuda()
      dist.all_reduce( total_loss_cuda, op=torch.distributed.ReduceOp.AVG )
      dist.all_reduce( total_losses_cuda, op=torch.distributed.ReduceOp.AVG )
      total_loss = total_loss_cuda.cpu()
      total_losses = total_losses_cuda.cpu()

    if 0 == cf.par_rank :
      logger.info('validation loss for strategy={} at epoch {} : {}',
                  BERT_test_strategy, epoch, total_loss)

    if cf.with_wandb and (0 == cf.par_rank) :
      loss_dict = {"val. loss {}".format(BERT_test_strategy) : total_loss}
      total_losses = total_losses.cpu().detach()
      for i, field in enumerate(cf.fields_prediction) :
        idx_name = 'val., {}, '.format(BERT_test_strategy) + field[0]
        loss_dict[idx_name] = total_losses[i]
        logger.info('validation loss for {} : {}', field[0], total_losses[i])
      wandb.log( loss_dict)
    batch_data = []
    torch.cuda.empty_cache()

    cf.BERT_strategy = BERT_strategy_train
    self.mode_test = False
    
    return total_loss

  ###################################################
  def test_loss( self, pred, target) :
    '''Hook for custom test loss'''
    pass

  ###################################################
  def loss( self, preds, batch_idx = 0, tmidx_list = None, weights_list = None) :

    # TODO: move implementations to individual files

    cf = self.cf
    mse_loss_total = torch.tensor( 0.,)
    losses = dict(zip(cf.losses,[[] for loss in cf.losses ]))
    
    for pred, idx in zip( preds, self.fields_prediction_idx) :
      target = self.targets[idx]

      mse_loss = self.MSELoss( pred[0], target = target) 
      mse_loss_total += mse_loss.cpu().detach()

      # MSE loss
      if 'mse' in self.cf.losses :
        losses['mse'].append( mse_loss)

      # MSE loss
      if 'mse_ensemble' in self.cf.losses :
        loss_en = torch.tensor( 0., device=target.device)
        for en in torch.transpose( pred[2], 1, 0) :
          loss_en += self.MSELoss( en, target = target) 
        losses['mse_ensemble'].append( loss_en / pred[2].shape[1])

      if 'weighted_mse' in self.cf.losses :
        loss_en = torch.tensor( 0., device=target.device)
        field_info = cf.fields[idx]
        token_size = field_info[4]

        weights = torch.Tensor(np.array([w for batch in weights_list[idx] for w in batch]))
        weights = weights.view(*weights.shape, 1, 1).repeat(1, 1, token_size[0], token_size[2]).swapaxes(1, 2)
        weights = weights.reshape([weights.shape[0], -1]).to(target.get_device())
       
        for en in torch.transpose( pred[2], 1, 0) :
          loss_en += weighted_mse( en, target, weights)          

        losses['weighted_mse'].append( loss_en / pred[2].shape[1]) 

      # Generalized cross entroy loss for continuous distributions
      if 'stats' in self.cf.losses :
        stats_loss = Gaussian( target, pred[0], pred[1])  
        diff = (stats_loss-1.) 
        # stats_loss = 0.01 * torch.mean( diff * diff) + torch.mean( torch.sqrt(torch.abs( pred[1])) )
        stats_loss = torch.mean( diff * diff) + torch.mean( torch.sqrt( torch.abs( pred[1])) )
        losses['stats'].append( stats_loss) 
      
      # Generalized cross entroy loss for continuous distributions
      if 'stats_area' in self.cf.losses :
        diff = torch.abs( torch.special.erf( (target - pred[0]) / (pred[1] * pred[1])) )
        stats_area = 0.2 * torch.mean( diff * diff) + torch.mean( torch.sqrt(torch.abs( pred[1])) )
        losses['stats_area'].append( stats_area)

      # CRPS score
      if 'crps' in self.cf.losses :
        crps_loss = torch.mean( CRPS( target, pred[0], pred[1]))
        losses['crps'].append( crps_loss)

      if 'kernel_crps' in self.cf.losses :
        kcrps_loss = torch.mean( kernel_crps( target,torch.transpose( pred[2], 1, 0)))
        losses['kernel_crps'].append( kcrps_loss)

        
    loss = torch.tensor( 0., device=self.device_out)
    tot_weight = torch.tensor( 0., device=self.device_out)
    for key in losses :
      logger.debug('LOSS : {} :: {}', key, losses[key])
      for ifield, val in enumerate(losses[key]) :
        loss += self.loss_weights[ifield] * val.to( self.device_out)
        tot_weight += self.loss_weights[ifield] 
    loss /= tot_weight
    mse_loss = mse_loss_total / len(self.cf.fields_prediction)

    return loss, mse_loss, losses

####################################################################################################
class Trainer_BERT( Trainer_Base) :

  ###################################################
  def __init__( self, cf, devices) :
    
    Trainer_Base.__init__( self, cf, devices)

    self.rng_seed = cf.rng_seed
    if not self.rng_seed :
      self.rng_seed = int(torch.randint( 100000000, (1,))) 
    # TODO: generate only rngs that are needed
    ll = len(cf.fields) * 8 #len(cf.vertical_levels)
    if cf.BERT_fields_synced :
      self.rngs = [np.random.default_rng(self.rng_seed) for _ in range(ll)]
    else : 
      self.rngs = [np.random.default_rng(self.rng_seed+i) for i in range(ll)]

    # batch preprocessing to be done in loader (mainly for performance reasons since it's 
    # parallelized there)
    self.pre_batch = functools.partial( prepare_batch_BERT_multifield, self.cf, self.rngs, 
                                                          self.cf.fields, self.cf.BERT_strategy )

  ###################################################
  def prepare_batch( self, xin) :
    '''Move data to device and some additional final preprocessing before model eval'''

    cf = self.cf
    devs = self.devices

    # unpack loader output
    # xin[0] since BERT does not have targets
    (sources, token_infos, targets, fields_tokens_masked_idx_list, _) = xin[0]
    (self.sources_idxs, self.sources_info) = xin[2]
    
    # network input
    batch_data = [ ( sources[i].to( devs[ cf.fields[i][1][3] ], non_blocking=True), 
                    self.tok_infos_trans(token_infos[i]).to( self.devices[0], non_blocking=True)) 
                                                                  for i in range(len(sources)) ]

    # store token number since BERT selects sub-cube (optionally)
    self.num_tokens = []
    for field_idx in range(len(batch_data)) :
      self.num_tokens.append( list(batch_data[field_idx][0].shape[2:5]))

    # target
    self.targets = []
    for ifield in self.fields_prediction_idx :
      self.targets.append( targets[ifield].to( devs[cf.fields[ifield][1][3]], non_blocking=True ))

    # idxs of masked tokens
    tmi_out = [ ]
    for i,tmi in enumerate(fields_tokens_masked_idx_list) :
      cdev = devs[cf.fields[i][1][3]]
      tmi_out += [ [torch.cat(tmi_l).to( cdev, non_blocking=True) for tmi_l in tmi] ]
    self.tokens_masked_idx = tmi_out
  
    return batch_data

  ###################################################
  def encoder_to_decoder( self, embeds_layers) :
    return ([embeds_layers[i][-1] for i in range(len(embeds_layers))] , embeds_layers )

  ###################################################
  def decoder_to_tail( self, idx_pred, pred) :
    '''Positional encoding of masked tokens for tail network evaluation'''

    field_idx = self.fields_prediction_idx[idx_pred]
    dev = self.devices[ self.cf.fields[field_idx][1][3] ]
    target_idx = self.tokens_masked_idx[field_idx]
    assert len(target_idx) > 0, 'no masked tokens but target variable'
    
    # select "fixed" masked tokens for loss computation
    
    # flatten token dimensions: remove space-time separation
    pred = torch.flatten( pred, 2, 3).to( dev)
    # extract masked token level by level
    pred_masked = []
    for lidx, level in enumerate(self.cf.fields[field_idx][2]) :
      # select masked tokens, flattened along batch dimension for easier indexing and processing
      pred_l = torch.flatten( pred[:,lidx], 0, 1)
      pred_masked.append( pred_l[ target_idx[lidx] ])
    
    # flatten along level dimension, for loss evaluation we effectively have level, batch, ...
    # as ordering of dimensions
    pred_masked = torch.cat( pred_masked, 0)

    return pred_masked

  ###################################################
  def log_validate( self, epoch, bidx, log_sources, log_preds) :
    '''Hook for logging: output associated with concrete training strategy.'''

    if not hasattr( self.cf, 'wandb_id') :
      return

    if 'forecast' in self.cf.BERT_strategy :
      self.log_validate_forecast( epoch, bidx, log_sources, log_preds)
    elif 'BERT' in self.cf.BERT_strategy or 'temporal_interpolation' == self.cf.BERT_strategy :
      self.log_validate_BERT( epoch, bidx, log_sources, log_preds)
    else :
      assert False
  
  ###################################################
  def log_validate_forecast( self, epoch, batch_idx, log_sources, log_preds) :
    '''Logging for BERT_strategy=forecast.'''

    cf = self.cf

    # save source: remains identical so just save ones
    (sources, targets, _) = log_sources

    sources_out, targets_out, preds_out, ensembles_out = [ ], [ ], [ ], [ ] 
    batch_size = len(self.sources_info)  
    # reconstruct geo-coords (identical for all fields)
    forecast_num_tokens = 1
    if hasattr( cf, 'forecast_num_tokens') :
      forecast_num_tokens = cf.forecast_num_tokens
  
    coords = []
    for fidx, field_info in enumerate(cf.fields) :
      # reshape from tokens to contiguous physical field
      num_levels = len(field_info[2])
      source = detokenize( sources[fidx].cpu().detach().numpy())
      # recover tokenized shape
      target = detokenize( targets[fidx].cpu().detach().numpy().reshape( [ num_levels, -1, 
                           forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(0,1))
     
      coords_b = []
  
      for bidx in range(batch_size):
        dates   = self.sources_info[bidx][0]
        lats    = self.sources_info[bidx][1]
        lons    = self.sources_info[bidx][2]
        dates_t = self.sources_info[bidx][0][ -forecast_num_tokens*field_info[4][0] : ]
        
        lats_idx = self.sources_idxs[bidx][1]
        lons_idx = self.sources_idxs[bidx][2]

        for vidx, _ in enumerate(field_info[2]) :
          normalizer, year_base = self.model.normalizer( fidx, vidx, lats_idx, lons_idx) 
          source[bidx,vidx] = denormalize( source[bidx,vidx], normalizer, dates, year_base)
          target[bidx,vidx] = denormalize( target[bidx,vidx], normalizer, dates_t, year_base)

        coords_b += [[dates, 90.-lats, lons, dates_t]]

      # append
      sources_out.append( [field_info[0], source])
      targets_out.append( [field_info[0], target])
      coords.append(coords_b)

    # process predicted fields
    for fidx, fn in enumerate(cf.fields_prediction) :
      field_info = cf.fields[ self.fields_prediction_idx[fidx] ]
      num_levels = len(field_info[2])
      # predictions
      pred = log_preds[fidx][0].cpu().detach().numpy()
      pred = detokenize( pred.reshape( [ num_levels, -1, 
                                    forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(0,1))
      # ensemble
      ensemble = log_preds[fidx][2].cpu().detach().numpy().swapaxes(0,1)
      ensemble = detokenize( ensemble.reshape( [ cf.net_tail_num_nets, num_levels, -1, 
                                            forecast_num_tokens, *field_info[3][1:], *field_info[4] ]).swapaxes(1, 2)).swapaxes(0,1)
      
      # denormalize
      for bidx in range(batch_size) : 
        lats  = self.sources_info[bidx][1]
        lons  = self.sources_info[bidx][2]
        dates_t = self.sources_info[bidx][0][ -forecast_num_tokens*field_info[4][0] : ]
       
        for vidx, vl in enumerate(field_info[2]) :
          normalizer, year_base = self.model.normalizer( self.fields_prediction_idx[fidx], vidx, lats_idx, lons_idx)
          pred[bidx,vidx] = denormalize( pred[bidx,vidx], normalizer, dates_t, year_base)
          ensemble[bidx,:,vidx] = denormalize(ensemble[bidx,:,vidx], normalizer, dates_t, year_base)

      # append
      preds_out.append( [fn[0], pred])
      ensembles_out.append( [fn[0], ensemble])

    levels = np.array(cf.fields[0][2])
    
    write_forecast( cf.wandb_id, epoch, batch_idx,
                                 levels, sources_out,
                                 targets_out, preds_out,
                                 ensembles_out, coords)
  
  ###################################################
  
  def split_data(self, data, idx_list, token_size) :
    lens_batches = [[len(t) for t in tt] for tt in idx_list]
    lens_levels = [torch.tensor( tt).sum() for tt in lens_batches]
    data_b = torch.split( data, lens_levels)    
    # split according to batch
    return [torch.split( data_b[vidx], lens) for vidx,lens in enumerate(lens_batches)]

  def get_masked_data(self, field_info, data, idx_list, ensemble = False):
  
    cf = self.cf
    batch_size = len(self.sources_info)  
    num_levels = len(field_info[2])
    num_tokens = field_info[3]
    token_size = field_info[4]
    data_b  =  self.split_data(data, idx_list, token_size)
   
    # recover token shape
    if ensemble:
      return [[data_b[vidx][bidx].reshape([-1, cf.net_tail_num_nets, *token_size])
                                                            for bidx in range(batch_size)]
                                                            for vidx in range(num_levels)]
    else:
      return [[data_b[vidx][bidx].reshape([-1, *token_size]) for bidx in range(batch_size)]
                                                             for vidx in range(num_levels)]

  ###################################################
  def log_validate_BERT( self, epoch, batch_idx, log_sources, log_preds) :
    '''Logging for BERT_strategy=BERT.'''

    cf = self.cf
    batch_size = len(self.sources_info) 

    # save source: remains identical so just save ones
    (sources, targets, tokens_masked_idx_list) = log_sources

    sources_out, targets_out, preds_out, ensembles_out = [ ], [ ], [ ], [ ]
    coords = []

    for fidx, field_info in enumerate(cf.fields) : 

      # reconstruct coordinates
      is_predicted = fidx in self.fields_prediction_idx
      num_levels = len(field_info[2])
      num_tokens = field_info[3]
      token_size = field_info[4]
      sources_b = detokenize( sources[fidx].numpy())
     
      if is_predicted :
        targets_b   = self.get_masked_data(field_info, targets[fidx], tokens_masked_idx_list[fidx])
        preds_mu_b  = self.get_masked_data(field_info, log_preds[fidx][0], tokens_masked_idx_list[fidx])
        preds_ens_b = self.get_masked_data(field_info, log_preds[fidx][2], tokens_masked_idx_list[fidx], ensemble = True)

      # for all batch items
      coords_b = []
      for bidx in range(batch_size):
        dates = self.sources_info[bidx][0]
        lats  = self.sources_info[bidx][1]
        lons  = self.sources_info[bidx][2]

        lats_idx = self.sources_idxs[bidx][1]
        lons_idx = self.sources_idxs[bidx][2]

        # target etc are aliasing targets_b which simplifies bookkeeping below
        if is_predicted :
          target   = [targets_b[vidx][bidx] for vidx in range(num_levels)]
          pred_mu  = [preds_mu_b[vidx][bidx] for vidx in range(num_levels)]
          pred_ens = [preds_ens_b[vidx][bidx] for vidx in range(num_levels)]

        coords_mskd_l = []
        for vidx, _ in enumerate(field_info[2]) :

          normalizer, year_base = self.model.normalizer( fidx, vidx, lats_idx, lons_idx)
          sources_b[bidx,vidx] = denormalize(sources_b[bidx,vidx], normalizer, dates, year_base = 2021) 

          if is_predicted :
            idx = tokens_masked_idx_list[fidx][vidx][bidx]
            grid = np.flip(np.array( np.meshgrid( lons, lats)), axis = 0) #flip to have lat on pos 0 and lon on pos 1
            grid_idx = np.flip(np.array( np.meshgrid( lons_idx, lats_idx)), axis = 0) #flip to have lat on pos 0 and lon on pos 1
       
            # recover time dimension since idx assumes the full space-time cube
            grid = torch.from_numpy( np.array( np.broadcast_to( grid,
                                shape = [token_size[0]*num_tokens[0], *grid.shape])).swapaxes(0,1))
            grid_lats_toked = tokenize( grid[0], token_size).flatten( 0, 2)
            grid_lons_toked = tokenize( grid[1], token_size).flatten( 0, 2)
          
            idx_loc = idx - np.prod(num_tokens) * bidx
            #save only useful info for each bidx. shape e.g. [n_bidx, lat_token_size*lat_num_tokens]
            lats_mskd = np.array([np.unique(t) for t in grid_lats_toked[ idx_loc ].numpy()])
            lons_mskd = np.array([np.unique(t) for t in grid_lons_toked[ idx_loc ].numpy()])

            #time: idx ranges from 0->863 12x6x12 
            t_idx = (idx_loc // (num_tokens[1]*num_tokens[2])) * token_size[0]
            #create range from t_idx-2 to t_idx
            t_idx = np.array([np.arange(t, t + token_size[0]) for t in t_idx])
            dates_mskd = dates[t_idx]
            
            for ii,(t,p,e,da,la,lo) in enumerate(zip( target[vidx], pred_mu[vidx], pred_ens[vidx],
                                                    dates_mskd, lats_mskd, lons_mskd)) :
              normalizer_ii = normalizer
              if len(normalizer.shape) > 2: #local normalization                                     
                lats_mskd_idx = np.where(np.isin(lats,la))[0]
                lons_mskd_idx = np.where(np.isin(lons,lo))[0]
                #normalizer_ii = normalizer[:, :, lats_mskd_idx, lons_mskd_idx] problems in python 3.9
                normalizer_ii = normalizer[:, :, lats_mskd_idx[0]:lats_mskd_idx[-1]+1, lons_mskd_idx[0]:lons_mskd_idx[-1]+1]
             
              targets_b[vidx][bidx][ii]   = denormalize(t, normalizer_ii, da, year_base)  
              preds_mu_b[vidx][bidx][ii]  = denormalize(p, normalizer_ii, da, year_base) 
              preds_ens_b[vidx][bidx][ii] = denormalize(e, normalizer_ii, da, year_base)
            
            coords_mskd_l += [[dates_mskd, 90.-lats_mskd, lons_mskd] ]
       
        coords_b += [ [dates, 90. - lats, lons] + coords_mskd_l ]
      
      coords += [ coords_b ]
      fn = field_info[0]
      sources_out.append( [fn, sources_b])

      targets_out.append([fn, [[t.numpy(force=True) for t in t_v] for t_v in targets_b]] if is_predicted else [fn, []])
      preds_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_mu_b]] if is_predicted else [fn, []] )
      ensembles_out.append( [fn, [[p.numpy(force=True) for p in p_v] for p_v in preds_ens_b]] if is_predicted else [fn, []] )

    levels = [[np.array(l) for l in field[2]] for field in cf.fields]
    write_BERT( cf.wandb_id, epoch, batch_idx, 
                levels, sources_out, targets_out,
                preds_out, ensembles_out, coords )

######################################################

  def log_attention( self, epoch, bidx, attention) : 
    '''Hook for logging: output attention maps.'''
    cf = self.cf

    attn_out = []
    for fidx, field_info in enumerate(cf.fields) : 
     
      # coordinates 
      coords_b = []
      for bidx in range(batch_size):
        dates = self.sources_info[bidx][0]
        lats  = 90. - self.sources_info[bidx][1]
        lons  = self.sources_info[bidx][2]
        coords_b += [ [dates, lats, lons] ]

      is_predicted = fidx in self.fields_prediction_idx
      attn_out.append([field_info[0], attention[fidx]] if is_predicted else [fn, []])
      
    levels = [[np.array(l) for l in field[2]] for field in cf.fields]
    write_attention(cf.wandb_id, epoch,
                    bidx, levels, attn_out,  coords_b )
