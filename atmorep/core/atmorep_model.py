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
import code
import os
# code.interact(local=locals())

# import horovod.torch as hvd
import atmorep.config.config as config
import atmorep.utils.utils as utils
from atmorep.utils.utils import identity
from atmorep.utils.utils import NetMode
from atmorep.utils.utils import get_model_filename

from atmorep.transformer.transformer_base import prepare_token
from atmorep.transformer.transformer_base import checkpoint_wrapper

from atmorep.datasets.multifield_data_sampler import MultifieldDataSampler

from atmorep.transformer.transformer_encoder import TransformerEncoder
from atmorep.transformer.transformer_decoder import TransformerDecoder
from atmorep.transformer.tail_ensemble import TailEnsemble


####################################################################################################
class AtmoRepData( torch.nn.Module) :

  def __init__( self, net) :
    '''Wrapper class for AtmoRep that handles data loading'''

    super( AtmoRepData, self).__init__()
    
    self.data_loader_test = None
    self.data_loader_train = None
    self.data_loader_iter = None

    self.net = net

    # ensure that all data loaders have the same seed and hence load the same data
    self.rng_seed = net.cf.rng_seed 
    if not self.rng_seed :
      self.rng_seed = int(torch.randint( 100000000, (1,))) 

  ###################################################
  def set_data( self, mode : NetMode, times_pos, batch_size = -1, num_loader_workers = -1) :

    cf = self.net.cf
    if batch_size < 0 :
      batch_size = cf.batch_size_train if mode == NetMode.train else cf.batch_size_test
    
    dataset = self.dataset_train if mode == NetMode.train else self.dataset_test
    dataset.set_data( times_pos, batch_size)
    
    self._set_data( dataset, mode, batch_size, num_loader_workers)

  ###################################################
  def set_global( self, mode : NetMode, times, batch_size = -1, num_loader_workers = -1) :

    cf = self.net.cf
    if batch_size < 0 :
      batch_size = cf.batch_size_train if mode == NetMode.train else cf.batch_size_test
    dataset = self.dataset_train if mode == NetMode.train else self.dataset_test
    dataset.set_global( times, batch_size, cf.token_overlap)

    self._set_data( dataset, mode, batch_size, num_loader_workers)

  ###################################################
  def set_location( self, mode : NetMode, pos, years, months, num_t_samples_per_month, 
                          batch_size = -1, num_loader_workers = -1) :

    cf = self.net.cf
    if batch_size < 0 :
      batch_size = cf.batch_size_train if mode == NetMode.train else cf.batch_size_test
    
    dataset = self.dataset_train if mode == NetMode.train else self.dataset_test
    dataset.set_location( pos, years, months, num_t_samples_per_month, batch_size)

    self._set_data( dataset, mode, batch_size, num_loader_workers)

  ###################################################
  def _set_data( self, dataset, mode : NetMode, batch_size = -1, loader_workers = -1) :
    '''Private implementation for set_data, set_global'''

    cf = self.net.cf
    if loader_workers < 0 :
      loader_workers = cf.num_loader_workers

    loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False, 
                      'num_workers': loader_workers, 'pin_memory': True}
    
    if mode == NetMode.train :
      self.data_loader_train = torch.utils.data.DataLoader( dataset, **loader_params, 
                                                            sampler = None)
    elif mode == NetMode.test :
      self.data_loader_test = torch.utils.data.DataLoader( dataset, **loader_params, 
                                                           sampler = None)
    else :
      assert False

  ###################################################
  def normalizer( self, field, vl_idx, lats_idx, lons_idx ) :

    if isinstance( field, str) :
      for fidx, field_info in enumerate(self.cf.fields) :
        if field == field_info[0] :
          break
      assert fidx < len(self.cf.fields), 'invalid field'
      normalizer = self.dataset_train.datasets[fidx].normalizer

    elif isinstance( field, int) :
      normalizer = self.dataset_train.normalizers[field][vl_idx]
      if len(normalizer.shape) > 2:
        normalizer = np.take( np.take( normalizer, lats_idx, -2), lons_idx, -1)
    else :
      assert False, 'invalid argument type (has to be index to cf.fields or field name)'
    
    year_base = self.dataset_train.year_base

    return normalizer, year_base

  ###################################################
  def mode( self, mode : NetMode) :
    
    if mode == NetMode.train :
      self.data_loader_iter = iter(self.data_loader_train)
      self.net.train()
    elif mode == NetMode.test :
      self.data_loader_iter = iter(self.data_loader_test)
      self.net.eval()
    else :
      assert False

    self.cur_mode = mode

  ###################################################
  def len( self, mode : NetMode) :
    if mode == NetMode.train :
      return len(self.data_loader_train)
    elif mode == NetMode.test :
      return len(self.data_loader_test)
    else :
      assert False

  ###################################################
  def next( self) :
    return next(self.data_loader_iter)

  ###################################################
  def forward( self, xin) :
    pred = self.net.forward( xin)
    return pred

  ###################################################
  def get_attention( self, xin) :
    attn = self.net.get_attention( xin)
    return attn

  ###################################################
  def create( self, pre_batch, devices, create_net = True, pre_batch_targets = None,
              load_pretrained=True) :

    if create_net :
      self.net.create( devices, load_pretrained)

    self.pre_batch = pre_batch
    self.pre_batch_targets = pre_batch_targets

    cf = self.net.cf
    loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False, 
                      'num_workers': cf.num_loader_workers, 'pin_memory': True}

    self.dataset_train = MultifieldDataSampler( config.path_data, cf.fields, cf.years_train,
                                                cf.batch_size,
                                                pre_batch, cf.n_size, cf.num_samples_per_epoch,
                                                with_shuffle = (cf.BERT_strategy != 'global_forecast'), 
                                                with_source_idxs = True, 
                                                compute_weights = (cf.losses.count('weighted_mse') > 0) )
    self.data_loader_train = torch.utils.data.DataLoader( self.dataset_train, **loader_params,
                                                          sampler = None)

    self.dataset_test = MultifieldDataSampler( config.path_data, cf.fields, cf.years_val,
                                               cf.batch_size_validation,
                                               pre_batch, cf.n_size, cf.num_samples_validate,
                                               with_shuffle = (cf.BERT_strategy != 'global_forecast'),
                                               with_source_idxs = True, 
                                               compute_weights = (cf.losses.count('weighted_mse') > 0) )                                               
    self.data_loader_test = torch.utils.data.DataLoader( self.dataset_test, **loader_params,
                                                          sampler = None)

    return self

####################################################################################################
class AtmoRep( torch.nn.Module) :

  def __init__(self, cf) :
    '''Constructor'''
 
    super( AtmoRep, self).__init__()

    self.cf = cf

  ###################################################
  def create( self, devices, load_pretrained=True) :
    '''Create network'''

    cf = self.cf
    self.devices = devices
    self.fields_coupling_idx = []

    self.fields_index = {}
    for ifield, field_info in enumerate(cf.fields) :
      self.fields_index[ field_info[0] ] = ifield 
    
    # # embedding network for global/auxiliary token infos
    # TODO: only for backward compatibility, remove
    self.embed_token_info = torch.nn.Linear( cf.size_token_info, cf.size_token_info_net)
    torch.nn.init.constant_( self.embed_token_info.weight, 0.0)

    self.embeds_token_info = torch.nn.ModuleList()
    for ifield, field_info in enumerate( cf.fields) :
      
      self.embeds_token_info.append( torch.nn.Linear( cf.size_token_info, cf.size_token_info_net))
      
      if len(field_info[1]) > 4 and load_pretrained :
        # TODO: inconsistent with embeds_token_info -> version that can handle both
        #       we could imply use the file name: embed_token_info vs embeds_token_info
        name = 'AtmoRep' + '_embeds_token_info'
        if not os.path.exists(get_model_filename( name, field_info[1][4][0], field_info[1][4][1])):
          name = 'AtmoRep' + '_embed_token_info'

        mloaded = torch.load( get_model_filename( name, field_info[1][4][0], field_info[1][4][1]))
      
        if "weight" not in mloaded.keys(): #TODO: get rid of this
          mloaded["weight"] = mloaded["0.weight"]
          mloaded["bias"] = mloaded["0.bias"]
          del mloaded["0.weight"]
          del mloaded["0.bias"]

        self.embeds_token_info[-1].load_state_dict( mloaded)
        print( 'Loaded embed_token_info from id = {}.'.format( field_info[1][4][0] ) )
      else :
        # initalization
        torch.nn.init.constant_( self.embeds_token_info[-1].weight, 0.0)
        self.embeds_token_info[-1].bias.data.fill_(0.0)

    # embedding and encoder

    self.embeds = torch.nn.ModuleList()
    self.encoders = torch.nn.ModuleList()

    for field_idx, field_info in enumerate(cf.fields) : 

      # encoder
      self.encoders.append( TransformerEncoder( cf, field_idx, True).create())
      # load pre-trained model if specified
      if len(field_info[1]) > 4 and load_pretrained :
        self.load_block( field_info, 'encoder', self.encoders[-1])
      self.embeds.append( self.encoders[-1].embed)
     
      # indices of coupled fields for efficient access in forward
      self.fields_coupling_idx.append( [field_idx])
      for field_coupled in field_info[1][2] : 
        if 'axial' in cf.encoder_att_type :
          self.fields_coupling_idx[field_idx].append( self.fields_index[field_coupled] )
        else :
          for _ in range(cf.coupling_num_heads_per_field) :
            self.fields_coupling_idx[field_idx].append( self.fields_index[field_coupled] )

    # decoder 

    self.decoders = torch.nn.ModuleList()
    self.field_pred_idxs = []
    for field in cf.fields_prediction :

      for ifield, field_info in enumerate(cf.fields) : 
        if field_info[0] == field[0] :
          self.field_pred_idxs.append( ifield)
          break

      self.decoders.append( TransformerDecoder( cf, field_info ) )
      # load pre-trained model if specified
      if len(field_info[1]) > 4 and load_pretrained :
        self.load_block( field_info, 'decoder', self.decoders[-1])

    # tail networks
    
    self.tails = torch.nn.ModuleList()
    for ifield, field in enumerate(cf.fields_prediction) :

      field_idx = self.field_pred_idxs[ifield]
      field_info = cf.fields[field_idx]
      self.tails.append( TailEnsemble( cf, field_info[1][1], np.prod(field_info[4]) ).create())
      # load pre-trained model if specified
      if len(field_info[1]) > 4 and load_pretrained:
        self.load_block( field_info, 'tail', self.tails[-1])

    # set devices

    for field_idx, field_info in enumerate(cf.fields) :
      # find determined device, use default if nothing specified
      device = self.devices[0]
      if len(field_info[1]) > 3 :
        assert field_info[1][3] < 4, 'Only single node model parallelism supported'
        print(devices, field_info[1][3])
        assert field_info[1][3] < len(devices), 'Per field device id larger than max devices'
        device = self.devices[ field_info[1][3] ]
      # set device
      self.embeds[field_idx].to(device)
      self.encoders[field_idx].to(device)

    for field_idx, field in enumerate(cf.fields_prediction) :
      field_info = cf.fields[ self.field_pred_idxs[field_idx] ]
      device = self.devices[0]
      if len(field_info[1]) > 3 :
        device = self.devices[ field_info[1][3] ]
      self.decoders[field_idx].to(device)
      self.tails[field_idx].to(device)

    # embed_token_info on device[0] since it is shared by all fields, potentially sub-optimal
    self.embed_token_info.to(devices[0])  # TODO: only for backward compatibility, remove
    self.embeds_token_info.to(devices[0])

    self.checkpoint = identity
    if cf.grad_checkpointing :
      self.checkpoint = checkpoint_wrapper

    return self

  ###################################################
  def load_block( self, field_info, block_name, block ) :

    # name = self.__class__.__name__ + '_' + block_name + '_' + field_info[0]
    name = 'AtmoRep_' + block_name + '_' + field_info[0]

    b_loaded = torch.load( get_model_filename(name, field_info[1][4][0], field_info[1][4][1]))

    # in coupling mode, proj_out of attention heads needs separate treatment: only the pre-trained
    # part can be loaded
    keys_del = []
    for name, param in block.named_parameters():
      if 'proj_out' in name :
        for k in b_loaded.keys() :
          if name == k :
            if param.shape[0] != param.shape[1] :   # non-square proj_out indicate deviation from pre-training
              with torch.no_grad() :
                # load pre-trained part
                param[ : , : b_loaded[k].shape[1] ] = b_loaded[k]
                # initalize remaining part to small random value
                param[ : , b_loaded[k].shape[1] : ] = 0.01 * torch.rand( param.shape[0],
                                                                         param.shape[1] - b_loaded[k].shape[1])
                keys_del += [ k ]

      #for backward compatibility. solved in new runs
      if 'proj_heads_other' in name: 
        for k in b_loaded.keys() :
          if name == k :
            if b_loaded[name].shape[0] == 0:
              keys_del += [ name ]
      
    for k in keys_del :
      del b_loaded[k]
    
    # use strict=False so that differing blocks, e.g. through coupling, are ignored
    mkeys, _ = block.load_state_dict( b_loaded, False)

    # missing keys = keys that are not pre-trained are initalized to small value
    [mkeys.remove(k) for k in keys_del]  # remove proj_out keys so that they are not over-written
    [utils.init_weights_uniform( block.state_dict()[k], 0.01) for k in mkeys]

    print( 'Loaded {} for {} from id = {} (ignoring/missing {} elements).'.format( block_name,
                                              field_info[0], field_info[1][4][0], len(mkeys) ) )

  ###################################################
  def translate_weights(self, mloaded, mkeys, ukeys):
    '''
    Function used for backward compatibility 
    '''
    cf = self.cf

    #encoder:
    for layer in range(cf.encoder_num_layers) :
  
      #shape([16, 3, 128, 2048])
      mw  = torch.cat([mloaded[f'encoders.0.heads.{layer}.heads_self.{head}.proj_{k}.weight'] for head in range(cf.encoder_num_heads) for k in ["qs", "ks", "vs"]])
      mloaded[f'encoders.0.heads.{layer}.proj_heads.weight'] = mw

      for head in range(cf.encoder_num_heads):
        del mloaded[f'encoders.0.heads.{layer}.heads_self.{head}.proj_qs.weight']
        del mloaded[f'encoders.0.heads.{layer}.heads_self.{head}.proj_ks.weight']
        del mloaded[f'encoders.0.heads.{layer}.heads_self.{head}.proj_vs.weight']

      #cross attention
      if f'encoders.0.heads.{layer}.heads_other.0.proj_qs.weight' in ukeys:
        mw  = torch.cat([mloaded[f'encoders.0.heads.{layer}.heads_other.{head}.proj_{k}.weight'] for head in range(cf.encoder_num_heads) for k in ["qs", "ks", "vs"]])
       
        for i in range(cf.encoder_num_heads):
          del mloaded[f'encoders.0.heads.{layer}.heads_other.{head}.proj_qs.weight']
          del mloaded[f'encoders.0.heads.{layer}.heads_other.{head}.proj_ks.weight']
          del mloaded[f'encoders.0.heads.{layer}.heads_other.{head}.proj_vs.weight']

        mloaded[f'encoders.0.heads.{layer}.proj_heads_other.0.weight'] = mw  
     
    #decoder
    for iblock in range(0, 19, 2) : 
      mw  = torch.cat([mloaded[f'decoders.0.blocks.{iblock}.heads.{head}.proj_{k}.weight'] for head in range(8) for k in ["qs", "ks", "vs"]])
      mloaded[f'decoders.0.blocks.{iblock}.proj_heads.weight'] = mw 

      qs = [mloaded[f'decoders.0.blocks.{iblock}.heads_other.{head}.proj_qs.weight'] for head in range(8)]
      mw  = torch.cat([mloaded[f'decoders.0.blocks.{iblock}.heads_other.{head}.proj_{k}.weight'] for head in range(8) for k in ["ks", "vs"]])
      
      mloaded[f'decoders.0.blocks.{iblock}.proj_heads_o_q.weight']  = torch.cat([*qs])
      mloaded[f'decoders.0.blocks.{iblock}.proj_heads_o_kv.weight'] = mw

      #self.num_samples_validate
      decoder_dim = self.decoders[0].blocks[iblock].ln_q.weight.shape #128
      mloaded[f'decoders.0.blocks.{iblock}.ln_q.weight'] = torch.tensor(np.ones(decoder_dim))
      mloaded[f'decoders.0.blocks.{iblock}.ln_k.weight'] = torch.tensor(np.ones(decoder_dim))
      mloaded[f'decoders.0.blocks.{iblock}.ln_q.bias'] = torch.tensor(np.ones(decoder_dim))
      mloaded[f'decoders.0.blocks.{iblock}.ln_k.bias'] = torch.tensor(np.ones(decoder_dim))

      for i in range(8):
        del mloaded[f'decoders.0.blocks.{iblock}.heads.{i}.proj_qs.weight']
        del mloaded[f'decoders.0.blocks.{iblock}.heads.{i}.proj_ks.weight']
        del mloaded[f'decoders.0.blocks.{iblock}.heads.{i}.proj_vs.weight']
        del mloaded[f'decoders.0.blocks.{iblock}.heads_other.{i}.proj_qs.weight']
        del mloaded[f'decoders.0.blocks.{iblock}.heads_other.{i}.proj_ks.weight']
        del mloaded[f'decoders.0.blocks.{iblock}.heads_other.{i}.proj_vs.weight']
    
    return mloaded

  ###################################################
  @staticmethod
  def load( model_id, devices, cf = None, epoch = -2, load_pretrained=False) :
    '''Load network from checkpoint'''

    if not cf : 
      cf = utils.Config()
      cf.load_json( model_id)

    model = AtmoRep( cf).create( devices, load_pretrained=False)
    mloaded = torch.load( utils.get_model_filename( model, model_id, epoch) )
    mkeys, ukeys = model.load_state_dict( mloaded, False )
    if (f'encoders.0.heads.0.proj_heads.weight') in mkeys:
      mloaded = model.translate_weights(mloaded, mkeys, ukeys)
      mkeys, ukeys = model.load_state_dict( mloaded, False )

    if len(mkeys) > 0 :
      print( f'Loaded AtmoRep: ignoring {len(mkeys)} elements: {mkeys}')

    # TODO: remove, only for backward 
    if model.embeds_token_info[0].weight.abs().max() == 0. :
      model.embeds_token_info = torch.nn.ModuleList()
      
    return model
    
  ###################################################
  def save( self, epoch = -2) :
    '''Save network '''

    # save entire network
    torch.save( self.state_dict(), utils.get_model_filename( self, self.cf.wandb_id, epoch) )

    # save parts also separately

    # name = self.__class__.__name__ + '_embed_token_info'
    # torch.save( self.embed_token_info.state_dict(),
    #             utils.get_model_filename( name, self.cf.wandb_id, epoch) )
    name = self.__class__.__name__ + '_embeds_token_info'
    torch.save( self.embeds_token_info.state_dict(),
                utils.get_model_filename( name, self.cf.wandb_id, epoch) )

    for ifield, enc in enumerate(self.encoders) :
      name = self.__class__.__name__ + '_encoder_' + self.cf.fields[ifield][0]
      torch.save( enc.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )

    for ifield, dec in enumerate(self.decoders) :
      name = self.__class__.__name__ + '_decoder_' + self.cf.fields_prediction[ifield][0]
      torch.save( dec.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )

    for ifield, tail in enumerate(self.tails) :
      name = self.__class__.__name__ + '_tail_' + self.cf.fields_prediction[ifield][0]
      torch.save( tail.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )

  ###################################################
  def forward( self, xin) :
    '''Evaluate network'''

    # embedding
    cf = self.cf
    
    fields_embed = self.get_fields_embed(xin)
    
    # attention maps (if requested)
    atts = [ [] for _ in cf.fields ]

    # encoder
    embeds_layers = [[] for i in self.field_pred_idxs]
    for ib in range(self.cf.encoder_num_layers) :
      fields_embed, att = self.forward_encoder_block( ib, fields_embed) 
      [embeds_layers[idx].append( fields_embed[i]) for idx,i in enumerate(self.field_pred_idxs)]
      [atts[i].append( att[i]) for i,_ in enumerate(cf.fields) ]
        
    # encoder-decoder coupling / token transformations
    (decoders_in, embeds_layers) = self.encoder_to_decoder( embeds_layers)

    preds = []
    for idx,i in enumerate(self.field_pred_idxs) :
    
      # decoder
      token_seq_embed, att = self.decoders[idx]( (decoders_in[idx], embeds_layers[idx]) )
      
      # tail net
      tail_in = self.decoder_to_tail( idx, token_seq_embed)
      pred = self.checkpoint( self.tails[idx], tail_in)
      
      preds.append( pred)
      [atts[i].append( a) for a in att]

    return preds, atts

  ###################################################
  def forward_encoder_block( self, iblock, fields_embed) :
    ''' evaluate one block (attention and mlp) '''

    # double buffer for commutation-invariant result (w.r.t evaluation order of transformers)
    fields_embed_cur, atts = [], []

    # attention heads
    for ifield in range( len(fields_embed)) :
      d = fields_embed[ifield].device
      fields_in =[fields_embed[i].to(d,non_blocking=True) for i in self.fields_coupling_idx[ifield]]
      # unpack list in argument for checkpointing
      y, att = self.checkpoint( self.encoders[ifield].heads[iblock], *fields_in)
      fields_embed_cur.append( y)
      atts.append( att)
      
    # MLPs 
    for ifield in range( len(fields_embed)) :
      fields_embed_cur[ifield] = self.checkpoint( self.encoders[ifield].mlps[iblock], 
                                                  fields_embed_cur[ifield] )
  
    return fields_embed_cur, atts

  ###################################################
  def get_fields_embed( self, xin ) :
    if 0 == len(self.embeds_token_info) :   # TODO: only for backward compatibility, remove
      emb_net_ti = self.embed_token_info
      return [prepare_token( field_data, emb_net, emb_net_ti )
                              for fidx,(field_data,emb_net) in enumerate(zip( xin, self.embeds))]
    else :
      embs_net_ti = self.embeds_token_info
      return [prepare_token( field_data, emb_net, embs_net_ti[fidx] )
                                for fidx,(field_data,emb_net) in enumerate(zip( xin, self.embeds))]
    
  ###################################################

  def get_attention( self, xin) : 

    cf = self.cf
    attn = []
    fields_embed = self.get_fields_embed(xin)
    #either accumulated attention or last layer attention:
    blocks = list(range(self.cf.encoder_num_layers)) if cf.attention_mode == 'accum' else [self.cf.encoder_num_layers-1]
    for idx, ifield in enumerate(self.field_pred_idxs) :      
      d = fields_embed[ifield].device
      fields_in =[fields_embed[i].to(d,non_blocking=True) for i in self.fields_coupling_idx[ifield]]
      attn_field = self.encoders[ifield].heads[blocks[0]].get_attention(fields_in)
      if cf.attention_mode == 'accum':
        for iblock in blocks[1:]:
          attn_layer = self.encoders[ifield].heads[iblock].get_attention(fields_in)
          attn_field = attn_field + attn_layer
        attn_field = torch.sum(attn_field, dim = 0, keepdim=True)
      attn.append(attn_field)
#      print("att FINAL", ifield, len(attn), attn[0].shape)
    return attn
