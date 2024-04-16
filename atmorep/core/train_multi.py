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
import os

import wandb

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
import atmorep.utils.utils as utils

import atmorep.config.config as config

####################################################################################################
def train_continue( model_id, model_epoch, Trainer, model_epoch_continue = -1) :

  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config().load_json( model_id)
  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.attention = False
  cf.optimizer_zero = False   
  if hasattr( cf, 'loader_num_workers') :
    cf.num_loader_workers = cf.loader_num_workers
  if not hasattr( cf, 'n_size'):
    cf.n_size = [36, 0.25*9*6, 0.25*9*12] 
  if not hasattr(cf, 'num_samples_per_epoch'):
    cf.num_samples_per_epoch = 1024
  if not hasattr(cf, 'num_samples_validate'):
    cf.num_samples_validate = 128
  if not hasattr(cf, 'with_mixed_precision'):
    cf.with_mixed_precision = True

  setup_wandb( cf.with_wandb, cf, par_rank, 'train-multi', mode='offline')  

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  if -1 == model_epoch_continue :
    model_epoch_continue = model_epoch

  # run
  trainer = Trainer.load( cf, model_id, model_epoch, device)
  print( 'Loaded run \'{}\' at epoch {}.'.format( model_id, model_epoch))
  trainer.run( model_epoch_continue)

####################################################################################################
def train_multi() :

  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config()
  # horovod 
  cf.num_accs_per_task = num_accs_per_task   # number of GPUs / accelerators per task
  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.back_passes_per_step = 4
  # general
  cf.comment = ''
  cf.file_format = 'grib'
  cf.file_path = str(config.path_data)
  cf.level_type = 'ml'
  
  cf.fields = [ [ 'vorticity', [ 1, 2048, ['divergence', 'temperature'], 0 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ],
                [ 'velocity_z', [ 1, 1536, ['vorticity', 'divergence'], 0 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ], 
                [ 'divergence', [ 1, 2048, ['vorticity', 'temperature'], 1 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ],
                [ 'specific_humidity', [ 1, 2048, ['vorticity', 'divergence', 'velocity_z'], 2 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ],
                [ 'temperature', [ 1, 1024, ['vorticity', 'divergence'], 3 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ],   
                [ 'total_precip', [ 1, 1536, ['vorticity', 'divergence', 'specific_humidity'], 3 ],
                                  [ 0 ], 
                                  [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ] ]
  cf.fields_prediction = [['vorticity', 0.25],  ['velocity_z', 0.1], 
                          ['divergence', 0.25], ['specific_humidity', 0.15],
                          ['temperature', 0.15], ['total_precip', 0.1] ]

  # cf.fields = [ [ 'velocity_u', [ 1, 2048, ['velocity_v', 'temperature'], 0, ['dys79lgw', 82] ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local' ],
  #               [ 'velocity_v', [ 1, 2048, ['velocity_u', 'temperature'], 1, ['22j6gysw', 82] ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local' ], 
  #               [ 'specific_humidity', [ 1, 2048, ['velocity_u', 'velocity_v', 'temperature'], 2, ['1565pb1f', 62] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local' ],
  #               [ 'velocity_z', [ 1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 3, ['3cb4q6sl', 190] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'global' ],
  #               [ 'temperature', [ 1, 1024, ['velocity_u', 'velocity_v', 'specific_humidity'], 3, ['2147fkco', 87] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local' ],
  #               ['total_precip', [1, 1536, ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity'], 3, ['3kdutwqb', 900]], 
  #                             [0], 
  #                             [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05]] ]
  # cf.fields_prediction = [['velocity_u', 0.225], ['velocity_v', 0.225], 
  #                         ['specific_humidity', 0.15], ['velocity_z', 0.1], ['temperature', 0.2],
  #                         ['total_precip', 0.1] ]

  cf.fields_targets = []
  cf.years_train = [2021] #list( range( 1980, 2018))
  cf.years_test = [2021] #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  # file and data parameter parameter
  cf.data_smoothing = 0
  cf.file_shape = (-1, 721, 1440)
  cf.num_t_samples = 31*24
  cf.num_patches_per_t_train = 4
  cf.num_patches_per_t_test = 1
  # random seeds
  cf.torch_seed = torch.initial_seed()
  # training params
  cf.batch_size_test = 8
  cf.batch_size_start = 2
  cf.batch_size_max = 2
  cf.batch_size_delta = 1
  cf.num_epochs = 128
  cf.num_files_train = 1
  cf.num_files_test = 1 # int( np.ceil( 12. / par_size ))
  cf.num_loader_workers = 2
  # additional infos
  cf.size_token_info = 8
  cf.size_token_info_net = 16
  cf.grad_checkpointing = True
  cf.with_cls = False
  # network config
  cf.with_layernorm = True
  cf.coupling_num_heads_per_field = 2
  cf.dropout_rate = 0.05
  cf.learnable_mask = False
  cf.with_qk_lnorm = True
  cf.num_neighborhoods = 1
  # encoder
  cf.encoder_num_layers = 10
  cf.encoder_num_heads = 16
  cf.encoder_num_mlp_layers = 2
  cf.encoder_att_type = 'dense'
  # decoder
  cf.decoder_num_layers = 10
  cf.decoder_num_heads = 16
  cf.decoder_num_mlp_layers = 2
  cf.decoder_self_att = False
  cf.decoder_cross_att_ratio = 0.5
  cf.decoder_cross_att_rate = 1.0
  cf.decoder_att_type = 'dense'
  # tail net
  cf.net_tail_num_nets = 16
  cf.net_tail_num_layers = 0
  cf.ensemble_weighted = False
  cf.ensemble_weighted_T = 50.
  # loss
  # supported: see Trainer for supported losses
  # cf.losses = ['mse', 'stats']
  cf.losses = ['mse_ensemble', 'stats']
  # cf.losses = ['stats']
  # training
  cf.optimizer_zero = False
  cf.lr_start = 0.00001
  cf.lr_max = 0.00002
  cf.lr_min = 0.00001
  cf.weight_decay = 0.025
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  cf.lat_sampling_weighted = False
  # BERT
  cf.BERT_strategy = 'BERT'      # 'BERT', 'forecast', 'identity', 'totalmask'
  cf.BERT_fields_synced = False   # apply synchronized / identical masking to all fields 
                                  # (fields need to have same BERT params for this to have effect)
  cf.BERT_mr_max = 2              # maximum reduction rate for resolution
  # debug / output
  cf.log_test_num_ranks = 0
  cf.save_grads = False
  cf.profile = False
  cf.test_initial = True
  cf.attention = False

  cf.rng_seed = None 

  # usually use %>wandb offline to switch to disable syncing with server
  cf.with_wandb = True
  setup_wandb( cf.with_wandb, cf, par_rank, 'train-multi', mode='offline')  

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  trainer = Trainer_BERT( cf, device).create()
  trainer.run()

####################################################################################################
if __name__ == '__main__':

  train_multi()

  # # Continue training run
  
#  model_id, model_epoch = '1jh2qvrx', -2
#  model_epoch_continue = 0
#  
#  Trainer = Trainer_BERT
#  train_continue( model_id, model_epoch, Trainer, model_epoch_continue)
