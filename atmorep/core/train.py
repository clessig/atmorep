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

import atmorep.config.config as config
from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
import atmorep.utils.utils as utils


####################################################################################################
def train_continue( wandb_id, epoch, Trainer, epoch_continue = -1) :

  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config().load_json( wandb_id)

  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.optimizer_zero = False
  cf.attention = False
  # name has changed but ensure backward compatibility
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
  
  cf.years_train = [2021] # list( range( 1980, 2018))
  cf.years_test = [2021]  #[2018] 
  
  # any parameter in cf can be overwritten when training is continued, e.g. we can increase the 
  # masking rate 
  # cf.fields = [ [ 'specific_humidity', [ 1, 2048, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.1, 0.05] ] ]

  setup_wandb( cf.with_wandb, cf, par_rank, project_name='train', mode='offline')  
  # resuming a run requires online mode, which is not available everywhere
  #setup_wandb( cf.with_wandb, cf, par_rank, wandb_id = wandb_id)  
  
  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  if -1 == epoch_continue :
    epoch_continue = epoch

  # run
  trainer = Trainer.load( cf, wandb_id, epoch, device)
  print( 'Loaded run \'{}\' at epoch {}.'.format( wandb_id, epoch))
  trainer.run( epoch_continue)

####################################################################################################
def train() :

  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  # torch.cuda.set_sync_debug_mode(1)
  torch.backends.cuda.matmul.allow_tf32 = True

  cf = Config()
  # parallelization
  cf.with_ddp = with_ddp
  cf.num_accs_per_task = num_accs_per_task   # number of GPUs / accelerators per task
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.back_passes_per_step = 4
  # general
  cf.comment = ''
  cf.file_format = 'grib'
  cf.data_dir = str(config.path_data)
  cf.level_type = 'ml'
  
  # format: list of fields where for each field the list is 
  # [ name , 
  #   [ dynamic or static field { 1, 0 }, embedding dimension, , device id ],
  #   [ vertical levels ],
  #   [ num_tokens],
  #   [ token size],
  #   [ total masking rate, rate masking, rate noising, rate for multi-res distortion]
  # ]

  # cf.fields = [ [ 'vorticity', [ 1, 2048, [ ], 0 ], 
  #                              [ 123 ], 
  #                              [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [ [ 'velocity_u', [ 1, 2048, [ ], 0], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.1, 0.05] ] ]
  
  # cf.fields = [ [ 'velocity_v', [ 1, 2048, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [ 12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [ [ 'velocity_z', [ 1, 1024, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [ [ 'specific_humidity', [ 1, 2048, [ ], 0 ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]

  #                             [12, 2, 4], [3, 27, 27], [0.5, 0.9, 0.1, 0.05], 'local' ] ]

  # cf.fields = [ [ 'total_precip', [ 1, 2048, [ ], 0 ],
  #                                 [ 0 ], 
  #                                 [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]

  # cf.fields = [  [ 'geopotential', [ 1, 1024, [], 0 ], 
  #                                 [ 0 ], 
  #                                 [12, 3, 6], [3, 18, 18], [0.25, 0.9, 0.1, 0.05] ] ]
  # cf.fields_prediction = [ ['geopotential', 1.] ]

  #cf.fields_prediction = [ [cf.fields[0][0], 1.0] ]

  # cf.fields = [ [ 'velocity_u', [ 1, 1024, [ ], 0], 
  #                               [ 114, 123, 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.1, 0.05] ],
  cf.fields = [
                [ 'velocity_v', [ 1, 1024, [ ], 0 ], 
                                [ 114, 123, 137 ], 
                                [ 12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ],
                [ 'total_precip', [ 1, 1536, [ ], 3 ],
                                  [ 0 ], 
                                  [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ] ]                
  #cf.fields_prediction = [ [cf.fields[0][0], 0.33],  [cf.fields[1][0], 0.33],  [cf.fields[2][0], 0.33]]
  # cf.fields = [ 
  #               [ 'total_precip', [ 1, 1536, [ ], 3 ],
  #                                 [ 0 ], 
  #                                 [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.2, 0.05] ] ]                
  cf.fields_prediction = [ [cf.fields[0][0], 0.5],[cf.fields[1][0], 0.5]  ] 
  cf.fields_targets = []
  cf.years_train = [2021] # list( range( 1980, 2018))
  cf.years_test = [2021]  #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  # file and data parameter parameter
  cf.data_smoothing = 0
  cf.file_shape = (-1, 721, 1440)
  cf.num_t_samples = 31*24
  cf.num_files_train = 5
  cf.num_files_test = 2
  cf.num_patches_per_t_train = 8
  cf.num_patches_per_t_test = 4
  # random seeds
  cf.torch_seed = torch.initial_seed()
  # training params
  cf.batch_size_test = 64
  cf.batch_size_start = 16
  cf.batch_size_max = 32
  cf.batch_size_delta = 8
  cf.num_epochs = 128
  # cf.num_loader_workers = 1#8
  # additional infos
  cf.size_token_info = 8
  cf.size_token_info_net = 16
  cf.grad_checkpointing = True
  cf.with_cls = False
  # network config
  cf.with_layernorm = True
  cf.coupling_num_heads_per_field = 1
  cf.dropout_rate = 0.05
  cf.learnable_mask = False
  cf.with_qk_lnorm = True
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
  # loss
  # supported: see Trainer for supported losses
  # cf.losses = ['mse', 'stats']
  cf.losses = ['mse_ensemble', 'stats']
  # cf.losses = ['mse']
  # cf.losses = ['stats']
  # cf.losses = ['crps']
  # training
  cf.optimizer_zero = False
  cf.lr_start = 5. * 10e-7
  cf.lr_max = 0.00005
  cf.lr_min = 0.00004
  cf.weight_decay = 0.05
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  cf.lat_sampling_weighted = True
  # BERT
  # strategies: 'BERT', 'forecast', 'temporal_interpolation', 'identity'
  cf.BERT_strategy = 'BERT'     
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
  setup_wandb( cf.with_wandb, cf, par_rank, 'train', mode='offline')  

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  #cf.levels = [114, 123, 137]
  cf.with_mixed_precision = True
  # cf.n_size = [36, 1*9*6, 1.*9*12]
  # in steps x lat_degrees x lon_degrees
  # cf.n_size = [36, 0.25*9*6, 0.25*9*12]
  cf.n_size = [36, 0.25*9*6, 0.25*9*12]
  cf.num_samples_per_epoch = 1024
  cf.num_samples_validate = 128
  cf.num_loader_workers = 1 #8

  cf.years_train = [2021] # list( range( 1980, 2018))
  cf.years_test = [2021]  #[2018] 

  trainer = Trainer_BERT( cf, device).create()
  trainer.run()

####################################################################################################
if __name__ == '__main__':

  train()

#  wandb_id, epoch = '1jh2qvrx', 392 #'4nvwbetz', -2 #392  #'4nvwbetz', -2
#  epoch_continue = epoch

#  Trainer = Trainer_BERT
#  train_continue( wandb_id, epoch, Trainer, epoch_continue)
