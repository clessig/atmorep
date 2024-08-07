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
import os
import sys
import traceback

import wandb

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch


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
  if not hasattr(cf, 'years_val'):
    cf.years_val = cf.years_test
  
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
  
  # format: list of fields where for each field the list is 
  # [ name , 
  #   [ dynamic or static field { 1, 0 }, embedding dimension, , device id ],
  #   [ vertical levels ],
  #   [ num_tokens],
  #   [ token size],
  #   [ total masking rate, rate masking, rate noising, rate for multi-res distortion]
  # ]

  cf.fields = [ [ 'temperature', [ 1, 1024, [ ], 0 ], 
                               [ 96, 105, 114, 123, 137 ], 
                               [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05] ] ]
  cf.fields_prediction = [ [cf.fields[0][0], 1.] ]

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

  cf.fields_targets = []

  cf.years_train = list( range( 2010, 2021))
  cf.years_val = [2021]  #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  # random seeds
  cf.torch_seed = torch.initial_seed()
  # training params
  cf.batch_size_validation = 1 #64
  cf.batch_size = 96 
  cf.num_epochs = 128
  cf.num_samples_per_epoch = 4096*12
  cf.num_samples_validate = 128*12
  cf.num_loader_workers = 8
  
  # additional infos
  cf.size_token_info = 8
  cf.size_token_info_net = 16
  cf.grad_checkpointing = True
  cf.with_cls = False
  # network config
  cf.with_mixed_precision = True
  cf.with_layernorm = True
  cf.coupling_num_heads_per_field = 1
  cf.dropout_rate = 0.05
  cf.with_qk_lnorm = False
  # encoder
  cf.encoder_num_layers = 6
  cf.encoder_num_heads = 16
  cf.encoder_num_mlp_layers = 2
  cf.encoder_att_type = 'dense'
  # decoder
  cf.decoder_num_layers = 6
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
  cf.losses = ['mse_ensemble', 'stats']  # mse, mse_ensemble, stats, crps, weighted_mse
  # training
  cf.optimizer_zero = False
  cf.lr_start = 5. * 10e-7
  cf.lr_max = 0.00005*3
  cf.lr_min = 0.00004 #0.00002
  cf.weight_decay = 0.05 #0.1
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  # BERT
  # strategies: 'BERT', 'forecast', 'temporal_interpolation'
  cf.BERT_strategy = 'BERT'
  cf.forecast_num_tokens = 2      # only needed / used for BERT_strategy 'forecast
  cf.BERT_fields_synced = False   # apply synchronized / identical masking to all fields 
                                  # (fields need to have same BERT params for this to have effect)
  cf.BERT_mr_max = 2              # maximum reduction rate for resolution
  
  # debug / output
  cf.log_test_num_ranks = 0
  cf.save_grads = False
  cf.profile = False
  cf.test_initial = False
  cf.attention = False

  cf.rng_seed = None 

  # usually use %>wandb offline to switch to disable syncing with server
  cf.with_wandb = True
  setup_wandb( cf.with_wandb, cf, par_rank, 'train', mode='offline')  

  # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res100_chunk32.zarr'
  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res100_chunk32.zarr'
  # # # in steps x lat_degrees x lon_degrees
  # cf.n_size = [36, 1*9*6, 1.*9*12]

  # # # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk16.zarr'
  # # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk32.zarr'
  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk32.zarr'
  # # # 
  # # # cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res025_chunk8.zarr'
  # # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk8_lat180_lon180.zarr'
  # # # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res025_chunk16.zarr'
  # cf.file_path = '/gpfs/scratch/ehpc03/era5_y2010_2021_res025_chunk8.zarr/'
  # # # in steps x lat_degrees x lon_degrees
  # cf.n_size = [36, 0.25*9*6, 0.25*9*12]

  # cf.file_path = '/ec/res4/scratch/nacl/atmorep/era5_y2021_res100_chunk16.zarr'
  cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y2021_res100_chunk16.zarr'
  cf.n_size = [36, 1*9*6, 1.*9*12]

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  trainer = Trainer_BERT( cf, device).create()
  trainer.run()

####################################################################################################
if __name__ == '__main__':
  
  try :

    train()

    #  wandb_id, epoch, epoch_continue = '1jh2qvrx', 392, 392
    #  Trainer = Trainer_BERT
    #  train_continue( wandb_id, epoch, Trainer, epoch_continue)

  except :
    
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)

