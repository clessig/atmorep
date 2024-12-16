####################################################################################################
#
#  Copyright (C) 2024
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
import torchinfo
import numpy as np

import torch
import os
import sys
import traceback
import pdb
import wandb

#from atmorep.core.trainer import Trainer_BERT
from atmorep.applications.downscaling.trainer import Trainer_Downscaling
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
from atmorep.utils.logger import logger

####################################################################################################
def train_continue( wandb_id, epoch,  epoch_continue = -1) :

  devices = init_torch()
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config().load_json( wandb_id)

  cf.num_accs_per_task = len(devices)   # number of GPUs / accelerators per task
  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.optimizer_zero = False
  cf.attention = False
  # name has changed but ensure backward compatibility
  if hasattr( cf, 'loader_num_workers') :
    cf.num_loader_workers = cf.loader_num_workers
  if not hasattr( cf, 'n_size'):
    cf.n_size = [36, 9*6, 9*12] 
  if not hasattr(cf, 'num_samples_per_epoch'):
    cf.num_samples_per_epoch = 1024
  if not hasattr(cf, 'num_samples_validate'):
    cf.num_samples_validate = 128
  if not hasattr(cf, 'with_mixed_precision'):
    cf.with_mixed_precision = True
  if not hasattr(cf, 'years_val'):
    cf.years_val = cf.years_test
 
  setup_wandb( cf.with_wandb, cf, par_rank, project_name='train', mode='offline')  
  # resuming a run requires online mode, which is not available everywhere
  #setup_wandb( cf.with_wandb, cf, par_rank, wandb_id = wandb_id)  
  
  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  if -1 == epoch_continue :
    epoch_continue = epoch

  # run
  trainer = Trainer_Downscaling.load( cf, wandb_id, epoch, devices)
  print( 'Loaded run \'{}\' at epoch {}.'.format( wandb_id, epoch))
  trainer.run( epoch_continue)

####################################################################################################

def train():

  #num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch()     #num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  # torch.cuda.set_sync_debug_mode(1)
  torch.backends.cuda.matmul.allow_tf32 = True

  #model_id = "3kdutwqb"  
  model_id = "wc5e2i3t"
  cf = Config().load_json( model_id)
  # parallelization
  cf.with_ddp = with_ddp
  #cf.num_accs_per_task = num_accs_per_task   # number of GPUs / accelerators per task
  cf.par_rank = par_rank
  cf.par_size = par_size

  for field in cf.fields:
      if len(field[1]) == 5:
          field[1].pop(4)
      field[1].append([model_id,61])
  
  #cf.model_id = "3kdutwqb" 

  cf.input_fields = cf.fields
 
  cf.downscaling_ratio = 3
  cf.fields_downscaling = [ ['total_precip', 
                            [1,1024,["velocity_u","velocity_v","velocity_z","specific_humidity"]],
                            [0],
                            [12,6,12],
                            [3,9*cf.downscaling_ratio,9*cf.downscaling_ratio], 
                            1.0 ] ]
  cf.target_fields = cf.fields_downscaling
  cf.input_file_path = "/p/scratch/atmo-rep/data/era5_1deg/months/era5_y1979_2021_res025_chunk8.zarr"
  cf.target_file_path = "/p/scratch/atmo-rep/data/imerg/imerg_regridded/imerg_regrid_y2003_2021_res083_chunk8.zarr"
  #cf.years_train = list( range( 2010, 2021))
  cf.years_train = [2003,2020]
  cf.years_val = [2021]  #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  
  # random seeds
  cf.torch_seed = torch.initial_seed()

  cf.rng_seed = None 
  
  cf.with_wandb = True
  cf.optimizer_zero = False
  cf.lr_start = 5. * 10e-7
  cf.lr_max = 0.00005*3
  cf.lr_min = 0.00004 #0.00002
  cf.weight_decay = 0.05 #0.1
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  cf.BERT_strategy = "BERT"
  cf.BERT_fields_synced = True
  
  #training params
  cf.batch_size_validation = 4 #64

  cf.batch_size = 16
  cf.num_epochs = 128
  cf.num_samples_per_epoch = 1024*12
  cf.num_samples_validate = 32*12
  cf.num_loader_workers = 6

  #additional infos
  cf.size_token_info = 8
  cf.size_token_info_net = 16
  cf.grad_checkpointing = True
  cf.with_cls = False

  #network config
  cf.with_mixed_precision = True
  cf.coupling_num_heads_per_field = 1
  cf.with_layernorm = True
  cf.dropout_rate = 0.05
  cf.with_qk_lnorm = False
 
  #encoder
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

  #perceiver
  cf.num_latent_queries = 4320
  cf.init_scale = 0.02
  cf.perceiver_num_layers = 6
  cf.perceiver_num_heads = 16
  cf.perceiver_num_mlp_layers = 2
  #cf.perceiver_output_emb = 256
  cf.perceiver_latent_dimension = 1024

  # tail net
  cf.net_tail_num_nets = 8
  cf.net_tail_num_layers = 0
  # loss
  cf.losses = ['mse_ensemble', 'stats']  # mse, mse_ensemble, stats, crps, weighted_mse
  # debug / output
  cf.log_test_num_ranks = 0
  cf.save_grads = False
  cf.profile = False
  cf.test_initial = False
  cf.attention = False
  cf.n_size = [36, 9*6, 9*12]
  
  setup_wandb(cf.with_wandb, cf, par_rank,'train', mode="offline")

  if cf.with_wandb and 0 == cf.par_rank:
      cf.write_json( wandb)
      cf.print()
  
  trainer = Trainer_Downscaling( cf, device).create()
  trainer.run()


if __name__ == "__main__":
  try :
  
    #train()
  
    wandb_id, epoch, epoch_continue = '6ff1d620', 31, 31
    train_continue( wandb_id, epoch, epoch_continue)
  
  except :
  
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
