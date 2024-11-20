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


def train():

  #num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch()     #num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  # torch.cuda.set_sync_debug_mode(1)
  torch.backends.cuda.matmul.allow_tf32 = True

  #model_id = "3kdutwqb"  
  model_id = "1jh2qvrx"
  cf = Config().load_json( model_id)
  # parallelization
  cf.with_ddp = with_ddp
  #cf.num_accs_per_task = num_accs_per_task   # number of GPUs / accelerators per task
  cf.par_rank = par_rank
  cf.par_size = par_size

  for field in cf.fields:
    field[1].pop(4)
    #field[1].append([model_id,45])
  
  #cf.model_id = "3kdutwqb" 

  cf.input_fields = cf.fields
 
  cf.downscaling_ratio = 3
  cf.fields_downscaling = [ ['total_precip', 
                            [1,1536,["velocity_u","velocity_v","specific_humidity"]],
                            [0],
                            [12,6,12],
                            [3,9*cf.downscaling_ratio,9*cf.downscaling_ratio], 
                            1.0 ] ]
  cf.target_fields = cf.fields_downscaling
  cf.input_file_path = "/p/scratch/atmo-rep/data/era5_1deg/months/era5_y1979_2021_res025_chunk8.zarr"
  cf.target_file_path = "/p/scratch/atmo-rep/data/imerg/imerg_regridded/imerg_regrid_y2003_2021_res008_chunk8.zarr"
  #cf.years_train = list( range( 2010, 2021))
  cf.years_train = [2003,2020]
  cf.years_val = [2021]  #[2018] 
  cf.month = None
  cf.geo_range_sampling = [[ -90., 90.], [ 0., 360.]]
  cf.time_sampling = 1   # sampling rate for time steps
  
  # random seeds
  cf.torch_seed = torch.initial_seed()
 
  cf.input_data = "/p/scratch/atmo-rep/data/era5/new_structure/total_precip/ml0/*"
  cf.output_data = "/p/scratch/atmo-rep/data/imerg_regridded"

  cf.num_epochs = 30
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
  cf.batch_size_validation = 1 #64

  cf.batch_size = 2
  cf.num_epochs = 128
  cf.num_samples_per_epoch = 8
  cf.num_samples_validate = 8
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
  cf.num_latent_queries = 64
  cf.init_scale = 0.02
  cf.perceiver_num_layers = 6
  cf.perceiver_num_heads = 16
  cf.perceiver_num_mlp_layers = 2
  cf.perceiver_output_emb = 256

  # tail net
  cf.net_tail_num_nets = 16
  cf.net_tail_num_layers = 0
  # loss
  cf.losses = ['mse_ensemble', 'stats']  # mse, mse_ensemble, stats, crps, weighted_mse
  # debug / output
  cf.log_test_num_ranks = 0
  cf.save_grads = False
  cf.profile = False
  cf.test_initial = False
  cf.attention = False

  setup_wandb(cf.with_wandb, cf, par_rank,'train', mode="offline")
  
  cf.file_path = '/p/scratch/atmo-rep/data/era5_1deg/months/era5_y1979_2021_res025_chunk8.zarr'
  cf.n_size = [36, 9*6, 9*12]

  trainer = Trainer_Downscaling( cf, device).create()
  trainer.run()


if __name__ == "__main__":
  try :
  
    train()
  
    #  wandb_id, epoch, epoch_continue = '1jh2qvrx', 392, 392
    #  Trainer = Trainer_BERT
    #  train_continue( wandb_id, epoch, Trainer, epoch_continue)
  
  except :
  
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
