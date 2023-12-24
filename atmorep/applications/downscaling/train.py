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
import torchinfo
import numpy as np

import os
import pathlib
import datetime
import time
import math

import functools

import wandb

import atmorep.config.config as config
from atmorep.applications.downscaling.trainer import Trainer_Downscaling
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
from atmorep.utils.utils import NetMode
import atmorep.utils.utils as utils

####################################################################################################
def evaluate( model_id, model_epoch, Trainer) :

  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  cf = Config().load_json( model_id)
  cf.par_rank = par_rank
  cf.par_size = par_size

  # TODO: set up wandb resume and/or overwrite settings stored in wandb (currently incorrect)
  setup_wandb( cf.with_wandb, cf, par_rank, 'downscaling')  

  # adjust parameters
  cf.num_files_test = 3
  cf.log_test_num_ranks = 8

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  cf.years_test = [2018] 
  
  # load model
  trainer = Trainer.load( cf, model_id, model_epoch, device)
  trainer.model.load_data( NetMode.test) 
  trainer.test( 0, cf.BERT_strategy)

  print( 'Finished evaluate().')

####################################################################################################
def train_continue( model_id, model_epoch, Trainer) :

  # device = init_torch( 1)
  # TODO: num_accs_per_task should be determined based on SLURM environment
  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  torch.backends.cuda.matmul.allow_tf32 = True

  cf = Config().load_json( model_id)
  cf.par_rank = par_rank
  cf.par_size = par_size
  cf.num_epochs = 256

  # TODO: set up wandb resume and/or overwrite settings stored in wandb (currently incorrect)
  setup_wandb( cf.with_wandb, cf, par_rank, 'downscaling')  

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  # run
  trainer = Trainer.load( cf, model_id, model_epoch, device)
  print( 'Loaded run \'{}\' at epoch {}.'.format( model_id, model_epoch))
  trainer.run( model_epoch)

####################################################################################################
def train() :

  # TODO: num_accs_per_task should be determined based on SLURM environment
  num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
  device = init_torch( num_accs_per_task)
  with_ddp = True
  par_rank, par_size = setup_ddp( with_ddp)

  # device = [ device[0] ]
  torch.backends.cuda.matmul.allow_tf32 = True

  # Config: only new parameters or those that should be overwritten are specified
  #         remainder is read from cf.base_model_id
  #   Caveats
  #   - temporal num_tokens and token_size has to match
  #   - cf.fields_targets overwrites cf.fields_prediction

  cf = Config()
  # horovod 
  cf.num_accs_per_task = num_accs_per_task   # number of GPUs / accelerators per task
  cf.with_ddp = with_ddp
  cf.par_rank = par_rank
  cf.par_size = par_size

  cf.base_model_id = '3cizyl1q'
  cf.base_epoch = -2
  cf.fields = [
      ['velocity_u', [1, 2048, [ 'velocity_v', 'temperature'], 0, ['dys79lgw', -2]], 
        [ 137], [12, 6, 12], [3, 9, 9], [0.0, 0.0, 0.0, 0.0], 'local'], 
        # [96, 105, 114, 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local'], 
        # [ 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local'], 
      ['velocity_v', [1, 2048, ['velocity_u', 'temperature'], 1, ['22j6gysw', -2]], 
        [ 137], [12, 6, 12], [3, 9, 9], [0.0, 0.0, 0.0, 0.0], 'local'], 
        # [96, 105, 114, 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local'], 
        # [ 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local'], 
      ['temperature', [1, 1024, ['velocity_u', 'velocity_v' ], 2, ['2147fkco', -2]], 
        [ 137], [12, 6, 12], [3, 9, 9], [0.0, 0.0, 0.0, 0.0], 'local']]
        # [96, 105, 114, 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local']]
        # [ 123, 137], [12, 6, 12], [3, 9, 9], [0.5, 0.9, 0.2, 0.05], 'local']]
  cf.fields_prediction = [['velocity_u', 0.5], ['velocity_v', 0.05], ['temperature', 0.5]]
  cf.fields_targets = [ [ 't2m' , [ 1, 2048, [ ], 3 ], [ 0 ], [12, 6, 12], [3, 4*9, 4*9],
                                  [0.25, 0.9, 0.0, 0.0], 'local',
                                  ['cosmo_rea6', [ -1, 685, 793], 
                                  [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
                                  ['temperature'], 1.0  ] ] ]
  # cf.fields_targets = [ [ 'u_10m' , [ 1, 3072, [ ], 3 ], [ 0 ], [12, 6, 12], [3, 4*9, 4*9],
  #                               [0.25, 0.9, 0.0, 0.0], 'global',
  #                               ['cosmo_rea6', [ -1, 685, 793], 
  #                               [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                               ['velocity_u'], 0.5 ] ] ]
                        #   ,
                        # [ 't2m' , [ 1, 2048, [ ], 3 ], [ 0 ], [12, 6, 12], [3, 4*9, 4*9],
                        #         [0.25, 0.9, 0.0, 0.0], 'global',
                        #         ['cosmo_rea6', [ -1, 685, 793], 
                        #         [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
                        #         ['temperature'], 1.0  ] ] ]


  # cf.base_model_id = '3qou60es'  # temperature, ml=137, local corrections
  # cf.base_epoch = 327
  # cf.fields = [['temperature', [1, 1536, [], 0, ['3qou60es', 327]], [137], 
  #                              [12, 2, 4], [3, 27, 27], [0.0, 0.0, 0.0, 0.0], 'local']]
  # cf.fields_prediction = [['temperature', 1.0]]
  # cf.fields_targets = [  [ 't2m' , [ 1, 4096, [ ], 0 ], [ 0 ], [12, 2, 4], [3, 4*27, 4*27],
  #                                 [0.25, 0.9, 0.0, 0.0], 'global',
  #                                 ['cosmo_rea6', [ -1, 685, 793], 
  #                                 [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                                 ['temperature'], 1.0  ] ] 
  #                                 ,
  #                       [ 'orography' , [ 0, 1024, [ ], 0 ], [ 0 ], [1, 2, 4], [1, 4*27, 4*27],
  #                                 [0.0, 0.0, 0.0, 0.0], 'global',
  #                                 ['cosmo_rea6', [ -1, 685, 793], 
  #                                 [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                                 [], 0.0 ] ] ]
  
  # # temperature with regular / small tokens
  # cf.base_model_id = '1za45vud'  # temperature, ml=137, local corrections
  # cf.base_epoch = 155
  # cf.fields = [['temperature', [1, 1024, [], 0, ['1za45vud', 155]], [137], 
  #                              [12, 6, 12], [3, 9, 9], [0.0, 0.0, 0.0, 0.0], 'local']]
  # cf.fields_prediction = [['temperature', 1.0]]
  # cf.fields_targets = [  [ 't2m' , [ 1, 2048, [ ], 0 ], [ 0 ], [12, 6, 12], [3, 4*9, 4*9],
  #                                 [0.25, 0.9, 0.0, 0.0], 'global',
  #                                 ['cosmo_rea6', [ -1, 685, 793], 
  #                                 [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                                 ['temperature'], 1.0  ] ] ]
  #                       #            ,
  #                       # [ 'orography' , [ 0, 1024, [ ], 0 ], [ 0 ], [1, 6, 12], [1, 4*9, 4*9],
  #                       #           [0.0, 0.0, 0.0, 0.0], 'global',
  #                       #           ['cosmo_rea6', [ -1, 685, 793], 
  #                       #           [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                       #           [], 0.0 ] ] ]

  # # cf.base_model_id = 'dys79lgw'  
  # cf.base_epoch = 82
  # cf.fields = [ [ 'velocity_u', [ 1, 2048, [], 0, ['dys79lgw', 82] ], 
  #                               [ 137 ], 
  #                               [12, 6, 12], [3, 9, 9], [0.0, 0.0, 0.0, 0.0], 'local' ] ]
  # cf.fields_prediction = [['velocity_u', 1.0]]
  # cf.fields_targets = [  [ 'u_10m' , [ 1, 3072, [ ], 0 ], [0], [12, 6, 12], [3, 4*9, 4*9],
  #                                 [0.25, 0.9, 0.0, 0.0], 'local',
  #                                 ['cosmo_rea6', [ -1, 685, 793], 
  #                                 [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                                 ['velocity_u'], 1.0  ] ] ]
  #                       #           ,
  #                       # [ 'orography' , [ 0, 1024, [ ], 0 ], [ 0 ], [1, 6, 12], [1, 4*9, 4*9],
  #                       #           [0.0, 0.0, 0.0, 0.0], 'global',
  #                       #           ['cosmo_rea6', [ -1, 685, 793], 
  #                       #           [ [27.5,70.25], [-12.5,37.0] ], 'netcdf', 
  #                       #           [], 0.0 ] ] ]

  cf.batch_size_test = 24
  cf.batch_size_start = 6
  cf.batch_size_max = 6
  cf.batch_size_delta = 1
  cf.num_files_train = 5
  cf.num_files_test = 1

  cf.data_dir = str(config.path_data)
  
  cf.years_train = list( range( 1997, 2017))
  cf.years_test = [2017] #[2018] 
  cf.month = None
  # center of COSMO domain adjusted to neighborhood size of pre-training 
  cf.geo_range_sampling = [ [42.125, 55.625], [-1.25, 25.75] ]  
  cf.num_patches_per_t_train = 2
  cf.num_patches_per_t_test = 1
  
  cf.downscaling_net_tail_num_nets = 4
  cf.losses = ['mse_ensemble', 'stats']
  # cf.downscaling_net_tail_num_nets = 1
  # cf.losses = ['mse']

  cf.learnable_mask = False
  cf.with_qk_lnorm = True
  cf.num_neighborhoods = 1
  cf.test_initial = True
  cf.size_token_info = 8
  cf.lat_sampling_weighted = False

  # usually use %>wandb offline to switch to disable syncing with server
  cf.with_wandb = True
  setup_wandb( cf.with_wandb, cf, par_rank, 'downscaling')

  # downscaling parameters

  cf.downscaling_optimize_encoder = True
  cf.downscaling_optimize_decoder = True

  cf.downscaling_num_layers = 6
  cf.downscaling_num_heads = 16
  cf.downscaling_num_mlp_layers = 2
  cf.downscaling_att_type = 'dense'   # 'dense', 'axial', 'axial_parallel'

  # cf.BERT_strategy = 'BERT' # 'identity'
  cf.BERT_strategy = 'identity' # 'identity'
  cf.BERT_window = False          # sample sub-region 
  cf.BERT_fields_synced = False   # apply synchronized / identical masking to all fields 
                                 # (fields need to have same BERT params for this to have effect)
  cf.BERT_mr_max = 2             # maximum reduction rate for resolution

  cf.grad_checkpointing = True
  
  cf.lr_start = 0.00001
  cf.lr_max = 0.000025
  cf.lr_min = 0.00001
  cf.lr_decay_rate = 1.1
  cf.lr_start_epochs = 3
  cf.log_test_num_ranks = 1
  cf.num_loader_workers = 8
  cf.optimizer_zero = 0

  cf.dropout_rate = 0.1
  cf.weight_decay = 0.1

  cf.attention = False

  # cf.test_initial = False
  cf.partial_load = 0

  cf.num_epochs = 256

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    # cf.print()

  trainer = Trainer_Downscaling( cf, device).load_create()
  trainer.run()

####################################################################################################
if __name__ == '__main__':

  train()

  # # Continue training or evaluate pre-trained model (the first two lines have to be set)
  # model_id = '333wsx3p'
  # model_epoch = 13
  # # model_id = 'js2r6vah'
  # # model_epoch = 9
  # # model_id = '13yzil5d'
  # # model_epoch = 42
  # model_id = '18o32zve'
  # model_epoch = 6
  # model_id = '28prx698'
  # # model_id = '1byg6xvi'
  # model_epoch = 19
  # model_id = '1w958gpn'
  # model_epoch = 24
  # model_id = '20qn8r8z'
  # model_epoch = 12
  # # model_id = '2iputsxy'
  # # model_epoch = 5
  # # model_id = '1w958gpn'
  # # model_epoch = 26
  # model_id = '37nxbqyj'
  # model_epoch = 23
  # model_id = 'nrnj7ctw'
  # model_epoch = 11
  # model_id = '1j851zsj'
  # model_epoch = 4
  # model_id = '1j851zsj'
  # model_epoch = -2
  # model_id = 'fm5kjhmc'
  # model_epoch = 4
  # model_id = '2803m4e8'
  # model_epoch = 10
  # Trainer = Trainer_Downscaling
  # # train_continue( model_id, model_epoch, Trainer)
  # evaluate( model_id, model_epoch, Trainer)

