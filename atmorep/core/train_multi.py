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
import sys
import traceback
import pdb
import wandb

import atmorep.config.config as config
from atmorep.core.trainer import Trainer_BERT
from atmorep.core.train import train_continue, initialize_atmorep
from atmorep.utils.utils import setup_wandb
import numpy as np


####################################################################################################
def train():
  with_ddp = True
  devices, par_rank, par_size, cf = initialize_atmorep(with_ddp)

  # parallelization
  cf.with_ddp = with_ddp
  cf.num_accs_per_task = len(devices)   # number of GPUs / accelerators per task
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

  cf.fields = [ 
                [ 'velocity_u', [ 1, 1024, ['velocity_v', 'temperature'], 0 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
                [ 'velocity_v', [ 1, 1024, ['velocity_u', 'temperature'], 1 ], 
                                [ 96, 105, 114, 123, 137 ], 
                                [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ], 
                [ 'specific_humidity', [ 1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 2 ], 
                              [ 96, 105, 114, 123, 137 ], 
                              [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
                [ 'velocity_z', [ 1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 3 ], 
                              [ 96, 105, 114, 123, 137 ], 
                              [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
                 [ 'temperature', [ 1, 1024, ['velocity_u', 'velocity_v', 'specific_humidity'], 3 ], 
                              [ 96, 105, 114, 123, 137 ], 
                              [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05], 'local' ],
              ]

  # cf.fields = [ 
  #               [ 'velocity_u', [ 1, 1024, ['velocity_v', 'temperature'], 0, ['j8dwr5qj', -2] ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
  #               [ 'velocity_v', [ 1, 1024, ['velocity_u', 'temperature'], 1, ['0tlnm5up', -2] ], 
  #                               [ 96, 105, 114, 123, 137 ], 
  #                               [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ], 
  #               [ 'specific_humidity', [ 1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 2, ['v63l01zu', -2] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
  #               [ 'velocity_z', [ 1, 1024, ['velocity_u', 'velocity_v', 'temperature'], 3, ['9l1errbo', -2] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 3, 6], [3, 18, 18], [0.5, 0.9, 0.2, 0.05] ],
  #                [ 'temperature', [ 1, 1024, ['velocity_u', 'velocity_v', 'specific_humidity'], 3, ['7ojls62c', -2] ], 
  #                             [ 96, 105, 114, 123, 137 ], 
  #                             [12, 2, 4], [3, 27, 27], [0.5, 0.9, 0.2, 0.05], 'local' ],
  #               # ['total_precip', [1, 1536, ['velocity_u', 'velocity_v', 'velocity_z', 'specific_humidity'], 3, ['3kdutwqb', 900]], 
  #               #               [0], 
  #               #               [12, 6, 12], [3, 9, 9], [0.25, 0.9, 0.1, 0.05]] 
  #             ]

  cf.fields_prediction = [
                          ['velocity_u', 0.225], ['velocity_v', 0.225], 
                          ['specific_humidity', 0.15], ['velocity_z', 0.1], ['temperature', 0.2]
             #             ['total_precip', 0.1] 
              ]

  cf.fields_targets = []

  cf.years_train = list( range( 1979, 2021))
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
  cf.with_qk_lnorm = True
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
  
  cf.lr_start = 0.00001 #5. * 10e-7
  cf.lr_max = 0.00002
  cf.lr_min = 0.00001
  cf.weight_decay = 0.025 #0.1
  cf.lr_decay_rate = 1.025
  cf.lr_start_epochs = 3
  cf.model_log_frequency = 256 #save checkpoint every X batches

  # BERT
  # strategies: 'BERT', 'forecast', 'temporal_interpolation'
  cf.BERT_strategy = 'BERT' #'BERT'
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
  
  #calculate n_size: same for all fields
  data_path_str = config.path_data.as_posix()
  assert "res" in data_path_str, Exception("Resolution not in file name. Please specify it.")
  size  = np.multiply(cf.fields[0][3], cf.fields[0][4]) #ntokens x token_size
  resol = int(data_path_str.split("res")[1].split("_")[0])/100
  cf.n_size = [float(cf.time_sampling*size[0]), float(resol*size[1]), float(resol*size[2])]

  if cf.with_wandb and 0 == cf.par_rank :
    cf.write_json( wandb)
    cf.print()

  trainer = Trainer_BERT( cf, devices).create()
  trainer.run()

####################################################################################################
if __name__ == "__main__":
  train_fresh = False

  try:
    if train_fresh:
        train()
    else:
      wandb_id, epoch, epoch_continue = (
        "uvrdtc0a",
        95,
        95,
      )  # multiformer from scratch
      Trainer = Trainer_BERT
      train_continue(wandb_id, epoch, Trainer, epoch_continue)

  except Exception:
    extype, value, tb = sys.exc_info()
    traceback.print_exc()
    pdb.post_mortem(tb)
