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

import numpy as np
import os
import code

import datetime

import wandb

from atmorep.core.trainer import Trainer_BERT
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
from atmorep.utils.utils import NetMode
import atmorep.utils.utils as utils

import atmorep.config.config as config

class Evaluator( Trainer_BERT) :

  ##############################################
  def __init__( self, cf, devices) :
    Trainer_BERT.__init__( self, cf, devices)

  ##############################################
  def parse_args( cf, args) :

    # set/over-write options as desired
    for (key,val) in args.items() :
      if '[' in key :  # handle lists, e.g. fields[0][2]
        key_split = key.split( '[')
        k, v = key_split[0], key_split[1:]
        v = [int(a[0]) for a in v]
        utils.list_replace_rec( getattr( cf, k), v, val)
      else :
        setattr( cf, key, val)

  ##############################################
  @staticmethod
  def run( cf, model_id, model_epoch, devices) :

    # set/over-write options as desired
    evaluator = Evaluator.load( cf, model_id, model_epoch, devices)
    evaluator.model.load_data( NetMode.test) 
    if 0 == cf.par_rank :
      cf.print()
      cf.write_json( wandb)
    evaluator.validate( 0, cf.BERT_strategy)

  ##############################################
  @staticmethod
  def evaluate( mode, model_id, args = {}, model_epoch=-2) :

    # SLURM_TASKS_PER_NODE is controlled by #SBATCH --ntasks-per-node=1; should be 1 for multiformer
    with_ddp = True
    if '-1' == os.environ.get('MASTER_ADDR', '-1') :
      with_ddp = False
      num_accs_per_task = 1 
    else :
      num_accs_per_task = int( 4 / int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] ))
    devices = init_torch( num_accs_per_task)
    par_rank, par_size = setup_ddp( with_ddp)

    cf = Config().load_json( model_id)
    cf.with_wandb = True
    cf.with_ddp = with_ddp
    cf.par_rank = par_rank
    cf.par_size = par_size
    # overwrite old config
    cf.data_dir = str(config.path_data)
    cf.attention = False
    setup_wandb( cf.with_wandb, cf, par_rank, '', mode='offline')
    if 0 == cf.par_rank :
      print( 'Running Evaluate.evaluate with mode =', mode)

    cf.num_loader_workers = cf.loader_num_workers
    cf.data_dir = './data/'
    cf.rng_seed = None 
    
    #backward compatibility
    if not hasattr( cf, 'n_size'):
      #cf.n_size = [36, 0.25*9*6, 0.25*9*12]
      cf.n_size = [36, 0.25*27*2, 0.25*27*4] 
    if not hasattr(cf, 'num_samples_per_epoch'):
      cf.num_samples_per_epoch = 1024
    if not hasattr(cf, 'with_mixed_precision'):
      cf.with_mixed_precision = False
    # cf.batch_size_start = 14
    func = getattr( Evaluator, mode)
    func( cf, model_id, model_epoch, devices, args)

  ##############################################
  @staticmethod
  def BERT( cf, model_id, model_epoch, devices, args = {}) :

    cf.lat_sampling_weighted = False
    cf.BERT_strategy = 'BERT'
    cf.log_test_num_ranks = 4
    if not hasattr(cf, 'num_samples_validate'):
      cf.num_samples_validate = 128
    Evaluator.parse_args( cf, args)

    Evaluator.run( cf, model_id, model_epoch, devices)

  ##############################################
  @staticmethod
  def forecast( cf, model_id, model_epoch, devices, args = {}) :

    cf.lat_sampling_weighted = False
    cf.BERT_strategy = 'forecast'
    cf.log_test_num_ranks = 4
    cf.forecast_num_tokens = 1  # will be overwritten when user specified
    if not hasattr(cf, 'num_samples_validate'):
      cf.num_samples_validate = 128 
    Evaluator.parse_args( cf, args)
    
    Evaluator.run( cf, model_id, model_epoch, devices)
  
  ##############################################
  @staticmethod
  def global_forecast( cf, model_id, model_epoch, devices, args = {}) :

    cf.BERT_strategy = 'forecast'
    cf.batch_size_test = 24
    cf.num_loader_workers = 1
    cf.log_test_num_ranks = 1
    cf.batch_size_start = 14
    if not hasattr(cf, 'num_samples_validate'):
      cf.num_samples_validate = 196 
    Evaluator.parse_args( cf, args)

    dates = args['dates']
    print("inside global forecast")
    evaluator = Evaluator.load( cf, model_id, model_epoch, devices)
    evaluator.model.set_global( NetMode.test, np.array( dates))
    if 0 == cf.par_rank :
      cf.print()
      cf.write_json( wandb)
    evaluator.validate( 0, cf.BERT_strategy)

  ##############################################
  @staticmethod
  def global_forecast_range( cf, model_id, model_epoch, devices, args = {}) :

    cf.forecast_num_tokens = 2
    cf.BERT_strategy = 'forecast'
    cf.token_overlap = [2, 6]

    cf.batch_size_test = 24
    cf.num_loader_workers = 1
    cf.log_test_num_ranks = 1
    cf.batch_size_start = 14
    if not hasattr(cf, 'num_samples_validate'):
      cf.num_samples_validate = 196 

    Evaluator.parse_args( cf, args)

    if 0 == cf.par_rank :
      cf.print()
      cf.write_json( wandb)

    # generate temporal sequence
    dates = [ ]
    num_steps = 31*2 
    cur_date = [2018, 1, 1, 0+6] #6h models
    for _ in range(num_steps) :
      tdate = datetime.datetime( cur_date[0], cur_date[1], cur_date[2], cur_date[3])
      tdate += datetime.timedelta( hours = 12 )
      cur_date = [tdate.year, tdate.month, tdate.day, tdate.hour]
      dates += [cur_date]

    evaluator = Evaluator.load( cf, model_id, model_epoch, devices)
    evaluator.model.set_global( NetMode.test, np.array( dates))
    evaluator.evaluate( 0, cf.BERT_strategy)

  ##############################################
  @staticmethod
  def temporal_interpolation( cf, model_id, model_epoch, devices, args = {}) :

    # set/over-write options
    cf.BERT_strategy = 'temporal_interpolation'
    cf.log_test_num_ranks = 4

    Evaluator.run( cf, model_id, model_epoch, devices)

  ##############################################
  @staticmethod
  def fixed_location( cf, model_id, model_epoch, devices, args = {}) :

    # set/over-write options
    cf.BERT_strategy = 'BERT'
    cf.num_files_test = 2
    cf.num_patches_per_t_test = 2
    cf.log_test_num_ranks = 4

    pos = [ 33.55 , 18.25 ]
    years = [2018]
    months = list(range(1,12+1))
    num_t_samples_per_month = 2

    evaluator = Evaluator.load( cf, model_id, model_epoch, devices)
    evaluator.model.set_location( NetMode.test, pos, years, months, num_t_samples_per_month)
    if 0 == cf.par_rank :
      cf.print()
      cf.write_json( wandb)
    evaluator.evaluate( 0)
