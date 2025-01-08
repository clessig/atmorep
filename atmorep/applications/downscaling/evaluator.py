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
import pytest
import datetime

import wandb

from atmorep.applications.downscaling.trainer import Trainer_Downscaling
from atmorep.utils.utils import Config
from atmorep.utils.utils import setup_ddp
from atmorep.utils.utils import setup_wandb
from atmorep.utils.utils import init_torch
from atmorep.utils.utils import NetMode
import atmorep.utils.utils as utils

import atmorep.config.config as config


class Evaluator( Trainer_Downscaling) :

    ####################################
    def __init__( self, cf, devices) :
        Trainer_Downscaling.__init__( self, cf, devices)

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

    ###################################
    @staticmethod
    def run( cf, model_id, model_epoch, devices) :

        cf.with_mixed_precision = True

        evaluator = Evaluator.load( cf, model_id, model_epoch, devices)

        if 0 ==  cf.par_rank :
            cf.print()
            cf.write_json( wandb)
        evaluator.validate( 0)

    @staticmethod
    def evaluate( model_id, args = {}, model_epoch=-2):

        devices = init_torch()
        with_ddp = True
        par_rank, par_size = setup_ddp( with_ddp)

        cf = Config().load_json( model_id)

        cf.num_accs_per_task = len(devices)
        cf.with_wandb = True
        cf.with_ddp = with_ddp
        cf.par_rank = par_rank
        cf.par_size = par_size
        cf.losses = cf.losses
        cf.attention = False
        setup_wandb( cf.with_wandb, cf, par_rank, '', mode='offline')
        if 0 == cf.par_rank :
            print( 'Running Evaluate.evaluate for downscaling')

        cf.num_loader_workers = 12
        cf.rng_seed = None

        #backward compatibility
        if not hasattr( cf, 'n_size'):
          cf.n_size = [36, 9*6, 9*12]
          #cf.n_size = [36, 0.25*27*2, 0.25*27*4] 
        if not hasattr(cf, 'num_samples_per_epoch'):
          cf.num_samples_per_epoch = 1024
        if not hasattr(cf, 'with_mixed_precision'):
          cf.with_mixed_precision = False
        if not hasattr(cf, 'with_pytest'):
          cf.with_pytest = False
        if not hasattr(cf, 'batch_size'):
          cf.batch_size = cf.batch_size_max
        if not hasattr(cf, 'batch_size_validation'):
          cf.batch_size_validation = cf.batch_size_max
        if not hasattr(cf, 'years_val'):
          cf.years_val = cf.years_test

        func = getattr( Evaluator, 'downscaling')
        func( cf, model_id, model_epoch, devices, args)

        if cf.with_pytest:
          fields = [field[0] for field in cf.fields_downscaling]
          for field in fields:
            pytest.main(["-x", "-s", "./atmorep/tests/applications/downscaling/validation_test.py", "--field", field, "--model_id", cf.wandb_id])

    @staticmethod
    def downscaling( cf, model_id, model_epoch, devices, args = {}) :

        cf.log_test_num_ranks = 1
        cf.num_samples_validate = 10
        
        dates = args['dates']
        token_overlap = args['token_overlap']
        evaluator = Evaluator.load( cf, model_id, model_epoch, devices)
        evaluator.model.set_source_idxs( NetMode.test, dates, token_overlap)

        if 0 == cf.par_rank :
            cf.print()
            cf.write_json( wandb)
        evaluator.validate( 0)
    

