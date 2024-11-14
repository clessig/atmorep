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

import datetime
import json
import os
from pathlib import Path
from enum import Enum
import wandb
import code
from calendar import monthrange
#import properscoring as ps
import numpy as np

import torch.distributed as dist
import torch.utils.data.distributed

import pandas as pd

import atmorep.config.config as config
from atmorep.utils.logger import logger

####################################################################################################
class NetMode( Enum) :
  indeterminate = 0
  train = 1
  test = 2

####################################################################################################
# Helper function to be able to json dump configs with classes 
# in which case the class name is dumped
# Note that loading a config then will lead to problems/not be possible
def json_default(o):
  if type(o) == type :
    return o.__name__
  else :
    return o.to_json()

####################################################################################################
class Config :

  def __init__( self, user_config=None) :
    if user_config is None:
      # original dir that venv is created from will be used
      user_config = config.UserConfig.from_path(config.ATMOREP_PROJECT_DIR)

    self.user_config = user_config

  def add_to_wandb( self, wandb) :
    wandb.config.update( self.__dict__)

  def get_self_dict( self) :
    return self.__dict__

  def print( self) :
    self_dict = self.__dict__
    for key, value in self_dict.items() : 
        print("{} : {}".format( key, value))

  def create_dirs( self, wandb) :
    # TODO why safe in 2 dirs ???
    dirname = self.user_config.results / "models" / f"id{wandb.run.id}"
    if not os.path.exists(dirname):
      os.makedirs( dirname)
      
    dirname = self.user_config.results / f"id{wandb.run.id}"
    if not os.path.exists(dirname):
      os.makedirs( dirname)
      
  def write_json( self, wandb) :

    if not hasattr( wandb.run, 'id') :
      return

    json_str = json.dumps(self.__dict__ )

    # save in directory with model files
    dirname = self.user_config.results / "models" / f"id{wandb.run.id}"
    if not os.path.exists(dirname):
      os.makedirs( dirname)
      
    fname = dirname / "model_id{wandb.run.id}.json"
    with open(fname, 'w') as f :
      f.write( json_str)

    # also save in results directory
    # TODO WHY ??? 
    dirname = self.user_config.results / f"id{wandb.run.id}"
    if not os.path.exists(dirname):
      os.makedirs( dirname)
    fname = dirname / f'model_id{wandb.run.id}.json'
    with open(fname, 'w') as f :
      f.write( json_str)

  def load_json( self, wandb_id) :
    if '/' in wandb_id :   # assumed to be full path instead of just id
      fname = wandb_id
    else :
      fname = self.user_config.models / f"id{wandb_id}" / f"model_id{wandb_id}.json"
    try :
      with open(fname, 'r') as f :
        json_str = f.readlines() 
    except (OSError, IOError) as e:
      # try path used for logging training results and checkpoints
      try :
        fname = self.user_config.results / 'models' / f'id{wandb_id}' / f'model_id{wandb_id}.json'
        with open(fname, 'r') as f :
          json_str = f.readlines()
      except (OSError, IOError) as e:
        print( f'Could not find fname due to {e}. Aborting.')
        quit()

    self.__dict__ = json.loads( json_str[0])

    # fix for backward compatibility
    if not hasattr( self, 'model_id') :
      self.model_id = self.wandb_id

    return self

####################################################################################################
def identity( func, *args) :
  return func( *args)

####################################################################################################

def str_to_tensor(modelid):
    return torch.tensor([ord(c) for c in modelid], dtype=torch.int32)

def tensor_to_str(tensor):
    return ''.join([chr(x) for x in tensor])

####################################################################################################
def init_torch() :
  
  torch.set_printoptions( linewidth=120)

  use_cuda = torch.cuda.is_available()
  if not use_cuda :
    return torch.device( 'cpu')

  num_accs_per_task = torch.cuda.device_count()
  if num_accs_per_task == '1' :
    devices = ['cuda']
  else :
    devices = [f'cuda:{i}' for i in range(num_accs_per_task)]
  logger.info( 'Using devices : {}'.format( devices) )

  torch.backends.cuda.matmul.allow_tf32 = True

  return devices 

####################################################################################################
def setup_ddp( with_ddp = True) :

  rank = 0
  size = 1

  master_node = os.environ.get('MASTER_ADDR', '-1')

  if with_ddp and (master_node != '-1'):

    local_rank = int(os.environ.get("SLURM_LOCALID"))
    ranks_per_node = int( os.environ.get('SLURM_TASKS_PER_NODE', '1')[0] )
    rank = int(os.environ.get("SLURM_NODEID")) * ranks_per_node + local_rank
    size = int(os.environ.get("SLURM_NTASKS"))

    master_node = os.environ.get('MASTER_ADDR', '-1')
    dist.init_process_group( backend='nccl', init_method='tcp://' + master_node + ':1345',
                              timeout=datetime.timedelta(seconds=10*8192),
                              world_size = size, rank = rank) 
    logger.info( f'Using DDP with MASTER_ADDR={master_node}.' )
  else :
    logger.info( 'DDP is not used.' )

  return rank, size

####################################################################################################
def setup_wandb( with_wandb, cf, rank, project_name = None, entity = 'atmorep', wandb_id = None,
                 mode='offline') :

  if with_wandb :
    wandb.require("service")
  
    if 0 == rank :

      slurm_job_id_node = os.environ.get('SLURM_JOB_ID', '-1')
      if slurm_job_id_node != '-1' :
        cf.slurm_job_id = slurm_job_id_node

      if None == wandb_id : 
        wandb.init( project = project_name, entity = entity,
                    mode = mode,
                    config = cf.get_self_dict() )
      else :
        wandb.init( id=wandb_id, resume='must',
                    mode = mode,
                    config = cf.get_self_dict() )
      wandb.run.log_code( root='./atmorep', include_fn=lambda path : path.endswith('.py'))
      
      # append slurm job id if defined
      if slurm_job_id_node != '-1' :
        wandb.run.name = 'atmorep-{}-{}'.format( wandb.run.id, slurm_job_id_node)
      else :
        wandb.run.name = 'atmorep-{}'.format( wandb.run.id)
      print( 'Wandb run: {}'.format( wandb.run.name))

      cf.wandb_id = wandb.run.id

  # communicate wandb id to all nodes
  wandb_id_int = torch.zeros( 8, dtype=torch.int32).cuda()
  if cf.with_wandb and cf.with_ddp:
    if 0 == rank :
      wandb_id_int = str_to_tensor( cf.wandb_id).cuda()
    dist.all_reduce( wandb_id_int, op=torch.distributed.ReduceOp.SUM )
    cf.wandb_id = tensor_to_str( wandb_id_int)

####################################################################################################
def init_weights_uniform( m, scale=0.01):
    '''Initialization of weights using uniform distribution'''

    classname = m.__class__.__name__

    if classname.find('ModuleList') != -1:
      for mm in m :
        mm.apply( lambda n: init_weights_uniform( n, scale) )

    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, scale)
        if m.bias is not None :
          m.bias.data.fill_(0)

####################################################################################################
def shape_to_str( shape) :
  ret ='{}'.format( list(shape)).replace(' ', '').replace(',','_').replace('(','s_').replace(')','')
  ret = ret.replace('[','s_').replace(']','')
  return ret

####################################################################################################

#TODO: should be a method of atmorep.core.atmorep_model.AtmoRep
def get_model_filename(user_config: config.UserConfig, model = None, model_id = '', epoch=-2, with_model_path = True) :
  if isinstance( model, str) :
    name = model 
  elif model :
    name = model.__class__.__name__
  else : # backward compatibility
    name = 'mod'

  if epoch > -2:
    filename = '{}_id{}_epoch{}.mod'.format(name, model_id, epoch)
  else:
    filename = '{}_id{}.mod'.format( name, model_id)

  # pretrained_models_path = config.path_models
  user_models_path = user_config.models
  if with_model_path:
    # pretrained_models_path = pretrained_models_path / f'id{model_id}'
    user_models_path = user_models_path / f'id{model_id}'

  # check both locations for matching model
  # model_file = pretrained_models_path / filename
  #if not model_file.is_file():
  model_file = user_models_path / filename
      
  return model_file

####################################################################################################
def relMSELoss( pred, target = None) :
  val = torch.mean( (pred - target) * (pred - target)) / torch.mean( target * target)
  return val
  
####################################################################################################
def days_in_month( year, month) :
  '''Days in month in specific year'''
  return monthrange( year, month)[1]

def days_until_month_in_year( year, month) :
  '''Days in year until month starts'''

  offset = 0
  for im in range( month - 1) :
    offset += monthrange( year, im+1)[1]
  
  return offset

####################################################################################################
def tokenize( data, token_size = [-1,-1,-1]) :

  data_tokenized = data
  if token_size[0] > -1 :

    data_shape = data.shape
    tok_tot_t = int( data_shape[-3] / token_size[0])
    tok_tot_x = int( data_shape[-2] / token_size[1])
    tok_tot_y = int( data_shape[-1] / token_size[2])

    if 5 == len(data_shape) :
      t2 = torch.reshape( data, (data.shape[0], data.shape[1], tok_tot_t, token_size[0], 
                                tok_tot_x, token_size[1], tok_tot_y, token_size[2]))
      data_tokenized = t2.permute( [0, 1, 2, 4, 6, 3, 5, 7])
    elif 4 == len(data_shape) :
      t2 = torch.reshape( data, (-1, tok_tot_t, token_size[0], 
                                tok_tot_x, token_size[1], tok_tot_y, token_size[2]))
      data_tokenized = t2.permute( [0, 1, 3, 5, 4, 3, 6])
    elif 3 == len(data_shape) :
      t2 = torch.reshape( data, (tok_tot_t, token_size[0], tok_tot_x, token_size[1], tok_tot_y, token_size[2]))
      data_tokenized = torch.transpose(torch.transpose( torch.transpose( t2, 4, 3), 2, 1), 3, 2)
    elif 2 == len(data_shape) :
      t2 = torch.reshape( data, (tok_tot_x, token_size[0], tok_tot_y, token_size[1]))
      data_tokenized = torch.transpose( t2, 1, 2)
    else :
      assert False

  return data_tokenized.contiguous()

####################################################################################################
def detokenize( data) :

  data = data.transpose( [*np.arange( len(data.shape)-5), -3, -5, -2, -4, -1])
  data = data.reshape( [*data.shape[:-6], np.prod( data.shape[-6:-4]), # time
                                          np.prod( data.shape[-4:-2]), # lat
                                          np.prod( data.shape[-2:])])  #lon
  return data 

####################################################################################################
def sgn_exp( x ) :
  '''exponential preserving sign'''
  return x.sign() * (torch.exp( x.abs() ) - 1.)

####################################################################################################
def token_info_to_time( token_info, return_pd = True) :
  str = f'{int(token_info[0])}-{int(np.floor(token_info[1]))+1}-{int(token_info[2])}'
  # correct for 1 day since %j strangely starts from 1
  date = pd.to_datetime( str, format='%Y-%j-%H')
  return date if return_pd else (date.year, date.month, date.day, date.hour)

####################################################################################################
def list_replace_rec( list, idxs, val) :
  if len(idxs) == 1 :
    list.__setitem__( idxs[0], val)
  else :
    list_replace_rec( list.__getitem__( idxs[0]), idxs[1:], val)
    list.__setitem__( idxs[0], list.__getitem__( idxs[0]) )

####################################################################################################
def Gaussian( x, mu=0., std_dev=1.) :
  # return (1 / (std_dev*np.sqrt(2.*np.pi))) * torch.exp( -0.5 * (x-mu)*(x-mu) / (std_dev*std_dev))
  # unnormalized Gaussian where maximum is one
  return torch.exp( -0.5 * (x-mu)*(x-mu) / (std_dev*std_dev))

def erf( x, mu=0., std_dev=1.) :
  c1 = torch.sqrt( torch.tensor(0.5 * np.pi) )
  c2 = torch.sqrt( 1. / torch.tensor(std_dev * std_dev))
  c3 = torch.sqrt( torch.tensor( 2.) )
  val = c1 * ( 1./c2 - std_dev * torch.special.erf( (mu - x) / (c3 * std_dev) ) )
  return val

########################################
# def CRPS_ps( y, mu, std_dev) :
#   val = ps.crps_gaussian(y.cpu().detach().numpy(), mu=mu.cpu().detach().numpy(), sig=std_dev.cpu().detach().numpy())
#   return torch.tensor(val)

def CRPS( y, mu, std_dev) :
   
  # see Eq. A2 in S. Rasp and S. Lerch. Neural networks for postprocessing ensemble weather forecasts. Monthly Weather Review, 146(11):3885 – 3900, 2018.
  c1 = np.sqrt(1./np.pi)
  t1 = 2. * erf( (y-mu) / std_dev) - 1.
  t2 = 2. * Gaussian( (y-mu) / std_dev)
  val = std_dev * ( (y-mu)/std_dev * t1 + t2 - c1 )
  return val

########################################
# def kernel_crps_ps( target, ens) :
#   val = ps.crps_ensemble(target.cpu().detach().numpy(), ens.permute([1,2,0]).cpu().detach().numpy())
#   return torch.tensor(val)

def kernel_crps( target, ens, fair = True) :
  ens_size = ens.shape[0]
  mae = torch.cat( [(target - mem).abs().mean().unsqueeze(0) for mem in ens], 0).mean()

  if ens_size == 1:
    return mae

  coef = -1.0 / (2.0 * ens_size * (ens_size - 1)) if fair else -1.0 / (2.0 * ens_size**2)
  temp = [(p1 - p2).abs().sum() for p1 in ens for p2 in ens]

  ens_var = coef * torch.tensor( [(p1 - p2).abs().sum() for p1 in ens for p2 in ens]).sum()
  ens_var /= (ens.shape[1]*ens.shape[2])

  return mae + ens_var

########################################

def get_weights(lats_idx, lat_min = -90., lat_max = 90., reso = 0.25):
  lat_range = lat_max - lat_min 
  bins = lat_range/reso+1

  theta_weight = np.array([np.cos(w) for w in np.arange( lat_max * np.pi/lat_range , lat_min * np.pi/lat_range, -np.pi/bins)], dtype = np.float32)
  
  return theta_weight[lats_idx]

########################################

def weighted_mse(x, target, weights):
        return torch.sum(weights * (x - target) **2 )/torch.sum(weights)

########################################

def check_num_samples(num_samples_validate, batch_size):
  assert num_samples_validate // batch_size > 0, f"Num samples validate: {num_samples_validate} is smaller than batch size: {batch_size}. Please increase it."
