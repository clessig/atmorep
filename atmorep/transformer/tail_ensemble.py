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

class TailEnsemble( torch.nn.Module) :

  def __init__( self, cf, dim_embed, dim_net_input, net_tail_num_nets = -1 ) :
    
    super( TailEnsemble, self).__init__()

    self.cf = cf
    self.dim_embed = dim_embed
    self.dim_input = dim_net_input
    self.net_tail_num_nets = net_tail_num_nets if net_tail_num_nets > 0 else cf.net_tail_num_nets

  ###################################################
  def create( self) :

    dim = self.dim_embed

    # tail networks: use class token to make prediction 
    nonlin = torch.nn.GELU()
    self.tail_nets = torch.nn.ModuleList()
    for inet in range( self.net_tail_num_nets) :
      self.tail_nets.append( torch.nn.ModuleList())
      self.tail_nets[-1].append( torch.nn.LayerNorm( dim, elementwise_affine=True))
      for _ in range( self.cf.net_tail_num_layers) :
        self.tail_nets[-1].append( torch.nn.Linear( dim, dim, bias=True))
        self.tail_nets[-1].append( nonlin)
      # un-embedding layer
      self.tail_nets[-1].append( torch.nn.Linear( dim, self.dim_input, bias=True)) 

    return self 

  ###################################
  def device( self):
    return next(self.parameters()).device

  ###################################################
  def forward( self, rep ) :

    rep.to( self.device())

    # evaluate ensemble of tail networks
    preds = []
    for tail_net in self.tail_nets : 
      cpred = rep
      for block in tail_net :
        cpred = block(cpred)
      preds.append( cpred.unsqueeze(1))
    preds = torch.cat( preds, 1)

    # # mean and variance of ensemble
    if 1 == len(self.tail_nets) :    # avoid that std_dev is NaN with 1 "ensemble" member
      dev = preds.device
      pred = ( torch.mean(preds,1), torch.zeros( torch.std(preds,1).shape, device=dev ), preds )
    else :
      pred = ( torch.mean(preds,1), torch.std(preds,1), preds )

    return pred