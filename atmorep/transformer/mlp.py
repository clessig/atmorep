
import torch

from atmorep.utils.utils import identity
from atmorep.transformer.transformer_base import checkpoint_wrapper

####################################################################################################
class MLP(torch.nn.Module):

  def __init__(self, dim_embed, num_layers = 2, with_lnorm = True, dim_embed_out = None, 
                     nonlin = torch.nn.GELU(), dim_internal_factor = 2, dropout_rate = 0.,
                     grad_checkpointing = False, with_residual = True) :
    """
    Multi-layer perceptron

    dim_embed   : embedding dimension
    num_layers  : number of layers
    nonlin      : nonlinearity
    dim_internal_factor : factor for number of hidden dimension relative to input / output
    """
    super(MLP, self).__init__()

    if not dim_embed_out :
      dim_embed_out = dim_embed

    self.with_residual = with_residual

    dim_internal = int( dim_embed * dim_internal_factor)
    if with_lnorm : 
      self.lnorm = torch.nn.LayerNorm( dim_embed, elementwise_affine=False)
    else :
      self.lnorm = torch.nn.Identity()

    self.blocks = torch.nn.ModuleList()
    self.blocks.append( torch.nn.Linear( dim_embed, dim_internal))
    self.blocks.append( nonlin)
    self.blocks.append( torch.nn.Dropout( p = dropout_rate))
    
    for _ in range( num_layers-2) :
      self.blocks.append( torch.nn.Linear( dim_internal, dim_internal))
      self.blocks.append( nonlin)
      self.blocks.append( torch.nn.Dropout( p = dropout_rate))
    
    self.blocks.append( torch.nn.Linear( dim_internal, dim_embed_out))
    self.blocks.append( nonlin)

    if dim_embed == dim_embed_out :
      self.proj_residual = torch.nn.Identity()
    else :
      self.proj_residual = torch.nn.Linear( dim_embed, dim_embed_out)

    self.checkpoint    = checkpoint_wrapper if grad_checkpointing else identity

  def forward( self, x, y = None) :
    
    x_in = x
    x = self.lnorm( x)
    
    for block in self.blocks:
      x = self.checkpoint( block, x)

    if self.with_residual :
      x += x_in

    return x