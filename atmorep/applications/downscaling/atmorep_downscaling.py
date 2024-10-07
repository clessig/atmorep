
import torch
import numpy as np
import code
# code.interact(local=locals())

# import horovod.torch as hvd

import atmorep.utils.utils as utils
from atmorep.utils.utils import identity
from atmorep.utils.utils import NetMode
from atmorep.utils.utils import get_model_filename

from atmorep.transformer.transformer_base import prepare_token
from atmorep.transformer.transformer_base import checkpoint_wrapper

from atmorep.datasets.multifield_data_sampler import MultifieldDataSampler

from atmorep.transformer.transformer_encoder import TransformerEncoder
from atmorep.transformer.transformer_decoder import TransformerDecoder
from atmorep.transformer.perceiver import Perceiver
from atmorep.transformer.tail_ensemble import TailEnsemble
from atmorep.core.atmorep_model import AtmoRep

from atmorep.utils.logger import logger


class AtmoRepDownscalingData( torch.nn.Module) :

    def __init__( self, net) :
        super( AtmoRepDownscalingData, self).__init__()
        
        self.data_loader_test = None
        self.data_loader_train = None
        self.data_loader_iter = None

        self.net = net

        self.rng_seed = self.net.cf.rng_seed
        if not self.rng_seed :
            self.rng_seed = int(torch.randint( 100000000, (1,))) 
    

    def mode( self, mode : NetMode) :

        if mode == NetMode.train :
            self.data_loader_iter = iter(self.data_loader_train)
            self.net.train()
        elif mode == NetMode.test :
            self.data_loader_iter = iter(self.data_loader_test)
            self.net.eval()
        else :
            assert False

        self.cur_mode = mode
  
    ###################################################
    def len( self, mode : NetMode) :

        if mode == NetMode.train :
            return len(self.data_loader_train)
        elif mode == NetMode.test :
            return len(self.data_loader_test)
        else :
            assert False

    ###################################################
    def next( self) :
        return next(self.data_loader_iter)

    ###################################################
    def forward( self, xin) :
        pred = self.net.forward( xin)
        return pred

    ###################################################
    def create( self, pre_batch, devices, create_net = True, pre_batch_targets = None,
            load_pretrained=True) :

        if create_net :
            self.net = self.net.create( devices, load_pretrained)

        self.pre_batch = pre_batch
        self.pre_batch_targets = pre_batch_targets

        cf = self.net.cf
        loader_params = { 'batch_size': None, 'batch_sampler': None, 'shuffle': False, 
                      'num_workers': cf.num_loader_workers, 'pin_memory': True}

        self.dataset_train = MultifieldDataSampler( cf.file_path, cf.fields, cf.years_train,
                                                cf.batch_size,
                                                pre_batch, cf.n_size, cf.num_samples_per_epoch,
                                                with_shuffle = (cf.BERT_strategy != 'global_forecast'), 
                                                with_source_idxs = True, 
                                                compute_weights = (cf.losses.count('weighted_mse') > 0) )
        self.data_loader_train = torch.utils.data.DataLoader( self.dataset_train, **loader_params,
                                                          sampler = None)

        self.dataset_test = MultifieldDataSampler( cf.file_path, cf.fields, cf.years_val,
                                               cf.batch_size_validation,
                                               pre_batch, cf.n_size, cf.num_samples_validate,
                                               with_shuffle = (cf.BERT_strategy != 'global_forecast'),
                                               with_source_idxs = True, 
                                               compute_weights = (cf.losses.count('weighted_mse') > 0) )                                               
        self.data_loader_test = torch.utils.data.DataLoader( self.dataset_test, **loader_params,
                                                          sampler = None)

        return self

class AtmoRepDownscaling( AtmoRep) :

    def __init__(self, cf) :
        
        super( AtmoRepDownscaling, self).__init__(cf)

    def create( self, devices, load_pretrained=True) :
        
        cf = self.cf
        self = super(AtmoRepDownscaling, self).create(devices, load_pretrained=load_pretrained)
        self.perceivers = torch.nn.ModuleList()
        self.downscaling_tails = torch.nn.ModuleList()

        for field_idx,field_info in enumerate(cf.fields_downscaling):

            self.perceivers.append(Perceiver(cf, field_info).create().to(devices[0]))
            self.downscaling_tails.append(TailEnsemble(cf, cf.perceiver_output_emb, np.prod(field_info[4])).create().to(devices[0]))

        return self


    ###################################################
    def load_block( self, field_info, block_name, block ) :

        super().load_block(field_info, block_name, block)

        
    ###################################################
    def translate_weights(self, mloaded, mkeys, ukeys) :

        mloaded = super().translate_weights(mloaded, mkeys, ukeys)
        return mloaded

    ###################################################
    @staticmethod
    def load( model_id, devices, cf = None, epoch = -2, load_pretrained=False) :
        '''Load network from checkpoint'''

        if not cf : 
            cf = utils.Config()
            cf.load_json( model_id)
      
        model = AtmoRepDownscaling( cf).create( devices, load_pretrained=False)
        mloaded = torch.load( utils.get_model_filename( model, model_id, epoch) )
        mkeys, ukeys = model.load_state_dict( mloaded, False )
        if (f'encoders.0.heads.0.proj_heads.weight') in mkeys:
            mloaded = model.translate_weights(mloaded, mkeys, ukeys)
            mkeys, ukeys = model.load_state_dict( mloaded, False )
        if len(mkeys) > 0 :
            print( f'Loaded AtmoRep: ignoring {len(mkeys)} elements: {mkeys}')

        # TODO: remove, only for backward 
        if model.embeds_token_info[0].weight.abs().max() == 0. :
            model.embeds_token_info = torch.nn.ModuleList()
        
        return model
  
    ###################################################
    def save( self, epoch = -2) :
        '''Save network '''

        # save entire network
        torch.save( self.state_dict(), utils.get_model_filename( self, self.cf.wandb_id, epoch) )

        # save parts also separately

        # name = self.__class__.__name__ + '_embed_token_info'
        # torch.save( self.embed_token_info.state_dict(),
        #             utils.get_model_filename( name, self.cf.wandb_id, epoch) )
        name = self.__class__.__name__ + '_embeds_token_info'
        torch.save( self.embeds_token_info.state_dict(),
                    utils.get_model_filename( name, self.cf.wandb_id, epoch) )

        for ifield, enc in enumerate(self.encoders) :
            name = self.__class__.__name__ + '_encoder_' + self.cf.fields[ifield][0]
            torch.save( enc.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )

        for ifield, dec in enumerate(self.decoders) :
            name = self.__class__.__name__ + '_decoder_' + self.cf.fields_prediction[ifield][0]
            torch.save( dec.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )

        for ifield, per in enumerate(self.perceivers) :
            name = self.__class__.__name__ + '_perceivers_' + self.cf.fields_downscaling[ifield][0]
            torch.save( per.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )
  
        for ifield, downscaling_tail in enumerate(self.downscaling_tails) :
            name = self.__class__.__name__ + '_downscaling_tails_' + self.cf.fields_downscaling[ifield][0]
            torch.save( downscaling_tail.state_dict(), utils.get_model_filename( name, self.cf.wandb_id, epoch) )
    ###################################################
    def forward( self, xin) :
        '''Evaluate network'''
        preds, atts = super().forward(xin)

        for idx_perceiver_net,perceiver_net in enumerate(self.perceivers):
            preds[idx_perceiver_net] = perceiver_net(preds[idx_perceiver_net])

        downscale_ensemble = []
        for idx_tail,tails in enumerate(self.downscaling_tails):
            downscale_ensemble.append(self.downscaling_tails[idx_tail](preds[idx_tail]))        

        return downscale_ensemble, atts
  
  
    ###################################################
    def get_fields_embed( self, xin ) :
        prepared_token = super().get_fields_embed(xin)
        return prepared_token    

    ###################################################
    def get_attention( self, xin) : 
        attn = super().get_attention( xin)
        return attn

