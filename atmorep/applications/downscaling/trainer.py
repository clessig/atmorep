

import torch
import torchinfo
import numpy as np
import time
import code

from pathlib import Path
import os
import datetime
import functools

import wandb

import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.utils.data.distributed

import atmorep.config.config as config

#from atmorep.core.atmorep_model import AtmoRep
#from atmorep.core.atmorep_model import AtmoRepData
from atmorep.training.bert import prepare_batch_BERT_multifield
from atmorep.transformer.transformer_base import positional_encoding_harmonic
from atmorep.core.trainer import Trainer_Base
from atmorep.applications.downscaling.atmorep_downscaling import AtmoRepDownscaling, AtmoRepDownscalingData

import atmorep.utils.token_infos_transformations as token_infos_transformations

from atmorep.utils.utils import Gaussian, CRPS, kernel_crps, weighted_mse, NetMode, tokenize, detokenize
from atmorep.utils.logger import logger

class Trainer_Downscaling( Trainer_Base):
    
    def __init__(self, cf, devices) :
        super(Trainer_Downscaling, self).__init__( cf, devices)
        self.cf = cf
        self.devices = devices

        self.fields_downscaling = cf.fields_downscaling
        self.loss_weights = torch.zeros( len(cf.fields_downscaling) )
        for ifield, field in enumerate(cf.fields_downscaling) :
            self.loss_weights[ifield] = self.cf.fields_downscaling[ifield][5]
                    
        self.MSELoss = torch.nn.MSELoss()
        self.rng_seed = cf.rng_seed
        if not self.rng_seed :
            self.rng_seed = int(torch.randint( 10000000000, (1,))) 

        if 0 == cf.par_rank :
            directory = Path( config.path_results, 'id{}'.format( cf.wandb_id))
        if not os.path.exists(directory):
            os.makedirs( directory)
            directory = Path( config.path_models, 'id{}'.format( cf.wandb_id))
        if not os.path.exists(directory):
            os.makedirs( directory)
        ll = len(cf.fields) * 8 #len(cf.vertical_levels)
        if cf.BERT_fields_synced :
            self.rngs = [np.random.default_rng(self.rng_seed) for _ in range(ll)]
        else :
            self.rngs = [np.random.default_rng(self.rng_seed+i) for i in range(ll)]
        self.pre_batch = functools.partial( prepare_batch_BERT_multifield, self.cf, self.rngs, 
                                                          self.cf.fields, self.cf.BERT_strategy )
    def create( self, load_embeds=True):
        net = AtmoRepDownscaling( self.cf)
        self.model = AtmoRepDownscalingData( net)

        self.model.create(self.pre_batch, self.devices)

        for idx, _ in enumerate(self.model.net.tails) :
            self.model.net.tails[idx] = torch.nn.Identity()


        self.model.net.encoder_to_decoder = self.encoder_to_decoder
        #self.model.decoder_to_preceiver = self.decoder_to_preceiver
        self.model.net.decoder_to_tail = self.decoder_to_tail
        return self

    @classmethod
    def load( Typename, cf, model_id, epoch, devices):
        trainer = Typename( cf, devices).create( load_embeds=False)
        trainer.model.net = trainer.model.net.load( model_id, devices, cf, epoch)

        
        trainer.model.net.encoder_to_decoder = self.encoder_to_decoder
        #trainer.model.decoder_to_preceiver = self.decoder_to_preceiver
        trainer.decoder_to_tail = self.decoder_to_tail

        print_statement = "Loaded model id = {}{}.".format(model_id, f' at epoch = {epoch}' if epoch > -2 else "")
        print(print_statement)
        return trainer

    def save( self, epoch) :
        self.model.net.save( epoch)

    def get_learn_rates( self) :
        cf = self.cf
        size_padding = 5
        learn_rates = np.zeros( cf.num_epochs + size_padding)
        learn_rates[:cf.lr_start_epochs] = np.linspace( cf.lr_start, cf.lr_max, num = cf.lr_start_epochs)
        lr = learn_rates[cf.lr_start_epochs-1]
        ic = 0
        for epoch in range( cf.lr_start_epochs, cf.num_epochs + size_padding) :
            lr = max( lr / cf.lr_decay_rate, cf.lr_min)
            learn_rates[epoch] = lr
            if ic > 9999 :  # sanity check
                assert "Maximum number of epochs exceeded."
        return learn_rates

    def run( self, epoch = -1) :
        
        cf = self.cf
        model = self.model
        print(model)

        learn_rates = self.get_learn_rates()
        if cf.with_ddp :
            self.model_ddp = torch.nn.parallel.DistributedDataParallel( model, static_graph=True)
            if not cf.optimizer_zero :
                self.optimizer = torch.optim.AdamW( self.model_ddp.parameters(), lr=cf.lr_start,
                                            weight_decay=cf.weight_decay)
            else :
                self.optimizer = ZeroRedundancyOptimizer(self.model_ddp.parameters(),
                                                              optimizer_class=torch.optim.AdamW,
                                                              lr=cf.lr_start )
        else :
            self.model_ddp = model
            self.optimizer = torch.optim.AdamW( self.model.parameters(), lr=cf.lr_start,
                                          weight_decay=cf.weight_decay)
    
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=cf.with_mixed_precision)

        if 0 == cf.par_rank :
            # print( self.model.net)
            model_parameters = filter(lambda p: p.requires_grad, self.model_ddp.parameters())
            num_params = sum([np.prod(p.size()) for p in model_parameters])
            print( f'Number of trainable parameters: {num_params:,}')


        if cf.test_initial :
            cur_test_loss = self.validate( epoch, cf.BERT_strategy).cpu().numpy()
            test_loss = np.array( [cur_test_loss])
        else :
              # generic value based on data normalization
            test_loss = np.array( [1.0]) 
        epoch += 1
           
        if cf.profile :
            lr = learn_rates[epoch]
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            self.profile()

        # training loop
        while True :

            if epoch >= cf.num_epochs :
                break
                
            lr = learn_rates[epoch]
            for g in self.optimizer.param_groups:
                g['lr'] = lr

            tstr = datetime.datetime.now().strftime("%H:%M:%S")
            print( '{} : {} :: batch_size = {}, lr = {}'.format( epoch, tstr, cf.batch_size, lr) )

            self.train( epoch)

            if cf.with_wandb and 0 == cf.par_rank :
                self.save( epoch)

            cur_test_loss = self.validate( epoch ).cpu().numpy()
            # self.validate( epoch, 'forecast')

            # save model 
            if cur_test_loss < test_loss.min() :
                self.save( -2)
            test_loss = np.append( test_loss, [cur_test_loss])

            epoch += 1

            tstr = datetime.datetime.now().strftime("%H:%M:%S")
            print( 'Finished training at {} with test loss = {}.'.format( tstr, test_loss[-1]) )

            # save final network
            if cf.with_wandb and 0 == cf.par_rank :
                self.save( -2)

            cur_test_loss = self.validate( epoch).cpu().numpy()

            if cur_test_loss < test_loss.min() :
                 self.save( -2)
            test_loss = np.append( test_loss, [cur_test_loss])

            epoch += 1

            tstr = datetime.datetime.now().strftime("%H:%M:%S")
            print( 'Finished training at {} with test loss = {}.'.format( tstr, test_loss[-1]) )

            # save final network
            if cf.with_wandb and 0 == cf.par_rank :
                self.save( -2)

    #########################################################
    def train( self, epoch) :
        model = self.model
        cf = self.cf
        model.mode( NetMode.train)
        self.optimizer.zero_grad()

        loss_total = [[] for i in range(len(cf.losses))]
        std_dev_total = [[] for i in range(len(self.fields_downscaling))]               
        mse_loss_total = []
        grad_loss_total = []
        ctr = 0

        self.optimizer.zero_grad()
        time_start = time.time()
        
        for batch_idx in range( model.len( NetMode.train)):

            batch_data = self.model.next()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cf.with_mixed_precision):
                batch_data,targets = self.prepare_batch_downscaling( batch_data)    #batch_data)
                preds, _ = self.model_ddp( batch_data)
                loss, mse_loss, losses = self.loss( preds, targets)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

            self.optimizer.zero_grad()


            [loss_total[idx].append( losses[key]) for idx, key in enumerate(losses.keys())]

            mse_loss_total.append( mse_loss.detach().cpu() )
            grad_loss_total.append( loss.detach().cpu() )
            [std_dev_total[idx].append( pred[1].detach().cpu()) for idx, pred in enumerate(preds)]


            if int((batch_idx * cf.batch_size) / 8) > ctr :
                # wandb logging
                if cf.with_wandb and (0 == cf.par_rank) :
                    loss_dict = { "training loss": torch.mean( torch.tensor( mse_loss_total)), 
                                    "gradient loss": torch.mean( torch.tensor( grad_loss_total))}
                    # log individual loss terms for individual fields
                    for idx, cur_loss in enumerate(loss_total) :
                        loss_name = self.cf.losses[idx]
                        lt = torch.tensor(cur_loss)
                        for i, field in enumerate(cf.fields_downscaling) :
                            idx_name = loss_name + ', ' + field[0]
                            idx_std_name = 'stddev, ' + field[0]
                            loss_dict[idx_name] = torch.mean( lt[:,i]).cpu().detach()
                            loss_dict[idx_std_name] = torch.mean(torch.cat(std_dev_total[i],0)).cpu().detach()
                    wandb.log( loss_dict )
                  
                    # console output
                    samples_sec = cf.batch_size / (time.time() - time_start)
                    str = 'epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:1.5f} : {:1.5f} :: {:1.5f} ({:2.2f} s/sec)'
                    print( str.format( epoch, batch_idx, model.len( NetMode.train),
                                        100. * batch_idx/model.len(NetMode.train), 
                                        torch.mean( torch.tensor( grad_loss_total)), 
                                        torch.mean(torch.tensor(mse_loss_total)),
                                        torch.mean( preds[0][1]), samples_sec ), flush=True)
                
                    # save model (use -2 as epoch to indicate latest, stored without epoch specification)
                    self.save( -2)

                # reset
                loss_total = [[] for i in range(len(cf.losses)) ]
                mse_loss_total = []
                grad_loss_total = []
                std_dev_total = [[] for i in range(len(self.fields_prediction_idx)) ]
                    
                ctr += 1
                time_start = time.time()

            # save gradients
        if cf.save_grads and cf.with_wandb and (0 == cf.par_rank) :
            
            dir_name = './grads/id{}'.format( cf.wandb_id)
            if not os.path.exists(dir_name) :
                os.makedirs(dir_name)

            rmsprop_ws = []
            for k in range( len(self.optimizer.state_dict()['state']) ) : 
                rmsprop_ws.append(self.optimizer.state_dict()['state'][k]['exp_avg_sq'].mean().unsqueeze(0))
            rmsprop_ws = torch.cat( rmsprop_ws)
            fname = '{}/{}_epoch{}_rmsprop.npy'.format( dir_name, cf.wandb_id, epoch)
            np.save( fname, rmsprop_ws.cpu().detach().numpy() ) 
             
            idx = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad : 
                    fname = '{}/{}_epoch{}_{:05d}_{}_grad.npy'.format( dir_name, cf.wandb_id, epoch, idx,name)
                    np.save( fname, param.grad.cpu().detach().numpy() ) 
                    idx += 1

        # clean memory
        self.optimizer.zero_grad()
        del batch_data, loss, loss_total, mse_loss_total, grad_loss_total, std_dev_total

    def validate( self, epoch):

        cf = self.cf
        self.model.mode( NetMode.test)
        total_loss = 0.
        total_losses = torch.zeros( len(self.fields_downscaling) )
        test_len = 0


        self.mode_test = True

        with torch.no_grad() :
            for it in range( self.model.len( NetMode.test)) :
                #batch_data = self.model.next()
                #if cf.par_rank < cf.log_test_num_ranks :
                #    inputs, targets = batch_data[0]
                #    log_sources = ( [inp.detach().clone().cpu() for inp in inputs])
                with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=cf.with_mixed_precision):
                    batch_data, targets = self.prepare_batch_downscaling()
                    preds, atts = self.model( batch_data)
                loss = torch.tensor( 0.)
                ifield = 0
                for  idx, pred in enumerate( preds) :
                    target = targets[idx]
                    self.test_loss( pred, target)
                    target = torch.flatten(torch.flatten(target,-3,-1),-3,-2)

                    cur_loss = self.MSELoss( pred[0], target = target ).cpu().item()

                    loss += cur_loss
                    total_losses[ifield] += cur_loss
                    ifield += 1

                total_loss += loss
                test_len += 1
                
                #logging    
                #if cf.par_rank < cf.log_test_num_ranks :
                #    log_preds


                if cf.with_ddp :
                    total_loss_cuda = total_loss.cuda()
                    total_losses_cuda = total_losses.cuda()
                    dist.all_reduce( total_loss_cuda, op=torch.distributed.ReduceOp.AVG )
                    dist.all_reduce( total_losses_cuda, op=torch.distributed.ReduceOp.AVG )
                    total_loss = total_loss_cuda.cpu()
                    total_losses = total_losses_cuda.cpu()


                if 0 == cf.par_rank :
                    print( 'Validation loss at epoch {} : {}'.format(epoch, total_loss), flush=True)


                if cf.with_wandb and (0 == cf.par_rank) :
                    loss_dict = {"val. loss"  : total_loss}
                    total_losses = total_losses.cpu().detach()
                    for i, field in enumerate(cf.fields_downscaling) :
                        idx_name = 'val_' + field[0]
                        loss_dict[idx_name] = total_losses[i]
                        print( 'validation loss for {} : {}'.format( field[0], total_losses[i] ))
                    wandb.log( loss_dict)
                batch_data = []
                torch.cuda.empty_cache()

                self.mode_test = False

                return total_loss


    def loss( self, preds, targets) :

        cf = self.cf
        mse_loss_total = torch.tensor( 0.,)
        losses = dict(zip(cf.losses,[[] for loss in cf.losses ]))

        for pred, target in zip( preds, targets) :

            target = torch.flatten(torch.flatten(target,-3,-1),-3,-2)
            mse_loss = self.MSELoss( pred[0], target = target)
            mse_loss_total += mse_loss.cpu().detach()

            
            if 'mse' in self.cf.losses :
                losses['mse'].append( mse_loss)

            if 'mse_ensemble' in cf.losses :
                loss_en = torch.tensor( 0., device=target.device)
                for en in torch.transpose( pred[2],1,0) :
                    loss_en += self.MSELoss( en, target = target)
                losses['mse_ensemble'].append( loss_en/ pred[2].shape[1])


            if 'stats' in self.cf.losses :
                stats_loss = Gaussian( target, pred[0], pred[1])
                diff = (stats_loss-1.)

                stats_loss = torch.mean( diff*diff) + torch.mean( torch.sqrt( torch.abs( pred[1])))
                losses['stats'].append( stats_loss)

        loss = torch.tensor( 0.)
        tot_weight = torch.tensor( 0.)
        for key in losses:
            for ifield, val in enumerate(losses[key]) :
                loss += self.loss_weights[ifield] * val.to(loss.device)
                tot_weight += self.loss_weights[ifield]
        loss /= tot_weight
        mse_loss = mse_loss_total / len(self.cf.fields_downscaling)

        return loss, mse_loss, losses

    ###################################################
    def encoder_to_decoder( self, embeds_layers) :
        return ([embeds_layers[i][-1] for i in range(len(embeds_layers))] , embeds_layers )

    ###################################################
    def decoder_to_tail( self, idx_pred, pred) :
        '''Positional encoding of masked tokens for tail network evaluation'''

        field_idx = self.fields_prediction_idx[idx_pred]
        dev = self.devices[ self.cf.fields[field_idx][1][3] ]
        
        # flatten token dimensions: remove space-time separation
        pred = torch.flatten( pred, 2, 3).to( dev)

        return pred


    ####################################################
    def prepare_batch_downscaling( self, xin):                  #( self, xin):
        cf = self.cf
        devs = self.devices

        (sources, token_infos) = xin[0]
        (self.sources_idxs, self.sources_info) = xin[1]
        (self.targets, self.target_token_infos) = xin[2]

        logger.info("sources_prepare_batch_script", sources.shape)
        logger.info("token_infos_batch_script", token_infos.shape)

        logger.info("sources_idxs_batch_script", self.sources_idxs.shape)
        logger.info("sources_info_batch_script", self.sources_info.shape)

        logger.info("targets_batch_script", self.targets.shape)
        logger.info("target_token_infos", self.target_token_infos.shape)

        # network input
        #batch_data = [ ( sources[i].to( devs[ cf.input_fields[i][1][3] ], non_blocking=True), 
        #                self.tok_infos_trans(token_infos[i]).to( self.devices[0], non_blocking=True)) 
        #                  for i in range(len(sources))  ]
        batch_data = [
                (torch.randn((16,5,12,6,12,3,9,9)).to( devs[ cf.input_fields[0][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,5,12*6*12,8)).to( self.devices[0], non_blocking=True))),

                (torch.randn((16,5,12,6,12,3,9,9)).to( devs[ cf.input_fields[1][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,5,12*6*12,8)).to( self.devices[0], non_blocking=True))),
        
                (torch.randn((16,5,12,6,12,3,9,9)).to( devs[ cf.input_fields[2][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,5,12*6*12,8)).to( self.devices[0], non_blocking=True))),
                
                (torch.randn((16,5,12,6,12,3,9,9)).to( devs[ cf.input_fields[3][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,5,12*6*12,8)).to( self.devices[0], non_blocking=True))),
                
                (torch.randn((16,5,12,2,4,3,27,27)).to( devs[ cf.input_fields[4][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,5,12*2*4,8)).to( self.devices[0], non_blocking=True))),
                
                (torch.randn((16,1,12,6,12,3,9,9)).to( devs[ cf.input_fields[5][1][3] ],non_blocking=True),
                self.tok_infos_trans(torch.randn((16,1,12*6*12,8)).to( self.devices[0], non_blocking=True)))]
        target = [torch.randn((16,12,6,3,27,27)).to(cf.input_fields[5][1][3])]

        assert False

        return batch_data, target
