from atmorep.utils.config import AtmorepConfig, ModelConfig,RunConfig, TrainingConfig
import atmorep.config.config as config

import pathlib as pl
import wandb
import json
import typing

class ConfigFacade(AtmorepConfig):
    """
    Facade that imitates Config from utils.utils but uses new AtmorepConfig utils.config internally.
    """

    def __init__(
        self,
        model: ModelConfig,
        run: RunConfig,
        training: TrainingConfig,
        user_config: config.UserConfig
    ):
        super().__init__(model, run, training)
        self.user_config = user_config
        
        # expose configuration arguments as expected from rest of system
        self.with_ddp: bool = self.run.with_ddp
        self.num_accs_per_task: int = self.run.num_accs
        self.par_rank: int = self.run.par_rank
        self.par_size: int = self.run.par_size
        self.fields: list[typing.Any] = [
            field.as_list() for field in self.training.fields
        ]
        self.fields_prediction: list[typing.Any] = [
            field.as_list() for field in self.training.fields_prediction
        ]
        self.fields_targets: list[typing.Any] = [] # TODO
        self.years_train: list[int] = self.training.years_training
        self.years_val: list[int] = self.training.years_validation
        self.geo_range_sampling: list[list[float]] = [
            list(self.training.sampling_range_lat),
            list(self.training.sampling_range_lon)
        ]
        self.time_sampling: int = self.training.sampling_time_rate
        self.torch_seed: int = self.run.torch_rng_seed
        self.batch_size_validation: int = self.training.batch_size_val
        self.batch_size: int = self.training.batch_size_train
        self.num_epochs: int = self.training.num_epochs
        self.num_samples_per_epoch: int = self.training.samples_per_epoch
        self.num_samples_validate: int = self.training.samples_validation
        self.num_loader_workers: int = self.training.num_workers
        self.size_token_info: int = self.model.token_info_size
        self.size_token_info_net: int = self.model.token_embed_size
        self.grad_checkpointing: bool = self.run.grad_checkpointing
        self.with_cls: bool = self.model.class_token
        self.with_mixed_precision: bool = self.model.mixed_prec
        self.with_layernorm: bool = self.model.layernorm
        self.coupling_num_heads_per_field: int = self.model.couple_heads
        self.dropout_rate: float = self.model.dropout_rate
        self.with_qk_lnorm: bool = self.model.qk_norm
        self.encoder_num_layers: int = self.model.encoder_layers
        self.encoder_num_heads: int = self.model.encoder_heads
        self.encoder_num_mlp_layers: int = self.model.encoder_mlp_layers
        self.encoder_att_type: str = self.model.encoder_attn_type
        self.decoder_num_layers: int = self.model.decoder_layers
        self.decoder_num_heads: int = self.model.decoder_heads
        self.decoder_num_mlp_layers: int = self.model.decoder_mlp_layers
        self.decoder_self_att: bool = self.model.decoder_self_att
        self.decoder_cross_att_ratio: float = self.model.decoder_cross_att_ratio
        self.decoder_cross_att_rate: float = self.model.decoder_cross_att_rate
        self.decoder_att_type: str = self.model.decoder_attn_type
        self.net_tail_num_nets: int = self.model.tail_nets
        self.net_tail_num_layers: int = self.model.tail_nets_layers
        self.losses: list[str] = self.training.losses
        self.optimizer_zero: bool = self.run.optimizer_zero
        self.lr_start: float = self.training.lr_start
        self.lr_max: float = self.training.lr_max
        self.lr_min: float = self.training.lr_min
        self.weight_decay: float = self.training.weight_decay
        self.lr_decay_rate: float = self.training.lr_decay
        self.lr_start_epochs: int = self.training.lr_start_epochs
        self.model_log_frequency: int = self.run.log_frequency
        self.BERT_strategy: str = self.training.strategy
        self.forecast_num_tokens: int = self.training.num_forecast_tokens
        self.BERT_fields_synced: bool = self.training.fields_synced
        self.BERT_mr_max: int = self.training.maximum_res_reduction
        self.log_test_num_ranks: int = self.run.log_num_ranks
        self.save_grads: bool = self.run.save_grads
        self.profile: bool = self.run.profiler
        self.test_initial: bool = self.run.test_initial
        self.attention: bool = self.run.log_att
        self.rng_seed: int | None = self.run.rng_seed
        self.with_wandb: bool = self.run.with_wandb
        self.slurm_job_id: str = self.run.slurm_id
        self.wandb_id: str = self.run.wandb_id
        self.n_size: list[int | float] = list(self.training.n_size)

    def add_to_wandb(self, wandb_id): # TODO fix unused argument
        """ Serialize config to wandb. """
        wandb.config.update(self.as_dict())
        
    def print(self):
        """ Serialize config to stdout. """
        for key, value in self.as_dict():
            print(f"{key} : {value}")

    def create_dirs(self, wandb_id: str):
        """
        Ensure directory with wandb_id of run exists.
        
        Arguments:
            wandb_id (str): unique identifier for the run used as directory name.
        """

        self._run_dir.mkdir(exist_ok=True)
        self._run_dir_alt.mkdir(exist_ok=True)

    def write_json(self):
        """ Serialize config into run specific directory. """

        # TODO really nessecairy ???
        self.create_dirs(self.wandb_id)

        config_file_name = f"model_id{self.wandb_id}.json"
        serialized_config = json.dumps(self.as_dict())

        with open(self._run_dir / config_file_name, "w") as fp:
            fp.write(serialized_config)

        with open(self._run_dir_alt / config_file_name, "w") as fp:
            fp.write(serialized_config)

    def load_json(self, wandb_id):
        """ Deserialize config from json file in run specific directory. """

        # possible file paths
        config_file_name = f"model_id{wandb_id}.json"
        config_pretrained = config.path_models / f"id{wandb_id}" / config_file_name
        config_user = self._run_dir / config_file_name
        config_user_alt = self._run_dir_alt / config_file_name

        if config_user_alt.is_file():
            config_file = config_user_alt

        if config_user.is_file():
            config_file = config_user

        if config_pretrained.is_file():
            config_file = config_pretrained

        # if wandb_id is a file
        # TODO: can be removed ???
        if pl.Path(wandb_id).is_file():
            config_file = pl.Path(wandb_id)
        
        return ConfigFacade.from_json(
            config_file, user_config=self.user_config, wandb_id=self.wandb_id
        )
    
    
    @property
    def _run_dir(self):
        """ Directory where data relevant to the run will be safed. """
        return self.user_config.results / f"id{self.wandb_id}"

    # TODO: phase out ???
    @property
    def _run_dir_alt(self):
        """ Alternative directory where data relevant to the run will be safed. """
        return self.user_config.results / "models" / f"id{self.wandb_id}"
