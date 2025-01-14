from atmorep.utils.config import (
    AtmorepConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    GeoRange,
    TimeLatLon,
    FieldConfig,
    PredictionFieldConfig,
    _get_empty_instance,
)
import atmorep.config.config as config

import pathlib as pl
import wandb
import json
import typing


class Config(AtmorepConfig):
    """
    Adapter that imitates Config from utils.utils but uses new AtmorepConfig utils.config internally.

    This class should only facilitate the incremental refactoring of code with references to the old utils.utils.Config class. Once every referce to the legacy configuration arguments has been updatet, this class should be removed.
    """

    def __init__(
        self,
        model: ModelConfig,
        run: RunConfig,
        training: TrainingConfig,
        user_config: config.UserConfig,
    ):
        super().__init__(model, run, training)
        self.user_config = user_config
        self._fields_targets = []

    # expose configuration arguments as expected from rest of system
    @property
    def with_ddp(self) -> bool:
        return self.run.with_ddp

    @with_ddp.setter
    def with_ddp(self, value: bool):
        self.run.with_ddp = value

    @property
    def par_rank(self) -> int:
        return self.run.par_rank

    @par_rank.setter
    def par_rank(self, value: int):
        self.run.par_rank = value

    @property
    def num_accs_per_task(self) -> int:
        return self.run.num_accs

    @num_accs_per_task.setter
    def num_accs_per_task(self, value: int):
        self.run.num_accs = value

    @property
    def par_size(self) -> int:
        return self.run.par_size

    @par_size.setter
    def par_size(self, value: int):
        self.run.par_size = value

    @property
    def fields(self) -> list[list[typing.Any]]:
        return [field.as_list() for field in self.training.fields]

    @fields.setter
    def fields(self, value: list[list[typing.Any]]):
        fields = [FieldConfig.from_list(field) for field in value]
        self.training.fields = fields

    @property
    def fields_prediction(self) -> list[typing.Any]:
        return [field.as_list() for field in self.training.fields_prediction]

    @fields_prediction.setter
    def fields_prediction(self, value: list[typing.Any]):
        fields = [PredictionFieldConfig.from_list(field) for field in value]
        self.training.fields_prediction = fields

    @property
    def fields_targets(self) -> list[typing.Any]:
        return [field for field in self._fields_targets]  # TODO

    @fields_targets.setter
    def fields_targets(self, value: list[typing.Any]):
        self._fields_targets = [field for field in value]  # TODO

    @property
    def years_train(self) -> list[int]:
        return self.training.years_training

    @years_train.setter
    def years_train(self, value: list[int]):
        self.training.years_training = value

    @property
    def years_val(self) -> list[int]:
        return self.training.years_validation

    @years_val.setter
    def years_val(self, value: list[int]):
        self.training.years_validation = value

    @property
    def geo_range_sampling(self) -> list[list[float]]:
        return [
            list(self.training.sampling_range_lat),
            list(self.training.sampling_range_lon),
        ]

    @geo_range_sampling.setter
    def geo_range_sampling(self, value: list[list[float]]):
        sampling_lat, sampling_lon = value
        self.training.sampling_range_lat = GeoRange(*sampling_lat)
        self.training.sampling_range_lon = GeoRange(*sampling_lon)

    @property
    def time_sampling(self) -> int:
        return self.training.sampling_time_rate

    @time_sampling.setter
    def time_sampling(self, value: int):
        self.training.sampling_time_rate = value

    @property
    def torch_seed(self) -> int:
        return self.run.torch_rng_seed

    @torch_seed.setter
    def torch_seed(self, value: int):
        self.run.torch_rng_seed = value

    @property
    def batch_size_validation(self) -> int:
        return self.training.batch_size_val

    @batch_size_validation.setter
    def batch_size_validation(self, value: int):
        self.training.batch_size_val = value

    @property
    def batch_size(self) -> int:
        return self.training.batch_size_train

    @batch_size.setter
    def batch_size(self, value: int):
        self.training.batch_size_train = value

    @property
    def num_epochs(self) -> int:
        return self.training.num_epochs

    @num_epochs.setter
    def num_epochs(self, value: int):
        self.training.num_epochs = value

    @property
    def num_samples_per_epoch(self) -> int:
        return self.training.samples_per_epoch

    @num_samples_per_epoch.setter
    def num_samples_per_epoch(self, value: int):
        self.training.samples_per_epoch = value

    @property
    def num_samples_validate(self) -> int:
        return self.training.samples_validation

    @num_samples_validate.setter
    def num_samples_validate(self, value: int):
        self.training.samples_validation = value

    @property
    def num_loader_workers(self) -> int:
        return self.training.num_workers

    @num_loader_workers.setter
    def num_loader_workers(self, value: int):
        self.training.num_workers = value

    @property
    def size_token_info(self) -> int:
        return self.model.token_info_size

    @size_token_info.setter
    def size_token_info(self, value: int):
        self.model.token_info_size = value

    @property
    def size_token_info_net(self) -> int:
        return self.model.token_embed_size

    @size_token_info_net.setter
    def size_token_info_net(self, value: int):
        self.model.token_embed_size = value

    @property
    def grad_checkpointing(self) -> bool:
        return self.run.grad_checkpointing

    @grad_checkpointing.setter
    def grad_checkpointing(self, value: bool):
        self.run.grad_checkpointing = value

    @property
    def with_cls(self) -> bool:
        return self.model.class_token

    @with_cls.setter
    def with_cls(self, value: bool):
        self.model.class_token = value

    @property
    def with_mixed_precision(self) -> bool:
        return self.model.mixed_prec

    @with_mixed_precision.setter
    def with_mixed_precision(self, value: bool):
        self.model.mixed_prec = value

    @property
    def with_layernorm(self) -> bool:
        return self.model.layernorm

    @with_layernorm.setter
    def with_layernorm(self, value: bool):
        self.model.layernorm = value

    @property
    def coupling_num_heads_per_field(self) -> int:
        return self.model.couple_heads

    @coupling_num_heads_per_field.setter
    def coupling_num_heads_per_field(self, value: int):
        self.model.couple_heads = value

    @property
    def dropout_rate(self) -> float:
        return self.model.dropout_rate

    @dropout_rate.setter
    def dropout_rate(self, value: float):
        self.model.dropout_rate = value

    @property
    def with_qk_lnorm(self) -> bool:
        return self.model.qk_norm

    @with_qk_lnorm.setter
    def with_qk_lnorm(self, value: bool):
        self.model.qk_norm = value

    @property
    def encoder_num_layers(self) -> int:
        return self.model.encoder_layers

    @encoder_num_layers.setter
    def encoder_num_layers(self, value: int):
        self.model.encoder_layers = value

    @property
    def encoder_num_heads(self) -> int:
        return self.model.encoder_heads

    @encoder_num_heads.setter
    def encoder_num_heads(self, value: int):
        self.model.encoder_heads = value

    @property
    def encoder_num_mlp_layers(self) -> int:
        return self.model.encoder_mlp_layers

    @encoder_num_mlp_layers.setter
    def encoder_num_mlp_layers(self, value: int):
        self.model.encoder_mlp_layers = value

    @property
    def encoder_att_type(self) -> str:
        return self.model.encoder_attn_type

    @encoder_att_type.setter
    def encoder_att_type(self, value: str):
        self.model.encoder_attn_type = value

    @property
    def decoder_num_layers(self) -> int:
        return self.model.decoder_layers

    @decoder_num_layers.setter
    def decoder_num_layers(self, value: int):
        self.model.decoder_layers = value

    @property
    def decoder_num_heads(self) -> int:
        return self.model.decoder_heads

    @decoder_num_heads.setter
    def decoder_num_heads(self, value: int):
        self.model.decoder_heads = value

    @property
    def decoder_num_mlp_layers(self) -> int:
        return self.model.decoder_mlp_layers

    @decoder_num_mlp_layers.setter
    def decoder_num_mlp_layers(self, value: int):
        self.model.decoder_mlp_layers = value

    @property
    def decoder_self_att(self) -> bool:
        return self.model.decoder_self_att

    @decoder_self_att.setter
    def decoder_self_att(self, value):
        self.model.decoder_self_att = value

    @property
    def decoder_cross_att_ratio(self) -> float:
        return self.model.decoder_cross_att_ratio

    @decoder_cross_att_ratio.setter
    def decoder_cross_att_ratio(self, value: float):
        self.model.decoder_cross_att_ratio = value

    @property
    def decoder_cross_att_rate(self) -> float:
        return self.model.decoder_cross_att_rate

    @decoder_cross_att_rate.setter
    def decoder_cross_att_rate(self, value: float):
        self.model.decoder_cross_att_rate = value

    @property
    def decoder_att_type(self) -> str:
        return self.model.decoder_attn_type

    @decoder_att_type.setter
    def decoder_att_type(self, value: str):
        self.model.decoder_attn_type = value

    @property
    def net_tail_num_nets(self) -> int:
        return self.model.tail_nets

    @net_tail_num_nets.setter
    def net_tail_num_nets(self, value: int):
        self.model.tail_nets = value

    @property
    def net_tail_num_layers(self) -> int:
        return self.model.tail_nets_layers

    @net_tail_num_layers.setter
    def net_tail_num_layers(self, value: int):
        self.model.tail_nets_layers = value

    @property
    def losses(self) -> list[str]:
        return self.training.losses

    @losses.setter
    def losses(self, value: list[str]):
        self.training.losses = value

    @property
    def optimizer_zero(self) -> bool:
        return self.run.optimizer_zero

    @optimizer_zero.setter
    def optimizer_zero(self, value: bool):
        self.run.optimizer_zero = value

    @property
    def lr_start(self) -> float:
        return self.training.lr_start

    @lr_start.setter
    def lr_start(self, value: float):
        self.training.lr_start = value

    @property
    def lr_max(self) -> float:
        return self.training.lr_max

    @lr_max.setter
    def lr_max(self, value: float):
        self.training.lr_max = value

    @property
    def lr_min(self) -> float:
        return self.training.lr_min

    @lr_min.setter
    def lr_min(self, value: float):
        self.training.lr_min = value

    @property
    def weight_decay(self) -> float:
        return self.training.weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float):
        self.training.weight_decay = value

    @property
    def lr_decay_rate(self) -> float:
        return self.training.lr_decay

    @lr_decay_rate.setter
    def lr_decay_rate(self, value: float):
        self.training.lr_decay = value

    @property
    def lr_start_epochs(self) -> int:
        return self.training.lr_start_epochs

    @lr_start_epochs.setter
    def lr_start_epochs(self, value: int):
        self.training.lr_start_epochs = value

    @property
    def model_log_frequency(self) -> int:
        return self.run.log_frequency

    @model_log_frequency.setter
    def model_log_frequency(self, value: int):
        self.run.log_frequency = value

    @property
    def BERT_strategy(self) -> str:
        return self.training.strategy

    @BERT_strategy.setter
    def BERT_strategy(self, value: str):
        self.training.strategy = value

    @property
    def forecast_num_tokens(self) -> int:
        return self.training.num_forecast_tokens

    @forecast_num_tokens.setter
    def forecast_num_tokens(self, value: int):
        self.training.num_forecast_tokens = value

    @property
    def BERT_fields_synced(self) -> bool:
        return self.training.fields_synced

    @BERT_fields_synced.setter
    def BERT_fields_synced(self, value: bool):
        self.training.fields_synced = value

    @property
    def BERT_mr_max(self) -> int:
        return self.training.maximum_res_reduction

    @BERT_mr_max.setter
    def BERT_mr_max(self, value: int):
        self.training.maximum_res_reduction = value

    @property
    def log_test_num_ranks(self) -> int:
        return self.run.log_num_ranks

    @log_test_num_ranks.setter
    def log_test_num_ranks(self, value: int):
        self.run.log_num_ranks = value

    @property
    def save_grads(self) -> bool:
        return self.run.save_grads

    @save_grads.setter
    def save_grads(self, value: bool):
        self.run.save_grads = value

    @property
    def profile(self) -> bool:
        return self.run.profiler

    @profile.setter
    def profile(self, value: bool):
        self.run.profiler = value

    @property
    def test_initial(self) -> bool:
        return self.run.test_initial

    @test_initial.setter
    def test_initial(self, value: bool):
        self.run.test_initial = value

    @property
    def attention(self) -> bool:
        return self.run.log_att

    @attention.setter
    def attention(self, value: bool):
        self.run.log_att = value

    @property
    def rng_seed(self) -> int | None:
        return self.run.rng_seed

    @rng_seed.setter
    def rng_seed(self, value: int | None):
        self.run.rng_seed = value

    @property
    def with_wandb(self) -> bool:
        return self.run.with_wandb

    @with_wandb.setter
    def with_wandb(self, value: bool):
        self.run.with_wandb = value

    @property
    def slurm_job_id(self) -> str:
        return self.run.slurm_id

    @slurm_job_id.setter
    def slurm_job_id(self, value: str):
        self.run.slurm_id = value

    @property
    def wandb_id(self) -> str:
        return self.run.wandb_id

    @wandb_id.setter
    def wandb_id(self, value: str):
        self.run.wandb_id = value

    @property
    def n_size(self) -> list[int | float]:
        return list(self.training.n_size)

    @n_size.setter
    def n_size(self, value: list[int | float]):
        self.training.n_size = TimeLatLon(*value)

    @property
    def with_pytest(self) -> bool | None:
        return self.run.with_pytest

    @with_pytest.setter
    def with_pytest(self, value: bool):
        self.run.with_pytest = value

    def add_to_wandb(self, wandb_id):  # TODO fix unused argument
        """Serialize config to wandb."""
        wandb.config.update(self.as_dict())

    def print(self):
        """Serialize config to stdout."""
        for key, value in self.as_dict().items():
            print(f"{key} : {value}")

    def create_dirs(self, wandb_id: str):
        """
        Ensure directory with wandb_id of run exists.

        Arguments:
            wandb_id (str): unique identifier for the run used as directory name.
        """

        self._run_dir.mkdir(exist_ok=True)
        self._run_dir_alt.mkdir(exist_ok=True)

    def write_json(self, wandb):
        """
        Serialize config into run specific directory.

        Arguments:
            wandb (Any): unused argument purely for compatibility DO NOT USE
        """

        # TODO really nessecairy ???
        self.create_dirs(self.wandb_id)

        config_file_name = f"model_id{self.wandb_id}.json"
        serialized_config = json.dumps(self.as_dict())

        with open(self._run_dir / config_file_name, "w") as fp:
            fp.write(serialized_config)

        with open(self._run_dir_alt / config_file_name, "w") as fp:
            fp.write(serialized_config)

    def load_json(self, wandb_id):
        """Deserialize config from json file in run specific directory."""

        # so that self._run_dir produces correct result
        self.run.wandb_id = wandb_id

        config_file_name = f"model_id{wandb_id}.json"
        
        potential_files = [
            config.path_models / f"id{wandb_id}" / config_file_name, # pretrained models
            self._run_dir / config_file_name, # user models at results/<modelid>
            self._run_dir_alt / config_file_name, # user models at results/models/<model_id>
            pl.Path(wandb_id) # legacy convention => can be removed ??
        ]
        
        for config_file in potential_files:
            if config_file.is_file():
                break
        else: # loop finishes naturally => no file found
            potential_files_ = "\n\t".join([str(file) for file in potential_files])
            msg = f"cant find config file for wandbid: {wandb_id} looked at:\n\t{potential_files_}"
            raise FileNotFoundError(msg)


        return Config.from_json(
            config_file, user_config=self.user_config
        )
    
    def get_self_dict(self): # used by setup_wandb
        return self.as_dict()

    @classmethod
    def init_empty(cls, user_config: config.UserConfig) -> typing.Self:
        return _get_empty_instance(cls, user_config=user_config)

    @property
    def _run_dir(self):
        """Directory where data relevant to the run will be safed."""
        return self.user_config.results / f"id{self.wandb_id}"

    # TODO: phase out ???
    @property
    def _run_dir_alt(self):
        """Alternative directory where data relevant to the run will be safed."""
        return self.user_config.results / "models" / f"id{self.wandb_id}"
