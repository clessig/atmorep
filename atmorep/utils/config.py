import dataclasses as dc
import pathlib as pl
import json
from typing import Iterable, Any, Self, Optional
from collections import namedtuple

TimeLatLon = namedtuple("TimeLatLon", ["time", "lat", "lon"])
GeoRange = namedtuple("GeoRange", ["start", "stop"])


@dc.dataclass
class PredictionFieldConfig:
    """
    Configuration class for a field to be predicted.

    Attributes:
        name (str): Name of the variable used in the Zarr dataset. Must be consistent with `FieldConfig`.
        weight (float): Weight of this variable in the loss function. The weights of all fields should sum to 1.
    """
    name: str # fields[x][0]
    """ Name of the variable used in the zarr. Must be consitent FieldConfig """
    
    weight: float
    """ Weight of this variable in the loss funtion. Weights of all fields shoud add up to 1 """
    
    @classmethod
    def from_list(cls, prediction_field: list[Any]) -> Self:
        """ deserialize from model config format. """
        return cls(prediction_field[0], prediction_field[1])
    
    def as_list(self) -> list[Any]:
        """ serialize into model config format. """
        return [self.name, self.weight]


@dc.dataclass
class FieldConfig:
    """
    Configuration class for a single field.

    Attributes:
        name (str): Name of the variable used in the Zarr dataset.
        dynamic (bool): Indicates whether the field is dynamic. If `False`, the field is static.
        embed_dim (int): Embedding dimension.
        dev_id (int): CUDA device ID within the process.
        vertical_levels (Iterable[int]): Vertical levels to be used. These must match the Zarr file convention.
        num_tokens (Iterable[int]): Number of tokens for each dimension, in the order [time, lon, lat].
        token_size (Iterable[int]): Size of the tokens for each dimension, in the order [time, lon, lat].
        total_mask_rate (float): Total masking rate.
        mask_rate (float): Local masking rate.
        noise_rate (float): Noising rate.
        dist_rate (float): Multi-resolution distortion rate.
    """


    name: str # fields[x][0]
    """ Name of the variable used in the zarr """

    dynamic: bool # field[x][1][0] # TODO: maybe ignore while parsing ??
    """ If true filed is dynamic, otherwise it's static """

    embed_dim: int # field[x][1][1]
    """ Embedding dimension """

    dev_id: int # field[x][1][2]
    """ CUDA device ID within the process """
    
    vertical_levels: Iterable[int] # fields[x][2]
    """ Vertical levels that are to be used. They have to match zarr file convention """

    num_tokens: TimeLatLon # fields[x][3]
    """ List containing number of tokens for each dimension in order [time, lon, lat] """

    token_size: TimeLatLon # fields[x][4] ()
    """ List containing sizes of the tokens for each dimension in order [time, lon, lat] """
    
    total_mask_rate: float # fields[x][5][0]
    """ Total masking rate """

    mask_rate: float # fields[x][5][1]
    """ Local masking rate """

    noise_rate: float # fields[x][5][2]
    """ Noising rate """

    dist_rate: float # fields[x][5][3]
    """ Multi-resolution distortion rate """

    def make_predictable(self, weight: float = 1) -> PredictionFieldConfig:
        """ Make PredictionFieldConfig for this field
        """
        return PredictionFieldConfig(self.name, weight)
    
    @classmethod
    def from_list(cls, field: list[Any]) -> Self:
        """ deserialize from model config format. """
        name = field[0]
        dynamic = field[1][0]
        embed_dim = field[1][1]
        dev_id = field[1][2]
        vertical_lvls = field[2]
        num_tokens = TimeLatLon(*field[3])
        token_size = TimeLatLon(*field[4])
        total_mask_rate = field[5][0]
        mask_rate = field[5][1]
        noise_rate = field[5][2]
        dist_rate = field[5][3]
        
        return cls(
            name,
            dynamic,
            embed_dim,
            dev_id,
            vertical_lvls,
            num_tokens,
            token_size,
            total_mask_rate,
            mask_rate,
            noise_rate,
            dist_rate
        )
    
    def as_list(self) -> list[Any]:
        """ serialize into model config format. """
        return [
            self.name,
            [self.dynamic, self.embed_dim, self.dev_id],
            self.vertical_levels,
            list(self.num_tokens),
            list(self.token_size),
            [self.total_mask_rate, self.mask_rate, self.noise_rate, self.dist_rate]
        ]


@dc.dataclass
class ModelConfig:
    """
    Configuration class for the Atmorep model.

    Attributes:
        mixed_prec (bool): Indicates whether the model uses mixed precision.
        layernorm (bool): Specifies whether the model uses layer normalization.
        couple_heads (int): Number of attention heads used for each field.
        dropout_rate (float): Dropout rate used during training.
        qk_norm (bool): Indicates whether the model uses Query-Key normalization.
        encoder_layers (int): Number of layers in the encoder.
        encoder_heads (int): Number of attention heads in the encoder.
        encoder_mlp_layers (int): Number of MLP layers in the encoder.
        encoder_attn_type (str): Type of attention in the encoder. Can be 'dense' or 'axial'.
        decoder_layers (int): Number of layers in the decoder.
        decoder_heads (int): Number of attention heads in the decoder.
        decoder_mlp_layers (int): Number of MLP layers in the decoder.
        decoder_attn_type (str): Type of attention in the decoder. Can be 'dense' or 'axial'.
        decoder_self_att (bool): Indicates whether the decoder uses self-attention.
        decoder_cross_att_ratio (float): Ratio of attention heads used for cross-attention with other fields.
        decoder_cross_att_rate (float): Ratio of attention heads used for cross-attention with the encoder.
        tail_nets (int): Number of tail networks.
        tail_nets_layers (int): Number of layers in each tail network.
        token_info_size (int): Size for the token_info array used for computing positional embeddings.
        token_embed_size (int): Number of elements in the token used for storing positional embedding.
        class_token (bool): If true model appends class token to the tokens created from input data.
    """
    mixed_prec: bool
    """ If true, model uses mixed precision """

    layernorm: bool 
    """ If true model uses layer normalisation """

    couple_heads: int
    """ Number of attention heads used for each field """

    dropout_rate: float
    """ Dropout rate for training """

    qk_norm: bool
    """ If true model uses Query-Key normalisation """

    encoder_layers: int
    """ Number of layers inside the encoder """

    encoder_heads: int
    """ Number of attention heads inside the encoder """

    encoder_mlp_layers: int
    """ Number of MLP layers inside the encoder """

    encoder_attn_type: str
    """ Type of attention for the encoder. Can be 'dense' or 'axial' """

    decoder_layers: int
    """ Number of layers inside the decoder """

    decoder_heads: int
    """ Number of attention heads inside the decoder """

    decoder_mlp_layers: int
    """ Number of MLP layers inside the decoder """

    decoder_attn_type: str
    """ Type of attention for the decoder. Can be 'dense' or 'axial' """

    decoder_self_att: bool
    """ If true decoder uses self attention """

    decoder_cross_att_ratio: float
    """ Set's rate of attention heads used for cross attention with other fields """

    decoder_cross_att_rate: float
    """ Set's rate of attention heads to be used for cross attention with encoder """

    tail_nets: int
    """ Number of tail networks """

    tail_nets_layers: int
    """ Number of layers in tail networks """

    token_info_size: int
    """ Size for the token_info tensor used for computing positional embeddings """

    token_embed_size: int
    """ Number of elements in the token used for storing positional embedding """

    class_token: bool
    """ If true model appends class token to the tokens created from input data. """

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """ Deserialize from model config format. """

        return cls(
            config_dict["with_mixed_precision"],
            config_dict["with_layer_norm"],
            config_dict["coupling_num_heads_per_field"],
            config_dict["dropout_rate"],
            config_dict["with_qk_lnorm"],
            config_dict["encoder_num_layers"],
            config_dict["encoder_num_heads"],
            config_dict["encoder_num_mlp_layers"],
            config_dict["encoder_att_type"],
            config_dict["decoder_num_layers"],
            config_dict["decoder_num_heads"],
            config_dict["decoder_num_mlp_layers"],
            config_dict["decoder_att_type"],
            config_dict["decoder_self_att"],
            config_dict["decoder_cross_att_ratio"],
            config_dict["decoder_cross_att_rate"],
            config_dict["net_tail_num_nets"],
            config_dict["net_tail_num_layers"],
            config_dict["size_token_info"],
            config_dict["size_token_info_net"],
            config_dict["with_cls"],
        )

    def as_dict(self) -> dict[str, Any]:
        """ Serialize into model config format. """

        return {
            "with_mixed_precision": self.mixed_prec,
            "with_layer_norm": self.layernorm,
            "coupling_num_heads_per_field": self.couple_heads,
            "dropout_rate": self.dropout_rate,
            "with_qk_lnorm": self.qk_norm,
            "encoder_num_layers": self.encoder_layers,
            "encoder_num_heads": self.encoder_heads,
            "encoder_num_mlp_layers": self.encoder_mlp_layers,
            "encoder_att_type": self.encoder_attn_type,
            "decoder_num_layers": self.decoder_layers,
            "decoder_num_heads": self.decoder_heads,
            "decoder_num_mlp_layers": self.decoder_mlp_layers,
            "decoder_att_type": self.decoder_attn_type,
            "decoder_self_att": self.decoder_self_att,
            "decoder_cross_att_ratio": self.decoder_cross_att_ratio,
            "decoder_cross_att_rate": self.decoder_cross_att_rate,
            "net_tail_num_nets": self.tail_nets,
            "net_tail_num_layers": self.tail_nets_layers,
            "size_token_info": self.token_info_size,
            "size_token_info_net": self.token_embed_size,
            "with_cls": self.class_token
        }


@dc.dataclass
class RunConfig:
    """
    Configuration class for run parameters.

    Attributes:
        wandb_id (str): Run ID assigned by the wandb
        slurm_id (str): Job ID assigned by the SLURM
        with_ddp (bool): Indicates whether Distributed Data Parallel (DDP) is used.
        num_accs (int): Number of accelerators available to each task.
        par_rank (int): Rank of the MPI parallel process. If DDP is not used, this is set to 0.
        par_size (int): Total number of MPI parallel processes. If DDP is not used, this is set to 1.
        log_num_ranks (int): Maximum number of tasks that log output in validation mode.
        save_grads (bool): Indicates whether gradients are saved along with weights and biases.
        profiler (bool): Indicates whether a profiler is used during the run.
        test_initial (bool): Indicates whether the initial test loss is computed. Defaults to 1.0 if false.
        log_att (bool): Indicates whether attention is being logged.
        rng_seed (Optional[int]): Seed for the random number generator. If not specified, a random seed is used.
        with_wandb (bool): Indicates whether Wandb is used for monitoring the run.
        torch_rng_seed (int): Seed for PyTorch's internal random number generator.
        log_frequency (int): Number of batches between saving checkpoints.
        grad_checkpointing (bool): Indicates whether gradient checkpointing is used during training.
    """
    wandb_id: str
    """ Run ID assigned by the wandb """

    slurm_id: str
    """ Job ID assigned by the SLURM """

    with_ddp: bool
    """ If true Distributed Data Parallel is used """

    num_accs: int
    """ Number of accelerators aviable to each task """

    par_rank: int
    """ Rank of used MPI parallel process. If DDP is false, it's 0 """

    par_size: int
    """ Number of all running MPI parallel processes. If DDP is false, it's 1 """

    log_num_ranks: int
    """ Upper limit of tasks that log output in validation mode. """

    save_grads: bool
    """ If true gradients are saved along weights and biases """

    profiler: bool
    """ If true, profiler is used during the run """

    test_initial: bool
    """ If true, initial test loss is computed, otherwise it's 1.0 """

    log_att: bool
    """ If true attention is being logged """

    rng_seed: Optional[int]
    """ Seed for the random number generator, if not specified random seed is used"""

    with_wandb: bool
    """ If true Wandb will be used for monitoring the run. """

    torch_rng_seed: int
    """ Seed for the torch's internal random number generator """

    log_frequency: int
    """ Number of batches between saving checkpoints """

    grad_checkpointing: bool
    """ If true, checkpointing is used in training """

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """ Deserialize from model config format. """

        return cls(
            config_dict["wandb_id"],
            config_dict["slurm_job_id"],
            config_dict["with_ddp"],
            config_dict["num_accs_per_task"],
            config_dict["par_rank"],
            config_dict["par_size"],
            config_dict["log_test_num_ranks"],
            config_dict["save_grads"],
            config_dict["profile"],
            config_dict["test_initial"],
            config_dict["attention"],
            config_dict["rng_seed"],
            config_dict["with_wandb"],
            config_dict["torch_seed"],
            config_dict["model_log_frequency"],
            config_dict["grad_checkpointing"]
        )

    def as_dict(self) -> dict[str, Any]:
        """ Serialize into model config format. """

        return {
            "wandb_id": self.wandb_id,
            "slurm_job_id": self.slurm_id,
            "with_ddp": self.with_ddp,
            "num_accs_per_task": self.num_accs,
            "par_rank": self.par_rank,
            "par_size": self.par_size,
            "log_test_num_ranks": self.log_num_ranks,
            "save_grads": self.save_grads,
            "profile": self.save_grads,
            "test_initial": self.test_initial,
            "attention": self.log_att,
            "rng_seed": self.rng_seed,
            "with_wandb": self.with_wandb,
            "torch_seed": self.torch_rng_seed,
            "model_log_frequency": self.log_frequency,
            "grad_checkpointing": self.grad_checkpointing
        }

@dc.dataclass
class TrainingConfig:
    """
    Configuration class for training parameters.

    Attributes:
        fields (Iterable[FieldConfig]): List of configuration objects for each field.
        fields_prediction (Iterable[PredictionFieldConfig]): Configuration object for predicted fields.
        field_targets (Iterable[PredictionFieldConfig]): Configuration object for fields to be targeted in downscaling applications.
        years_training (Iterable[int]): List of years to be used for training.
        years_validation (Iterable[int]): List of years to be used for validation.
        sampling_range_lat (GeoRange): Range of sampling for latitude.
        sampling_range_lon (GeoRange): Range of sampling for longitude.
        sampling_time_rate (int): Sampling rate for timesteps.
        batch_size_train (int): Batch size for training.
        batch_size_val (int): Batch size for validation.
        num_epochs (int): Number of training epochs.
        samples_per_epoch (int): Number of samples per epoch.
        samples_validation (int): Number of samples for validation.
        num_workers (int): Number of workers to be used by data loaders.
        losses (Iterable[str]): List of loss functions to be used. Available options: 'mse', 'mse_ensemble', 'stats', 'crps', 'weighted_mse'.
        lr_start (float): Initial learning rate for computing learning rates.
        lr_max (float): Maximum learning rate for computing learning rates.
        lr_min (float): Minimum learning rate for computing learning rates.
        lr_decay (float): Learning rate decay for computing learning rates.
        lr_start_epochs (int): Number of epochs used to test learning rates from `lr_start` to `lr_max`.
        weight_decay (float): Weight decay for the optimizer.
        strategy (str): Strategy used for BERT training. Options: 'BERT', 'forecast', 'temporal_interpolation'.
        num_forecast_tokens (int): Number of tokens to forecast when the strategy is set to 'forecast'.
        fields_synced (bool): Indicates whether identical masking is applied to all fields.
        maximum_res_reduction (int): Maximum reduction for the resolution.
    """

    fields: Iterable[FieldConfig]
    """ List of configuration objects for each field """

    fields_prediction: Iterable[PredictionFieldConfig]
    """ Configuration object for predicted fields """

    field_targets: Iterable[PredictionFieldConfig]
    """ Configuration object for fields that are to be target in downscaling application"""

    years_training: Iterable[int]
    """ List of years to be used for training """

    years_validation: Iterable[int]
    """ List of years to be used for validation """

    sampling_range_lat: GeoRange
    """ Range of sampling for latitude """

    sampling_range_lon: GeoRange
    """ Range of sampling for longitude """

    sampling_time_rate: int
    """ Sampling rate for timesteps """

    batch_size_train: int
    """ Batch size for training """

    batch_size_val: int
    """ Batch size for validation """

    num_epochs: int
    """ Number of epochs """

    samples_per_epoch: int
    """ Number of samples per epoch """

    samples_validation: int
    """ Number of samples per validation """

    num_workers: int
    """ Number of workers for to be used by dataloaders """

    losses: Iterable[str]
    """ List of loss functions to be used. Available are: mse, mse_ensemble, stats, crps, weighted_mse """

    lr_start: float
    """ Initial learining rate for computing learning rates """

    lr_max: float
    """ Maximal learining rate for computing learning rates """

    lr_min: float
    """ Minimal learining rate for computing learning rates """

    lr_decay: float
    """ Learing rate decay for computing learning rates """

    lr_start_epochs: int
    """ Set number of epochs used for testing learning rates from lr_start to lr_max """

    weight_decay: float
    """ Weight decay for the optimizer """

    strategy: str
    """ Strategy used for BERT training. Available are: 'BERT', 'forecast', 'temporal_interpolation' """

    num_forecast_tokens: int
    """ Number of tokens to be forecasten when strategy is set to 'forecast' """

    fields_synced: bool
    """ If true identical masking is applied to all fields """

    maximum_res_reduction: int
    """ Maximum reduction for the resolution """

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """ Deserialize from model config format. """

        fields = [FieldConfig.from_list(field) for field in config_dict["fields"]]
        fields_prediction = [
            PredictionFieldConfig.from_list(field)
            for field in config_dict["fields_prediction"]
        ]
        fields_targets = [
            PredictionFieldConfig.from_list(field)
            for field in config_dict["fields_targets"]
        ]
        sampling_lat, sampling_lon = config_dict["geo_range_sampling"]
        sampling_range_lat = GeoRange(*sampling_lat)
        sampling_range_lon = GeoRange(*sampling_lon)

        return cls(
            fields,
            fields_prediction,
            fields_targets,
            config_dict["years_train"],
            config_dict["years_val"],
            sampling_range_lat,
            sampling_range_lon,
            config_dict["time_sampling"],
            config_dict["batch_size"],
            config_dict["batch_size_validation"],
            config_dict["num_epochs"],
            config_dict["num_samples_per_epoch"],
            config_dict["num_samples_validate"],
            config_dict["num_loader_workers"],
            config_dict["losses"],
            config_dict["lr_start"],
            config_dict["lr_max"],
            config_dict["lr_min"],
            config_dict["lr_decay_rate"],
            config_dict["lr_start_epochs"],
            config_dict["weight_decay"],
            config_dict["BERT_strategy"],
            config_dict["forecast_num_tokens"],
            config_dict["BERT_fields_synced"],
            config_dict["BERT_mr_max"]
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": [field.as_list() for field in self.fields],
            "fields_prediction": [
                field.as_list() for field in self.fields_prediction
            ],
            "fields_targets": [field.as_list() for field in self.field_targets],
            "years_train": self.years_training,
            "years_val": self.years_validation,
            "time_sampling": self.sampling_time_rate,
            "batch_size": self.batch_size_train,
            "batch_size_validation": self.batch_size_val,
            "num_epochs": self.num_epochs,
            "num_samples_per_epoch": self.samples_per_epoch,
            "num_samples_validate": self.samples_validation,
            "num_loader_workers": self.num_workers,
            "losses": self.losses,
            "lr_start": self.lr_start,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "lr_decay_rate": self.lr_decay,
            "lr_start_epochs": self.lr_start_epochs,
            "weight_decay": self.weight_decay,
            "BERT_strategy": self.strategy,
            "forecast_num_tokens": self.num_forecast_tokens,
            "BERT_fields_synced": self.fields_synced,
            "BERT_mr_max": self.maximum_res_reduction
        }

@dc.dataclass
class AtmorepConfig:
    model: ModelConfig
    run: RunConfig
    training: TrainingConfig

    @classmethod
    def from_dict(cls, config_dict) -> Self:
        return cls(
            ModelConfig.from_dict(config_dict),
            RunConfig.from_dict(config_dict),
            TrainingConfig.from_dict(config_dict),
        )

    @classmethod
    def from_json(cls, config_file: pl.Path) -> Self:
        """ deserialize Config from model.json file. """

        with open(config_file, "r") as config:
            config_dict = json.load(config)
        
        return cls.from_dict(config_dict)

    def as_dict(self) -> dict[str, Any]:
        return self.model.as_dict() | self.run.as_dict() | self.training.as_dict()
    
    def to_json(self, config_file: pl.Path):
        """ serialize Config into model.json file. """

        with open(config_file, "w") as config:
            json.dump(self.as_dict(), config)
