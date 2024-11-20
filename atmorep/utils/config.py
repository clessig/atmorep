import dataclasses as dc
from typing import Iterable


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

    dynamic: bool # field[x][1][0]
    """ If true filed is dynamic, otherwise it's static """

    embed_dim: int # field[x][1][1]
    """ Embedding dimension """

    dev_id: int # field[x][1][2]
    """ CUDA device ID within the process """
    
    vertical_levels: Iterable[int] # fields[x][2]
    """ Vertical levels that are to be used. They have to match zarr file convention"""

    num_tokens: Iterable[int] # fields[x][3]
    """ List containing number of tokens for each dimension in order [time, lon, lat]"""

    token_size: Iterable[int] # fields[x][4] ()
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
    


@dc.dataclass
class PredictionConfig:
    """
    Configuration class for the fields to be predicted.

    Attributes:
        fields (Iterable[PredictionFieldConfig]): List of configuration objects for each field to be predicted.
    """
    fields: Iterable[PredictionFieldConfig]
    """ List of configuration object for each field """
