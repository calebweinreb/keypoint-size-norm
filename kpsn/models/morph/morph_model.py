from typing import NamedTuple, Protocol, Callable, Tuple, Optional
from jaxtyping import Array, Float, Scalar, Integer
import jax.numpy as jnp

from ..pose import Observations


class MorphTrainedParams(Protocol):
    """
    Parameter set for a general morph model.
    """
    def with_hyperparams(self, hyperparams):
        return MorphParameters(self, hyperparams)


class MorphHyperparams(Protocol):
    """
    Hyperparameters describing a morph model.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    """
    N: int
    M: int


class MorphParameters(Protocol):
    params: MorphTrainedParams
    hyperparams: MorphHyperparams


class MorphModel(NamedTuple):
    """
    In all types, $N$ is the number of sessions, $M$ ($=KD$) is the
    dimensionality of pose (and keypoint) space and $K$ is one or many
    batch dimensions.
    """
    ParameterClass: type
    sample_hyperparams: Callable[..., MorphHyperparams]
    sample_parameters: Callable[..., MorphTrainedParams]
    transform: Callable[
        [MorphParameters, Float[Array, "*#K M"], Integer[Array, "*#K"]],
        Float[Array, "*#K M"]]
    inverse_transform: Callable[
        [MorphParameters, Float[Array, "*#K M"], Integer[Array, "*#K"]],
        Float[Array, "*#K M"]]
    log_prior: Callable[
        [MorphParameters],
        dict]
    init_hyperparams: Callable[..., MorphHyperparams]
    init: Callable[..., MorphTrainedParams]
    reports: Callable[[MorphParameters], dict]