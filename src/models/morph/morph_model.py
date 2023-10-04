from typing import NamedTuple, Protocol, Callable, Tuple, Optional
from jaxtyping import Array, Float, Scalar
import jax.numpy as jnp

from ..pose import Observations

class MorphParameters(Protocol):
    """
    Parameter set for a general morph model.
    """
    pass

class MorphHyperparams(Protocol):
    """
    Hyperparameters describing a morph model.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    """
    N: int
    M: int


class MorphModel(NamedTuple):
    sample_parameters: Callable[..., MorphParameters]
    get_transform: Callable[
        [MorphParameters, MorphHyperparams],
        Tuple[Float[Array, "N KD M"], Float[Array, "N KD"]]]
    log_prior: Callable[
        [MorphParameters, MorphHyperparams,
         Optional[Float[Array, "N KD M"]], Optional[Float[Array, "N KD"]]],
        Scalar]
    init: Callable[..., MorphParameters]

    def pose_mle(
        self,
        observations: Observations,
        params: MorphParameters,
        hyperparams: MorphHyperparams
        ) -> Float[Array, "Nt M"]:
        C, d = self.get_transform(params, hyperparams)
        Cinv = jnp.linalg.inv(C)[observations.subject_ids]
        d = d[observations.subject_ids]
        return (Cinv @ (observations.keypts - d)[..., None])[..., 0]
    