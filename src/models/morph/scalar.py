from typing import NamedTuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import jax.random as jr
import jax.numpy as jnp

from .morph_model import MorphModel



class ScalarMorphParameters(NamedTuple):
    """
    Parameter set describing a subject-wise scalar morph.

    :param scale: Scale factor for each subject.
    :param M: Pose space dimension.
    """
    scale_log: Float[Array, "N"]


class ScalarMorphHyperparams(NamedTuple):
    """
    Hyperparameters describing a subject-wise scalar morph.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    """
    N: int
    M: int





def get_transform(
    params: ScalarMorphParameters,
    hyperparams: ScalarMorphHyperparams,
    ) -> Float[Array, "N 1 1"]:
    """
    Calculate the linear transform defining the morph model
    using a given set of parameters.

    Returns
    -------
    morph_transform:
        Linear transformation broadcastable to shape (N, M, M).
    """
    return (
        jnp.eye(hyperparams.M)[None] *
        jnp.exp(params.scale_log)[:, None, None]
    )


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: ScalarMorphHyperparams,
    log_std: Scalar,
    ) -> ScalarMorphParameters:
    """
    Sample a set of `ScalarMorphParameters`.
    
    The parameters are sampled according to the generative model:
    $$
    \mathrm{scale_log} \sim \mathcal{N}(0, \mathrm{log_std}^2)
    $$

    :param rkey: JAX random key.
    :param hyperparams: Hyperparameters of the resulting morph model.
    :param log_std: Standard deviation of log scale factors.
    """
    return ScalarMorphParameters(
        scale_log = log_std * jr.normal(rkey,
            shape = (hyperparams.N,))
    )


ScalarMorph = MorphModel(
    sample_parameters = sample_parameters,
    get_transform = get_transform
)
