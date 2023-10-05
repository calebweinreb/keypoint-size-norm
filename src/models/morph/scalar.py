from typing import NamedTuple, Tuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import jax.random as jr
import jax.numpy as jnp

from .morph_model import MorphModel
from ..pose.pose_model import Observations
from ...util import pca


class ScalarMorphParameters(NamedTuple):
    """
    Parameter set describing a subject-wise scalar morph.

    :param scale: Scale factor for each subject.
    :param M: Pose space dimension.
    """
    scale_log: Float[Array, "N"]
    offsets: Float[Array, "N M"]


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
    ), params.offsets


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: ScalarMorphHyperparams,
    log_std: Scalar,
    offset_std: Scalar
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
    rkey = jr.split(rkey, 2)
    return ScalarMorphParameters(
        scale_log = log_std * jr.normal(rkey[0],
            shape = (hyperparams.N,)),
        offsets = offset_std * jr.normal(rkey[1],
            shape = (hyperparams.N, hyperparams.M))
    )

def init(
    hyperparams: ScalarMorphHyperparams,
    observations: Observations,
    reference_subject: int,
    seed: int = 0
    ) -> ScalarMorphParameters:
    """
    Initialize `ScalarMorphParameters` and pose latents
    """
    subjwise_keypts = observations.unstack(observations.keypts)
    offsets = jnp.stack([
        subj_kpts.mean(axis = 0) for subj_kpts in subjwise_keypts])
    scales = jnp.stack([
        (jnp.linalg.norm(subj_kpts - subj_ofs[None], axis = 1) ** 2).mean()
        for subj_kpts, subj_ofs
        in zip(subjwise_keypts, offsets)])
    scales_log = jnp.log(scales) / 2
    scales_log = scales_log - scales_log.mean()
    return ScalarMorphParameters(
        scale_log = scales_log,
        offsets = offsets
    )

def log_prior(
    params: ScalarMorphParameters,
    hyperparams: ScalarMorphHyperparams,
    morph_matrix: Float[Array, "N KD M"] = None,
    morph_ofs: Float[Array, "N KD"] = None
    ):
    avg_offset_sqnorm = -((params.offsets.mean(axis = 0)) ** 2).sum() / 2
    avg_scale_sqnorm = -(params.scale_log.mean() ** 2) / 2
    return avg_offset_sqnorm + avg_scale_sqnorm
    

ScalarMorph = MorphModel(
    sample_parameters = sample_parameters,
    get_transform = get_transform,
    init = init,
    log_prior = log_prior,
)
