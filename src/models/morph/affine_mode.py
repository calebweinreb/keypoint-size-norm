from typing import NamedTuple, Union, Tuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
import scipy.stats

from .morph_model import MorphModel
from ..pose import Observations
from ...util import pca


class AffineModeMorphParameters(NamedTuple):
    """
    Parameter set describing an affine mode morph.

    :param uniform_scale: Logarithm of factor across all dimensions.
    :param modal_scale: Logarithm of factor across each morph dimension.
    :param modes: Morph dimensions across the full population.
    :param updates: Updates to the morph modes for each subject.
    """
    uniform_scale: Float[Array, "N"]
    modes: Float[Array, "M L"]
    updates: Float[Array, "N M L"]
    offsets: Float[Array, "N M"]


class AffineModeMorphHyperparams(NamedTuple):
    """
    Hyperparameters describing an affine mode morph.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of morph dimensions.
    :param update_scale: Standard deviation of the spherical normal prior on
        mode updates.
    :param modes: Morph dimensions across the full population (optional).
    
    """
    N: int
    M: int
    L: int
    update_scale: Scalar
    modes: Union[Float[Array, "M L"], None]




def get_transform(
    params: AffineModeMorphParameters,
    hyperparams: AffineModeMorphHyperparams,
    ) -> Tuple[Float[Array, "N M M"], Float[Array, "N M"]]:
    """
    Calculate the linear transform defining the morph model
    using a given set of parameters.

    Returns
    -------
    morph_transform:
        Linear transformation broadcastable to shape (N, M, M).
    """

    # Pesudoinverse and projection onto orthogonal complement of U
    mp_inv = jla.pinv(params.modes) # (U^T U)^{-1} U^T, shape: (L, M)
    orthog_proj = ( # I - U (U^T U)^{-1} U^T, shape: (M, M)
        jnp.eye(hyperparams.M) -
        params.modes @ mp_inv
    )
    
    # Reconstruction matrix, U + \hat{U}
    reconst = params.modes[None] + params.updates # (N, M, L)

    # reconst @ U^+, shape: (N, M, M)
    rot = reconst @ mp_inv
    # rot_scale = jnp.einsum( # old: uses modewise scaling
    #     # i=1...N, j=1...M, k=1...L, l=1...M
    #     "ijk,ik,kl->ijl", # jk,k,kl batched over i
    #     reconst, jnp.exp(params.modal_scale), mp_inv)
    
    return (
        # (N, M, M) + (1, M, M)
        (rot + orthog_proj[None]) *
        # (N, 1, 1)
        jnp.exp(params.uniform_scale)[:, None, None]
    ), params.offsets


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: AffineModeMorphHyperparams,
    uniform_scale_std: Scalar,
    mode_std: Scalar,
    update_std: Scalar,
    offset_std: Scalar,
    ) -> AffineModeMorphParameters:
    r"""
    Sample a set of `AffineModeMorphParameters`.
    
    The parameters are sampled according to the generative model: $$
    \begin{align} \mathrm{uniform_scale} &\sim
        \mathcal{N}(0, \mathrm{uniform_scale_std}^2) //
    \mathrm{modes} \sim
        \mathrm{SO}(M)_{:, :L} + \mathcal{N}(0, \mathrm{mode_std}^2 I_{ML})
    \mathrm{updates} \sim
        \mathcal{N}(0, \mathrm{update_std}^2 I_{NML})
    \end{align} $$

    :param rkey: JAX random key.
    :param hyperparams: Hyperparameters of the resulting morph model.
    :param uniform_scale_std: Standard deviation of log uniform scale factor.
    :param mode_std: Standard deviation of independent normal noise atop random
        orthogonal vectors defining morph dimensions.
    :param update_std: Standard deviation of independent normal mode update
        vectors.
    """
    rkey = jr.split(rkey, 6)
    ret = AffineModeMorphParameters(

        uniform_scale = uniform_scale_std * jr.normal(rkey[1],
            shape = (hyperparams.N,)),
        
        modes = (
            None if hyperparams.modes is not None else
            scipy.stats.special_ortho_group.rvs(
                hyperparams.M,
                random_state = np.array(rkey)[3, 0]
            )[:, :hyperparams.L] + 
            mode_std * jr.normal(rkey[3],
                shape = (hyperparams.M, hyperparams.L))
        ),
        
        updates = update_std * jr.normal(rkey[4],
            shape = (hyperparams.N, hyperparams.M, hyperparams.L)),
        
        offsets = offset_std * jr.normal(rkey[5],
            shape = (hyperparams.N, hyperparams.M))
    
    )
    return ret


def log_prior(
    params: AffineModeMorphParameters,
    hyperparams: AffineModeMorphHyperparams,
    morph_matrix: Float[Array, "N KD M"] = None,
    morph_ofs: Float[Array, "N KD"] = None
    ):

    # Logpdfs of N(0, I) evaluated at mean offset and mean uniform_scale
    avg_offset_sqnorm = -((params.offsets.mean(axis = 0)) ** 2).sum() / 2
    avg_scale_sqnorm = -(params.uniform_scale.mean() ** 2) / 2

    # Logpdf of N(0, 1) evaluated at mode log-norms
    mode_norms = jnp.linalg.norm(params.modes, axis = 0) # (L,)
    mode_logpdf = -(jnp.log(mode_norms) ** 2).sum() / 2

    # Logpdf of N(0, update_scale * I) evaluated at each
    # (normalized) update vector
    normed_updates = params.updates / mode_norms[None, None] # (N, M, L)
    update_sqnorms = (normed_updates ** 2).sum(axis = 1) # (N, L)
    update_logpdf = -(update_sqnorms.sum() / 
        hyperparams.update_scale ** (2 * hyperparams.M)) / 2
    
    return (avg_offset_sqnorm + avg_scale_sqnorm +
            mode_logpdf + update_logpdf)



def init(
    hyperparams: AffineModeMorphHyperparams,
    observations: Observations,
    reference_subject: int,
    seed: int = 0
    ) -> AffineModeMorphParameters:
    
    # Calculate offsets
    subjwise_keypts = observations.unstack(observations.keypts)
    offsets = jnp.stack([
        subj_kpts.mean(axis = 0) for subj_kpts in subjwise_keypts])
    
    # Calculate uniform_scale
    keypts_centered = [
        subj_kpts - subj_ofs[None]
        for subj_kpts, subj_ofs
        in zip(subjwise_keypts, offsets)]
    scales = jnp.stack([
        (jnp.linalg.norm(kpts, axis = 1) ** 2).mean()
        for kpts in keypts_centered])
    scales_log = jnp.log(scales) / 2
    scales_log = scales_log - scales_log.mean()
    
    # Calculate modes
    pcs = pca.fit(keypts_centered[reference_subject], centered = True)
    modes = pcs.pcs()[:hyperparams.L, :].T

    return AffineModeMorphParameters(
        uniform_scale = scales_log,
        modes = modes,
        updates = jnp.zeros([hyperparams.N, hyperparams.M, hyperparams.L]),
        offsets = offsets
    )
    
AffineModeMorph = MorphModel(
    sample_parameters = sample_parameters,
    get_transform = get_transform,
    log_prior = log_prior,
    init = init
)
