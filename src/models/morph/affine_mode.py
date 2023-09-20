from typing import NamedTuple, Union
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import jax.random as jr
import jax.numpy as jnp
import jax.numpy.linalg as jla
import scipy.stats

from .morph_model import MorphModel



class AffineModeMorphParameters(NamedTuple):
    """
    Parameter set describing an affine mode morph.

    :param uniform_scale: Logarithm of factor across all dimensions.
    :param modal_scale: Logarithm of factor across each morph dimension.
    :param modes: Morph dimensions across the full population.
    :param updates: Updates to the morph modes for each subject.
    """
    uniform_scale: Float[Array, "N"]
    modal_scale: Float[Array, "N L"]
    modes: Float[Array, "M L"]
    updates: Float[Array, "N M L"]


class AffineModeMorphHyperparams(NamedTuple):
    """
    Hyperparameters describing an affine mode morph.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of morph dimensions.
    :param modes: Morph dimensions across the full population (optional).
    
    """
    N: int
    M: int
    L: int
    modes: Union[Float[Array, "M L"], None]




def get_transform(
    params: AffineModeMorphParameters,
    hyperparams: AffineModeMorphHyperparams,
    ) -> Float[Array, "N M M"]:
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

    # reconst @ diag(exp(s)) @ U^+, shape: (N, M, M)
    rot_scale = jnp.einsum(
        # i=1...N, j=1...M, k=1...L, l=1...M
        "ijk,ik,kl->ijl", # jk,k,kl batched over i
        reconst, jnp.exp(params.modal_scale), mp_inv)
    
    return (
        # (N, M, M) + (1, M, M)
        (rot_scale + orthog_proj[None]) *
        # (N, 1, 1)
        jnp.exp(params.uniform_scale)[:, None, None]
    )


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: AffineModeMorphHyperparams,
    uniform_scale_std: Scalar,
    modal_scale_std: Scalar,
    mode_std: Scalar,
    update_std: Scalar,
    ) -> AffineModeMorphParameters:
    r"""
    Sample a set of `AffineModeMorphParameters`.
    
    The parameters are sampled according to the generative model: $$
    \begin{align} \mathrm{uniform_scale} &\sim
        \mathcal{N}(0, \mathrm{uniform_scale_std}^2) //
    \mathrm{modal_scale} &\sim
        \mathcal{N}(0, \mathrm{modal_scale_std}^2 I_N)
    \mathrm{modes} \sim
        \mathrm{SO}(M)_{:, :L} + \mathcal{N}(0, \mathrm{mode_std}^2 I_{ML})
    \mathrm{updates} \sim
        \mathcal{N}(0, \mathrm{update_std}^2 I_{NML})
    \end{align} $$

    :param rkey: JAX random key.
    :param hyperparams: Hyperparameters of the resulting morph model.
    :param uniform_scale_std: Standard deviation of log uniform scale factor.
    :param modal_scale_std: Standard deviation of log modal scale factors.
    :param mode_std: Standard deviation of independent normal noise atop random
        orthogonal vectors defining morph dimensions.
    :param update_std: Standard deviation of independent normal mode update
        vectors.
    """
    rkey = jr.split(rkey, 4)
    ret = AffineModeMorphParameters(
        uniform_scale = uniform_scale_std * jr.normal(rkey[1],
            shape = (hyperparams.N,)),
        modal_scale = modal_scale_std * jr.normal(rkey[2],
            shape = (hyperparams.N, hyperparams.L)),
        modes = (
            None if hyperparams.modes is not None else
            scipy.stats.special_ortho_group.rvs(
                hyperparams.M)[:, :hyperparams.L] + 
            mode_std * jr.normal(rkey[3],
                shape = (hyperparams.M, hyperparams.L))
            #jnp.eye(hyperparams.M)[:, :hyperparams.L]
        ),
        updates = update_std * jr.normal(rkey[4],
            shape = (hyperparams.N, hyperparams.M, hyperparams.L))
    )
    return ret


AffineModeMorph = MorphModel(
    sample_parameters = sample_parameters,
    get_transform = get_transform,
    init = None
)
