from typing import NamedTuple, Union, Tuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
import scipy.stats

from .morph_model import *
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
    
    @staticmethod
    def create(
        uniform_scale: Float[Array, "N"],
        modes: Float[Array, "M L"],
        updates: Float[Array, "N M L"],
        offsets: Float[Array, "N M"]):
        return AffineModeMorphParameters(
            uniform_scale = AffineModeMorphAllParameters.normalize_scale(uniform_scale),
            modes = AffineModeMorphAllParameters.normalize_modes(modes),
            updates = updates,
            offsets = AffineModeMorphAllParameters.normalize_offsets(offsets))

    def with_hyperparams(self, hyperparams):
        return AffineModeMorphAllParameters(self, hyperparams)


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

    def as_static_dynamic_parts(self):
        return ((self.N, self.M, self.L),
                (self.update_scale, self.modes))
    
    @staticmethod
    def from_static_dynamic_parts(static, dynamic):
        return AffineModeMorphHyperparams(
            N = static[0], M = static[1], L = static[2],
            update_scale = dynamic[0], modes = dynamic[1])


class AffineModeMorphAllParameters(NamedTuple):
    params: AffineModeMorphParameters
    hyperparams: AffineModeMorphHyperparams

    # hyperparameter passthroughs
    N = property(lambda self: self.hyperparams.N)
    M = property(lambda self: self.hyperparams.M)
    L = property(lambda self: self.hyperparams.L)
    update_scale = property(lambda self: self.hyperparams.update_scale)

    # normalized parameters
    def uniform_scale(self): 
        return AffineModeMorphAllParameters.normalize_scale(self.params.uniform_scale)
    def modes(self):
        if self.hyperparams.modes is None: unnormalized = self.params.modes
        else: unnormalized = self.hyperparams.modes
        return AffineModeMorphAllParameters.normalize_modes(unnormalized)
    def updates(self): return self.params.updates
    def offsets(self):
        return AffineModeMorphAllParameters.normalize_offsets(self.params.offsets)
    
    @staticmethod
    def normalize_scale(scales):
        return scales - scales.mean()
    
    @staticmethod
    def normalize_modes(modes):
        return modes / jnp.linalg.norm(modes, axis = -2, keepdims = True)

    @staticmethod
    def normalize_offsets(offsets):
        return offsets #- offsets.mean(axis = -1, keepdims = True)
    
    HyperparamClass = AffineModeMorphHyperparams
    ParamClass = AffineModeMorphParameters


import platform
__use_explicit_pseudoinverse = ( # avoid M1 chip crashes
    platform.processor() == 'arm' and
    (not jax.default_backend() == 'gpu')
)

def get_transform(
    params: AffineModeMorphAllParameters,
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
    modes = params.modes()
    if __use_explicit_pseudoinverse:
        modes_rows = jnp.swapaxes(modes, -2, -1)
        mp_inv = jla.inv(modes_rows @ modes) @ modes_rows
    else:
        mp_inv = jla.pinv(modes) # (U^T U)^{-1} U^T, shape: (L, M)
    orthog_proj = ( # I - U (U^T U)^{-1} U^T, shape: (M, M)
        jnp.eye(params.M) -
        modes @ mp_inv
    )
    
    # Reconstruction matrix, U + \hat{U}
    reconst = modes[None] + params.updates() # (N, M, L)

    # reconst @ U^+, shape: (N, M, M)
    rot = reconst @ mp_inv
    # rot_scale = jnp.einsum( # old: uses modewise scaling
    #     # i=1...N, j=1...M, k=1...L, l=1...M
    #     "ijk,ik,kl->ijl", # jk,k,kl batched over i
    #     reconst, jnp.exp(params.modal_scale), mp_inv)
    
    return (
        # (N, M, M) + (1, M, M)
        # (rot + orthog_proj[None]) *
        # (N, 1, 1)
        # jnp.exp(params.uniform_scale())[:, None, None]
        jnp.stack([
            jnp.eye(params.M)
            for _ in range(params.N)])
    ), params.offsets()


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
    ret = AffineModeMorphParameters.create(

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
    params: AffineModeMorphAllParameters,
    morph_matrix: Float[Array, "N KD M"] = None,
    morph_ofs: Float[Array, "N KD"] = None
    ):

    # Logpdf of N(0, update_scale * I) evaluated at each
    # (normalized) update vector
    update_sqnorms = (params.updates() ** 2).sum(axis = 1) # (N, L)
    update_logpdf = -(update_sqnorms.sum() / 
        params.update_scale ** 2) / 2
    
    return dict(
        update_norm = update_logpdf,)


def reports(
    params: AffineModeMorphAllParameters
    ) -> dict:
    return dict(
        priors = log_prior(params))


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
    pcs = pca.fit(keypts_centered[reference_subject], sign_correction = None)
    modes = pcs.pcs()[:hyperparams.L, :].T

    return AffineModeMorphParameters.create(
        uniform_scale = scales_log,
        modes = modes,
        updates = jnp.zeros([hyperparams.N, hyperparams.M, hyperparams.L]),
        offsets = offsets
    )

  
AffineModeMorph = MorphModel(
    ParameterClass = AffineModeMorphAllParameters,
    sample_parameters = sample_parameters,
    get_transform = get_transform,
    log_prior = log_prior,
    init = init,
    reports = reports
)


def mode_components(
    params: AffineModeMorphAllParameters,
    poses: Float[Array, "*#K M"]
    ) -> Tuple[Float[Array, "*#K L"], Float[Array, "*#K M L"], Float[Array, "*#K M-L"]]:
    """
    Returns:
        components: 
            Components of poses along the subspace of each morph mode.
        complement:
            Component of poses in the complement of the span of the morph
            modes.
    """
    dim_expand = (None,) * (len(poses.shape) - 1)
    modes = params.modes()[dim_expand]
    coords = (poses[..., None] * modes).sum(axis = 1) # (*#K, L)
    components = coords[..., None, :] * modes # (*#K, M L)
    complement = poses - components.sum(axis = -1)
    return coords, components, complement


def as_offset_only(
    params: AffineModeMorphAllParameters,
    ) -> AffineModeMorphAllParameters:
    params, hyperparams = params.params, params.hyperparams
    return AffineModeMorphAllParameters(
        params = AffineModeMorphParameters(
            uniform_scale = 0 * params.uniform_scale,
            modes = params.modes,
            updates = 0 * params.updates,
            offsets = params.offsets),
        hyperparams = hyperparams)

    
def as_uniform_scale(
    params: AffineModeMorphAllParameters,
    ) -> AffineModeMorphAllParameters:
    params, hyperparams = params.params, params.hyperparams
    return AffineModeMorphAllParameters(
        params = AffineModeMorphParameters(
            uniform_scale = params.uniform_scale,
            modes = params.modes,
            updates = 0 * params.updates,
            offsets = params.offsets),
        hyperparams = hyperparams)


def with_scaled_modes(
    scale_factor: float,
    params: AffineModeMorphAllParameters,
    ) -> AffineModeMorphAllParameters:
    params, hyperparams = params.params, params.hyperparams
    return AffineModeMorphAllParameters(
        params = AffineModeMorphParameters(
            uniform_scale = params.uniform_scale,
            modes = params.modes,
            updates = scale_factor * params.updates,
            offsets = params.offsets),
        hyperparams = hyperparams)


def with_sliced_modes(
    params: AffineModeMorphAllParameters,
    slc: slice
    ) -> AffineModeMorphAllParameters:
    params, hyperparams = params.params, params.hyperparams
    if hyperparams.modes is None:
        param_modes = params.modes[:, slc]
        hyper_modes = None
        L = param_modes.shape[-1]
    else:
        param_modes = params.modes
        hyper_modes = hyperparams.modes[:, slc]
        L = hyper_modes.shape[-1]
    return AffineModeMorphAllParameters(
        params = AffineModeMorphParameters(
            uniform_scale = params.uniform_scale,
            modes = param_modes,
            updates = params.updates[:, :, slc],
            offsets = params.offsets),
        hyperparams = AffineModeMorphHyperparams(
            N = hyperparams.N,
            M = hyperparams.M,
            L = L,
            update_scale = hyperparams.update_scale,
            modes = hyper_modes))

def with_locked_modes(
    params: AffineModeMorphAllParameters,
    ) -> AffineModeMorphAllParameters:
    params, hyperparams = params.params, params.hyperparams
    return AffineModeMorphAllParameters(
        params = AffineModeMorphParameters(
            uniform_scale = params.uniform_scale,
            modes = None,
            updates = params.updates,
            offsets = params.offsets),
        hyperparams = AffineModeMorphHyperparams(
            N = hyperparams.N,
            M = hyperparams.M,
            L = hyperparams.L,
            update_scale = hyperparams.update_scale,
            modes = params.modes))

