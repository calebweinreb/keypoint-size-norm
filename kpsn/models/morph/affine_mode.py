from typing import NamedTuple, Union, Tuple
from jaxtyping import Array, Float, Scalar, PRNGKeyArray as PRNGKey
import tensorflow_probability.substrates.jax as tfp
import numpy as np
import jax.random as jr
import jax.numpy as jnp
import jax.numpy.linalg as jla
import jax
import scipy.stats

from .morph_model import *
from ..pose import Observations
from ...util import pca, computations
from ...util.computations import (
    broadcast_batch, stacked_take)


class AFMTrainedParameters(NamedTuple):
    """
    Parameter set describing an affine mode morph.

    :param uniform_scale: Logarithm of factor across all dimensions.
    :param modal_scale: Logarithm of factor across each morph dimension.
    :param modes: Morph dimensions across the full population.
    :param updates: Updates to the morph modes for each subject.
    """
    mode_updates: Float[Array, "N M L"]
    offset_updates: Float[Array, "N M"]
    
    @staticmethod
    def create(
        hyperparams: 'AFMHyperparams',
        mode_updates: Float[Array, "N M L"],
        offset_updates: Float[Array, "N M"]) -> 'AFMTrainedParameters':
        return AFMTrainedParameters(
            mode_updates = AFMParameters.normalize_mode_updates(
                mode_updates, hyperparams.identity_sess),
            offset_updates = AFMParameters.normalize_mode_updates(
                offset_updates, hyperparams.identity_sess))

    def with_hyperparams(self, hyperparams):
        return AFMParameters(self, hyperparams)


class AFMHyperparams(NamedTuple):
    """
    Hyperparameters describing an affine mode morph.

    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of morph dimensions.
    :param upd_var_modes: Standard deviation of the spherical normal prior on
        mode updates.
    :param modes: Morph dimensions across the full population (optional).
    
    """
    N: int
    M: int
    L: int
    upd_var_modes: Scalar
    upd_var_ofs: Scalar
    modes: Float[Array, "M L"]
    offset: Float[Array, "M"]
    identity_sess: int

    def as_static_dynamic_parts(self):
        return ((self.N, self.M, self.L),
                (self.upd_var_ofs, self.upd_var_modes,
                 self.modes, self.offset,
                 self.identity_sess))
    
    @staticmethod
    def from_static_dynamic_parts(static, dynamic):
        return AFMHyperparams(
            N = static[0], M = static[1], L = static[2],
            upd_var_ofs = dynamic[0], upd_var_modes = dynamic[1],
            modes = dynamic[2], offset = dynamic[3],
            identity_sess = dynamic[4])


class AFMParameters(NamedTuple):
    
    trained_params: AFMTrainedParameters
    hyperparams: AFMHyperparams
    
    HyperparamClass = AFMHyperparams
    TrainedParamClass = AFMTrainedParameters


    # parameter passthroughs
    # ----------------------
    N = property(lambda self: self.hyperparams.N)
    M = property(lambda self: self.hyperparams.M)
    L = property(lambda self: self.hyperparams.L)
    upd_var_ofs = property(lambda self: self.hyperparams.upd_var_ofs)
    upd_var_modes = property(lambda self: self.hyperparams.upd_var_modes)
    modes = property(lambda self: self.hyperparams.modes)
    offset = property(lambda self: self.hyperparams.offset)

    # normalized parameters
    # ---------------------
    @property
    def offset_updates(self):
        return AFMParameters.normalize_offset_updates(
            self.trained_params.offset_updates,
            self.hyperparams.identity_sess)
    
    @staticmethod
    def normalize_offset_updates(offset_updates, identity_sess):
        if identity_sess is None:
            return offset_updates
        return offset_updates.at[identity_sess].set(0)
    
    @property
    def mode_updates(self):
        return AFMParameters.normalize_mode_updates(
            self.trained_params.mode_updates,
            self.hyperparams.identity_sess)
    
    @staticmethod
    def normalize_mode_updates(mode_updates, identity_sess):
        if identity_sess is None:
            return mode_updates
        return mode_updates.at[identity_sess].set(0)
    


def get_transform(
    params: AFMParameters
    ) -> Tuple[Float[Array, "N M M"],
               Float[Array, "M"],
               Float[Array, "N M"]]:

    mode_pinv = params.modes.T[None] # (1, L, M)
    I = jnp.eye(mode_pinv.shape[-1])
    linear_parts = I + params.mode_updates @ mode_pinv

    pop_offset = params.offset
    sess_offsets = params.offset[None] + params.offset_updates
    
    return linear_parts, pop_offset, sess_offsets


def transform(
    params: AFMParameters,
    poses: Float[Array, "*#K M"],
    sess_ids = Integer[Array, "*#K"]
    ) -> Float[Array, "*#K M"]:
    """
    Calculate the linear transform defining the morph model
    using a given set of parameters.

    Parameters
    ----------
    sess_ids: jax array, integer
        Session indices ($N$, in parameter matrices) to apply to each
        batch element.

    Returns
    -------
    morphed_poses:
        Array of poses under the morph transform.
    """ 
    
    sess_ids = jnp.array(sess_ids)
    
    linear_parts, pop_offset, sess_offsets = get_transform(params)

    # broadcast transform arrays
    batch_shape = sess_ids.shape
    linear_parts = linear_parts[sess_ids] # (batch, M, M)
    pop_offset = broadcast_batch(pop_offset, batch_shape) # (batch, M)
    sess_offsets = sess_offsets[sess_ids] # (batch, M)

    # apply transform
    centered = (poses - pop_offset)[..., None] # (batch, M, 1)
    updated = (linear_parts @ centered)[..., 0] # (batch, M)
    uncentered = updated + sess_offsets # (batch, M)

    return uncentered


def inverse_transform(
    params: AFMParameters,
    keypts: Float[Array, "*#K M"],
    sess_ids: Integer[Array, "*#K"],
    return_determinants: bool = False
    ) -> Tuple[Float[Array, "*#K M"], Float[Array, "*#K"]]:

    sess_ids = jnp.array(sess_ids)

    linear_parts, pop_offset, sess_offsets = get_transform(params)
    linear_invs = jla.inv(linear_parts)

    # broadcast transform arrays
    batch_shape = sess_ids.shape
    linear_inv = linear_invs[sess_ids] # (batch, M, M)
    pop_offset = broadcast_batch(pop_offset, batch_shape) # (batch, M)
    sess_offsets = sess_offsets[sess_ids] # (batch, M)

    # apply transform
    centered = (keypts - sess_offsets)[..., None] # (batch, M, 1)
    updated = (linear_inv @ centered)[..., 0] # (batch, M)
    uncentered = updated + pop_offset # (batch, M)
    
    if return_determinants:
        linear_invs_logdet = jnp.log(jla.det(linear_invs))
        # jax.debug.print("linvsh {}", linear_invs_det)
        linear_inv_logdet = linear_invs_logdet[sess_ids]
        return uncentered, linear_inv_logdet
    else: return uncentered
    


def sample_hyperparams(
    rkey: PRNGKey,
    N: int,
    M: int,
    L: int,
    upd_var_modes: float,
    upd_var_ofs: float,
    offset_std: float,
    identity_sess: int
    ) -> AFMHyperparams:
    r"""
    Sample a set of `AFMHyperparams`
    
    The parameters are sampled according to the generative model: $$
    \begin{align}
    \mathrm{modes} &\sim
        \mathrm{SO}(M)_{:, :L} //
    \mathrm{offset} &\sim
        \mathcal{N}(0, \mathrm{offset_std}^2) //
    \end{align}
    """
    rkey = jr.split(rkey, 2)
    return AFMHyperparams(
        N = N, M = M, L = L,
        identity_sess = identity_sess,
        upd_var_modes = upd_var_modes, upd_var_ofs = upd_var_ofs,
        modes = jnp.array(scipy.stats.special_ortho_group.rvs(
                M, random_state = np.array(rkey)[0, 0]
            )[:, :L]),
        offset = jnp.random.randn(M) * offset_std,
    )


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: AFMHyperparams,
    update_std: Scalar,
    offset_std: Scalar,
    ) -> AFMTrainedParameters:
    r"""
    Sample a set of `AFMTrainedParameters`.
    
    The parameters are sampled according to the generative model: $$
    \begin{align}
    \mathrm{offsets} \sim
        \mathcal{N}(0, \mathrm{update_std}^2 I_{NM})
    \mathrm{updates} \sim
        \mathcal{N}(0, \mathrm{update_std}^2 I_{NML})
    \end{align} $$

    :param rkey: JAX random key.
    :param hyperparams: Hyperparameters of the resulting morph model.
    :param mode_std: Standard deviation of independent normal noise atop random
        orthogonal vectors defining morph dimensions.
    :param update_std: Standard deviation of independent normal mode update
        vectors.
    """
    rkey = jr.split(rkey, 6)
    ret = AFMTrainedParameters.create(
        hyperparams = hyperparams,
        mode_updates = update_std * jr.normal(rkey[4],
            shape = (hyperparams.N, hyperparams.M, hyperparams.L)),
        offset_updates = offset_std * jr.normal(rkey[5],
            shape = (hyperparams.N, hyperparams.M))
    
    )
    return ret


def log_prior(params: AFMParameters) -> dict:

    offset_logp = tfp.distributions.MultivariateNormalDiag(
        scale_diag = params.upd_var_ofs * jnp.ones(params.M)
    ).log_prob(params.offset_updates)

    flat_updates = params.mode_updates.reshape([params.N, params.M * params.L])
    mode_logp = tfp.distributions.MultivariateNormalDiag(
        scale_diag = params.upd_var_modes * jnp.ones(params.M * params.L)
    ).log_prob(flat_updates)

    return dict(
        offset = offset_logp,
        mode = mode_logp,)


def reports(
    params: AFMParameters
    ) -> dict:
    return dict(
        priors = log_prior(params))


def init_hyperparams(
    observations: Observations,
    N: int, M: int, L: int,
    upd_var_modes: float,
    upd_var_ofs: float,
    identity_sess: int,
    reference_subject: int,
    seed: int = 0
    ):
    
    ref_keypts = stacked_take(
        observations.keypts, observations.subject_ids, reference_subject)
    pcs = pca.fit_with_center(ref_keypts)

    return AFMHyperparams(
        N = N, M = M, L = L,
        upd_var_modes = upd_var_modes,
        upd_var_ofs = upd_var_ofs,
        identity_sess = identity_sess,
        modes = pcs._pcadata.pcs()[:L, :].T,
        offset = pcs._center
    )


def init(
    hyperparams: AFMHyperparams,
    observations: Observations,
    reference_subject: int,
    seed: int = 0,
    init_offsets = True
    ) -> AFMTrainedParameters:
    
    # Calculate offsets
    if init_offsets:
        subjwise_keypts = observations.unstack(observations.keypts)
        offset_updates = jnp.stack([
            subj_kpts.mean(axis = 0) - hyperparams.offset
            for subj_kpts in subjwise_keypts])

    return AFMTrainedParameters.create(
        hyperparams = hyperparams,
        offset_updates = (offset_updates if init_offsets
                          else jnp.zeros([hyperparams.N, hyperparams.M])),
        mode_updates = jnp.zeros([hyperparams.N, hyperparams.M, hyperparams.L]),
    )

  
AffineModeMorph = MorphModel(
    ParameterClass = AFMParameters,
    sample_hyperparams = sample_hyperparams,
    sample_parameters = sample_parameters,
    transform = transform,
    inverse_transform = inverse_transform,
    log_prior = log_prior,
    init_hyperparams = init_hyperparams,
    init = init,
    reports = reports
)


def mode_components(
    params: AFMParameters,
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
    modes = params.modes[dim_expand]
    coords = (poses[..., None] * modes).sum(axis = 1) # (*#K, L)
    components = coords[..., None, :] * modes # (*#K, M L)
    complement = poses - components.sum(axis = -1)
    return coords, components, complement


def as_offset_only(
    params: AFMParameters,
    ) -> AFMParameters:
    params, hyperparams = params.trained_params, params.hyperparams
    return AFMParameters(
        trained_params = AFMTrainedParameters(
            mode_updates = 0 * params.mode_updates,
            offset_updates = params.offset_updates),
        hyperparams = hyperparams)


def with_scaled_modes(
    scale_factor: float,
    params: AFMParameters,
    ) -> AFMParameters:
    params, hyperparams = params.params, params.hyperparams
    return AFMParameters(
        trained_params = AFMTrainedParameters(
            uniform_scale = params.uniform_scale,
            modes = params.modes,
            updates = scale_factor * params.updates,
            offsets = params.offsets),
        hyperparams = hyperparams)


def with_sliced_modes(
    params: AFMParameters,
    slc: slice
    ) -> AFMParameters:
    params, hyperparams = params.trained_params, params.hyperparams
    hyper_modes = hyperparams.modes[:, slc]
    L = hyper_modes.shape[-1]
    return AFMParameters(
        trained_params = AFMTrainedParameters(
            mode_updates = params.mode_updates[:, :, slc],
            offset_updates = params.offset_updates),
        hyperparams = AFMHyperparams(
            N = hyperparams.N,
            M = hyperparams.M,
            L = L,
            upd_var_ofs = hyperparams.upd_var_ofs,
            upd_var_modes = hyperparams.upd_var_modes,
            modes = hyper_modes,
            offset = hyperparams.offset,
            identity_sess = hyperparams.identity_sess))


def with_locked_modes(
    params: AFMParameters,
    ) -> AFMParameters:
    params, hyperparams = params.params, params.hyperparams
    return AFMParameters(
        params = AFMTrainedParameters(
            uniform_scale = params.uniform_scale,
            modes = None,
            updates = params.updates,
            offsets = params.offsets),
        hyperparams = AFMHyperparams(
            N = hyperparams.N,
            M = hyperparams.M,
            L = hyperparams.L,
            upd_var_modes = hyperparams.upd_var_modes,
            upd_var_ofs = hyperparams.upd_var_ofs,
            modes = params.modes))

