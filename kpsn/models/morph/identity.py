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


class IDMTrainedParameters(NamedTuple):
    """
    Parameter set describing an affine mode morph.

    :param uniform_scale: Logarithm of factor across all dimensions.
    :param modal_scale: Logarithm of factor across each morph dimension.
    :param modes: Morph dimensions across the full population.
    :param updates: Updates to the morph modes for each subject.
    """
 
    def with_hyperparams(self, hyperparams):
        return IDMParameters(self, hyperparams)


class IDMHyperparams(NamedTuple):
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

    def as_static_dynamic_parts(self):
        return ((self.N, self.M, self.L),
                ())
    
    @staticmethod
    def from_static_dynamic_parts(static, dynamic):
        return IDMHyperparams(
            N = static[0], M = static[1], L = static[2])


class IDMParameters(NamedTuple):
    
    trained_params: IDMTrainedParameters
    hyperparams: IDMHyperparams
    
    HyperparamClass = IDMHyperparams
    TrainedParamClass = IDMTrainedParameters


    # parameter passthroughs
    # ----------------------
    N = property(lambda self: self.hyperparams.N)
    M = property(lambda self: self.hyperparams.M)
    L = property(lambda self: self.hyperparams.L)

    # no normalized parameters



def transform(
    params: IDMParameters,
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

    return poses


def inverse_transform(
    params: IDMParameters,
    keypts: Float[Array, "*#K M"],
    sess_ids: Integer[Array, "*#K"]
    ) -> Float[Array, "*#K M"]:

    return keypts
    


def sample_hyperparams(
    rkey: PRNGKey,
    N: int,
    M: int,
    L: int,
    ) -> IDMHyperparams:
    r"""
    Sample a set of `IDMHyperparams`
    
    The parameters are sampled according to the generative model: $$
    \begin{align}
    \mathrm{modes} &\sim
        \mathrm{SO}(M)_{:, :L} //
    \mathrm{offset} &\sim
        \mathcal{N}(0, \mathrm{offset_std}^2) //
    \end{align}
    """
    return IDMHyperparams(
        N = N, M = M, L = L
    )


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: IDMHyperparams,
    update_std: Scalar,
    offset_std: Scalar,
    ) -> IDMTrainedParameters:
    r"""
    Sample a set of `IDMTrainedParameters`.
    
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
    return IDMTrainedParameters()


def log_prior(params: IDMParameters) -> dict:

    return dict()


def reports(
    params: IDMParameters
    ) -> dict:
    return dict()


def init_hyperparams(
    observations: Observations,
    N: int, M: int, L: int,
    reference_subject: int,
    seed: int = 0
    ):
    
    return IDMHyperparams(
        N = N, M = M, L = L
    )


def init(
    hyperparams: IDMHyperparams,
    observations: Observations,
    reference_subject: int,
    seed: int = 0
    ) -> IDMTrainedParameters:

    return IDMTrainedParameters()

  
IdentityMorph = MorphModel(
    ParameterClass = IDMParameters,
    sample_hyperparams = sample_hyperparams,
    sample_parameters = sample_parameters,
    transform = transform,
    inverse_transform = inverse_transform,
    log_prior = log_prior,
    init_hyperparams = init_hyperparams,
    init = init,
    reports = reports
)

