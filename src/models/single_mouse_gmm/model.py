from dataclasses import dataclass
from typing import Any
from typing import NamedTuple
from jaxtyping import Array, Float, Float32, Integer, Scalar, jaxtyped
from typeguard import typechecked
import jax.numpy as jnp
import jax.nn as jnn
import jax

from ...util.computations import expand_tril_cholesky, expand_tril

class SingleMouseGMMArch(NamedTuple):
    """
    Architecture specification for a SingleMouseGMM.

    NOTE: Morphs cannot be meaningfully fit for a single mouse, so keypoint
    and latent spaces must be the same dimemsion. Therefore we exclude $K$ and $D$
    parameters here and assume that $K\times D = M$.

    :param N: Number of mixture components.
    :param M: Dimensionality of latent and keypoint space.
    :param T: Number of timepoints.
    """
    N: int
    M: int
    T: int


class SingleMouseGMMLatents(NamedTuple):
    """
    Latent variable values for a SingleMouseGMM.

    :param z: Mixture component identities at each timepoint.
    :param x: Latent pose space locations at each timepoint.
    """
    z: Integer[Array, "T"]
    x: Float[Array, "T M"]


class SingleMouseGMMObservables(NamedTuple):
    """
    Observable variable values for a SingleMouseGMM.

    :param y: Observed keypoint locations at each timepoint.
    """
    y: Float[Array, "T M"]



class SingleMouseGMMParameters(NamedTuple):
    """
    Parameter values for a SingleMouseGMM.
    
    :param pi: Mixture component weights.
    :param m: Mixture component means.
    :param lq: Flat triangular form of Cholesky factors of component covariances.
    """
    pibar: Float[Array, "N"]
    m: Float[Array, "N M"]
    lq: Float[Array, "N M*(M+1)/2"]

    def L(self) -> Float[Array, "N M M"]:
        return expand_tril(self.lq, n = self.m.shape[1])
    
    def Q(self) -> Float32[Array, "N M M"]:
        return expand_tril_cholesky(self.lq, n = self.m.shape[1])

    def pi(self):
        return jnn.softmax(self.pibar)


class SingleMouseGMMHyperparams(NamedTuple):
    """
    Hyperparameter values for a SingleMouseGMM.
    
    :param eps: Variance of noise on observables from latents.
    """
    eps: Scalar


class SingleMouseGMM(NamedTuple):
    """
    Variables specifying a realized SingleMouseGMM model.
    """

    hyperparams: SingleMouseGMMHyperparams
    latents: SingleMouseGMMLatents
    observables: SingleMouseGMMObservables
    params: SingleMouseGMMParameters

