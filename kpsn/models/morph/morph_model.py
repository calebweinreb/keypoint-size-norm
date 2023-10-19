from typing import NamedTuple, Protocol, Callable, Tuple, Optional
from jaxtyping import Array, Float, Scalar, Integer
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
    reports: Callable[..., dict]

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
    
    def pose_mle_from_array(
        self,
        hyperparams: MorphHyperparams,
        params: MorphParameters,
        keypoints: Float[Array, "*#K KD"],
        subject_ids: Integer[Array, "*#K"]
        ) -> Float[Array, "*#K M"]:
        keypoints = jnp.array(keypoints)
        subject_ids = jnp.array(subject_ids)
        return self.pose_mle(Observations(
            keypoints.reshape(-1, keypoints.shape[-1]),
            subject_ids.reshape(-1,)
        ), params, hyperparams).reshape(subject_ids.shape + (hyperparams.M,))

    def transform(
        self,
        hyperparams: MorphHyperparams,
        params: MorphParameters,
        poses: Float[Array, "*#K M"],
        subject_ids: Integer[Array, "*#K"]
        ) -> Float[Array, "*#K KD"]:
        C, d = self.get_transform(params, hyperparams)
        return (C[subject_ids] @ poses[..., None])[..., 0] + d[subject_ids]
    