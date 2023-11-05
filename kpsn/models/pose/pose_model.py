from typing import NamedTuple, Tuple, Callable, Any, Protocol
from jaxtyping import Array, Float, Integer, Scalar
import jax.numpy as jnp
import jax

from ...util.computations import unstack


class PoseSpaceTrainedParams(Protocol):
    def with_hyperparams(self, hyperparams):
        return PoseSpaceParameters(self, hyperparams)


class PoseSpaceHyperparams(Protocol):
    """
    Hyperparameters for a pose space model.
    
    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of mixture components.
    :param eps: Observation error variance.
    """
    N: int
    M: int
    L: int
    eps: int


class PoseSpaceParameters(Protocol):
    params: PoseSpaceTrainedParams
    hyperparams: PoseSpaceHyperparams


class PoseStates(Protocol):
    """
    Latent variables arising from a pose space model.

    :param poses: Pose space states resulting from the GMM.
    """
    poses: Float[Array, "Nt M"]



class Observations(NamedTuple):
    """
    Observations from a combined pose space and morph model.

    :param keypts: Array of observed keypoint data.
        First dimension is the result of concatenating sessions,
        and second is the flattened dimenions (K, D) over keypoints
        and spatial dimensions.
    :param subject_slices: Slices grabbing subjects from keypts array
    """
    keypts: Float[Array, "Nt KD"]
    subject_ids: Integer[Array, "Nt"]

    def unstack(
        self,
        arr,
        N = None,
        axis = 0
        ) -> Tuple:
        """
        Convert subject by time axis to list of time axes.

        Args:
            arr: Array-like (Na..., Nt, Nb...) Array to split by subject.
            N: Number of subjects.
                If not given, then will be inferred from `subject_ids`.
            axis: Axis along which to split `arr`

        Returns:
            arrs: Arrays (Na..., T, Nb...) for each subject.
        """
        return unstack(arr, self.subject_ids, N = N, axis = axis)
    

class EMAuxPDF(NamedTuple):
    """
    Constants and supporting values for calculating the auxiliary distribution
    in EM under a set of estimated parameters.

    :param consts: Array-like, shape (Nt, L)
        Probabilities of data points given the discrete pose state varaible and
        estimated parameters.
    """
    consts: Float[Array, "Nt L"]


class PoseSpaceModel(NamedTuple):
    ParameterClass: type
    discrete_mle: Callable[..., PoseStates]
    sample_poses: Callable[..., Tuple[PoseStates, Integer[Array, "Nt"]]]
    sample_hyperparams: Callable[..., PoseSpaceHyperparams]
    sample_parameters: Callable[..., PoseSpaceTrainedParams]
    pose_logprob: Callable[..., Float[Array, "Nt L"]]
    aux_distribution: Callable[..., Float[Array, "Nt L"]]
    log_prior: Callable[
        [PoseSpaceParameters], 
        Scalar]
    init_hyperparams: Callable[..., PoseSpaceHyperparams]
    init: Callable[..., PoseSpaceTrainedParams]
    reports: Callable[..., dict]





# def common_logprob_expectations(
#     observations: Observations,
#     query_discrete_probs: Float[Array, "N L"],
#     ) -> Float[Array, "Nt L"]: 
#     """
#     Compute objective function terms shared across pose space models.
#     """
    
#     return jnp.log(query_discrete_probs)[observations.subject_ids]



# def objective(
#     posespace_model: PoseSpaceModel,
#     params: PoseSpaceParameters,
#     poses: Float[Array, "Nt L"],
#     aux_pdf: Float[Array, "Nt L"],
#     ) -> Scalar:
#     """
#     Objective function for M-step to maximize.

#     NOTE: only invertable morphs are supported at the moment.
#     """
    
#     # ----- Compute terms of the objective function

#     : Float[Array, "Nt L"] = posespace_model.pose_logprob(
#         params, poses)
#     common_logprob_expect: Float[Array, "Nt L"] = common_logprob_expectations(
#         observations,
#         posespace_model.discrete_prob(params)
#     )

#     # ----- Sum terms and return
#     return (
#         term_weights * (model_log_exp + common_logprob_expect)
#     ).sum() 
