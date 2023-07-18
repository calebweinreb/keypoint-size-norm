from typing import NamedTuple, Tuple, Callable, Any, Protocol
from jaxtyping import Array, Float, Integer, Scalar
import jax.numpy as jnp
import jax

from ...util.computations import unstack


class PoseSpaceParameters(Protocol): pass


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
        


class PoseSpaceModel(NamedTuple):
    sample: Callable[..., Tuple[PoseStates, Integer[Array, "Nt"]]]
    sample_parameters: Callable[..., PoseSpaceParameters]
    logprob_expectations: Callable[..., Float[Array, "Nt L"]]
    discrete_prob: Callable[..., Float[Array, "N L"]]
    aux_distribution: Callable[..., Tuple[Float[Array, "Nt L"], Any]]



def point_weights(
    aux_dist_consts: Float[Array, "Nt L"],
    discrete_probs: Float[Array, "N L"],
    subject_ids: Integer[Array, "Nt"],
    ) -> Float[Array, "Nt L"]:
    """
    Calculate term weights in the objective function.

    This function computes the weights for each data point and
    discrete pose state pairing in the objective function, with
    weights normalized over discrete state space values.

    Args:
        aux_dist_consts: Constants from the model `aux_distribution`
        discrete_probs: Array from the model `discrete_probs`
        subject_ids: Subject assignments in `Nt` axis.

    Returns:
        weights: Weights for each term in the objective function.
    """

    # shape: (Nt, L)
    unnormalized = aux_dist_consts * discrete_probs[subject_ids]
    # shape: (1, L)
    normalizers = unnormalized.sum(axis = 1, keepdims = True)

    return unnormalized / normalizers


def common_logprob_expectations(
    observations: Observations,
    query_discrete_probs: Float[Array, "N L"],
    ) -> Float[Array, "Nt L"]: 
    """
    Compute objective function terms shared across pose space models.
    """

    # log Z_R constant in parameters, so ignored
    # R_norm = ...
    
    z_term = jnp.log(query_discrete_probs)[observations.subject_ids]

    return z_term



def objective(
    posespace_model: PoseSpaceModel,
    observations: Observations,
    query_morph_matrix: Float[Array, "N KD M"],
    est_morph_matrix: Float[Array, "N KD M"],
    query_params: PoseSpaceParameters,
    estimated_params: PoseSpaceParameters,
    hyperparams: PoseSpaceHyperparams,
    ) -> Scalar:
    """
    Objective function for M-step to maximize.

    NOTE: only invertable morphs are supported at the moment.
    """
    
    # ----- Compute terms of the objective function

    consts, aux_pdf = posespace_model.aux_distribution(
        observations, est_morph_matrix, estimated_params, hyperparams
    )
    est_discrete_probs = posespace_model.discrete_prob(
        estimated_params, hyperparams)
    term_weights: Float[Array, "Nt L"] = point_weights(
        consts, est_discrete_probs, observations.subject_ids)
    model_log_exp: Float[Array, "Nt L"] = posespace_model.logprob_expectations(
        observations, query_morph_matrix, query_params, hyperparams, aux_pdf
    )
    common_logprob_expect: Float[Array, "Nt L"] = common_logprob_expectations(
        observations,
        posespace_model.discrete_prob(query_params, hyperparams)
    )

    # ----- Sum terms and return
    return (
        term_weights * (model_log_exp + common_logprob_expect)
    ).sum()
