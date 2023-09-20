from typing import NamedTuple, Tuple, Protocol
from jaxtyping import Scalar, Float, Array, Integer
from functools import partial
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import jax

from ..models import pose
from ..models import joint_model

def _check_should_stop_early(loss_hist, tol):
    """
    Check for average decrease in loss that is worse than `tol`.

    Args:
        loss_hist: jax.ndarray, shape: (N,)
            Last `N` observations of loss. NaNs indicate an incomplete history
            that should always continue.
        tol: Scalar
            Required average improvement to continue.
    """
    mean = jnp.diff(loss_hist).mean()
    return np.isfinite(mean) and mean < -tol
    

def _point_weights(
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


def _estep(
    model: joint_model.JointModel,
    observations: pose.Observations,
    estimated_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams
    ) -> pose.PoseSpaceParameters:

    est_morph_matrix = model.morph.get_transform(
        estimated_params.morph,
        hyperparams.morph)
    aux_pdf = model.posespace.aux_distribution(
        observations, est_morph_matrix,
        estimated_params.posespace, hyperparams.posespace
    )
    est_discrete_probs = model.posespace.discrete_prob(
        estimated_params.posespace, hyperparams.posespace)
    term_weights: Float[Array, "Nt L"] = _point_weights(
        aux_pdf.consts,
        est_discrete_probs,
        observations.subject_ids)
    
    return aux_pdf, term_weights


def _mstep_objective(
    model: joint_model.JointModel,
    observations: pose.Observations,    
    query_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams,    
    aux_pdf: pose.EMAuxPDF,
    term_weights: Float[Array, "Nt L"],
    ) -> Scalar:
    """
    Calculate objective for M-step to maximize.
    """

    morph_matrix = model.morph.get_transform(
        query_params.morph,
        hyperparams.morph)

    return pose.objective(
        model.posespace,
        observations,
        morph_matrix,
        query_params.posespace,
        hyperparams.posespace,
        aux_pdf,
        term_weights
    )


def _mstep_loss(
    model: joint_model.JointModel,
    ) -> Scalar:
    def step_loss_(
        params: optax.Params, 
        emissions: pose.Observations,
        hyperparams: joint_model.JointParameters,
        aux_pdf: pose.EMAuxPDF,
        term_weights: Float[Array, "Nt L"], 
        ):

        return -_mstep_objective(
            model, emissions, params, hyperparams, aux_pdf, term_weights)
    return step_loss_


def _mstep(
    model: joint_model.JointModel,
    init_params: optax.Params,
    aux_pdf: pose.EMAuxPDF,
    term_weights: Float[Array, "Nt L"], 
    emissions: pose.Observations,
    hyperparams: joint_model.JointParameters,
    optimizer: optax.GradientTransformation,
    n_steps: int,
    log_every: int,
    progress = False,
    ) -> Tuple[Float[Array, "n_steps"], joint_model.JointParameters]:

    # ----- Define the step function with weight update

    loss_func = _mstep_loss(model)

    @partial(jax.jit, static_argnums = (3,))
    def step(opt_state, params, emissions, hyperparams, aux_pdf, term_weights):
        loss_value, grads = jax.value_and_grad(loss_func, argnums = 0)(
            params, emissions, hyperparams, aux_pdf, term_weights)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    # ---- Run M-step iterations

    curr_params = init_params
    opt_state = optimizer.init(init_params)
    loss_hist = np.empty([n_steps])
    iter = range(n_steps) if not progress else tqdm.trange(n_steps)
    for step_i in iter:
        curr_params, opt_state, loss_value = step(
            opt_state, curr_params,
            emissions, hyperparams, aux_pdf, term_weights)
        loss_hist[step_i] = loss_value
        
        if (log_every > 0) and (not step_i % log_every):
            print(f"Step {step_i} : loss = {loss_value}")

    return loss_hist, curr_params