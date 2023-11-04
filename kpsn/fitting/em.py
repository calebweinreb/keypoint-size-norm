from typing import NamedTuple, Tuple, Protocol
from jaxtyping import Scalar, Float, Array, Integer
from functools import partial
import jax.tree_util as pt
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import optax
import tqdm
import jax

from ..models import pose
from ..models import joint_model
from ..util import logging
from ..util import computations


def _pytree_sum(tree):
    return pt.tree_reduce(
        lambda x, y: x.sum() + y.sum(),
        tree)


def _check_should_stop_early(loss_hist, tol):
    """
    Check for average decrease in loss that is worse than `tol`.

    Args:
        loss_hist: np.ndarray, shape: (N,)
            Last `N` observations of loss.
        tol: Scalar
            Required average improvement to continue.
    """
    return np.diff(loss_hist).mean() > -tol


def _point_weights(
    aux_dist_log_consts: Float[Array, "Nt L"],
    discrete_logits: Float[Array, "N L"],
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
    unnormalized = aux_dist_log_consts + discrete_logits[subject_ids]

    return jnn.softmax(unnormalized, axis = 1)


def _estep(
    model: joint_model.JointModel,
    observations: pose.Observations,
    estimated_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams,
    ) -> pose.PoseSpaceParameters:

    estimated_params = estimated_params.with_hyperparams(hyperparams)

    est_morph_matrix, est_morph_ofs = model.morph.get_transform(
        estimated_params.morph)
    aux_pdf = model.posespace.aux_distribution(
        observations, est_morph_matrix, est_morph_ofs,
        estimated_params.posespace
    )
    
    est_discrete_logits = model.posespace.discrete_logits(
        estimated_params.posespace)
    term_weights: Float[Array, "Nt L"] = _point_weights(
        aux_pdf.consts,
        est_discrete_logits,
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

    params = query_params.with_hyperparams(hyperparams)

    morph_matrix, morph_ofs = model.morph.get_transform(
        params.morph)
    
    morph_prior = _pytree_sum(model.morph.log_prior(
        params.morph))
    posespace_prior = _pytree_sum(model.posespace.log_prior(
        params.posespace))
    
    # return morph_prior
    return pose.objective(
        model.posespace,
        observations,
        morph_matrix,
        morph_ofs,
        params.posespace,
        aux_pdf,
        term_weights
    ) + morph_prior + posespace_prior


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


def construct_jitted_mstep(
    model: joint_model.JointModel,
    optimizer: optax.GradientTransformation):

    loss_func = _mstep_loss(model)

    @partial(jax.jit, static_argnums = (3,))
    def step(opt_state, params, emissions,
             hyperparams_static, hyperparams_dynamic,
             aux_pdf, term_weights):
        hyperparams = joint_model.JointHyperparams.from_static_dynamic_parts(
            model, hyperparams_static, hyperparams_dynamic)
        loss_value, grads = jax.value_and_grad(loss_func, argnums = 0)(
            params, emissions, hyperparams, aux_pdf, term_weights)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value

    return step


def construct_jitted_estep(
    model: joint_model.JointModel
    ):
    @partial(jax.jit, static_argnums = (2,))
    def step(observations, estimated_params,
             hyperparams_static, hyperparams_dynamic):
        hyperparams = joint_model.JointHyperparams.from_static_dynamic_parts(
            model, hyperparams_static, hyperparams_dynamic)
        return _estep(model, observations, estimated_params, hyperparams)
    return step
    


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
    tol: float = None,
    stop_window: int = 10,
    jitted_step  = None,
    ) -> Tuple[Float[Array, "n_steps"], joint_model.JointParameters]:
    """
    Args:
        tol: float > 0
            Stop iterations if average improvement over `stop_window` is
            not better than `tol`.
        stop_window: int
            Number of iterations over which to assess early stopping. 
    """

    # ----- Define the step function with weight update

    step = jitted_step
    if step is None:
        step = construct_jitted_mstep(model, optimizer)

    # ----- Set up variables for iteration

    curr_params = init_params
    opt_state = optimizer.init(init_params)
    loss_hist = np.empty([n_steps])
    iter = range(n_steps) if not progress else tqdm.trange(n_steps)
    hyper_stat, hyper_dyna = hyperparams.as_static_dynamic_parts()
    
    # ---- Run M-step iterations

    for step_i in iter:

        curr_params, opt_state, loss_value = step(
            opt_state, curr_params,
            emissions, hyper_stat, hyper_dyna,
            aux_pdf, term_weights)
        loss_hist[step_i] = loss_value
        
        if (log_every > 0) and (not step_i % log_every):
            print(f"Step {step_i} : loss = {loss_value}")

        # evaluate early stopping
        if (tol is not None and step_i > stop_window and 
            _check_should_stop_early(
                loss_hist[step_i-stop_window:step_i+1], tol)
            ):
            loss_hist = loss_hist[:step_i+1]
            break

    return loss_hist, curr_params


def iterate_em(
    model: joint_model.JointModel,
    init_params: joint_model.JointParameters,
    emissions: pose.Observations,
    hyperparams: joint_model.JointHyperparams,
    n_steps: int,
    log_every: int,
    progress = False,
    tol: float = None,
    stop_window: int = 5,
    batch_size: int = None,
    batch_seed: int = 23,
    mstep_learning_rate: float = 5e-3,
    mstep_n_steps: int = 2000,
    mstep_log_every: int = -1,
    mstep_progress = False,
    mstep_tol: float = None,
    mstep_stop_window: int = 10,
    return_mstep_losses = False,
    return_param_hist = False,
    return_reports = False,
    ) -> Tuple[
        Float[Array, "n_steps"], joint_model.JointParameters,
        Float[Array, "n_steps mstep_n_steps"],
        Tuple[joint_model.JointParameters]]:
    """
    Perform EM on a JointModel. 
    """

    curr_params = init_params
    loss_hist = np.empty([n_steps])
    iter = range(n_steps) if not progress else tqdm.trange(n_steps)
    report_trace = logging.ReportTrace(n_steps)
    if return_mstep_losses:
        mstep_losses = np.full([n_steps, mstep_n_steps], np.nan)
    if return_param_hist:
        param_hist = [curr_params.with_hyperparams(hyperparams)]

    optimizer = optax.adam(learning_rate = mstep_learning_rate)
    jitted_mstep = construct_jitted_mstep(model, optimizer)
    jitted_estep = construct_jitted_estep(model)
    hyper_stat, hyper_dyna = hyperparams.as_static_dynamic_parts()

    if batch_size is not None:
        batch_rkey_seed = jr.PRNGKey(batch_seed)
        unstacked_ixs = computations.unstacked_ixs(
            emissions.subject_ids,
            N = hyperparams.posespace.N)

    for step_i in iter:
        
        if batch_size is not None:
            batch_rkey_seed, batch_rkey = jr.split(batch_rkey_seed, 2)
            batch = computations.stacked_batch(
                batch_rkey, unstacked_ixs, batch_size)
            step_obs = pose.Observations(
                keypts = emissions.keypts[batch],
                subject_ids = emissions.subject_ids[batch])
        else:
            step_obs = emissions

        aux_pdf, term_weights = jitted_estep(
            observations = step_obs,
            estimated_params = curr_params,
            hyperparams_static = hyper_stat,
            hyperparams_dynamic = hyper_dyna)
        loss_hist_mstep, fit_params_mstep = _mstep(
            model = model,
            init_params = curr_params,
            aux_pdf = aux_pdf,
            term_weights = term_weights,
            emissions = step_obs,
            hyperparams = hyperparams,
            optimizer = optimizer,
            n_steps = mstep_n_steps,
            log_every = mstep_log_every,
            progress = mstep_progress,
            tol = mstep_tol,
            stop_window = mstep_stop_window,
            jitted_step = jitted_mstep
        )
        loss_hist[step_i] = loss_hist_mstep[-1]
        curr_params = fit_params_mstep
        if return_mstep_losses:
            mstep_losses[step_i, :len(loss_hist_mstep)] = loss_hist_mstep
        if return_param_hist:
            param_hist.append(curr_params.with_hyperparams(hyperparams))

        if return_reports:
            report_trace.record(dict(
                morph = model.morph.reports(
                    curr_params.morph.with_hyperparams(
                        hyperparams.morph)),
                posespace = model.posespace.reports(
                    curr_params.posespace.with_hyperparams(
                        hyperparams.posespace)),
                total_logprob = _mstep_objective(
                    model, step_obs, curr_params,
                    hyperparams, aux_pdf, term_weights)),
                step_i)

        
        if (log_every > 0) and (not step_i % log_every):
            print(f"Step {step_i} : loss = {loss_hist[step_i]}")

        # evaluate early stopping and divergence
        if (tol is not None and step_i > stop_window and 
            _check_should_stop_early(
                loss_hist[step_i-stop_window:step_i+1], tol)
            ) or (not np.isfinite(loss_hist[step_i])):
            loss_hist = loss_hist[:step_i+1]
            print("Stopping due to early convergence or divergence.")
            break

        
    ret = loss_hist, curr_params.with_hyperparams(hyperparams)
    if return_mstep_losses: ret = ret + (mstep_losses,)
    if return_param_hist: ret = ret + (param_hist,)
    if return_reports: ret = ret + (report_trace,)
    return ret 