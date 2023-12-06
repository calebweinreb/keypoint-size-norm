from typing import NamedTuple, Tuple, Protocol
from jaxtyping import Scalar, Float, Array, Integer
from functools import partial
from fnmatch import fnmatch
import jax.tree_util as pt
import jax.random as jr
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import functools
import optax
import tqdm
import jax

from ..models import pose
from ..models import joint_model
from ..util import logging
from ..util import computations

from kpsn_test.routines.util import load_routine


def _pytree_sum(tree):
    return pt.tree_reduce(
        lambda x, y: x.sum() + y.sum(),
        tree)


def _check_should_stop_early(loss_hist, tol):
    """
    Check for median decrease in loss that is worse than `tol`.

    Args:
        loss_hist: np.ndarray, shape: (N,)
            Last `N` observations of loss.
        tol: Scalar
            Required average improvement to continue.
    """
    return np.median(np.diff(loss_hist)) > -tol


def _estep(
    model: joint_model.JointModel,
    observations: pose.Observations,
    estimated_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams,
    ) -> pose.PoseSpaceParameters:

    estimated_params = estimated_params.with_hyperparams(hyperparams)

    est_poses = model.morph.inverse_transform(
        estimated_params.morph,
        observations.keypts,
        observations.subject_ids)
    aux_pdf = model.posespace.aux_distribution(
        estimated_params.posespace,
        est_poses,
        observations.subject_ids)
    
    return aux_pdf


def _mstep_objective(
    model: joint_model.JointModel,
    observations: pose.Observations,    
    query_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams,    
    aux_pdf: Float[Array, "Nt L"],
    ) -> Scalar:
    """
    Calculate objective for M-step to maximize.
    """

    params = query_params.with_hyperparams(hyperparams)

    poses, jacobian_logdets = model.morph.inverse_transform(
        params.morph, observations.keypts, observations.subject_ids,
        return_determinants = True)
    pose_probs = model.posespace.pose_logprob(
        params.posespace, poses, observations.subject_ids)
    jacobian_logdets = jacobian_logdets[..., None] # (Nt, L)
    keypt_probs = pose_probs + jacobian_logdets
    
    dataset_prob = (keypt_probs * aux_pdf).sum()
    
    morph_prior = model.morph.log_prior(params.morph)
    posespace_prior = model.posespace.log_prior(params.posespace)

    aux_entropy = -jnp.where(aux_pdf > 1e-12,
        aux_pdf * jnp.log(aux_pdf),
        0).sum()
    elbo = dataset_prob + aux_entropy
    
    # return morph_prior
    return {
        'objectives': {
            'dataset': dataset_prob,
            'morph': morph_prior,
            'pose': posespace_prior},
        'aux': {
            'logjac': jacobian_logdets.mean(),
            'elbo': elbo}
    }


def _mstep_loss(
    model: joint_model.JointModel,
    use_priors: bool = True
    ) -> Scalar:
    def step_loss_(
        params: optax.Params, 
        emissions: pose.Observations,
        hyperparams: joint_model.JointParameters,
        aux_pdf: Float[Array, "Nt L"], 
        ):
        scalars = _mstep_objective(
            model, emissions, params, hyperparams, aux_pdf)
        if use_priors:
            loss = -_pytree_sum(scalars['objectives'])
        else:
            print("Ignored priors.")
            loss = -scalars['objectives']['dataset']
        objectives = {**scalars["objectives"], **scalars['aux']}
        return loss, objectives
    return step_loss_


    
def lr_const(step_i, lr, n_samp, **kw):
    return (lr/n_samp)
    
def lr_exp(step_i, n_samp, lr, hl, min = 0, **kw):
    return (lr/n_samp) * 2 ** (-step_i / hl) + min/n_samp
        
rates = dict(
    const = lr_const,
    exp = lr_exp)


def construct_jitted_mstep(
    model: joint_model.JointModel,
    optimizer: optax.GradientTransformation,
    update_max: float,
    update_blacklist: list = None,
    use_priors: bool = True):

    loss_func = _mstep_loss(model, use_priors)

    @partial(jax.jit, static_argnums = (3,))
    def step(opt_state, params, emissions,
             hyperparams_static, hyperparams_dynamic,
             aux_pdf):
        
        hyperparams = joint_model.JointHyperparams.from_static_dynamic_parts(
            model, hyperparams_static, hyperparams_dynamic)
        (loss_value, objectives), grads = jax.value_and_grad(loss_func,
                argnums = 0, has_aux = True)(
            params, emissions, hyperparams, aux_pdf)
        
        if update_max is not None:
            grads = pt.tree_map(
                (lambda a: a.clip(-update_max, update_max)),
                grads)
        
        if update_blacklist is not None:
            pt.tree_map_with_path(
                lambda pth, grad: (
                    print("Locking param:", logging._keystr(pth))
                    if any(fnmatch(logging._keystr(pth), p)
                           for p in update_blacklist)
                    else None),
                grads)
            grads = pt.tree_map_with_path(
                lambda pth, grad: (
                    jnp.zeros_like(grad)
                    if any(fnmatch(logging._keystr(pth), p)
                           for p in update_blacklist)
                    else grad),
                grads)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, (loss_value, objectives)

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
    aux_pdf: Float[Array, "Nt L"], 
    emissions: pose.Observations,
    hyperparams: joint_model.JointHyperparams,
    optimizer: optax.GradientTransformation,
    opt_state,
    n_steps: int,
    reinit_opt = True,
    batch_size: int = None,
    batch_seed = 29,
    log_every: int = -1,
    progress = False,
    tol: float = None,
    stop_window: int = 10,
    update_max = None,
    update_blacklist = None,
    use_priors = True,
    jitted_step = None
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
        step = construct_jitted_mstep(
            model, optimizer,
            update_max, update_blacklist, use_priors)

    # ----- Set up variables for iteration

    curr_params = init_params
    if reinit_opt:
        opt_state = optimizer.init(init_params)
    loss_hist = np.full([n_steps], np.nan)
    iter = range(n_steps) if not progress else tqdm.trange(n_steps)
    hyper_stat, hyper_dyna = hyperparams.as_static_dynamic_parts()
    param_trace = logging.ReportTrace(n_steps)

    if batch_size is not None:
        batch_rkey_seed = jr.PRNGKey(batch_seed)
        unstacked_ixs = computations.unstacked_ixs(
            emissions.subject_ids,
            N = hyperparams.posespace.N)
        generate_batch = lambda rkey: computations.stacked_batch(
            rkey, unstacked_ixs, batch_size)
    
    # ---- Run M-step iterations

    for step_i in iter:

        if batch_size is not None:
            batch_rkey_seed, batch = generate_batch(batch_rkey_seed)
            step_obs = pose.Observations(
                keypts = emissions.keypts[batch],
                subject_ids = emissions.subject_ids[batch])
            step_aux = aux_pdf[batch]
        else:
            step_obs = emissions
            step_aux = aux_pdf

        curr_params, opt_state, (loss_value, objectives) = step(
            opt_state, curr_params,
            step_obs, hyper_stat, hyper_dyna,
            step_aux)
        loss_hist[step_i] = loss_value
        param_trace.record(curr_params, step_i)
        
        if (log_every > 0) and (not step_i % log_every):
            print(f"Step {step_i} : loss = {loss_value}")

        # evaluate early stopping
        if (tol is not None and step_i > stop_window and 
            _check_should_stop_early(
                loss_hist[step_i-stop_window:step_i+1], tol)
            ):
            loss_hist = loss_hist[:step_i+1]
            break

    return loss_hist, curr_params, objectives, param_trace, opt_state


def iterate_em(
    model: joint_model.JointModel,
    init_params: joint_model.JointParameters,
    emissions: pose.Observations,
    n_steps: int,
    log_every: int,
    progress = False,
    tol: float = None,
    stop_window: int = 5,
    batch_size: int = None,
    batch_seed: int = 23,
    update_blacklist = None,
    use_priors = True,
    learning_rate: float = dict(kind='const', lr=5e-3),
    scale_lr: bool = False,
    full_dataset_objectives = True,
    mstep_reinit_opt = True,
    mstep_n_steps: int = 2000,
    mstep_log_every: int = -1,
    mstep_progress = False,
    mstep_tol: float = None,
    mstep_stop_window: int = 10,
    mstep_batch_size = None,
    mstep_batch_seed = 29,
    mstep_update_max = None,
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

    hyperparams = init_params.hyperparams
    curr_params = init_params.trained_params
    loss_hist = np.empty([n_steps])
    iter = range(n_steps) if not progress else tqdm.trange(n_steps)
    report_trace = logging.ReportTrace(n_steps)

    if return_mstep_losses:
        mstep_losses = np.full([n_steps, mstep_n_steps], np.nan)
    if return_param_hist is True:
        param_hist = [curr_params]
    elif return_param_hist == 'trace':
        param_trace = logging.ReportTrace(n_steps + 1)
        param_trace.record(curr_params, 0)
    elif return_param_hist == 'mstep':
        param_trace = logging.ReportTrace(n_steps)
    
    if isinstance(learning_rate, int) or isinstance(learning_rate, float):
        learning_rate = dict(kind = 'const', lr = learning_rate)

    if scale_lr:
        if mstep_batch_size is not None: step_data_size = mstep_batch_size
        elif batch_size is not None: step_data_size = batch_size
        else: step_data_size = emissions.keypts.shape[0]
        print("Adjusting learning rate:",
            f"{learning_rate['lr']} -> {learning_rate['lr'] / step_data_size}")
    print(f"Loading schedule: {learning_rate['kind']}")

    lr_sched = functools.partial(rates[learning_rate['kind']],
        n_steps = n_steps, n_samp = step_data_size, **learning_rate)

    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate['lr'])
    opt_state = optimizer.init(curr_params)
    jitted_mstep = construct_jitted_mstep(model, optimizer,
        mstep_update_max, update_blacklist, use_priors)
    jitted_estep = construct_jitted_estep(model)
    hyper_stat, hyper_dyna = hyperparams.as_static_dynamic_parts()

    if batch_size is not None:
        batch_rkey_seed = jr.PRNGKey(batch_seed)
        unstacked_ixs = computations.unstacked_ixs(
            emissions.subject_ids,
            N = hyperparams.posespace.N)
        generate_batch = lambda rkey: computations.stacked_batch(
            rkey, unstacked_ixs, batch_size)

    for step_i in iter:
        
        aux_pdf = jitted_estep(
            observations = emissions,
            estimated_params = curr_params,
            hyperparams_static = hyper_stat,
            hyperparams_dynamic = hyper_dyna)
        
        if batch_size is not None:
            batch_rkey_seed, batch = generate_batch(batch_rkey_seed)
            step_obs = pose.Observations(
                keypts = emissions.keypts[batch],
                subject_ids = emissions.subject_ids[batch])
            step_aux = aux_pdf[batch]
        else:
            step_obs = emissions
            step_aux = aux_pdf

        opt_state.hyperparams['learning_rate'] = lr_sched(step_i)
        loss_hist_mstep, fit_params_mstep, mstep_end_objective, mstep_param_trace, opt_state = _mstep(
            model = model,
            init_params = curr_params,
            aux_pdf = step_aux,
            emissions = step_obs,
            hyperparams = hyperparams,
            optimizer = optimizer,
            opt_state = opt_state,
            n_steps = mstep_n_steps,
            reinit_opt = mstep_reinit_opt,
            batch_size = mstep_batch_size,
            batch_seed = mstep_batch_seed + step_i,
            log_every = mstep_log_every,
            progress = mstep_progress,
            tol = mstep_tol,
            stop_window = mstep_stop_window,
            update_max = mstep_update_max,
            update_blacklist = update_blacklist,
            jitted_step = jitted_mstep
        )

        loss_hist[step_i] = loss_hist_mstep[-1]
        start_params = curr_params
        curr_params = fit_params_mstep
        if return_mstep_losses:
            mstep_losses[step_i, :len(loss_hist_mstep)] = loss_hist_mstep
        if return_param_hist is True:
            param_hist.append(curr_params)
        elif return_param_hist == 'trace':
            param_trace.record(curr_params, step_i + 1)
        elif return_param_hist == 'mstep':
            param_trace.record(mstep_param_trace.as_dict(), step_i)

        if return_reports:
            aux_reports = {}
            if full_dataset_objectives:
                aux_reports['dataset_logprob'] = _mstep_objective(
                    model, emissions, curr_params, hyperparams,
                    aux_pdf = jitted_estep(
                        observations = emissions,
                        estimated_params = start_params,
                        hyperparams_static = hyper_stat,
                        hyperparams_dynamic = hyper_dyna)
                )['objectives']['dataset']
            report_trace.record(dict(
                logprobs = mstep_end_objective,
                lr = jnp.array(lr_sched(step_i)),
                **aux_reports),
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
    if return_param_hist is True: ret = ret + (param_hist,)
    elif return_param_hist in ['trace', 'mstep']: ret = ret + (param_trace,)
    if return_reports: ret = ret + (report_trace,)
    return ret 