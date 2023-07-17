from typing import NamedTuple
from jaxtyping import Scalar
from functools import partial
import numpy as np
import optax
import tqdm
import jax

from ..models import pose
from ..models import joint_model

def objective(
    model: joint_model.JointModel,
    observations: pose.Observations,    
    query_params: joint_model.JointParameters,
    estimated_params: joint_model.JointParameters,
    hyperparams: joint_model.JointHyperparams,    
    ) -> Scalar:
    """
    Calculate objective for M-step to maximize.
    """

    est_morph_matrix = model.morph.get_transform(
        estimated_params.morph,
        hyperparams.morph)
    morph_matrix = model.morph.get_transform(
        query_params.morph,
        hyperparams.morph)

    return pose.objective(
        model.posespace,
        observations,
        morph_matrix,
        est_morph_matrix,
        query_params.posespace,
        estimated_params.posespace,
        hyperparams.posespace
    )



def step_loss(
    model: joint_model.JointModel,
    ) -> Scalar:
    def step_loss_(
        params: optax.Params, 
        estimated_params: joint_model.JointParameters,
        emissions: pose.Observations,
        hyperparams: joint_model.JointParameters):

        return -objective(model, emissions, params, estimated_params, hyperparams)
    return step_loss_

def mstep(
    model: joint_model.JointModel,
    init_params: optax.Params,
    estimated_params: joint_model.JointParameters,
    emissions: pose.Observations,
    hyperparams: joint_model.JointParameters,
    optimizer: optax.GradientTransformation,
    n_steps: int,
    log_every: int,
    progress = False,
    ):

    # ----- Define the step function with weight update

    loss_func = step_loss(model)

    @partial(jax.jit, static_argnums = (4,))
    def step(opt_state, params, estimated_params, emissions, hyperparams):
        loss_value, grads = jax.value_and_grad(loss_func, argnums = 0)(
            params, estimated_params, emissions, hyperparams)
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
            estimated_params, emissions, hyperparams)
        loss_hist[step_i] = loss_value
        
        if (log_every > 0) and (not step_i % log_every):
            print(f"Step {step_i} : loss = {loss_value}")

    return loss_hist, curr_params