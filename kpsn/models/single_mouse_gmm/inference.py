from jaxtyping import Float, Array
from typing import Tuple

from tensorflow_probability.substrates import jax as tfp
import jax.numpy as jnp
import jax.nn as jnn
import jax

from ...util import quadform
from .model import *



def gaussian_product(
    y: Float[Array, "T M"],
    m_star: Float[Array, "N M"],
    Q_star: Float[Array, "N M M"],
    R_star: Float[Array, "M M"]
    ) -> Tuple[Float[Array, "T N M"], Float[Array, "N M M"], Float[Array, "T N"]]:
    r"""Calculate terms that summarize product of normal pdfs.
    
    When formulating $q_t(x, z)$, the product of two normal PDFs are combined into
    a single evaluation of a normal pdf with location $\mu_{z, t}^*$ and covariance
    $\Sigma_z^*, and a proportionality constant $P(y_t\mid z, \theta^*). This
    function computes those terms so that they may be used in evaluating the
    objective function.
    
    Args:
        y: Emissions / observations
        m_star: Current estimated component means
        Q_star: Current estimated component covariances
        R_star: Current estimated observation noise covariance

    Returns:
        mu_star: Combined normal PDF mean, $\mu_{z, t}^*$
        sigma_star: Combined normal PDF covariance, $\Sigma_z,^*
        prob_obs_given_z: Proportionality constant, $P(y_t\mid z,\theta^*)$.
    """

    # ----- \Sigma^*_{z, t}
    sigma_star = R_star[None] @ jnp.linalg.inv(R_star[None] + Q_star) @ Q_star

    # ----- \mu^*_{z, t}
    observable_part = jnp.linalg.inv(R_star)[None, None] @ y[:, None, :, None]
    latent_part = jnp.linalg.inv(Q_star)[None] @ m_star[None, :, :, None]
    mu_star = (sigma_star[None] @ (observable_part + latent_part))[..., 0]

    # ----- P(y_t | z, \theta^*)
    M = mu_star.shape[-1]
    pi_part = (2 * jnp.pi) ** M
    numer_det_part = jnp.linalg.det(sigma_star)
    denom_det_part = (R_star[0, 0] ** M) * jnp.linalg.det(Q_star)
    
    observable_part: Float[Array, "T"] = quadform(y, jnp.linalg.inv(R_star)[None])
    latent_part: Float[Array, "N"] = quadform(m_star, jnp.linalg.inv(Q_star))
    combined_part: Float[Array, "T N"] = quadform(mu_star, jnp.linalg.inv(sigma_star)[None])

    exp_part: Float[Array, "T N"] = jnp.exp(
          observable_part[:, None]
        + latent_part[None, :]
        - combined_part)
    
    prob_obs_given_z = ( 
        numer_det_part / 
        (pi_part * denom_det_part * exp_part)
    ) ** (1/2)

    
    return mu_star, sigma_star, prob_obs_given_z



def objective(
    query_params: SingleMouseGMMParameters,
    estimated_params: SingleMouseGMMParameters,
    emissions: SingleMouseGMMObservables,
    hyperparams: SingleMouseGMMHyperparams
    ) -> Scalar:
    """Objective function to maximize during the M-step"""

    # ===== Calculate supporting values

    M = estimated_params.m.shape[-1]
    R_star: Float[Array, "M M"] = (hyperparams.eps * jnp.eye(M))
    pi_star = estimated_params.pi()
    est_Q = estimated_params.Q()
    query_Q = query_params.Q()
    query_pi = query_params.pi()

    mu_star, sigma_star, prob_obs_given_z = gaussian_product(
        emissions.y,
        estimated_params.m,
        est_Q,
        R_star)
    
    joint_prob_obs_z = prob_obs_given_z * pi_star[None, :]
    prob_obs: Float[Array, "T"] = joint_prob_obs_z.sum(axis = -1)
    

    # ===== Calculate summand terms
    
    # No log |R| term, since R is not being fit
    # log_R = ...
    log_Q = jnp.log(jnp.linalg.det(query_Q))
    # Normal pdf terms
    # norm_obs: Float[Array, "T N"] = jnp.swapaxes(
    #     tfp.distributions.MultivariateNormalFullCovariance(
    #         emissions.y, R_star[None], validate_args = True
    #     ).prob(jnp.swapaxes(mu_star, 0, 1)),
    #     0, 1)
    # norm_clust: Float[Array, "T N"] = tfp.distributions.MultivariateNormalFullCovariance(
    #     query_params.m, query_Q, validate_args = True
    # ).prob(mu_star)
    norm_obs: Float[Array, "T N"] = quadform(
        mu_star - emissions.y[:, None], jnp.linalg.inv(R_star))
    norm_clust: Float[Array, "T N"] = quadform(
        mu_star - query_params.m[None], jnp.linalg.inv(query_Q)[None])
    # Trace terms
    tr_r: Float[Array, "N"] = jnp.trace(
        sigma_star,
        axis1 = -2, axis2 = -1
    ) / R_star[0, 0]
    tr_q: Float[Array, "N"] = jnp.einsum(
        'kij,kji->k',
        sigma_star,
        jnp.linalg.inv(query_Q))
    # pibar terms
    logpi = -2 * jnp.log(query_pi)

    # ===== Calculate sum
    # Group by shape to avoid unnecessary additions on broadcast arrays
    obj = -0.5 * (
        (joint_prob_obs_z /
         prob_obs[:, None] * (
            norm_obs + norm_clust +
            (tr_r[None, :] + tr_q[None, :] +
            logpi[None, :] + log_Q[None, None])
        )).sum(axis = 0) # sum over t before multiplying by \pi_z^*
    ).sum()

    return obj
    # return obj, dict(
    #     pi_star = pi_star,
    #     prob_obs_given_z = prob_obs_given_z,
    #     prob_obs = prob_obs,
    #     norm_obs = norm_obs,
    #     norm_clust = norm_clust,
    #     tr_r = tr_r,
    #     tr_q = tr_q,
    #     logpi = logpi,
    #     log_Q = log_Q,
    #     )