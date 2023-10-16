
from jaxtyping import Float, Array, Scalar, PRNGKeyArray as Key
import jax.numpy as jnp
import jax.random as jr
import jax.nn as jnn

from .model import *
from ...util.computations import extract_tril_cholesky

def sample(
        key: Key,
        arch: SingleMouseGMMArch,
        hyperparams: SingleMouseGMMHyperparams,
        pi_logit_means: Float[Array, "N"],
        pi_logit_vars: Float[Array, "N"],
        m_norm_center: Scalar,
        m_norm_spread: Scalar,
        q_var_center: Scalar,
        q_var_spread: Scalar
    ) -> SingleMouseGMM:
    r"""
    Sample a SingleMouseGMM for testing data.

    We use the following generative framework for $\theta$, the parameters of the
    SingleMouseGMM:

    $$
    \pi \sim \mathrm{Softmax}(
        \mathcal{N}(\mathrm{pi_logit_means}, \mathrm{diag}(\mathrm{pi_logit_vars}))
    ) \\
    m \sim \mathrm{Uniform}[S^{M}] \cdot \exp(
        \mathcal{N}(\mathrm{m_norm_center}, \mathrm{m_norm_spread})
    ) \\
    Q \sim \mathrm{diag}(\exp(
        \mathcal{N}(\mathrm{q_var_center}, \mathrm{q_var_spread})
    )) \\\
    $$

    :param key: JAX random key to for deterministic output.
    :param arch: Architecture variables for the SingleMouseGMM.
    :param hyperparams: Hyperparameters, in particular observable noise variance.
    :param pi_logit_means: Mean vector of Gaussian logits for component weights.
    :param pi_logit_vars: Diagonal covariance of Gaussian logits for component weights.
    :param m_norm_center: Median of log-normally distributed norms of cluster centers.
    :param m_norm_spread: Spread of log-normally distributed norms of cluster centers.
    :param q_var_center: Median of log-normally distributed diagonals of cluster coviariances.
    :param q_var_spread: Spread of log-normally distributed diagonals of cluster coviariances.
    """ 

    keys = jr.split(key, 7)

    # ===== Generate parameters

    pi_logits = jr.multivariate_normal(
        keys[0],
        pi_logit_means, jnp.diag(pi_logit_vars))
    pi = jnn.softmax(pi_logits)

    m_direction = jr.multivariate_normal(
        keys[1],
        jnp.zeros(arch.M), jnp.diag(jnp.ones(arch.M)),
        shape = (arch.N,)
    )
    m_norm = jnp.exp(m_norm_center + m_norm_spread * jr.normal(
        keys[2],
        shape = (arch.N,)
    ))
    m = m_direction * m_norm[:, None] / jnp.linalg.norm(m_direction, axis = 1)[:, None]

    q_sigma = jnp.exp(q_var_center * q_var_spread * jr.normal(
        keys[3],
        shape = (arch.N, arch.M),
    ))
    Q = jnp.zeros([arch.N, arch.M, arch.M])
    Q = Q.at[:, jnp.arange(arch.M), jnp.arange(arch.M)].set(q_sigma)
    lq = extract_tril_cholesky(Q)


    # ===== Generate latents and observables

    z = jr.choice(
        keys[4],
        arch.N,
        shape = (arch.T,),
        replace = True,
        p = pi
    )

    x = jr.multivariate_normal(
        keys[5],
        m[z], Q[z],
    )

    y = x + (hyperparams.eps**2) * jr.normal(
        keys[6],
        shape = (arch.T, arch.M)
    )

    # ===== Output

    return SingleMouseGMM(
        hyperparams = hyperparams,
        latents = SingleMouseGMMLatents(z, x),
        observables = SingleMouseGMMObservables(y),
        params = SingleMouseGMMParameters(pi_logits, m, lq)
    )



ExampleSMGMMArch = SingleMouseGMMArch(
    N = 5, M = 2, T = 100
)
ExampleSMGMMHyperparams = SingleMouseGMMHyperparams(
    eps = 0.05
)
ExampleSMGMMDatasetParams = dict(
    pi_logit_means = 2 * jnp.ones(ExampleSMGMMArch.N),
    pi_logit_vars = 0.06 * jnp.ones(ExampleSMGMMArch.N),
    m_norm_center = 2,
    m_norm_spread = 0.4,
    q_var_center = 0.5,
    q_var_spread = 0.1,
)





def pertub_parameters(
    rngkey: Key,
    params: SingleMouseGMMParameters,
    nudge: float
    ) -> SingleMouseGMMParameters:

    keys = jr.split(rngkey, 3)

    # Add normal(scale = nudge * std[pi_logits]) to pi_logits
    pibar = params.pibar + (nudge
        * params.pibar.std()
        * jr.normal(keys[0], shape = params.pibar.shape))
    
    # Add normal(scale = nudge * std[log(diag(Q))]) to diag(Q)
    M = params.m.shape[-1]
    diag_ixs = jnp.arange(M)
    Q = params.Q().copy()
    qdiag = Q[:, diag_ixs, diag_ixs]
    Q = Q.at[:, diag_ixs, diag_ixs].set(
        qdiag * jnp.exp(
            nudge
          * jnp.log(qdiag.std())
          * jr.normal(keys[1], shape = (qdiag.shape))
        )
    )
    lq = extract_tril_cholesky(Q)
    
    # Add normal(scale = nudge * std(m)) to m
    m = params.m + (nudge
        * params.m.std()
        * jr.normal(keys[2], shape = params.m.shape))
    
    return SingleMouseGMMParameters(pibar = pibar, m = m, lq = lq)
    