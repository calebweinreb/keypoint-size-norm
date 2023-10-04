from typing import NamedTuple, Tuple
from jaxtyping import Array, Float, Float32, Integer, Scalar, PRNGKeyArray as PRNGKey
from tensorflow_probability.substrates import jax as tfp
from sklearn import mixture
import numpy.random as nr
import numpy as np
import jax.numpy as jnp
import jax.numpy as jnp
import jax.numpy as jnp
from sklearn import mixture
import jax.random as jr
import jax.nn as jnn

from .pose_model import *
from ...util.computations import (
    expand_tril_cholesky, extract_tril_cholesky,
    gaussian_product, normal_quadform_expectation,
    linear_transform_gaussian, sq_mahalanobis)


class GMMParameters(NamedTuple):
    """
    Parameters for Gaussian mixture pose space model.

    :param subj_weight_logits: Probability logits over mixture components
        for each subject. Observable layer of the heirarchical Dirichlet prior.
    :param pop_weight_logits: Probability logits over mixture components
        shared across the population. Hidden layer of the heirarchical
        Dirichlet prior.
    :param means: Mean for each mixture component.
    :param cholesky: Flat cholesky decomposition of covariance matrix
        for each mixture component.
    """
    subj_weight_logits: Float[Array, "N L"]
    pop_weight_logits: Float[Array, "L"]
    means: Float[Array, "L M"]
    cholesky: Float[Array, "L (M+1)*M/2"]

    def covariances(self) -> Float32[Array, "L M M"]:
        return expand_tril_cholesky(self.cholesky, n = self.means.shape[1])

    @classmethod
    def cholesky_from_covariances(self, covariances: Float32[Array, "L M M"]):
        return extract_tril_cholesky(covariances) 

    def weights(self) -> Float[Array, "N L"]:
        return jnn.softmax(self.subj_weight_logits, axis = 1)
    
    def pop_weights(self) -> Float[Array, "L"]:
        return jnn.softmax(self.pop_weight_logits)


class GMMHyperparams(NamedTuple):
    """
    Hyperparameters for a Gaussian mixture pose space model.
    
    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of mixture components.
    :param eps: Observation error variance.
    :pop_weight_uniformity: Tighness of the distribution of expected cluster
        weights. Variance of population cluster weights is inversely
        proportional to this parameter plus one. Positive float.
    :subj_weight_uniformity: Tightness of the distribution of subject cluster
        weights around the population expected cluster weight. Variance of
        subject-wise cluster weights inversely proportional to this parameter
        plus one. Positive float.
    """
    N: int
    M: int
    L: int
    eps: int
    pop_weight_uniformity: float
    subj_weight_uniformity: float


class GMMPoseStates(NamedTuple):
    """
    Pose states for a Gaussian mixuture pose space model.
    
    :param components: Component likelihoods for each sample.
    :param poses: Pose space states resulting from the GMM.
    """
    components: Float[Array, "Nt L"]
    poses: Float[Array, "Nt M"]


class GMMAuxPDF(NamedTuple):
    """
    Values defining the auxiliary distribution up to a constant.

    In the theory document, we outline how the standard EM auxiliary
    distribution is broken down into a constant in $x$ times a PDF in
    $x$. A `GMMAuxPDF` contains precomputed values defining
    that PDF, from which expectations against log-probabilty terms
    may be evaluated.

    For the GMM pose space model, the PDF is Gaussian, and is therefore
    defined by a mean and covariance.

    :param mean: Mean of the normal PDF.
    :param cov: Covariance matrix of the normal PDF.
    """
    consts: Float[Array, "Nt L"]
    mean: Float[Array, "Nt L M"]
    cov: Float[Array, "Nt L M M"]


def aux_distribution(
    observations: Observations,
    morph_matrix: Float[Array, "N KD M"],
    morph_ofs: Float[Array, "N M"],
    params: GMMParameters,
    hyperparams: GMMHyperparams,
    morph_inv: Float[Array, "N M KD"] = None,
    ) -> Tuple[Float[Array, "Nt L"], GMMAuxPDF]:
    """
    Calculate the auxiliary distribution for EM.
    
    As is described in the theory document, we break the auxiliary
    distribution into a set of constants independent of the
    continuous pose state $x$ and a distribution in $x$ to be
    integrated against log probability terms arising in the objective
    function. This function calculates those constants and any helper
    values for the distribution that arise naturally during the
    computation of those constants.
    
    Args:
        observations: Observations from combined model.
        morph_matrix: Subject-wise linear transform from poses to
            keypoints.
        morph_ofs: Subject-wise affine component of transform from
            poses to keypoints.
        params, hyperparams: Parameters of the mixture model.

    Returns:
        consts: Array of proportionality constants.
        helpers: Helper values for the distribution.
    """

    # See Pose space models > Gaussian mixture in theory doc.

    # optionally use precomputed values
    if morph_inv is None:
        morph_inv = jnp.linalg.inv(morph_matrix)

    # Compute distribution in pose space including observation noise
    KD: int = morph_matrix.shape[1]
    ps_mean, ps_cov, ps_cov_inv, normalizer = linear_transform_gaussian(
        query_point = observations.keypts, # (Nt, KD)
        cov = hyperparams.eps * jnp.eye(KD)[None], # (1, KD, KD)
        cov_inv = jnp.eye(KD)[None] / hyperparams.eps, # (1, KD, KD)
        A = morph_matrix[observations.subject_ids], # (Nt, KD, M)
        Ainv = morph_inv[observations.subject_ids], # (Nt, KD, M)
        d = morph_ofs[observations.subject_ids], # (Nt, M) 
        return_cov_inv = True,
        return_normalizer = True
    )
    
    # Compute mean and covariances going into Gaussian product
    a: Float[Array, "Nt 1 M"] = ps_mean[:, None]
    A: Float[Array, "Nt 1 M M"] = ps_cov[:, None]
    Ainv: Float[Array, "Nt 1 M M"] = ps_cov_inv[:, None]
    b: Float[Array, "1 L M"] = params.means[None, :]
    B: Float[Array, "1 L M M"] = params.covariances()[None, :]

    # Compute gaussian product with normalizer terms
    # The transformation that turns the PDF in y into a PDF in x introduces
    # normalizer terms that cancel Z_A = Z_{C^{-1} R C^{-1}^T} and replace
    # it with a normalizer Z_R.
    K, c, C, gp_norm = gaussian_product(a, A, b, B, Ainv = Ainv,
        return_normalizer = True)
    
    # more efficient normalizer calculation that removes two determinant calls
    # custom_norm = jnp.sqrt(
    #     jnp.linalg.det(C) / jnp.linalg.det(B) / (hyperparams.eps ** KD) *
    #     (2 * jnp.pi) ** (C.shape[-1] - B.shape[-1] - KD)
    # )
    
    combined_norm = normalizer[:, None] * gp_norm
    K *= combined_norm
    
    return GMMAuxPDF(consts = K, mean = c, cov = C)


def discrete_prob(
    params: GMMParameters,
    hyperparams: GMMHyperparams
    ) -> Float[Array, "N L"]:
    """
    Probabilities across discrete pose space for each subject.
    
    Args:
        params, hyperparams: Parameters of the mixture model.
    
    Returns:
        probs: Probability distributions for each subject.
    """
    return params.weights()


def logprob_expectations(
    observations: Observations,
    morph_matrix: Float[Array, "N KD M"],
    morph_ofs: Float[Array, "N M"],
    query_params: GMMParameters,
    hyperparams: GMMHyperparams,
    aux_pdf: GMMAuxPDF,
    morph_inv: Float[Array, "N M KD"] = None,
    ) -> Float[Array, "Nt L"]:
    """
    Compute
        int_x s(x) log N(x; y) and int_x s(x) log F(x)
    as well as:
        terms from F constant in x (included in int_x s(x) log F(x))
    
    terms from swapping x, y in N() do not care what pose space
    model is being used and so will be dealt with elsewhere.

    Args:
        morph_inv: Avoid recalculating inverse if precomputed elsewhere
    """

    # ----- Fill out optionally precomputed terms

    if morph_inv is None:
        morph_inv = jnp.linalg.inv(morph_matrix)


    # Compute distribution in pose space including observation noise
    # This yields (the mean and covariance of) a normal distribution over pose
    # space, the log PDF of which will be integrated against the auxiliary
    # distribution.
    # Note that since we calculate the log PDF explicitly here (later, in
    # normal_quadform_expectation) instead of calculating the normal PDF itself,
    # we do not use the normalizer term from linear_transform_gaussian which is
    # used to correct for the new covariance of the resulting normal PDF.
    KD: int = morph_matrix.shape[1]
    ps_mean, ps_cov, ps_cov_inv = linear_transform_gaussian(
        query_point = observations.keypts, # (Nt, KD)
        cov = hyperparams.eps * jnp.eye(KD)[None], # (1, KD, KD)
        cov_inv = jnp.eye(KD)[None] / hyperparams.eps, # (1, KD, KD)
        A = morph_matrix[observations.subject_ids], # (Nt, KD, M)
        Ainv = morph_inv[observations.subject_ids], # (Nt, KD, M)
        d = morph_ofs[observations.subject_ids], # (Nt, M)
        return_cov_inv = True,
        return_normalizer = False
    )

    # ----- Compute expectation terms

    obs_term = normal_quadform_expectation(
        aux_pdf.mean, aux_pdf.cov,
        ps_mean[:, None],
        ps_cov_inv[:, None]
    )

    query_Q = query_params.covariances()
    posespace_term = normal_quadform_expectation(
        aux_pdf.mean, aux_pdf.cov,
        query_params.means,
        jnp.linalg.inv(query_Q)
    )
    posespace_norm = jnp.log(jnp.linalg.det(query_Q))
    
    return (obs_term + posespace_term + posespace_norm) / (-2)


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: GMMHyperparams,
    m_norm_center: Scalar,
    m_norm_spread: Scalar,
    q_var_center: Scalar,
    q_var_spread: Scalar,
    pi_logit_means: Float[Array, "N L"] = None,
    pi_logit_vars: Float[Array, "N L"] = None,
    ) -> GMMParameters:
    r"""
    We use the following generative framework for $\theta$, the
    parameters of the GMM pose space model:

    $$
    \mathrm{weight_logits} \sim 
        \mathcal{N}(\mathrm{pi_logit_means},
                    \mathrm{diag}(\mathrm{pi_logit_vars}))
    \\
    \mathrm{means} \sim \mathrm{Uniform}[S^{M}] \cdot \exp(
        \mathcal{N}(\mathrm{m_norm_center}, \mathrm{m_norm_spread})
    ) \\
    \mathrm{covariances} \sim \mathrm{diag}(\exp(
        \mathcal{N}(\mathrm{q_var_center}, \mathrm{q_var_spread})
    ))
    $$

    If `pi_logit_means` or `pi_logit_vars` is omitted, then the component
    weights will be sampled from a heirarchical Dirichlet distribution
    according to the hyperparameters `pop_weight_uniformity` and
    `subj_weight_uniformity`.

    $$
    \beta \sim \mathrm{Dir}(\mathrm{pop_weight_uniformity}) \\
    \mathrm{weight_logits} \sim
    \mathrm{Dir}(\mathrm{subj_weight_uniformity\ }\cdot\, \beta)
    $$
    
    :param rkey: JAX random key.
    :param hyperparams: Hyperparameters of a GMM pose space model.
    :param m_norm_center: Median of log-normally distributed norms of
        cluster centers.
    :param m_norm_spread: Spread of log-normally distributed norms of
        cluster centers.
    :param q_var_center: Median of log-normally distributed diagonals
        of cluster coviariances.
    :param q_var_spread: Spread of log-normally distributed diagonals
        of cluster coviariances.
    :param pi_logit_means: Mean vector of Gaussian logits for
        component weights. Allows additional specification of component
        weights not conforming to the heirarchical Dirichlet prior.
    :param pi_logit_vars: Diagonal covariance of Gaussian logits for
        component weights. Allows additional specification of component
        weights not conforming to the heirarchical Dirichlet prior.
    """

    rkey = jr.split(rkey, 5)

    # --- Component weights
    if pi_logit_means is not None and pi_logit_vars is not None:
        subj_weight_logits = (
            pi_logit_means +
            jnp.sqrt(pi_logit_vars) * jr.normal(
                rkey[0],
                shape = pi_logit_means.shape))
        subj_weights = jnn.softmax(subj_weight_logits)
        pop_weights = subj_weights.mean(axis = 0)
        pop_weight_logits = jnp.log(pop_weights)
    else:
        pop_weights = jr.dirichlet(
            rkey[0],
            jnp.ones([hyperparams.L]) *
                hyperparams.pop_weight_uniformity /
                hyperparams.L
        )
        pop_weight_logits = jnp.log(pop_weights)
        subj_weight_logits = jnp.log(jr.dirichlet(
            rkey[1],
            pop_weights * hyperparams.subj_weight_uniformity,
            shape = (hyperparams.N,)
        ))
    # logits are defined up to an additive constant
    pop_weight_logits = pop_weight_logits - pop_weight_logits.mean()
    subj_weight_logits = (
        subj_weight_logits - 
        subj_weight_logits.mean(axis = 1, keepdims = True))

    # --- Component means
    m_direction: Float[Array, "L"] = jr.multivariate_normal(
        rkey[2],
        jnp.zeros(hyperparams.M), jnp.diag(jnp.ones(hyperparams.M)),
        shape = (hyperparams.L,)
    )
    m_norm = jnp.exp(m_norm_center + m_norm_spread * jr.normal(
        rkey[3],
        shape = (hyperparams.L,)
    ))
    m = (m_direction * m_norm[:, None] /
         jnp.linalg.norm(m_direction, axis = 1)[:, None])

    # --- Component covairances
    q_sigma = jnp.exp(q_var_center * q_var_spread * jr.normal(
        rkey[4],
        shape = (hyperparams.L, hyperparams.M),
    ))
    Q = jnp.zeros([hyperparams.L, hyperparams.M, hyperparams.M])
    q_diag = jnp.arange(hyperparams.M)
    Q = Q.at[:, q_diag, q_diag].set(q_sigma)
    Q_chol = GMMParameters.cholesky_from_covariances(Q)

    return GMMParameters(
        pop_weight_logits = pop_weight_logits,
        subj_weight_logits = subj_weight_logits,
        means = m,
        cholesky = Q_chol)


def sample(
    rkey: PRNGKey,
    params: GMMParameters,
    hyperparams: GMMHyperparams,
    T: int
    ) -> GMMPoseStates:

    rkey = jr.split(rkey, 2)

    comp_weights = params.weights()
    z: Integer[Array, "N*T"] = jnp.stack([
        jr.choice(k, hyperparams.L, shape = (T,), replace = True, p = w)
        for k, w, in zip(jr.split(rkey[0], hyperparams.N), comp_weights)
        ]).flatten()

    x: Integer[Array, "N*T M"] = jr.multivariate_normal(
        rkey[2],
        params.means[z], params.covariances()[z],
    )

    subject_ids = jnp.broadcast_to(
        jnp.arange(hyperparams.N)[:, None],
        (hyperparams.N, T)).flatten()

    return GMMPoseStates(components = z, poses = x), subject_ids


def init_parameters_and_latents(
    hyperparams: GMMHyperparams,
    observations: Observations,
    poses: Float[Array, "Nt M"],
    reference_subject: int,
    seed: int = 0,
    count_eps: float = 1e-3
    ) -> Tuple[GMMParameters, GMMPoseStates]:
    """
    Initialize a GMMPoseSpaceModel based on observed keypoint data.

    This function uses a single subject's keypoint data to initialize a
    `GMMPoseSpaceModel` based on a standard Gaussian Mixture.

    Args:
        hyperparams: GMMHyperparams
            Hyperparameters of the pose space model.
        poses:
            Pose state latents as given by initialization of a morph model.
        observations: pose.Observations
            Keypoint-space observations to estimate the pose space model from.
        reference_sibject: int
            Subject ID in `observations` to initialize
        count_eps: float
            Count representing no observations in a component. Component weights
            are represented as logits (logarithms) which cannot capture zero
            counts, so instances of zero counts are replaced with `count_eps` to
            indicate a very small cluster weight.
    Returns:
        init_params: GMMHyperparams
            Initial parameters for a `GMMPoseSpaceModel`
        init_states: GMMPoseStates
            Initial pose states for a `GMMPoseSpaceModel`
    """

    # fit GMM to reference subject
    init_pts = observations.unstack(poses)[reference_subject]
    init_mix = mixture.GaussianMixture(
        n_components = hyperparams.L, 
        random_state = nr.RandomState(seed),
    ).fit(init_pts)

    # get component labels & counts across all subjects
    init_components = observations.unstack(
        init_mix.predict(poses)) # List<subj>[(T,)]
    init_counts = np.zeros([hyperparams.N, hyperparams.L])
    for i_subj in range(hyperparams.N):
        uniq, count = np.unique(
            init_components[i_subj],
            return_counts = True)
        init_counts[i_subj][uniq] = count
    init_counts[init_counts == 0] = count_eps
    
    return GMMParameters(
        subj_weight_logits = jnp.log(init_counts),
        pop_weight_logits = jnp.log(init_counts[reference_subject]),
        means = init_mix.means_,
        cholesky = extract_tril_cholesky(init_mix.covariances_)
    )


def discrete_mle(
    poses: Float[Array, "Nt M"],
    hyperparams: GMMHyperparams,
    estimated_params: GMMParameters
    ) -> GMMPoseStates:
    """
    Estimate pose model discrete latents given poses.
    """
    dists = sq_mahalanobis(
        poses[:, None],
        estimated_params.means[None, :],
        estimated_params.covariances()[None, :])
    return GMMPoseStates(
        components = dists.argmin(axis = 1),
        poses = poses
    )


def log_prior(
    params: GMMParameters,
    hyperparams: GMMHyperparams,
    ):
    # Heirarchical dirichlet prior on component weights
    pop_weights = params.pop_weights()
    pop_logpdf = tfp.distributions.Dirichlet(
        jnp.ones([hyperparams.L]) *
            hyperparams.pop_weight_uniformity /
            hyperparams.L,
    ).log_prob(pop_weights)
    subj_logpdf = tfp.distributions.Dirichlet( 
        hyperparams.subj_weight_uniformity *
        pop_weights[None]
    ).log_prob(params.weights()).sum()
    # Remove symmetry of logits up to additive constant
    pop_zero = -(params.pop_weight_logits.mean() ** 2) / 2
    subj_zero = -(params.subj_weight_logits.mean(axis = 1) ** 2).sum() / 2

    return pop_logpdf + subj_logpdf + pop_zero + subj_zero
    


    


GMMPoseSpaceModel = PoseSpaceModel(
    discrete_mle = discrete_mle,
    sample = sample,
    sample_parameters = sample_parameters,
    logprob_expectations = logprob_expectations,
    discrete_prob = discrete_prob,
    aux_distribution = aux_distribution,
    log_prior = log_prior,
    init = init_parameters_and_latents
)



