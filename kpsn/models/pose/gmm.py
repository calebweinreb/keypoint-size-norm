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
    sq_mahalanobis)


class GMMTrainedParams(NamedTuple):
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

    LOGIT_MAX = 5
    
    @staticmethod
    def create(
        subj_weight_logits: Float[Array, "N L"],
        pop_weight_logits: Float[Array, "L"],
        means: Float[Array, "L M"],
        cholesky: Float[Array, "L (M+1)*M/2"]) -> 'GMMTrainedParams':

        return GMMTrainedParams(
            subj_weight_logits =
                GMMParameters.normalize_logits(subj_weight_logits),
            pop_weight_logits = 
                GMMParameters.normalize_pop_logits(pop_weight_logits),
            means = means,
            cholesky = cholesky)

    def with_hyperparams(self, hyperparams):
        return GMMParameters(self, hyperparams)



class GMMHyperparams(NamedTuple):
    """
    Hyperparameters for a Gaussian mixture pose space model.
    
    :param N: Number of subjects.
    :param M: Pose space dimension.
    :param L: Number of mixture components.
    :param diag_eps: Factor to add to diagonal of component covariance matrices
        or `None`.
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
    diag_eps: float
    pop_weight_uniformity: float
    subj_weight_uniformity: float

    def as_static_dynamic_parts(self):
        return (self, None)
    
    @staticmethod
    def from_static_dynamic_parts(static, dynamic):
        return static


class GMMParameters(NamedTuple):
    trained_params: GMMTrainedParams
    hyperparams: GMMHyperparams
    
    # hyperparameter passthrough
    N = property(lambda self: self.hyperparams.N)
    M = property(lambda self: self.hyperparams.M)
    L = property(lambda self: self.hyperparams.L)
    diag_eps = property(lambda self: self.hyperparams.diag_eps)
    pop_weight_uniformity = property(lambda self:
        self.hyperparams.pop_weight_uniformity)
    subj_weight_uniformity = property(lambda self:
        self.hyperparams.subj_weight_uniformity)
    # passthrough of some parameters
    means = property(lambda self: self.trained_params.means)
    cholesky = property(lambda self: self.trained_params.cholesky)
    
    def covariances(self) -> Float32[Array, "L M M"]:
        covs = expand_tril_cholesky(self.cholesky, n = self.M)
        # addition of diagonal before extracting cholesky
        if self.diag_eps is not None:
            diag_ixs = jnp.diag_indices(self.M)
            covs = covs.at[
                ..., diag_ixs[0], diag_ixs[1]
            ].add(self.diag_eps)
        return covs

    @staticmethod
    def cholesky_from_covariances(
        covariances: Float32[Array, "L M M"],
        diag_eps: float):
        # undo addition of diagonal before extracting cholesky
        if diag_eps is not None:
            diag_ixs = jnp.diag_indices(covariances.shape[-1])
            covariances = covariances.at[
                ..., diag_ixs[0], diag_ixs[1]
            ].add(-diag_eps)
        return extract_tril_cholesky(covariances) 

    def weights(self) -> Float[Array, "N L"]:
        return jnn.softmax(self.logits(), axis = -1)
    
    def pop_weights(self) -> Float[Array, "L"]:
        return jnn.softmax(self.pop_logits(), axis = -1)
    
    def logits(self) -> Float[Array, "N L"]:
        return GMMParameters.normalize_logits(self.trained_params.subj_weight_logits)
    def pop_logits(self) -> Float[Array, "N L"]:
        return GMMParameters.normalize_pop_logits(self.trained_params.pop_weight_logits)
    
    @staticmethod
    def normalize_logits(logits):
        centered = logits - logits.mean(axis = -1)[..., None]
        saturated = GMMTrainedParams.LOGIT_MAX * jnp.tanh(
            centered / GMMTrainedParams.LOGIT_MAX)
        return saturated
    
    @staticmethod
    def normalize_pop_logits(pop_logits):
        centered = pop_logits - pop_logits.mean(axis = -1)[..., None]
        saturated = GMMTrainedParams.LOGIT_MAX * jnp.tanh(
            centered / GMMTrainedParams.LOGIT_MAX)
        return saturated
    
    HyperparamClass = GMMHyperparams
    ParamClass = GMMTrainedParams


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
    params: GMMParameters,
    poses: Float[Array, "*#K M"],
    sess_ids: Integer[Array, "*#K M"]
    ) -> Tuple[Float[Array, "*#K L"]]:
    """
    Calculate the auxiliary distribution for EM.
    
    Args:
        observations: Observations from combined model.
        morph_matrix: Subject-wise linear transform from poses to
            keypoints.
        morph_ofs: Subject-wise affine component of transform from
            poses to keypoints.
        params, hyperparams: Parameters of the mixture model.

    Returns:
        probs: Array component probabilities at each data point.
    """

    pose_logits = pose_logprob(params, poses, sess_ids)
    return jnn.softmax(pose_logits, axis = -1)


def pose_logprob(
    params: GMMParameters,
    poses: Float[Array, "*#K M"],
    sess_ids: Integer[Array, "*#K"]
    ) -> Float[Array, "*#K L"]:

    norm_probs = tfp.distributions.MultivariateNormalFullCovariance(
        loc = params.means,
        covariance_matrix = params.covariances(),
    ).log_prob(poses[..., None, :])

    component_logprobs = jnp.log(params.weights())[sess_ids]
    
    return norm_probs + component_logprobs


def sample_hyperparams(
    rkey: PRNGKey,
    N: int,
    M: int,
    L: int,
    diag_eps: float,
    pop_weight_uniformity: float,
    subj_weight_uniformity: float) -> GMMHyperparams:
    return GMMHyperparams(
        N, M, L,
        diag_eps,
        pop_weight_uniformity,
        subj_weight_uniformity)
    


def sample_parameters(
    rkey: PRNGKey,
    hyperparams: GMMHyperparams,
    m_norm_center: Scalar,
    m_norm_spread: Scalar,
    q_var_center: Scalar,
    q_var_spread: Scalar,
    pi_logit_means: Float[Array, "N L"] = None,
    pi_logit_vars: Float[Array, "N L"] = None,
    ) -> GMMTrainedParams:
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
    Q_chol = GMMParameters.cholesky_from_covariances(Q, diag_eps = hyperparams.diag_eps)

    return GMMTrainedParams.create(
        pop_weight_logits = pop_weight_logits,
        subj_weight_logits = subj_weight_logits,
        means = m,
        cholesky = Q_chol)


def sample_poses(
    rkey: PRNGKey,
    params: GMMParameters,
    T: int
    ) -> GMMPoseStates:

    rkey = jr.split(rkey, 2)

    comp_weights = params.weights()
    z: Integer[Array, "N*T"] = jnp.stack([
        jr.choice(k, params.L, shape = (T,), replace = True, p = w)
        for k, w, in zip(jr.split(rkey[0], params.N), comp_weights)
        ]).flatten()

    x: Integer[Array, "N*T M"] = jr.multivariate_normal(
        rkey[2],
        params.means[z], params.covariances()[z],
    )

    subject_ids = jnp.broadcast_to(
        jnp.arange(params.N)[:, None],
        (params.N, T)).flatten()

    return GMMPoseStates(components = z, poses = x), subject_ids


def init_hyperparams(
    N: int,
    M: int,
    L: int,
    diag_eps: float,
    pop_weight_uniformity: float,
    subj_weight_uniformity: float) -> GMMHyperparams:
    return GMMHyperparams(
        N, M, L,
        diag_eps,
        pop_weight_uniformity,
        subj_weight_uniformity)


def init(
    hyperparams: GMMHyperparams,
    observations: Observations,
    poses: Float[Array, "Nt M"],
    reference_subject: int,
    seed: int = 0,
    count_eps: float = 1e-3,
    fit_to_all_subj: bool = False,
    subsample: float = False,
    cov_eigenvalue_eps = 1e-3,
    uniform = False
    ) -> Tuple[GMMTrainedParams]:
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
        uniform: bool
            Do not initialize cluster weights
    Returns:
        init_params: GMMHyperparams
            Initial parameters for a `GMMPoseSpaceModel`
    """

    # fit GMM to reference subject

    if not fit_to_all_subj:
        init_pts = observations.unstack(poses)[reference_subject]
    else: init_pts = poses

    if subsample is not False and subsample < 1:
        init_pts = jr.choice(
            jr.PRNGKey(seed),
            init_pts,
            (int(len(init_pts) * subsample),),
            replace = False)
    
    init_mix = mixture.GaussianMixture(
        n_components = hyperparams.L, 
        random_state = nr.RandomState(seed),
    )
    init_mix = init_mix.fit(init_pts)

    # get component labels & counts across all subjects
    if not uniform:
        init_components = observations.unstack(
            init_mix.predict(poses)) # List<subj>[(T,)]
        init_counts = np.zeros([hyperparams.N, hyperparams.L])
        for i_subj in range(hyperparams.N):
            uniq, count = np.unique(
                init_components[i_subj],
                return_counts = True)
            init_counts[i_subj][uniq] = count
        init_counts[init_counts == 0] = count_eps
    else:
        init_counts = np.ones([hyperparams.N, hyperparams.L])

    # Correct any negative eigenvalues
    # In the case of a non positive semidefinite covariance output by
    # the GMM fit (happens in moderate dimensionality), snap to the
    # nearest (in Frobenius norm) symmetric matrix with eigenvalues
    # not less than `cov_eigenvalue_eps`
    cov_vals, cov_vecs = jnp.linalg.eigh(init_mix.covariances_)
    if jnp.any(cov_vals < 0):
        clipped_vals = jnp.clip(cov_vals, cov_eigenvalue_eps)
        init_mix.covariances_ = (
            (cov_vecs[1] * clipped_vals[..., None, :]) @
            jnp.swapaxes(cov_vecs[1], -2, -1))
    return GMMTrainedParams(
        subj_weight_logits = jnp.log(init_counts),
        pop_weight_logits = jnp.log(init_counts[reference_subject]),
        means = jnp.array(init_mix.means_),
        cholesky = GMMParameters.cholesky_from_covariances(
            jnp.array(init_mix.covariances_),
            hyperparams.diag_eps)
    )


def discrete_mle(
    estimated_params: GMMParameters,
    poses: Float[Array, "Nt M"],
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
    ):
    # Heirarchical dirichlet prior on component weights
    pop_weights = params.pop_weights()
    pop_logpdf = tfp.distributions.Dirichlet(
        jnp.ones([params.L]) *
            params.pop_weight_uniformity /
            params.L,
    ).log_prob(pop_weights)
    subj_logpdf = tfp.distributions.Dirichlet( 
        params.subj_weight_uniformity *
        pop_weights
    ).log_prob(params.weights())

    return dict(
        pop_weight  = pop_logpdf,
        subj_weight = subj_logpdf,
    )
    

def reports(
    params: GMMParameters,
    ):
    return dict(
        priors = log_prior(params)
    )
    


GMMPoseSpaceModel = PoseSpaceModel(
    ParameterClass = GMMParameters,
    discrete_mle = discrete_mle,
    sample_poses = sample_poses,
    sample_parameters = sample_parameters,
    sample_hyperparams = sample_hyperparams,
    pose_logprob = pose_logprob,
    aux_distribution = aux_distribution,
    log_prior = log_prior,
    init_hyperparams = init_hyperparams,
    init = init,
    reports = reports
)



