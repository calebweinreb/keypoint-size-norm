from typing import Tuple, Optional
from jaxtyping import Float, Array, Integer, PRNGKeyArray as PRNGKey
import jax.numpy as jnp
import jax.random as jr
import jax

def quadform(
    a: Float[Array, "*#K M"],
    B: Float[Array, "*#K M M"]
    ) -> Float[Array, "*#K"]:
    """
    Batched computation of `a^T B a`.

    :param a: Batch of vectors input to the quatradic form defined by `B`.
    :param B: Batch of symmetric matrices defining the quadratic forms.
    """
    # a: (K, M)
    # B: (K, M, M)
    # a[..., None, :] @ B: (K, 1, M) stacked row vectors a_k^T B
    # ($_[..., 0, :] * a).sum(axis = -1), dot product of each such row with a_k
    return ((a[..., None, :] @ B)[..., 0, :] * a).sum(axis = -1)



def gaussian_product(
    a: Float[Array, "*#K M"],
    A: Float[Array, "*#K M M"],
    b: Float[Array, "*#K M"],
    B: Float[Array, "*#K M M"],
    Ainv: Optional[Float[Array, "*#K M M"]] = None,
    Binv: Optional[Float[Array, "*#K M M"]] = None,
    return_normalizer: bool = False,
    ) -> Tuple[Float[Array, "*#K"],
               Float[Array, "*#K M"],
               Float[Array, "*#K M M"]]:
    """
    Batched computation of Gaussian PDF product.
    
    Computes K, c, and C in the equation
    $$
    K N(x; c, C) = N(x; a, A) N(x; b, B)
    $$

    Args:
        a: Mean of the first Gaussian PDF term.
        A: Covariance of the first Gaussian PDF term.
        b: Mean of the second Gaussian PDF term.
        B: Covariance of the second Gaussian PDF term.
        Ainv: Optional precomputed inverse of A.
        Binv: Optional precomputed inverse of B.
        return_normalizer: Do not compute Gaussian normalizer terms for `const`.
    
    Returns:
        const: Normalization constant for the resulting expression.
        c: Mean of the resulting Gaussian PDF.
        C: Covariance of the resulting Gaussian PDF.
    """
    if Ainv is None: Ainv = jnp.linalg.inv(A)
    if Binv is None: Binv = jnp.linalg.inv(B)

    sum_inv = jnp.linalg.inv(A + B)
    C = A @ sum_inv @ B
    Cinv = Ainv + Binv
    c = (B @ sum_inv @ a[..., None] + A @ sum_inv @ b[..., None])[..., 0]

    K = jnp.exp((
        quadform(a, Ainv) + quadform(b, Binv) - quadform(c, Cinv)
    ) / (-2))

    if return_normalizer:
        normalizer = jnp.sqrt(
            jnp.linalg.det(C) / jnp.linalg.det(A) / jnp.linalg.det(B) *
            (2 * jnp.pi) ** (C.shape[-1] - A.shape[-1] - B.shape[-1])
        )
        return K, c, C, normalizer
    
    return K, c, C


def normal_quadform_expectation(
    a: Float[Array, "*#K M"],
    A: Float[Array, "*#K M M"],
    b: Float[Array, "*#K M"],
    Binv: Float[Array, "*#K M M"],):
    r"""
    Compute expextation of a quadratic form in a normal RV.

    $$
    E_{x~\mathcal{N}(a, A)}\left[ (x - b)^TB^{-1}(x - b) \right]
    $$

    Args:
        a: Mean of the normal random variable.
        A: Covariance matrix of the normal random varable.
        b: Center of the quadratic form.
        B: Inverse of the matrix of the quadratic form.

    Returns:
        avg: Expectation of the quatratic form.
    """

    A, Binv = jnp.broadcast_arrays(A, Binv)
    orig_shp = A.shape
    tr = jnp.einsum(
        'kij,kji->k',
        A.reshape((-1,) + orig_shp[-2:]),
        Binv.reshape((-1,) + orig_shp[-2:])
    ).reshape(orig_shp[:-2])
    
    return quadform(a - b, Binv) + tr







def expand_tril(
    tril_values: Float[Array, "*#K n*(n+1)/2"],
    n: int):

    """
    Lower-triangular flat-form to dense lower-triangular
    """

    tmp = jnp.zeros(tril_values.shape[:-1] + (n, n))
    tril = jnp.tril_indices(n)
    return tmp.at[..., tril[0], tril[1]].set(tril_values)




def expand_tril_cholesky(
    tril_values: Float[Array, "*#K n*(n+1)/2"],
    n: int
    ) -> Float[Array, "*#K n n"]:

    """
    Lower-triangular flat-form cholesky decomposition to positive-definite
    """

    tmp = jnp.zeros(tril_values.shape[:-1] + (n, n))
    tril = jnp.tril_indices(n)
    L = tmp.at[..., tril[0], tril[1]].set(tril_values)
    return L @ jnp.swapaxes(L, -2, -1)


def extract_tril_cholesky(
    A: Float[Array, "*#K n n"],
    ) -> Float[Array, "*#K n(n+1)/2"]:

    """Lower-triangular flat-form cholesky decomposition of positive-definite"""

    L = jnp.linalg.cholesky(A)
    n = L.shape[-1]
    tril = jnp.tril_indices(n)
    return L[..., tril[0], tril[1]]



    
def unstack(
    arr,
    ixs,
    N = None,
    axis = 0
    ) -> Tuple:
    """
    Convert stacked axis to list of original axes.

    Args:
        arr: Array-like (Na..., Nt, Nb...) Array to split by subject.
        ixs: Indices in the unstacked array.
        N: Number of unique indices.
            If not given, then will be inferred from `ixs`.
        axis: Axis along which to split `arr`

    Returns:
        arrs: Arrays (Na..., T, Nb...) for each subject.
    """
    if N is None: N = ixs.max() + 1
    return tuple(
        jnp.take(arr, jnp.where(ixs == i)[0], axis = axis)
        for i in range(N))


def restack(
    arrs,
    axis = 0
    ) -> Array:
    """
    Convert unstacked array back to stacked.
    
    Equivalent to flatten, but reliably mimicks unstack

    To get `ixs`, perform
        restack(broadcast_to(arange(|A|)[:, None]), [|A|, |B|])
    or similar for |B|.
    
    Args:
        arrs: Tuple-like of array-like, shape List[(..., b_i, ...)]
            Arrays to be stacked. All dimensions must match except `axis`.
        axis: int
            Axis along which to stack the arrays

    Returns:
        stacked: Array-like of shpe (sum b_i, ...)
    """
    return jnp.concatenate(arrs, axis = axis)


def stack_ixs(
    arrs
    ) -> Array:
    """
    Convert unstacked list of arrays to stacked array of indices.
    
    This function is the other defines an inverse pair between to `restack` and
    `unstack`. In particular,
        unstack(restack(arrs), stack_ixs(arrs)) == arrs
    """
    return restack([
        jnp.broadcast_to(jnp.array([i]), len(arr))
        for i, arr in enumerate(arrs)])


def unstacked_ixs(
    ixs,
    N: int = None
    ) -> Tuple[Integer[Array, 'M']]:
    """
    Convert stacked array of indices to unstacked list of of indices
    into stacked array.
    """
    return unstack(jnp.arange(len(ixs)), ixs, N = N)


def stacked_batch(
    rkey: PRNGKey,
    unstacked_ixs: Tuple[Integer[Array, "M"]],
    batch_size: int,
    replace: bool = False,
    ) -> Integer[Array, "n_stacked*batch_size"]:
    """
    Sample stacked batch indices from a list of index arrays
    """
    return restack([
        jr.choice(rkey, ixs, (batch_size,), replace = replace)
        for ixs in unstacked_ixs])


def linear_transform_gaussian(
    query_point: Float[Array, "*#K M"],
    cov: Float[Array, "*#K M M"],
    A: Float[Array, "*#K M M"],
    d: Float[Array, "*#K M"] = None,
    Ainv: Float[Array, "*#K M M"] = None,
    cov_inv: Float[Array, "*#K M M"] = None,
    return_cov_inv = False,
    return_normalizer = False,
    ) -> Tuple[Float[Array, "*#K M"], Float[Array, "*#K M M"]]:
    """
    Precompute the necessary values for $f(x)$ to be computed as a normal PDF
    in $x$:
    f(x) = N(query_point | A(x - d), cov)
         = normalizer * N(x | Ainv @ query_point + d, Ainv @ cov @ Ainv^T)

    In particular:
    new_mean = Ainv @ query_point + d
    new_cov = Ainv @ cov @ Ainv^T
    normalizer = |cov| / |Ainv @ cov @ Ainv^T|
    """
    
    if Ainv is None: Ainv = jnp.linalg.inv(A)

    if d is not None:
        new_mean = (Ainv @ (query_point - d)[..., None])[..., 0]
    else:
        new_mean = (Ainv @ query_point[..., None])[..., 0]
    new_cov = Ainv @ cov @ jnp.swapaxes(Ainv, -2, -1)
    
    if return_cov_inv:
        if cov_inv is None: cov_inv = jnp.linalg.inv(cov)
        new_cov_inv = jnp.swapaxes(A, -2, -1) @ cov_inv @ A
        
        if return_normalizer:
            normalizer = (jnp.linalg.det(new_cov) / jnp.linalg.det(cov)) ** 0.5
            return new_mean, new_cov, new_cov_inv, normalizer
        
        return new_mean, new_cov, new_cov_inv

    if return_normalizer:
        normalizer = (jnp.linalg.det(new_cov) / jnp.linalg.det(cov)) ** 0.5
        return new_mean, new_cov, normalizer
    
    return new_mean, new_cov


def sq_mahalanobis(
    x: Float[Array, "*#K M"],
    y: Float[Array, "*#K M"],
    cov: Optional[Float[Array, "*#K M M"]] = None,
    cov_inv: Optional[Float[Array, "*#K M M"]] = None
    ):
    """
    Args:
        x, y: Array-like, shape (..., M)
            Batch of vectors do calculate distances between
        cov: Array-like, PSD, shape (..., M)
            Covariance matrix defining the Mahalanobis distance.
        cov_inv: Array-like, shape (..., M)
            Inverse covariance matrix. Required if `cov` is not provided.
    Returns:
        dists: Array-like, shape (...)
            Scalar squared distances.
    """
    diff = x - y
    assert cov is not None or cov_inv is not None, (
        "One of `cov` or `cov_inv` required.")
    if cov_inv is None:
        cov_inv = jnp.linalg.inv(cov)
    return (diff[..., None, :] @ cov_inv @ diff[..., :, None])[..., 0, 0]