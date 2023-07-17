from typing import Tuple, Optional
from jaxtyping import Float, Array
import jax.numpy as jnp
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
    unnormalized: bool = False,
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
        unnormalized: Do not compute Gaussian normalizer terms in `const`.
    
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

    if not unnormalized:
        K *= jnp.sqrt(
            jnp.linalg.det(C) / jnp.linalg.det(A) / jnp.linalg.det(B) *
            (2 * jnp.pi) ** (C.shape[-1] - A.shape[-1] - B.shape[-1])
        )
    
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



    

