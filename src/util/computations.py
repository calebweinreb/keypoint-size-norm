from jaxtyping import Float, Array
import jax.numpy as jnp

def quadform(
    a: Float[Array, "*#K M"],
    B: Float[Array, "*#K M M"]
    ) -> Float[Array, "*#K M M"]:
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



    

