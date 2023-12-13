import jax.numpy as jnp
from jax import vmap
from jax.scipy.linalg import solve_triangular
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class InverseWishart(tfd.TransformedDistribution):

    def __new__(cls, *args, **kwargs):
        # Patch for tfp 0.18.0.
        # See https://github.com/tensorflow/probability/issues/1617
        return tfd.Distribution.__new__(cls)

    def __init__(self, df, scale):
        r"""Implementation of an inverse Wishart distribution as a transformation of
        a Wishart distribution. This distribution is defined by a scalar degrees of
        freedom `df` and a scale matrix, `scale`.

        #### Mathematical Details
        The probability density function (pdf) is,
        ```none
        pdf(X; df, scale) = det(X)**(-0.5 (df+k+1)) exp(-0.5 tr[inv(X) scale]) / Z
        Z = 2**(0.5 df k) Gamma_k(0.5 df) |det(scale)|**(-0.5 df)
        ```

        where
        * `df >= k` denotes the degrees of freedom,
        * `scale` is a symmetric, positive definite, `k x k` matrix,
        * `Z` is the normalizing constant, and,
        * `Gamma_k` is the [multivariate Gamma function](
            https://en.wikipedia.org/wiki/Multivariate_gamma_function).

        Args:
            df (_type_): _description_
            scale (_type_): _description_
        """
        self._df = df
        self._scale = scale
        # Compute the Cholesky of the inverse scale to parameterize a
        # Wishart distribution
        dim = scale.shape[-1]
        eye = jnp.broadcast_to(jnp.eye(dim), scale.shape)
        cho_scale = jnp.linalg.cholesky(scale)
        inv_scale_tril = solve_triangular(cho_scale, eye, lower=True)

        # model p(y) as y = g(x) and p(x)
        # x ~ WishartTril
        # y = [Cholesky->Standard]( CholeskyInv( [Standard->Cholesky]( x )))
        super().__init__(
            tfd.WishartTriL(df, scale_tril=inv_scale_tril),
            tfb.Chain([tfb.CholeskyOuterProduct(),
                       tfb.CholeskyToInvCholesky(),
                       tfb.Invert(tfb.CholeskyOuterProduct())]))

        self._parameters = dict(df=df, scale=scale)

    @classmethod
    def _parameter_properties(self, dtype, num_classes=None):
        return dict(
            # Annotations may optionally specify properties, such as `event_ndims`,
            # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
            # the `ParameterProperties` documentation for details.
            df=tfp.util.ParameterProperties(event_ndims=0),
            scale=tfp.util.ParameterProperties(event_ndims=2))

    @property
    def df(self):
        return self._df

    @property
    def scale(self):
        return self._scale

    def _mean(self):
        dim = self.scale.shape[-1]
        df = jnp.array(self.df)[..., None, None]  # at least 2d on the right
        assert self.df > dim + 1, "Mean only exists if df > dim + 1"
        return self.scale / (df - dim - 1)

    def _mode(self):
        dim = self.scale.shape[-1]
        df = jnp.array(self.df)[..., None, None]  # at least 2d on the right
        return self.scale / (df + dim + 1)

    def _variance(self):
        """Compute the marginal variance of each entry of the matrix.
        """

        def _single_variance(df, scale):
            assert scale.ndim == 2
            assert df.shape == scale.shape
            dim = scale.shape[-1]
            diag = jnp.diag(scale)
            rows = jnp.arange(dim)[:, None].repeat(3, axis=1)
            cols = jnp.arange(dim)[None, :].repeat(3, axis=0)
            numer = (df - dim + 1) * scale**2 + (df - dim - 1) * diag[rows] * diag[cols]
            denom = (df - dim) * (df - dim - 1)**2 * (df - dim - 3)
            return numer / denom

        dfs, scales = jnp.broadcast_arrays(jnp.array(self.df)[..., None, None], self.scale)
        if scales.ndim == 2:
            return _single_variance(dfs, scales)
        else:
            return vmap(_single_variance)(dfs, scales)