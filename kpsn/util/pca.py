"""
Getting crashes using sklearn PCA on M1 chip - custom implementation.

For fitting with only a partial set of PCs, see scipy.sparse.linalg.svds
"""

from typing import NamedTuple, Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
import numpy as np

class PCAData(NamedTuple):
    n_fit: int
    s: Float[Array, "*#K n_feats"]
    v: Float[Array, "*#K n_feats n_feats"]

    def __getitem__(self, slc):
        return PCAData(self.n_fit, self.s[..., slc], self.v[..., slc])

    def pcs(self) -> Float[Array, "*#K n_feats n_feats"]:
        """
        Returns
        -------
        pcs: (..., n_components = n_feats, n_feats)
            Array of normalized principal components.
            The `i`th component vector is at position `i` along the
            second to last dimension.
        """
        return jnp.swapaxes(self.v, -2, -1)
    
    def variances(self):
        """
        Returns:
        vars: (..., n_feats)
            Variance explained by each principal component.
        """
        return self.s ** 2 / (self.n_fit)
    

    def coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        coords: (..., n_pts, n_feats)
            Coordinates in principal component space.
        Each row (second to last dimension) of the array is a vector
        that when right-multiplied by $V$, reconstructs the
        corresponding row of `arr`.
        """

        """
        Derivation:
        X = USV'
        columns of V are singular vectors
        coords in are column vectors to left-multiply by V
            or row vectors to right-multiply by V'
        CV' = X => C = XV = USV'V = US
        For a different matrix of observations, we do the same:
        CV' = A => AV
        """
        return arr @ self.v
    

    def whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Whitened coords have second-moment 1 alond each axis. 
        Returns:
        coords: (..., n_pts, n_feats)
            Coordinates in whitened principal component space.
        Each row (second to last dimension) of the array is a vector
        that when right-multiplied by $V$, reconstructs the
        corresponding row of `arr`.
        """

        """
        Derivation:
        lambda_i = s_i ** 2 / (n - 1) are variances of PCs
        sqrt(lambda_i) = s_i / sqrt(n - 1) are stddevs of PCs
        AV are coords for data matrix A of row-vector points
        diag(s_i / sqrt(n - 1))^{-1} @ AV gives whitened coords
        """
        norm = jnp.sqrt(self.n_fit)
        return self.coords(arr) / (self.s[..., None, :] / norm)
    

    def from_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        pts: (..., n_pts, n_feats)
            Points transformed back from PC space.
        
        Coordinate vector achieved via C = AV
        To retrieve data points, calculate A = CV'
        """
        return arr @ jnp.swapaxes(self.v, -2, -1)
        

    def from_whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        """
        Returns:
        pts: (..., n_pts, n_feats)
            Points transformed back from whitened PC space.
        
        Coordinate vector achieved via C = diag(s_i / sqrt(n - 1))^{-1} @ AV
        To retrieve data points, calculate A = diag(s_i / sqrt(n - 1)) @CV'
        """
        norm = jnp.sqrt(self.n_fit)
        vt = jnp.swapaxes(self.v, -2, -1)
        return (arr * (self.s[..., None, :] / norm)) @ vt
    
    

class CenteredPCA():
    def __init__(self, center, pcadata):
        self._pcadata = pcadata
        self._center = center
    
    def from_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.from_coords(arr) + self._center[..., None, :]
    
    def whitened_coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.whitened_coords(arr - self._center[..., None, :])
    
    def coords(self, arr: Float[Array, "*#K n_pts n_feats"]):
        return self._pcadata.coords(arr - self._center[..., None, :])


def fit(
    data: Float[Array, "*#K n_samples n_feats"],
    sign_correction: str = None,
    ) -> PCAData:
    """
    Parameters:
        centered: boolean
            Whether sample mean of the data is zero in all features.
            If it is not, then components will be given a canonical
            orientation (+/-) such that the mean of the data has positive
            coordinates."""
    
    cov = data.T @ data
    _, s2, vt = np.linalg.svd(cov)

    if sign_correction is not None:
        if sign_correction == 'mean':
            # coords for mean of data
            standard_vec = data.mean(axis = -2)
        if sign_correction == 'ones':
            standard_vec = jnp.ones(data.shape[:-2] + (data.shape[-1],))
        # standard_vec: (..., n_components)
        # coord_directions: (..., n_components)
        coord_directions = jnp.sign(
            standard_vec[..., None, :] @ jnp.swapaxes(vt, -2, -1)
        )[..., 0, :]
        # coords for vector of ones
        # flip PCs acoording to sign of mean coords
        # (..., n_components, n_feats)
        vt = coord_directions[..., :, None] * vt
    
    return PCAData(
        data.shape[-2],
        np.sqrt(s2),
        jnp.array(jnp.swapaxes(vt, -2, -1)))


def fit_with_center(
    data: Float[Array, "*#K n_samples n_feats"],
    sign_correction: str = None,
    ) -> CenteredPCA:
    center = data.mean(axis = -2)
    pcs = fit(
        data - center[..., None, :],
        sign_correction = sign_correction)
    return CenteredPCA(center, pcs)


def second_moment(arr: Float[Array, "*#K n_samples n_feats"]):
    return jnp.swapaxes(arr, -2, -1) @ arr / (arr.shape[-2])