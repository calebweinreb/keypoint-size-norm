from jaxtyping import Float, Array
from typing import Tuple, Iterable
from sklearn import mixture
import numpy as np

def gaussian_mle(
    x: Iterable[Float[Array, "N M"]]
    ) -> Tuple[Iterable[Float[Array, "M"]], Iterable[Float[Array, "M M"]]]:
    """
    Compute mean and covariance of Gaussian MLE for multiple samples.
    
    Args:
        x: List, tuple, or array-like of samples
    
    Returns:
        means: Array of mean vectors.
        covs: Array of covariance matrices.
    """
    M = x[0].shape[-1]
    means = np.empty([len(x), M])
    covs = np.empty([len(x), M, M])
    for i in range(len(x)):
        gm = mixture.GaussianMixture(n_components = 1,).fit(x[i])
        means[i] = gm.means_
        covs[i] = gm.covariances_
    return means, covs