
from scipy import spatial
from scipy import special
import numpy as np
import logging
import tqdm


def ball_volume(r, d):
    """Volume of a d-dimensional ball of radius r"""
    normalizer = np.pi ** (d / 2) / special.gamma(d / 2 + 1)
    return normalizer * (r ** d)


class PointCloudDensity:
    """Density estimation in a point cloud using a kdtree"""

    def __init__(self, k, eps = 1e-10):
        """Initialize a point cloud density estimator
        
        Parameters
        ----------
        k : int
            k-th nearest neighbor to use for density estimation
        eps : float
            Small number to avoid division by zero.
        distance_eps : float
            Distance for max independent set reduction when querying a function
            to be averaged. If None, no reduction is performed.
        """
        self._k = k
        self.is_fitted = False
        self._eps = eps
        self._pdf = None
        self._internal_dist = None


    def fit(self, data):
        """Fit a point cloud density estimator"""
        self._tree = spatial.KDTree(data)
        self._n = len(data)
        self._d = data.shape[-1]
        self.is_fitted = True
        return self
    

    def predict(self, x):
        """Density estimation in a point cloud with pre-prepocessed kdtree
    
        Parameters
        ----------
        x : array_like (n, d)
            Points to estimate density for
        
        Returns
        -------
        densities : array_like (n,)
            Estimated densities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        distances, _ = self._tree.query(x, self._k)
        distances = distances[:, -1]
        volumes = ball_volume(distances, self._d)
        if (np.mean(volumes / self._eps) < 1e-3) > 0.5:
            logging.warn("More than half of measured densities are smaller " +
                         "than epsilon. Consider increasing.")
        return (self._k - 1) / (self._n * (volumes + self._eps))
    

    def normalized_distance(self, x):
        """Normalized distance measure to nearest points in the cloud
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        distances = self._tree.query(x, self._k)[0][:, -1]
        return distances / self._n
    

    def measure(self, func):
        """Evaluate func at points in the cloud, for example to compute expectation

        todo: maximal independent set reduction and weighted average
    
        Parameters
        ----------
        func : callable[array]
            Function to measure. Should take a single argument x of shape (n, d)
            and return a scalar.
        return_evals : bool
            If True, return the values of func at each queried point and weights
            of the queried points in the expectation.
        
        Returns
        -------
        expectation : float
        evaluations : array_like (n_queried,)
            Values of func at the query points
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # evaluate function at each point
        query_points = self._tree.data        
        evaluations = func(query_points)
        return evaluations
    

    def __len__(self):
        return self._n
    

    def pdf(self):
        """Density at each point in the cloud
        
        Returns
        -------
        pdf : array_like (n_pts,)
            Densities
        """
        if self._pdf is None:
            self._pdf = self.measure(self.predict)
        return self._pdf
    
    def internal_distances(self):
        if self._internal_dist is None:
            self._internal_dist = self.measure(self.normalized_distance)
        return self._internal_dist


def cloud_kl(
    cloud_a,
    cloud_b,
    ):
    """Compute pairwise KL(a, b) between PointCloudDensity estimators
    
    Parameters
    ----------
    cloud_a, cloud_b : PointCloudDensity
        Point cloud density estimators
    """
    cloud_a_pdf = cloud_a.pdf()
    prob_b_at_a = cloud_a.measure(cloud_b.predict)
    relative_surprise = np.log(prob_b_at_a / cloud_a_pdf)
    return -np.mean(relative_surprise)


def cloud_js(cloud_a, cloud_b):
    """
    Compute the mixture distrubution from two PointCloudDensity estimators
    
    Parameters
    ----------
    cloud_a, cloud_b : PointCloudDensity
        Point cloud density estimators
    cloud_a_pdf, cloud_b_pdf : array_like (n_pts,)
        Precomputed densities at cloud's own points
    """

    cloud_a_pdf = cloud_a.pdf()
    cloud_b_pdf = cloud_b.pdf()

    # probability of each cloud at the other cloud's points
    prob_b_at_a = cloud_a.measure(cloud_b.predict)
    prob_a_at_b = cloud_b.measure(cloud_a.predict)

    # normalizers
    prob_a_normalizer = len(prob_a_at_b) / (prob_a_at_b / cloud_b_pdf).sum()
    prob_b_normalizer = len(prob_b_at_a) / (prob_b_at_a / cloud_a_pdf).sum()
    # prob_a_at_b *=  prob_a_normalizer
    # prob_b_at_a *=  prob_b_normalizer
    
    # mixture logprob at sample points of cloud_a and cloud_b
    logmix_at_a = np.log((cloud_a_pdf + prob_b_at_a) / 2)
    logmix_at_b = np.log((cloud_b_pdf + prob_a_at_b) / 2)
    
    # kl divergences
    kl_a_to_mix = np.mean((np.log(cloud_a_pdf) - logmix_at_a))
    kl_b_to_mix = np.mean((np.log(cloud_b_pdf) - logmix_at_b))
    
    return 0.5 * (kl_a_to_mix + kl_b_to_mix)


def pairwise_cloud_metric(
    clouds,
    metric,
    progress = False,
    symmetric = True):

    # optionally show progress bar
    if progress:
        pbar = lambda: tqdm.tqdm(total = len(clouds) ** 2)
    else: pbar = lambda: None
    n = len(clouds)
    if symmetric: _pbar = tqdm.tqdm(total = int((n * (n - 1)) / 2))
    else: _pbar = tqdm.tqdm(total = n * (n - 1))


    # Compute js distances
    ret = np.zeros((len(clouds), len(clouds),)).tolist()
    with _pbar as pbar:
        for j, cloud_b in enumerate(clouds):
            for i, cloud_a in enumerate(clouds):
                
                if j == i: continue
                if j >= i and symmetric:
                    ret[i][j] = ret[j][i]
                    continue
                
                metric_val = metric(cloud_a, cloud_b)
                ret[i][j] = metric_val

                pbar.update(1)
                
    return ret


def ball_cloud_js(cloud_a, cloud_b):
    a = cloud_a
    b = cloud_b
    
    a_dists = a.internal_distances() * a._n
    pdf_a = a._k / a._n / ball_volume(a_dists, a._d)
    count_b_at_a = b._tree.query_ball_point(a._tree.data, a_dists, return_length = True)
    pdf_b_at_a = count_b_at_a / a._n / ball_volume(a_dists, a._d)

    b_dists = b.internal_distances() * b._n
    pdf_b = b._k / b._n / ball_volume(b_dists, b._d)
    count_a_at_b = a._tree.query_ball_point(b._tree.data, b_dists, return_length = True)
    pdf_a_at_b = count_a_at_b / b._n / ball_volume(b_dists, b._d)
    
    kl_a_to_mix = np.log(pdf_a).mean() - np.log(0.5 * (pdf_b + pdf_a_at_b)).mean()
    kl_b_to_mix = np.log(pdf_b).mean() - np.log(0.5 * (pdf_a + pdf_b_at_a)).mean()

    return 0.5 * (kl_a_to_mix + kl_b_to_mix)
