
from scipy import spatial
from scipy import special
import numpy as np
import logging
import tqdm

def ball_volume(r, d):
    """Volume of a d-dimensional ball of radius r"""
    normalizer = np.pi ** (d / 2) / special.gamma(d / 2 + 1)
    return normalizer * (r ** d)


def inverse_ball_volume(v, d):
    """Radius of a d-dimensional ball with volume v"""
    normalizer = np.pi ** (d / 2) / special.gamma(d / 2 + 1)
    return (v / normalizer) ** (1/d)


def max_independent_set(data, distance, tree = None):
    """Select a subset of points such that no two points are closer than distance
    
    Parameters
    ----------
    data : array_like (n, d)
        Points to select from
    distance : float
        Minimum distance between points in the subset
    tree : scipy.spatial.KDTree (optional)
        Precomputed KDTree of data
    
    Returns
    -------
    mask : array_like (n,)
        Boolean mask of selected points
    new_data : array_like (n', d)
        Selected points
    weights : array_like (n,)
        Normalized number of points originally within distance of each selected
        point.
    """
    if tree is None:
        tree = spatial.KDTree(data)
    
    selected_ixs = []
    ix_counts = []
    remaining_ixs = np.arange(len(data))
    while len(remaining_ixs) > 0:
        # select a point
        ix = remaining_ixs[0]
        remaining_ixs = remaining_ixs[1:]
        # find all points within distance
        neighbors = tree.query_ball_point(data[ix], distance)
        # remove them from the remaining points
        # remaining_ixs = np.setdiff1d(remaining_ixs, neighbors)
        to_keep = np.isin(remaining_ixs, neighbors,
                          invert = True, assume_unique = True,)
        remaining_ixs = remaining_ixs[to_keep]
        # add the point to the selected points
        selected_ixs.append(ix)
        ix_counts.append(len(neighbors) + 1)
    
    mask = np.zeros(len(data), dtype = bool)
    mask[selected_ixs] = True
    weights = np.array(ix_counts) / np.sum(ix_counts)
    return mask, data[mask], weights


class PointCloudDensity:
    """Density estimation in a point cloud using a kdtree"""

    def __init__(self, k, eps = 1e-10, distance_eps = None):
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
        self._distance_eps = distance_eps
        self._sparse_pts = None
        self._sparse_pt_weights = None
        self._sparse_pts_mask = None


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
    

    def pdf_to_radius(self, pdf):
        """Convert pdf value to knn radius that would produce it"""
        volumes = self._n * pdf / (self.k - 1) - self._eps
        return inverse_ball_volume(volumes, self._d)
        
    

    def measure(self, func, return_evals = False):
        """Expectation of a function in the density.

        Queries func for each point in the cloud and computes the mean. If
        distance_eps is not None, the expectation is computed over a reduced
        set of points.
    
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
            Expectation of func in the density
        mask : array_like (n_fit_pts,)
            Boolean mask of queried points
        evaluations : array_like (n_queried,)
            Values of func at the query points
        weights : array_like (n_queried,)
            Normalized weight of each queried point in the expectation.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        # optionally reduce the number of points
        # or use precomputed reduction
        if self._distance_eps is not None:
            self._compute_sparse_set()
            query_points = self._sparse_pts
            weights = self._sparse_pt_weights
            mask = self._sparse_pts_mask
        else:
            query_points = self._tree.data
            weights = np.full(len(query_points), 1 / len(query_points))
            mask = np.ones(len(query_points), dtype = bool)
        
        # evaluate function at each point
        evaluations = func(query_points)

        # compute expectation and format return
        ret = (np.sum(evaluations * weights),)
        if return_evals:
            ret = ret + (mask, evaluations, weights)
        return ret
    
    def _compute_sparse_set(self, recompute = False):
        """Compute a sparse set of points to query for expectation
        computation"""
        if ((self._sparse_pts is None or recompute)
            and self._distance_eps is not None):
            maxind = max_independent_set(self._tree.data, self._distance_eps)
            self._sparse_pts_mask = maxind[0]
            self._sparse_pts = maxind[1]
            self._sparse_pt_weights = maxind[2]


def cloud_kl(
    cloud_a,
    cloud_b,
    cloud_a_pdf,
    cloud_b_pdf = None,
    ):
    """Compute pairwise KL(a, b) between PointCloudDensity estimators
    
    Parameters
    ----------
    cloud_a, cloud_b : PointCloudDensity
        Point cloud density estimators
    cloud_a_pdf : array_like (len(cloud_a.data),)
        Precomputed densities of cloud_a's max independent set
    distance_eps : float
        Distance for max independent set reduction. If None, no reduction is
        performed.
    """
    _, _, b_pdf, weights = cloud_a.measure(
        cloud_b.predict, return_evals = True)
    relative_surprise = np.log(b_pdf / cloud_a_pdf)
    return -np.sum(relative_surprise * weights)


def cloud_js(cloud_a, cloud_b, cloud_a_pdf = None, cloud_b_pdf = None):
    """
    Compute the mixture distrubution from two PointCloudDensity estimators
    
    Parameters
    ----------
    cloud_a, cloud_b : PointCloudDensity
        Point cloud density estimators
    cloud_a_pdf, cloud_b_pdf : array_like (n_pts,)
        Precomputed densities at cloud's own points
    """

    if cloud_a_pdf is None:
        cloud_a_pdf = cloud_a.measure(cloud_a.predict, return_evals = True)[2]
    if cloud_b_pdf is None:
        cloud_b_pdf = cloud_b.measure(cloud_b.predict, return_evals = True)[2]

    # probability of each cloud at the other cloud's points
    _, _, prob_b_at_a, a_pt_weights = cloud_a.measure(
        cloud_b.predict, return_evals = True)
    _, _, prob_a_at_b, b_pt_weights = cloud_b.measure(
        cloud_a.predict, return_evals = True)
    
    # mixture logprob at sample points of cloud_a and cloud_b
    # logmix_at_a = np.log(( prob_b_at_a) )
    # logmix_at_b = np.log(( prob_a_at_b) )
    logmix_at_a = np.log((cloud_a_pdf + prob_b_at_a) / 2)
    logmix_at_b = np.log((cloud_b_pdf + prob_a_at_b) / 2)
    
    # kl divergences
    kl_a_to_mix = np.sum((np.log(cloud_a_pdf) - logmix_at_a) * a_pt_weights)
    kl_b_to_mix = np.sum((np.log(cloud_b_pdf) - logmix_at_b) * b_pt_weights)
    
    return 0.5 * (kl_a_to_mix + kl_b_to_mix)


def predict_mixture(cloud_a, cloud_b, x, a_dists = None, b_dists = None):

    k = cloud_a._k
    if k != cloud_b._k: print("density.predict_mixture: k mismatch!")
    if cloud_a._n != cloud_b._n: print("density.predict_mixture: n mismatch!")

    # distance to k nearest neighbors for each cloud
    if a_dists is None: a_dists = cloud_a._tree.query(x, k = k)[0]
    if b_dists is None: b_dists = cloud_b._tree.query(x, k = k)[0]

    # k nearest out of these 2k neighbors
    all_dists = np.concatenate([a_dists, b_dists], axis = -1)
    kth_dist_mix = np.partition(all_dists, [k - 1, k], axis = -1)[:, k-1]
    
    return a_dists[:, -1], b_dists[:, -1], kth_dist_mix


def entropic_jsd(cloud_a, cloud_b):
    a_dists, _, mix_dists_a = predict_mixture(cloud_a, cloud_b, cloud_a._tree.data)
    _, b_dists, mix_dists_b = predict_mixture(cloud_a, cloud_b, cloud_b._tree.data)
    mix_normalizer = (cloud_a._k - 1) / (cloud_a._n + cloud_b._n)
    pdf_a = (cloud_a._k - 1) / cloud_a._n / ball_volume(a_dists, 1)
    pdf_b = (cloud_b._k - 1) / cloud_b._n / ball_volume(b_dists, 1)
    mix_pdf_a = mix_normalizer / ball_volume(mix_dists_a, 1)
    mix_pdf_b = mix_normalizer / ball_volume(mix_dists_b, 1)
    a_entropy = -np.mean(np.log(pdf_a))
    b_entropy = -np.mean(np.log(pdf_b))
    mix_entropy = -np.mean(np.log(np.concatenate([mix_pdf_a, mix_pdf_b])))
    print(mix_entropy, a_entropy, b_entropy)
    return mix_entropy - 0.5 * (a_entropy + b_entropy)


def pairwise_mixture_logprob(cloud_a, cloud_b, cloud_a_pdf, cloud_b_pdf):
    """Relative surprise distribution of each cloud against their mixture.
    
    This computes the two distributions that are averaged to arrive at the JS
    divergence.
    """
    # probability of each cloud at the other cloud's points
    prob_b_at_a = np.log(cloud_b.predict(cloud_a._tree.data))
    prob_a_at_b = np.log(cloud_a.predict(cloud_b._tree.data))
    return prob_b_at_a, prob_a_at_b, cloud_a_pdf, cloud_b_pdf



def pairwise_cloud_jss(clouds, progress = False):
    """Compute pairwise JS divergence between PointCloudDensity estimators
    
    Parameters
    ----------
    clouds : list[PointCloudDensity]
        Point cloud density estimators
    """
    ret = np.zeros((len(clouds), len(clouds)))
    
    # precompute densities of a cloud's own points
    own_densities = zip(*[
        cloud.measure(cloud.predict, return_evals = True)
        for cloud in clouds])

    # optionally show progress bar
    if progress:
        pbar = lambda: tqdm.tqdm(total = len(clouds) ** 2)
    else: pbar = lambda: None
    
    # Compute js distances
    with tqdm.tqdm(total = len(clouds) ** 2) as pbar:
        for i, cloud_a in enumerate(clouds):
            for j, cloud_b in enumerate(clouds):
                pbar.update(1)
                if j >= i:
                    continue
                ret[i, j] = pairwise_cloud_js(
                    cloud_a, cloud_b, own_densities[i], own_densities[j])
                
    return ret


def pairwise_cloud_metric(
    clouds,
    metric,
    progress = False,
    symmetric = True):
    
    # precompute densities of a cloud's own points
    own_densities = [
        cloud.measure(cloud.predict, return_evals = True)[2]
        for cloud in clouds]

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
                
                metric_val = metric(
                    cloud_a, cloud_b, own_densities[i], own_densities[j])
                ret[i][j] = metric_val

                pbar.update(1)
                
    return ret