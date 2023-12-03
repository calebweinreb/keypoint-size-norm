
import sklearn.cluster
import sklearn.preprocessing
import numpy as np
import jax.random as jr

from kpsn.util.keypt_io import apply_across_flat_array


def train_clusters(n_clusters, features, seed = None, whiten = True):
    """
    features: (n_samp, n_feat)
    seed: int, RandomState or None
    Returns:
    - clusterer: tuple(StandardScaler, KMeans)"""
    
    clusters = sklearn.cluster.KMeans(
        n_clusters, n_init = 'auto',
        random_state = seed)
    return clusters.fit(features)


def masks_and_logits(all_labels, N):
    """
    all_labels: int (n_samp,)
    N: int
        number of clusters
    Returns
    -------
    label_ixs: list(int (n_in_clust,))
        Indices in each cluster
    counts: int (N,)
    logits: float (N,)
    """
    label_ixs = [np.where(all_labels == c)[0] for c in range(N)]
    counts = np.array([l.size for l in label_ixs])
    logits = np.log(counts / np.sum(counts))
    return label_ixs, counts, logits


def max_resamp(arr, group_ixs, logits, temperature, seed = 0):
    """
    Generate new arrays resampled such that each group is predominant in one
    session.
    
    arr: (n_samp, *feature_dims)
    group_ixs: list[int (n_in_clust,)]
    logits: list[float (n_in_clist,)]
    temperature: float
        Factor by which predominant group should outweigh others in log scale
    seed: int

    Returns
    -------
    data: dict[str, (n_new_samp, *feature_dims)]
    samp_counts[str, int (n_clust)]
    samp_logits[str, int (n_clust)]
    """
    L = logits.size
    samp_data = {}
    samp_counts = {}
    samp_logits = {}
    rngk = jr.PRNGKey(seed)

    for i in range(L):
        new_logits = logits.copy()
        new_logits[i] = logits.max() + temperature
        new_counts = (np.exp(new_logits) / np.exp(new_logits).sum() * len(arr)
                      ).astype('int')
        rngk, rngk_use = jr.split(rngk, 2)
        samp_data[f'm{i}'] = np.concatenate([
            jr.choice(rngk_use, arr[group_ixs[i]], (new_counts[c],), replace = True)
            for c in range(L)
        ])
        samp_counts[f'm{i}'] = new_counts
        samp_logits[f'm{i}'] = new_logits

    return samp_data, samp_counts, samp_logits