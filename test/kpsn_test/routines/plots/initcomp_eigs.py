import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
from kpsn_test.clustering import apply_clusters, masks_and_logits

def plot(
    plot_name,
    init,
    cfg,
    **kwargs
    ):
    
    vals, vecs = np.linalg.eigh(init.posespace.covariances())

    fig, ax = plt.subplots(1, 2, figsize = (6, 3))
    pal = sns.color_palette("Set1", len(vals))
    ndim = vals.shape[-1]
    for i in range(len(vals)):
        ax[0].plot(np.arange(1, ndim + 1), vals[i], color = pal[i])
        ax[1].plot(np.arange(1, ndim + 1), vals[i], label = f'{i}', color = pal[i])
    ax[0].set_yscale('log')
    ax[0].set_ylabel("Init covariance eigenvalue")
    ax[1].axhline(0, color = 'k', lw = 1)
    ax[1].set_xlim(1, 8)
    ax[1].set_ylim(-0.05, 0.5)
    plt.legend(frameon = False, bbox_to_anchor = (1, 0.5), loc = 'center left', title = 'Component')

    sns.despine()
    fig.tight_layout()
        
    return {plot_name: fig}

defaults = dict()