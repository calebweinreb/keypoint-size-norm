import matplotlib.pyplot as plt
import seaborn as sns

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io

def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    centers = dataset['metadata']['shared']['clusters']
    L = len(centers)
    fig, ax = plt.subplots(
        L, 2, figsize = (3 * 2, 2 * L),
        sharex = 'row')
    
    if centers.shape[-1] < 42:
        to_kpt = lambda arr: alignment.sagittal_align_insert_redundant_subspace(
            arr, cfg['origin_keypt'], skeleton.default_armature)
    else:
        to_kpt = lambda arr: arr


    for i_clust in range(L):
        
        for row, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
        
            viz.diagram_plots.plot_mouse(
                ax[i_clust, row],
                to_kpt(centers[i_clust]).reshape([skeleton.default_armature.n_kpts, 3]),
                xaxis, yaxis,
                scatter_kw = {'color': 'k'},
                line_kw = {'color': 'k', 'lw': 1})
            sns.despine(ax = ax[i_clust, row])

        meanlab = "\ncentroid" if i_clust == 0 else ""
        ax[i_clust, 0].set_ylabel(f"Cluster {i_clust}{meanlab}")

    fig.tight_layout()

    return {plot_name: fig}

defaults = dict(
    origin_keypt = 'hips'
)