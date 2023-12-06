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

    meta = dataset['metadata']
    obs = {sess: dataset['keypts'][meta['session_slice'][sess]]
           for sess in meta['session_slice']}
    means = {sess: arr.mean(axis = 0) for sess, arr in obs.items()}

    N = len(obs)
    fig, ax = plt.subplots(
        N, 2, figsize = (3 * 2, 2 * N),
        sharex = True, sharey = True)
    
    if dataset['keypts'].shape[-1] < 42:
        to_kpt = lambda arr: alignment.sagittal_align_insert_redundant_subspace(
            arr, cfg['origin_keypt'], skeleton.default_armature)
    else:
        to_kpt = lambda arr: arr


    for i_sess, sess in enumerate(means):
        
        for row, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
        
            viz.diagram_plots.plot_mouse(
                ax[i_sess, row],
                to_kpt(means[sess]).reshape([skeleton.default_armature.n_kpts, 3]),
                xaxis, yaxis,
                scatter_kw = {'color': 'k'},
                line_kw = {'color': 'k', 'lw': 1})
            sns.despine(ax = ax[i_sess, row])

        meanlab = "\ncentroid" if i_sess == 0 else ""
        ax[i_sess, 0].set_ylabel(f"{sess}{meanlab}")

    fig.tight_layout()

    return {plot_name: fig}

defaults = dict(
    origin_keypt = 'hips'
)