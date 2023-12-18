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

    hyperparams = init.posespace.hyperparams
    fig, ax = plt.subplots(
        hyperparams.L, 2, figsize = (3 * 2, 2 * hyperparams.L),
        sharex = True, sharey = True)
    print("ax:", ax.shape)
    ax = ax.reshape([-1, 2])
    print("new ax:", ax.shape)

    for i_comp in range(hyperparams.L):
        for (curr_param, color, label, scatt_kw) in [
            (init.posespace, '.6', "init", {'s': 10}),
            (fit['fit_params'].posespace, 'k', 'fit', {'s': 0})]:
            
            if dataset['keypts'].shape[-1] < 42:
                kpts = alignment.sagittal_align_insert_redundant_subspace(
                    curr_param.means[i_comp], cfg['origin_keypt'], skeleton.default_armature)
            else:
                kpts = curr_param.means[i_comp]
            
            for row, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
            
                viz.diagram_plots.plot_mouse(
                    ax[i_comp, row],
                    kpts.reshape([skeleton.default_armature.n_kpts, 3]),
                    xaxis, yaxis,
                    scatter_kw = {'color': color, **scatt_kw},
                    line_kw = {'color': color, 'lw': 1},
                    label = label)
                sns.despine(ax = ax[i_comp, row])

        meanlab = "\nmean" if i_comp == 0 else ""
        ax[i_comp, 0].set_ylabel(f"Component {i_comp}{meanlab}")

    ax[0, 1].legend(bbox_to_anchor = (1, 0.5), frameon = False, loc = 'center left')
    fig.tight_layout()

    return {plot_name: fig}

defaults = dict(
    origin_keypt = 'hips'
)