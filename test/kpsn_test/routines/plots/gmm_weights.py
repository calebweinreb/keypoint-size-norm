import matplotlib.pyplot as plt
import seaborn as sns

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
import numpy as np

def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    hyperparams = init.posespace.hyperparams
    meta = dataset['metadata']

    fig, ax = plt.subplots(hyperparams.L, 2, 
        sharey = True, sharex = True,
        figsize = (3 * 2, hyperparams.L * 1.5))
    ax = ax.reshape([-1, 2])
    group_keys, groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    for i_param, (curr_param, desat, ltn) in enumerate([
            (init, 0.5, 0.5),
            (fit['fit_params'], 1., 0.)]):

        weights = curr_param.posespace.weights()
        pop_weights = curr_param.posespace.pop_weights()
        for i_comp in range(hyperparams.L):
            for group_key, group in zip(group_keys, groups):
                session_ixs = [meta['session_ix'][sess] for sess in group]
                ax[i_comp, i_param].bar(
                    session_ixs,
                    weights[session_ixs, i_comp],
                    width = 0.8, color = 
                        np.array(sns.desaturate(pal[group_key], desat)) * (1 - ltn) + 
                        np.array([1., 1., 1.]) * ltn)
                ax[i_comp, i_param].axhline(
                    pop_weights[i_comp], color = '.5', lw = 1)
            
            ax[i_comp, 0].set_ylim(0, 1.05)
            ax[i_comp, 1].set_ylim(0, 1.05)
                
            prob_lab = "P(component)\n" if i_comp == 0 else ""
            ax[i_comp, 0].set_ylabel(f"{prob_lab}Component {i_comp}")
            sns.despine(ax = ax[i_comp, i_param])

    ax[-1, 0].set_xlabel("Session")
    ax[0, 0].set_title("Init params", fontsize = 10)
    ax[0, 1].set_title("Fit params", fontsize = 10)

    fig.tight_layout()
    return {plot_name: fig}

defaults = dict(
    colorby = 'age'
)