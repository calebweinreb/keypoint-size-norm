import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kpsn_test.visualize import fitting


def plot(
    plot_name,
    dataset,
    fit,
    cfg,
    **kwargs,
    ):

    mstep_lengths = fitting.mstep_lengths(fit['mstep_losses'])
    mstep_lengths = [m for m in mstep_lengths if m > 1]
    hyperparams = fit['fit_params'].hyperparams.posespace
    steps = np.arange(0, len(mstep_lengths), cfg['stepsize'])
    pal = sns.color_palette('Set1', n_colors = hyperparams.L)

    fig, ax = plt.subplots(4 * hyperparams.L, len(steps),
        figsize = (2 * len(steps), 4 * (hyperparams.L * 2)),
        sharex = 'col', sharey = False)

    for col, step in enumerate(steps):
        for comp_i in range(hyperparams.L):
            if not (comp_i == 0 and col == 0):
                for i in range(4):
                    ax[comp_i + i *  hyperparams.L, col].sharey(ax[i * hyperparams.L, 0])

    line_kw = dict(lw = 0.3)
    for col, step in enumerate(steps):
        step_params = fit['param_hist'][step]
        step_len = mstep_lengths[step]
        diag_mask = np.eye(hyperparams.M).astype('bool')
        for comp_i in range(hyperparams.L):
            ax[comp_i, col].plot(
                step_params.posespace.means[:step_len, comp_i],
                color = pal[comp_i], **line_kw)
            ax[comp_i + hyperparams.L, col].plot(
                step_params.posespace.with_hyperparams(hyperparams
                    ).covariances()[:step_len, comp_i][:, diag_mask],
                color = pal[comp_i], **line_kw)
            ax[comp_i + 2 * hyperparams.L, col].plot(
                step_params.posespace.with_hyperparams(hyperparams
                    ).covariances()[:step_len, comp_i][:, ~diag_mask][:, ::51],
                color = pal[comp_i], **line_kw)
            ax[comp_i + 3 * hyperparams.L, col].plot(
                np.linalg.det(step_params.posespace.with_hyperparams(hyperparams
                    ).covariances()[:step_len, comp_i]),
                color = pal[comp_i], **line_kw)
            
            ax[comp_i + hyperparams.L, col].set_yscale('log')
            ax[comp_i + 3 * hyperparams.L, col].set_yscale('log')
        ax[0, col].set_title(f"Step {step}")


    ax[0, 0].set_ylabel(f"Component means")
    ax[hyperparams.L, 0].set_ylabel(f"Component cov diag")
    ax[2 * hyperparams.L, 0].set_ylabel(f"Component cov off-diag")
    ax[3 * hyperparams.L, 0].set_ylabel(f"Component cov det")
        
    sns.despine()
    fig.tight_layout()
    
    return {plot_name: fig}

defaults = dict(
    stepsize = 1,
    colorby = 'age'
)