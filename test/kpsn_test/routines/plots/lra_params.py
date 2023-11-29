import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kpsn_test.visualize import fitting
from kpsn_test.visualize import defaults as viz_defaults


def plot(
    plot_name,
    dataset,
    fit,
    cfg,
    **kwargs,
    ):

    mstep_lengths = fitting.mstep_lengths(fit['mstep_losses'])
    mstep_lengths = [m for m in mstep_lengths if m > 1]
    hyperparams = fit['fit_params'].hyperparams.morph
    steps = np.arange(0, len(mstep_lengths), cfg['stepsize'])
    pal = viz_defaults.age_pal(dataset['metadata'][cfg['colorby']])

    fig, ax = plt.subplots(2, len(steps),
        figsize = (4 * len(steps), 3 * (hyperparams.L + 1)),
        sharex = 'col', sharey = 'row')
    if ax.ndim < 2:
        ax = ax.reshape([2, len(steps)])

    line_kw = dict(lw = 0.3)
    for col, step in enumerate(steps):
        step_params = fit['param_hist'][step]
        for sess_i in range(hyperparams.N):
            sess_name = dataset['metadata']['session_ix'].inv[sess_i]
            sess_color = dataset['metadata'][cfg['colorby']][sess_name]
            step_len = mstep_lengths[step]
            print("lra:", step_params.morph.mode_updates.shape, step_params.morph.offset_updates.shape)
            ax[0, col].plot(
                step_params.morph.offset_updates[:step_len, sess_i],
                color = pal[sess_color], **line_kw)
            for mode_i in range(hyperparams.L):
                ax[mode_i + 1, col].plot(
                    step_params.morph.mode_updates[:step_len, sess_i, :, mode_i],
                    color = pal[sess_color], **line_kw)

        ax[0, col].set_title(f"Step {step}")


    ax[0, 0].set_ylabel(f"Offset updates")
    ax[1, 0].set_ylabel(f"Mode updates")
        
    sns.despine()
    fig.tight_layout()
    
    return {plot_name: fig}

defaults = dict(
    stepsize = 1,
    colorby = 'age'
)