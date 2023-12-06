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

    fig, ax = plt.subplots(2, 1,
        figsize = (12, 4),
        sharex = 'col', sharey = 'row')
    ax = ax[:, None]
    # if ax.ndim < 2:
        # ax = ax.reshape([2, len(steps)])

    line_kw = dict(lw = 0.3)
    global_step = 0
    for col, step in enumerate(steps):
        step_params = fit['param_hist'][step]
        ax[0, 0].axvline(global_step, lw = 0.2, color = '.8')
        ax[1, 0].axvline(global_step, lw = 0.2, color = '.8')
        for sess_i in range(hyperparams.N):
            sess_name = dataset['metadata']['session_ix'].inv[sess_i]
            sess_color = dataset['metadata'][cfg['colorby']][sess_name]
            step_len = mstep_lengths[step]
            step_x = np.arange(global_step, global_step + step_len)
            ax[0, 0].plot(
                step_x,
                step_params.morph.offset_updates[:step_len, sess_i],
                color = pal[sess_color], **line_kw)
            for mode_i in range(hyperparams.L):
                ax[mode_i + 1, 0].plot(
                    step_x,
                    step_params.morph.mode_updates[:step_len, sess_i, :, mode_i],
                    color = pal[sess_color], **line_kw)
        global_step += step_len

        # ax[0, 0].set_title(f"Step {step}")


    ax[0, 0].set_ylabel(f"Offset updates")
    ax[1, 0].set_ylabel(f"Mode updates")
        
    sns.despine()
    fig.tight_layout()
    
    return {plot_name: fig}

defaults = dict(
    stepsize = 1,
    colorby = 'age'
)