import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import seaborn as sns
from kpsn_test.visualize import fitting
from kpsn_test.visualize import defaults as viz_defaults
from jax import tree_util as pt

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
    steps =np.arange(0, len(mstep_lengths), cfg['stepsize'])
    pal = viz_defaults.age_pal(dataset['metadata'][cfg['colorby']])
    if fit['param_hist'].as_dict().morph.offset_updates.ndim > 3:
        hist_mode = 'mstep-line'
    else: hist_mode = 'step-point'

    params = ['offs', 'mode']
    fig, axes = plt.subplots(1 + hyperparams.L, 1,
        figsize = (12, 4),
        sharex = 'col')
    ax = {'offs': axes[0], 'mode': axes[1:]}

    xs = []
    global_steps = [0]
    offs_lines = [[] for j in range(hyperparams.N)]
    mode_lines = [[[] for j in range(hyperparams.L)] for i in range(hyperparams.N)]
    def insert_nans():
        for i, lines in enumerate([offs_lines, mode_lines]):
            for line in lines:
                line.append(np.full((1,) + line[-1].shape[1:], np.nan))
        xs.append(xs[-1][[-1]])

    global_step = 0
    for col, step in enumerate(steps):
        step_params = fit['param_hist'][step]
        
        if hist_mode == 'mstep-line':
            step_x = np.arange(global_step, global_step + step_len)
            step_slice = (slice(None, step_len),)
        else:
            step_x = np.array([global_step])
            step_slice = (None,)

        xs.append(step_x)
        global_steps.append(global_step)
        
        for sess_i in range(hyperparams.N):
            sess_name = dataset['metadata']['session_ix'].inv[sess_i]
            sess_color = dataset['metadata'][cfg['colorby']][sess_name]
            step_len = mstep_lengths[step]

            offs_lines[sess_i].append(
                step_params.morph.offset_updates[step_slice][..., sess_i, :])
            for mode_i in range(hyperparams.L):
                mode_lines[sess_i][mode_i].append(
                    step_params.morph.mode_updates[step_slice][..., sess_i, :, mode_i])

        if hist_mode == 'mstep-line':
            insert_nans()
        global_step += step_len

        # ax[0, 0].set_title(f"Step {step}")

    xs = np.concatenate(xs)
    offs_lines = pt.tree_map(np.concatenate, offs_lines)
    mode_lines = pt.tree_map(np.concatenate, mode_lines)
    
    line_kw = dict(lw = 0.3)
    for sess_i in range(hyperparams.N):
        sess_name = dataset['metadata']['session_ix'].inv[sess_i]
        sess_color = dataset['metadata'][cfg['colorby']][sess_name]
        ax['offs'].plot(
            xs, offs_lines[sess_i],
            color = pal[sess_color], **line_kw)
        for mode_i in range(hyperparams.L):
            ax['mode'][mode_i].plot(
                xs, mode_lines[sess_i][mode_i],
                color = pal[sess_color], **line_kw)

    ax['offs'].set_ylabel(f"Offset\nupdates")
    ax['mode'][0].set_ylabel(f"Mode\nupdates")

    for a in axes:
        ylim = a.get_ylim()
        trans = mt.blended_transform_factory(a.transData, a.transAxes)
        a.vlines(
            global_steps, ymin=0, ymax=1, transform = trans,
            lw = 0.2, color = '.8')
        a.set_ylim(*ylim)
    
    sns.despine()
    fig.tight_layout()
    
    return {plot_name: fig}

defaults = dict(
    stepsize = 1,
    colorby = 'age'
)