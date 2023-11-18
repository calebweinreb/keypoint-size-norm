
from kpsn.util.logging import _all_paths, _index

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


def mstep_lengths(mstep_losses):
    mstep_lengths = []
    for i in range(0, len(mstep_losses)):
        if np.any(~np.isfinite(mstep_losses[i])):
            mstep_lengths.append(np.argmax(~np.isfinite(mstep_losses[i])))
        else:
            mstep_lengths.append(len(mstep_losses[i]))
    return mstep_lengths


def em_loss(loss_hist, mstep_losses, mstep_relative = True):
    fig, ax = plt.subplots(figsize = (9, 1.7), ncols = 3)
    
    pal = sns.hls_palette(len(mstep_losses) * 5, h = 0.7, l = 0.4)[:len(mstep_losses)]

    mstep_lengths = []
    for i in range(0, len(mstep_losses)):
        
        if np.any(~np.isfinite(mstep_losses[i])):
            curr_loss = mstep_losses[i][:np.argmax(~np.isfinite(mstep_losses[i]))]
            if len(curr_loss) == 0: continue
        else:
            curr_loss = mstep_losses[i]
        mstep_lengths.append(len(curr_loss))
        
        if mstep_relative:
            plot_y = (curr_loss - curr_loss.min()) / (curr_loss.max() - curr_loss.min())
        else: plot_y = curr_loss

        ax[1].plot(
            np.linspace(0, 1, len(curr_loss)),
            plot_y,
            color = pal[i], lw = 1)
        
    if not mstep_relative:
        ax[1].set_yscale('log')
        
    ax[0].plot(loss_hist, 'k-')
    if loss_hist.max() > 2 * loss_hist[0]:
        ax[0].set_ylim(None, 2 * loss_hist[0])
    ax[2].plot(np.arange(len(mstep_lengths)), mstep_lengths, 'k-')

    ax[1].set_xlabel("Loss profile")
    ax[2].set_ylabel("M step length")
    ax[2].set_xlabel("Step")
    ax[0].set_ylabel("Loss")
    ax[0].set_xlabel("Step")
    
    fig.tight_layout()
    sns.despine()
    return fig


def report_plots(reports):
    n_col = 5
    n_row = int(np.ceil(reports.n_leaves() / n_col))
    fig, ax = plt.subplots(n_row, n_col, figsize = (3 * n_col, 2 * n_row))
    reports.plot(ax.ravel()[:reports.n_leaves()], color = 'k', lw = 1)
    for a in ax.ravel()[reports.n_leaves():]: a.set_axis_off()
    sns.despine()
    fig.tight_layout()
    return fig



def training_param_traces(
    param_hist # iterable of pytrees
    ):
    raise NotImplementedError

    morph_keys = _all_paths(param_hist[0].morph.trainable_params)
    pose_keys = _all_paths(param_hist[0].posespace.trainable_params)

    param_hist = fit['param_hist']
    mode_hist = np.stack([p.morph.mode_updates for p in param_hist])
    ofs_hist = np.stack([p.morph.offset_updates for p in param_hist])

    pop_weight_hist = np.stack([p.posespace.pop_weight_logits for p in param_hist])
    subj_weight_hist = np.stack([p.posespace.subj_weight_logits for p in param_hist])

    fig, ax = plt.subplots(3, init_params.morph.N, figsize = (6, 3), sharey = 'row')
    for subj_i in range(init_params.morph.N):
        ax[0, subj_i].plot(mode_hist[:, subj_i, :, 0], 'k-', lw = 1)
        ax[1, subj_i].plot(ofs_hist[:, subj_i], 'k-', lw = 1)
        ax[2, subj_i].plot(subj_weight_hist[:, subj_i], 'k-', lw = 1)
        for i in [0, 1, 2]:
            ax[i, subj_i].axhline(0, ls = '--', color = '.6', lw = 0.5)
    ax[0, 0].set_ylabel("Mode\nupdate")
    ax[1, 0].set_ylabel("Offset\nupdate")
    ax[2, 0].set_ylabel("Subject\nweights")
    sns.despine()
    fig.tight_layout()
    subjwise_fig = fig

    fig, ax = plt.subplots(1, 1, figsize = (3, 1.5))
    ax.plot(pop_weight_hist, 'k-', lw = 1)
    ax.axhline(0, ls = '--', color = '.6', lw = 0.5)
    ax.set_ylabel("Population\nweights")
    sns.despine()
    fig.tight_layout()
    pop_fig = fig

    return subjwise_fig, pop_fig