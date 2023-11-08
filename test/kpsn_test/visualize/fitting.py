import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np



def em_loss(loss_hist, mstep_losses):
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
        
        ax[1].plot(
            np.linspace(0, 1, len(curr_loss)),
            (curr_loss - curr_loss.min()) / (curr_loss.max() - curr_loss.min()),
            color = pal[i], lw = 1)
        
    ax[0].plot(loss_hist, 'k-')
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