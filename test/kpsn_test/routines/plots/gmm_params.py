import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mt
import seaborn as sns
from kpsn_test.visualize import fitting
import kpsn_test.visualize as viz
import tqdm


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
    subj_pal = viz.defaults.age_pal(dataset['metadata'][cfg['colorby']])
    if fit['param_hist'].as_dict().morph.offset_updates.ndim > 3:
        hist_mode = 'mstep-line'
    else: hist_mode = 'step-point'

    params = ['mean', 'diag', 'det', 'wt']
    n_param = 5
    fax = {
        param: plt.subplots(hyperparams.L, 1,
            figsize = (12, hyperparams.L),
            sharex = True, sharey = True)
        for param in params
    }
    fig = {param: fax[param][0] for param in params}
    ax  = {param: fax[param][1] for param in params}


    # ----- empty lists of parameter changes over m steps
    xs = []
    global_steps = [0]
    mean_lines = [[] for i in range(hyperparams.L)]
    diag_lines = [[] for i in range(hyperparams.L)]
    det_lines = [[] for i in range(hyperparams.L)]
    pop_lines = [[] for i in range(hyperparams.L)]
    subj_lines = [[] for i in range(hyperparams.L)]
    def insert_nans():
        for i, lines in enumerate([mean_lines, diag_lines, det_lines, pop_lines, subj_lines]):
            for line in lines:
                line.append(np.full((1,) + line[-1].shape[1:], np.nan))
        xs.append(xs[-1][[-1]])


    # ----- assemble parameter changes over m steps
    line_kw = dict(lw = 0.3)
    global_step = 0
    for step in tqdm.tqdm(steps):
        step_params = fit['param_hist'][step]
        step_params = step_params.posespace.with_hyperparams(hyperparams)
        step_len = mstep_lengths[step]

        if hist_mode == 'mstep-line':
            step_x = np.arange(global_step, global_step + step_len)
        else:
            step_x = np.array([global_step])
        xs.append(step_x)
        global_steps.append(global_step)

        for comp_i in range(hyperparams.L):

            if hist_mode == 'mstep-line':
                step_slice = (slice(None, step_len), comp_i)
            else:
                step_slice = (None, comp_i,)

            # means
            mean_lines[comp_i].append(step_params.means[step_slice])
            
            # eigs or diag
            cov_eig = np.log(np.linalg.eigvalsh(step_params.covariances()[step_slice]))
            diag_lines[comp_i].append(cov_eig)
            
            # determinant
            det_lines[comp_i].append(
                np.log10(np.linalg.det(step_params.covariances()[step_slice])))
            
            # weights
            pop_lines[comp_i].append(step_params.pop_weights()[step_slice])
            subj_lines_ = np.zeros([
                step_len if hist_mode == 'mstep-line' else 1,
                len(dataset['metadata'][cfg['colorby']])])
            for sess in dataset['metadata'][cfg['colorby']]:
                ix = dataset['metadata']['session_ix'][sess]
                weights = np.swapaxes(step_params.weights(), -2, -1)
                subj_lines_[:, ix] = weights[step_slice][..., ix]
            subj_lines[comp_i].append(subj_lines_)
            
        if hist_mode == 'mstep-line':
            insert_nans()
        global_step += step_len

    

    # concatenate arrays from each m step
    xs = np.concatenate(xs)
    mean_lines = [np.concatenate(arr) for arr in mean_lines]
    diag_lines = [np.concatenate(arr) for arr in diag_lines]
    det_lines = [np.concatenate(arr) for arr in det_lines]
    pop_lines = [np.concatenate(arr) for arr in pop_lines]
    subj_lines = [np.concatenate(arr) for arr in subj_lines]

    print(xs.shape, mean_lines[0].shape, diag_lines[0].shape, det_lines[0].shape, pop_lines[0].shape, subj_lines[0].shape)

    # plot concatenated lines
    for comp_i in range(hyperparams.L):
        ax['mean'][comp_i].plot(xs, mean_lines[comp_i],
            color = pal[comp_i], **line_kw)
        ax['diag'][comp_i].plot(xs, diag_lines[comp_i],
            color = pal[comp_i], **line_kw)
        ax['det'][comp_i].plot(xs, det_lines[comp_i],
            color = pal[comp_i], **line_kw)
        for clr, zorder, lw in [('k', 0, 0.9), (pal[comp_i], 1, 0.8)]:
            ax['wt'][comp_i].plot(xs, pop_lines[comp_i],
                color = clr, **{**line_kw, 'zorder': zorder, 'lw': lw})
        for sess in dataset['metadata'][cfg['colorby']]:
            clr = subj_pal[dataset['metadata'][cfg['colorby']][sess]]
            ix = dataset['metadata']['session_ix'][sess]
            ax['wt'][comp_i].plot(xs, subj_lines[comp_i][:, ix],
                color = clr, **line_kw)

    # set axis scales / limits
    for comp_i in range(hyperparams.L):
        for param in params:
            a = ax[param][comp_i]
            trans = mt.blended_transform_factory(a.transData, a.transAxes)
            a.vlines(
                global_steps, ymin=0, ymax=1, transform = trans,
                lw = 0.2, color = '.8')
            sns.despine(ax = ax[param][comp_i])
        ax['wt'][comp_i].set_ylim(0, 1)
        
    # set axis labels
    ax['mean'][0].set_ylabel(f"Component \n means")
    if cfg['eigs']:
        ax['diag'][0].set_ylabel(f"Cov eigs")
    else:
        ax['diag'][0].set_ylabel(f"Cov diag")
    ax['det'][0].set_ylabel(f"Cov det")
    ax['wt'][0].set_ylabel(f"Pop and subj\nweights")
        
    for param in params:
        fig[param].tight_layout()
    
    return {f'{plot_name}-{param}': fig[param] for param in params}

defaults = dict(
    stepsize = 3,
    colorby = 'age',
    logdet = True,
    eigs = False
)