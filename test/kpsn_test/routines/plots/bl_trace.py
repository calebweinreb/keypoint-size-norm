from kpsn.models.morph import affine_mode as afm
from kpsn.models.morph import linear_skeletal as ls
from kpsn.util import skeleton, alignment
from kpsn_test import visualize as viz
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


def plot(
    plot_name,
    dataset,
    fit,
    cfg,
    **kwargs
    ):

    morph_model = afm.AffineModeMorph
    skel = skeleton.default_armature
    if cfg['origin_keypt'] != skel.root:
        skel = skeleton.reroot(skel, cfg['origin_keypt'])
    ls_mat = ls.construct_transform(skel, skel.keypt_by_name[skel.root])
    param_hist = fit['param_hist'].copy()

    to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])
    
    # if passed full m-step parameter traces: select last entry from m-step
    if param_hist[0].posespace.means.ndim > 2:
        mstep_lengths = np.array(viz.fitting.mstep_lengths(fit['mstep_losses']))
        param_hist.map(lambda arr: arr[np.arange(len(arr)), mstep_lengths - 1])

    # ------------------------- calculate: bone lengths at each step

    metadata = dataset['metadata']
    sessions = metadata['session_ix'].keys()
    mean_lengths = {sess_name: [] for sess_name in sessions}
    steps = np.arange(0, len(param_hist), cfg['stepsize'])
    for step_i in steps:
        # map onto reference subject body
        params = param_hist[step_i].with_hyperparams(fit['fit_params'].hyperparams)
        poses = morph_model.inverse_transform(
            params.morph,
            dataset['keypts'],
            dataset['subject_ids'])
        feats = morph_model.transform(
            params.morph,
            poses,
            np.full([len(poses)], metadata['session_ix'][cfg['ref_sess']]))
        keypts = to_kpt(feats)

        # get bones and calculate mean length by subject
        all_roots, all_bones = ls.transform(
            keypts.reshape([-1, skel.n_kpts, 3]), ls_mat)
        all_lengths = np.linalg.norm(all_bones, axis = -1)
        for sess_name in sessions:
            slc = metadata['session_slice'][sess_name]
            mean_lengths[sess_name].append(all_lengths[slc].mean(axis = 0))
    mean_lengths = {k: np.array(v) for k, v in mean_lengths.items()}

    # ------------------------- calculate: base bone lengths for each session


    all_roots, all_bones = ls.transform(
        to_kpt(dataset['keypts']).reshape([-1, skel.n_kpts, 3]),
        ls_mat)
    base_lengths = {
        sess_name: np.linalg.norm(
            all_bones[metadata['session_slice'][sess_name]],
            axis = -1).mean(axis = 0)
        for sess_name in sessions
    }

    # -------------------- finally! plot

    age_pal = viz.defaults.age_pal(metadata[cfg['groupby']])

    fig, axes, grid = viz.struct.flat_grid(len(skel.bones), 5, ax_size = (2, 2), 
        subplot_kw = dict(sharex = True, sharey = False), return_grid = True)
    for i_bone, ax in enumerate(axes):
        for sess_name in sessions:
            ax.plot(
                steps, mean_lengths[sess_name][:, i_bone],
                lw = 2 if sess_name == cfg['ref_sess'] else 1,
                color = age_pal[metadata[cfg['groupby']][sess_name]],
                label = sess_name)
        ax.set_title(skel.bone_name(i_bone), fontsize = 10)

        x0, x1 = ax.get_xlim()
        for sess_name in sessions:
            ax.plot([x0], [base_lengths[sess_name][i_bone]],
            'd', ms = 6, mew = 0.5, mec = 'w',
            zorder = 10, clip_on = False,
            color = age_pal[metadata[cfg['groupby']][sess_name]])
        ax.set_xlim(x0, x1)

    grid[0, -1].legend(frameon = False, bbox_to_anchor = (1, 0.5), loc = 'center left')
    grid[0, 0].set_ylabel("Projected bone length")

    sns.despine()
    fig.tight_layout()

    return {
        plot_name: fig
    }

defaults = dict(
    ref_sess = None,
    stepsize = 5,
    groupby = 'age',
    origin_keypt = 'hips'
    )