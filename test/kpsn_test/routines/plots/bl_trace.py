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
    if cfg['reroot'] is not None:
        skel = skeleton.reroot(skel, cfg['reroot'])
    ls_mat = ls.construct_transform(skel, skel.keypt_by_name[skel.root])
    param_hist = fit['param_hist']
    print("orig:", param_hist.as_dict().posespace.means.shape)
    if param_hist[0].posespace.means.ndim > 2:
        param_hist.map(lambda arr: arr[:, -1])
        print("new:", param_hist.as_dict().posespace.means.shape)

    metadata = dataset['metadata']
    sessions = metadata['session_ix'].keys()
    mean_lengths = {sess_name: [] for sess_name in sessions}
    steps = np.arange(0, len(fit['param_hist']), cfg['stepsize'])
    for step_i in steps:
        params = fit['param_hist'][step_i].with_hyperparams(fit['fit_params'].hyperparams)
        poses = morph_model.inverse_transform(
            params.morph,
            dataset['keypts'],
            dataset['subject_ids'])
        feats = morph_model.transform(
            params.morph,
            poses,
            np.full([len(poses)], metadata['session_ix'][cfg['ref_sess']]))
        keypts = alignment.sagittal_align_insert_redundant_subspace(
            feats, skel.root, skel)
        
        all_roots, all_bones = ls.transform(
            keypts.reshape([-1, skel.n_kpts, 3]), ls_mat)
        all_lengths = np.linalg.norm(all_bones, axis = -1)
        for sess_name in sessions:
            slc = metadata['session_slice'][sess_name]
            mean_lengths[sess_name].append(all_lengths[slc].mean(axis = 0))
    mean_lengths = {k: np.array(v) for k, v in mean_lengths.items()}

    age_pal = viz.defaults.age_pal(metadata[cfg['groupby']])
    fig, axes = viz.struct.flat_grid(len(skel.bones), 5, ax_size = (2, 2), 
        subplot_kw = dict(sharex = True, sharey = False))
    for i_bone, ax in enumerate(axes):
        for sess_name in sessions:
            ax.plot(
                steps, mean_lengths[sess_name][:, i_bone],
                lw = 1,
                color = age_pal[metadata[cfg['groupby']][sess_name]])
        ax.set_title(skel.bone_name(i_bone), fontsize = 10)
    sns.despine()
    fig.tight_layout()

    return {
        plot_name: fig
    }

defaults = dict(
    ref_sess = None,
    stepsize = 5,
    reroot = 'hips',
    groupby = 'age',
    )