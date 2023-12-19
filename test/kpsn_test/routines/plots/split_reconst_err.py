import numpy as np
from kpsn.models.morph import affine_mode as afm
from kpsn.models.morph import linear_skeletal as ls
from kpsn.util import skeleton, alignment, logging, keypt_io
from kpsn_test import visualize as viz
import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns
import jax.tree_util as pt


def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    # ------------------------- setup: organize data

    ref_sess = cfg['ref_sess']
    origin_keypt = cfg['origin_keypt']
    stepsize = cfg['stepsize']
    colorby = cfg['colorby']

    hyperparams = init.hyperparams
    meta = dataset['metadata']
    sessions = list(meta['session_slice'].keys())

    to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], origin_keypt)

    # if passed full m-step parameter traces: select last entry from m-step
    param_hist = fit['param_hist'].copy()
    mstep_lengths = np.array(viz.fitting.mstep_lengths(fit['mstep_losses']))
    if param_hist[0].posespace.means.ndim > 2:
        param_hist.map(lambda arr: arr[np.arange(len(arr)), mstep_lengths - 2])
    
    # grab features from the reference session to transform according to
    # use_frames
    slices = meta['session_slice']
    src_feats = dataset['keypts'][slices[ref_sess]]
    src_kpts = to_kpt(src_feats).reshape([-1, 14, 3])

    steps = np.arange(0, len(mstep_lengths), stepsize)
    err_trace = logging.ReportTrace(n_steps = len(steps))


    # ------------------------- compute: measure errors

    for step_i, step_num in enumerate(steps):
        step_params = param_hist[step_num]
        step_params = step_params.with_hyperparams(hyperparams).morph

        reconst, subj_ids = viz.affine_mode.reconst_feat_with_params(
            src_feats, meta['session_ix'][ref_sess],  
            step_params, len(slices), return_subj_ids = True)
        reconst_kpts = {
            s: to_kpt(reconst[subj_ids == meta['session_ix'][s]]).reshape([-1, 14, 3])
            for s in sessions}
        
        pseudoslices = {s: s for s in slices}
        errs = viz.model_compare.keypt_errs(
            reconst_kpts, src_kpts, pseudoslices, single_b = True, is_kpt = True)
        err_trace.record(errs, step_i)


    # ------------------------- finally! plot

    pal = viz.defaults.age_pal(meta[colorby])
    errs = err_trace.as_dict()
    avg_errs = np.stack(list(
        e for k, e in errs.items() if k != ref_sess)).mean(axis = 0)
    sess_order = list(meta[colorby].keys())
    keypt_names = skeleton.default_armature.keypt_names

    fig, ax, ax_grid = viz.struct.flat_grid(
        len(keypt_names), n_col = 5, ax_size = (2.5, 2),
        subplot_kw = dict(sharex = True, sharey = cfg['sharey']),
        return_grid = True)
    
    for i_kp, kp_name in enumerate(keypt_names):
        ax[i_kp].set_title(kp_name)
        if kp_name == origin_keypt: continue
        for sess in sess_order:
            if sess == ref_sess: continue
            sess_clr = pal[meta[colorby][sess]]

            ax[i_kp].plot( # subj: err over fit time
                steps, errs[sess][:, i_kp],
                color = sess_clr,
                lw = 1,
                label = sess)
        
        ax[i_kp].plot( # avg: err over fit time
            steps, avg_errs[:, i_kp],
            color = 'k', lw = 2,
            label = 'mean')
        
    ax[0].set_ylabel("Mean reconstructed dist")
    ax_grid[0, -1].legend(bbox_to_anchor = (1, 0.5), loc = 'center left', frameon = False)

    fig.tight_layout()
    sns.despine()

    return {plot_name: fig}


defaults = dict(
    ref_sess = '3wk_m0',
    origin_keypt = 'hips',
    stepsize = 3,
    colorby = 'tgt_age',
    sharey = False
)