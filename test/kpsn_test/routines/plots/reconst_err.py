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
    N = hyperparams.posespace.N
    sessions = list(meta['session_slice'].keys())
    tgt_sessions = [s for s in sessions if s != ref_sess]

    to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], origin_keypt)

    # ensure we have frame mapping
    if 'frame_ids' not in meta:
        print('warning: added frame identity map')
        shared_frame_ids = np.arange(dataset['keypts'].shape[0] // N)
        meta['frame_ids'] = {k: shared_frame_ids for k in meta['session_slice']}

    # if passed full m-step parameter traces: select last entry from m-step
    param_hist = fit['param_hist'].copy()
    mstep_lengths = np.array(viz.fitting.mstep_lengths(fit['mstep_losses']))
    if param_hist[0].posespace.means.ndim > 2:
        param_hist.map(lambda arr: arr[np.arange(len(arr)), mstep_lengths - 2])

    # find frame indices that match across videos
    valid_frames = viz.diagram_plots.valid_display_frames(meta['frame_ids'].values())
    use_frames = {
        k: viz.diagram_plots.frame_examples(valid_frames, ids)
        for k, ids in meta['frame_ids'].items()}

    # grab features from the reference session to transform according to use_frames
    obs_feats = {
        sess: dataset['keypts'][slc][use_frames[sess]]
        for sess, slc in meta['session_slice'].items()}
    src_feats = obs_feats[ref_sess]
    obs_kpts = {
        sess: to_kpt(obs_feats[sess]).reshape([-1, 14, 3])
        for sess in obs_feats}
    slices, all_kpts = keypt_io.to_flat_array(obs_kpts)
    sess_ix, subj_ids = keypt_io.ids_from_slices(all_kpts, slices)
    src_kpts = obs_kpts[ref_sess]


    steps = np.arange(0, len(mstep_lengths), stepsize)
    err_trace = logging.ReportTrace(n_steps = len(steps))


    # ------------------------- compute: measure errors

    for step_i, step_num in enumerate(steps):
        step_params = param_hist[step_num]
        step_params = step_params.with_hyperparams(hyperparams).morph

        pose = afm.inverse_transform(
            step_params, src_feats, meta['session_ix'][ref_sess])
        copied_poses = jnp.concatenate([pose for _ in range(N)])
        reconst = afm.transform(
            step_params, copied_poses, subj_ids)
        reconst_kpts = {
            s: to_kpt(reconst[slc]).reshape([-1, 14, 3])
            for s, slc in slices.items()}
        errs = {
            s: jnp.linalg.norm(reconst_kpts[s] - all_kpts[slc], axis = -1).mean(axis = 0)
            for s, slc in slices.items()}

        err_trace.record(errs, step_i)


    # ------------------------- calculate: base errors betwen subjs
    
    base_errs = {
        s: jnp.linalg.norm(all_kpts[slc] - src_kpts, axis = -1).mean(axis = 0)
        for s, slc in slices.items()}
    

    # ------------------------- finally! plot

    pal = viz.defaults.age_pal(meta[colorby])
    errs = err_trace.as_dict()
    avg_errs = np.stack(list(
        e for k, e in errs.items() if k != ref_sess)).mean(axis = 0)
    avg_base_errs = np.stack(list(
        e for k, e in base_errs.items() if k != ref_sess)).mean(axis = 0)
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
        
        # second pass: have to do once we have fixed xlim
        x0, x1 = ax[i_kp].get_xlim()
        for sess in sess_order:
            if sess == ref_sess: continue
            sess_clr = pal[meta[colorby][sess]]

            ax[i_kp].plot( # subj: base err
                [x0], [base_errs[sess][i_kp]],
                'd', ms = 6.5, mew = 0.5, mec = 'w', color = sess_clr,
                zorder = 10, clip_on = False
            )
            
        ax[i_kp].plot(  # avg: base err
            [x0],
            [avg_base_errs[i_kp]],
            'd', ms = 7, mew = 0.25, mec = 'w', color = 'k',
            zorder = 10, clip_on = False
        )

        ax[i_kp].set_xlim(x0, x1)

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