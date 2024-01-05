import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
from kpsn.models.morph import affine_mode as afm


def plot_for_params(dataset, params, cfg):

    meta = dataset['metadata']
    sessions = list(meta['session_slice'].keys())

    # --- setup: find representative frames    
    to_kpt, to_feat = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])

    feats, match_dataset, match_meta = viz.diagram_plots.matching_dataset(
        dataset['keypts'], meta['session_slice'], meta.get('frame_ids'))
    slices = match_meta['session_slice']

    ref_sessions = {}; ref_feats = {}; selected_feats = {}
    selected_frames = {}
    for sess in sessions:
        # ref_sess = sess.split('.')[0] + '.0' # 3wk_m0.1 -> 3wk_m0.0
        ref_sess = sess.split('-')[-1] # 52wk-3wk_m0 -> 3wk_m0
        ref_sessions[sess] = ref_sess
        
        if ref_sess not in selected_frames:
            frames = list(viz.diagram_plots.pose_gallery_ixs(
                to_kpt(feats[ref_sess]), skeleton.default_armature).values())
            selected_frames[ref_sess] = frames
        else:
            frames = selected_frames[ref_sess]

        ref_feats[sess] = feats[ref_sess][frames]
        selected_feats[sess] = feats[sess][frames]
    n_plot_frames = len(frames)

    ref_slices, ref_all_feats = keypt_io.to_flat_array(ref_feats)
    tgt_sess_ix, tgt_sess_ids = keypt_io.ids_from_slices(ref_all_feats, ref_slices)
    
    # ------------------------------------- reconstruct frames
    topose_sess_ids = viz.affine_mode.map_ids(
        tgt_sess_ix, meta['session_ix'],
        {s: ref_sessions[s] for s in sessions})[tgt_sess_ids]
    tokpt_sess_ids = viz.affine_mode.map_ids(
        tgt_sess_ix, meta['session_ix'],
        {s: s for s in sessions})[tgt_sess_ids]
    
    reconst = to_kpt(viz.affine_mode.map_morphology(
        ref_all_feats, topose_sess_ids, tokpt_sess_ids, params))
    
    # ------------------------------------------  plot!

    # create figure etc
    fig, ax = plt.subplots(
        nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
        figsize = (3 * len(frames), 4 * len(meta['session_slice'])), sharex = 'col')
    
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    # --- iterate: get keypoints of first session per group
    for i_frame in range(n_plot_frames):
        for i_sess, sess in enumerate(sessions):

            ref_sess = ref_sessions[sess]
            fg_color = pal[meta[cfg['colorby']][sess]]
            
            for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                curr_ax = ax[2*i_sess + row_ofs, i_frame]
                kpt_shape = [skeleton.default_armature.n_kpts, 3]
                dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                # reference
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    to_kpt(ref_feats[sess][i_frame]).reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': '.6'},
                    line_kw = {'color': '.6'},
                    label = ref_sess)
                
                # ground truth target
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    to_kpt(selected_feats[sess][i_frame]).reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': 'k', 's': 6},
                    line_kw = {'color': 'k', 'lw': 1},
                    label = f'{sess}, datset')
                # morphed target
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    reconst[ref_slices[sess]][i_frame].reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': fg_color, 's': 0},
                    line_kw = {'color': fg_color, 'lw': 1},
                    label = f'{sess} reconst.')
                
                if dolabel:
                    curr_ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5,), frameon = False)
                sns.despine(ax = curr_ax)
    
    fig.tight_layout()
    return fig


def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    return {
        f'{plot_name}-init': plot_for_params(dataset, init.morph, cfg),
        f'{plot_name}-fit': plot_for_params(dataset, fit['fit_params'].morph, cfg),
    }


defaults = dict(
    ref_sess = 'subj0',
    colorby = 'body',
    origin_keypt = 'hips',
    match_mode = 'frame',
        # frame: assume frames correspond across videos
        # same: use keypoints from the reference session
        # meta: use frame_id from metadata to meta
)   