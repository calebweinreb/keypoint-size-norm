import matplotlib.pyplot as plt
import seaborn as sns

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
from kpsn.models.morph import affine_mode as afm



def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    meta = dataset['metadata']

    # ----------------------- setup: find reference sessions and frames to plot
    to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])
    slices = meta['session_slice']
    sessions = list(slices.keys())
    feats = {s : dataset['keypts'][slices[s]] for s in sessions}

    ref_sessions = {}; ref_feats = {}
    for sess in sessions:
        ref_sess = sess.split('.')[0] + '.0'
        # ref_sess = sess.split('-')[-1]
        ref_sessions[sess] = ref_sess
        frames = list(viz.diagram_plots.pose_gallery_ixs(
            to_kpt(feats[ref_sess]), skeleton.default_armature).values())
        ref_feats[sess] = feats[ref_sess][frames]
    n_plot_frames = len(frames)

    ref_slices, ref_all_feats = keypt_io.to_flat_array(ref_feats)
    tgt_sess_ix, tgt_sess_ids = keypt_io.ids_from_slices(ref_all_feats, ref_slices)

    # ------------------------------ plot!
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    def plot_for_params(curr_param):

        topose_sess_ids = viz.affine_mode.map_ids(
            tgt_sess_ix, meta['session_ix'],
            {s: ref_sessions[s] for s in sessions})[tgt_sess_ids]
        tokpt_sess_ids = viz.affine_mode.map_ids(
            tgt_sess_ix, meta['session_ix'],
            {s: s for s in sessions})[tgt_sess_ids]
        
        reconst = to_kpt(viz.affine_mode.map_morphology(
            ref_all_feats, topose_sess_ids, tokpt_sess_ids, curr_param))

        fig, ax = plt.subplots(
            nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
            figsize = (3 * len(frames), 4 * len(slices)), sharex = 'col')

        for i_frame in range(n_plot_frames):
            for i_sess, sess in enumerate(sessions):
                # --- hop into (src_sess morph) and out of (tgt_sess morph) pose
                # space
                ref_sess = ref_sessions[sess]
                fg_color = pal[meta[cfg['colorby']][sess]]

                # --- plot results of trasnform
                for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                    curr_ax = ax[2*i_sess + row_ofs, i_frame]
                    dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                    # plot both reference and transformed
                    reconst_frame = reconst[ref_slices[sess]][i_frame]
                    ref_frame = reconst[ref_slices[ref_sess]][i_frame]
                    for kpts, color, label, zorder in [
                            (ref_frame, '.6', ref_sess, -1),
                            (reconst_frame, fg_color, sess, 1)]:
                        
                        viz.diagram_plots.plot_mouse(
                            curr_ax,
                            kpts.reshape([14, 3]),
                            xaxis, yaxis,
                            scatter_kw = {'color': color},
                            line_kw = {'color': color, 'lw': 1},
                            label = label)
                    
                    if dolabel:
                        curr_ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5,), frameon = False)
                    if row_ofs == 0 and i_frame == 0:
                        curr_ax.set_ylabel(sess)

                    sns.despine(ax = curr_ax)

        fig.tight_layout()
        return fig
    
    return {f'{plot_name}-init': plot_for_params(init.morph),
            f'{plot_name}-fit': plot_for_params(fit['fit_params'].morph)}


defaults = dict(
    colorby = 'body',
    origin_keypt = 'hips'
)
