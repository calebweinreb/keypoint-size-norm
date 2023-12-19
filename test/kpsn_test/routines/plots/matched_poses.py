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

    # --- find frames to plot
    if dataset['keypts'].shape[-1] < 42:
        to_kpt = lambda arr: alignment.sagittal_align_insert_redundant_subspace(
            arr, cfg['origin_keypt'], skeleton.default_armature)
    else:
        to_kpt = lambda arr: arr
    slices = meta['session_slice']
    ref_sess = cfg['ref_sess']
    all_src_kpts = to_kpt(dataset['keypts'][slices[ref_sess]])
    frames = viz.diagram_plots.pose_gallery_ixs(
        all_src_kpts, skeleton.default_armature)

    # --- set up grouping and figure
    bodies, body_groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    def plot_for_params(curr_param):

        reconst, subj_ids = viz.affine_mode.reconst_feat_with_params(
            dataset['keypts'][slices[ref_sess]], meta['session_ix'][ref_sess],  
            curr_param, len(slices), return_subj_ids = True)
        reconst_kpts = {
            s: to_kpt(reconst[subj_ids == meta['session_ix'][s]]).reshape([-1, 14, 3])
            for s in slices}

        fig, ax = plt.subplots(
            nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
            figsize = (3 * len(frames), 4 * len(slices)), sharex = 'col')

        for i_frame, (frame_name, frame) in enumerate(frames.items()):
            for i_body, (body, body_group) in enumerate(zip(bodies, body_groups)):

                tgt_sess = body_group[0]

                # --- hop into (src_sess morph) and out of (tgt_sess morph) pose
                # space
                src_kpt = all_src_kpts[frame]

                # --- plot results of trasnform
                for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                    curr_ax = ax[2*i_body + row_ofs, i_frame]
                    dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                    # plot both reference and transformed
                    
                    for kpts, color, label in [(reconst_kpts[tgt_sess][frame], pal[body], tgt_sess),
                                               (src_kpt, '.6', ref_sess)]:
                        
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
                        curr_ax.set_ylabel(tgt_sess)

                    sns.despine(ax = curr_ax)

        fig.tight_layout()
        return fig
    
    return {f'{plot_name}-init': plot_for_params(init.morph),
            f'{plot_name}-fit': plot_for_params(fit['fit_params'].morph)}


defaults = dict(
    ref_sess = 'subj0',
    colorby = 'body',
    origin_keypt = 'hips'
)
