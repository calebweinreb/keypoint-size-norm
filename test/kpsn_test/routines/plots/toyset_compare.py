import matplotlib.pyplot as plt
import seaborn as sns

from kpsn_test import visualize as viz
from kpsn.util import skeleton, alignment, keypt_io
from kpsn.models.morph import affine_mode as afm


def plot_for_params(dataset, params, cfg):

    meta = dataset['metadata']

    # --- setup: find representative frames    
    to_kpt = lambda arr: alignment.sagittal_align_insert_redundant_subspace(
        arr, cfg['origin_keypt'], skeleton.default_armature)

    src_slc = meta['session_slice'][cfg['ref_sess']]
    src_all_kpts = to_kpt(dataset['keypts'][src_slc])

    frames = viz.diagram_plots.pose_gallery_ixs(
        src_all_kpts, skeleton.default_armature)

    # --- setup: create groups and figure
    bodies, body_groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    fig, ax = plt.subplots(
        nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
        figsize = (3 * len(frames), 4 * len(meta['session_slice'])), sharex = 'col')

    # --- iterate: get keypoints of first session per group
    for i_frame, (frame_name, frame) in enumerate(frames.items()):
        for i_body, (body, body_group) in enumerate(zip(bodies, body_groups)):

            tgt_sess = body_group[0]
            tgt_slc = meta['session_slice'][tgt_sess]
            
            src_kpts = src_all_kpts[frame]
            tgt_kpts = to_kpt(dataset['keypts'][tgt_slc][frame]) 

            # --- morph from reference subject to target
            pose = afm.inverse_transform(
                params, dataset['keypts'][src_slc][frame], meta['session_ix'][cfg['ref_sess']])
            reconst = afm.transform(
                params, pose, meta['session_ix'][tgt_sess])
            reconst_kpts = to_kpt(reconst)
            
            for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                curr_ax = ax[2*i_body + row_ofs, i_frame]
                kpt_shape = [skeleton.default_armature.n_kpts, 3]
                dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                # reference
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    src_kpts.reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': '.6'},
                    line_kw = {'color': '.6'},
                    label = cfg['ref_sess'])
                # ground truth target
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    tgt_kpts.reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': 'k', 's': 6},
                    line_kw = {'color': 'k', 'lw': 1},
                    label = f'{tgt_sess}, datset')
                # morphed target
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    reconst_kpts.reshape(kpt_shape),
                    xaxis, yaxis,
                    scatter_kw = {'color': pal[body], 's': 0},
                    line_kw = {'color': pal[body], 'lw': 1},
                    label = f'{tgt_sess} reconst.')
                
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
    origin_keypt = 'hips'
)   