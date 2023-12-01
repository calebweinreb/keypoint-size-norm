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
    
    params = fit['fit_params'].morph
    meta = dataset['metadata']

    # --- find frames to plot
    if dataset['keypts'].shape[-1] < 42:
        to_kpt = lambda arr: alignment.sagittal_align_insert_redundant_subspace(
            arr, cfg['origin_keypt'], skeleton.default_armature)
    else:
        to_kpt = lambda arr: arr
    all_src_kpts = to_kpt(dataset['keypts'][meta['session_slice'][cfg['ref_sess']]])
    frames = viz.diagram_plots.pose_gallery_ixs(
        all_src_kpts, skeleton.default_armature)

    # --- set up grouping and figure
    bodies, body_groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    fig, ax = plt.subplots(
        nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
        figsize = (3 * len(frames), 4 * len(meta['session_slice'])), sharex = 'col')

    for i_frame, (frame_name, frame) in enumerate(frames.items()):
        for i_body, (body, body_group) in enumerate(zip(bodies, body_groups)):

            tgt_sess = body_group[0]
            src_slc = meta['session_slice'][cfg['ref_sess']]

            # perform pose space transformation for both init and fitted params
            for (curr_param, color, label) in [
                    (init.morph, '.6', "init" if tgt_sess != cfg['ref_sess'] else None),
                    (params, pal[body], "fit")]:

                # --- hop into (src_sess morph) and out of (tgt_sess morph) pose space
                pose = afm.inverse_transform(
                    curr_param,
                    dataset['keypts'][src_slc][frame],
                    meta['session_ix'][cfg['ref_sess']])
                reconst = afm.transform(
                    curr_param, pose, meta['session_ix'][tgt_sess])
                reconst_kpts = to_kpt(reconst)

                # --- plot results of trasnform
                for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                    curr_ax = ax[2*i_body + row_ofs, i_frame]
                    dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                    viz.diagram_plots.plot_mouse(
                        curr_ax,
                        reconst_kpts.reshape([14, 3]),
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
    
    return {plot_name: fig}


defaults = dict(
    ref_sess = 'subj0',
    colorby = 'body',
    origin_keypt = 'hips'
)
