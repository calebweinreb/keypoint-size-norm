import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    to_kpt, to_feat = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])
    all_keypts = to_kpt(dataset['keypts'])
    params = init.morph

    if cfg['match_mode'] == 'meta':
        valid_frames = viz.diagram_plots.valid_display_frames(meta['frame_ids'].values())
    else:
        valid_frames = None
    frames = viz.diagram_plots.pose_gallery_ixs(
        all_keypts[meta['session_slice'][cfg['ref_sess']]],
        skeleton.default_armature,
        valid_frames = valid_frames)

    bodies, body_groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    fig, ax = plt.subplots(
        nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
        figsize = (3 * len(frames), 4 * len(meta['session_slice'])), sharex = 'col')

    for i_frame, (frame_name, frame_id) in enumerate(frames.items()):
        for i_body, (body, body_group) in enumerate(zip(bodies, body_groups)):

            tgt_sess = body_group[0]
            tgt_slc = meta['session_slice'][tgt_sess]
            src_slc = meta['session_slice'][cfg['ref_sess']]
            if cfg['match_mode'] == 'meta':
                tgt_frame = np.where(meta['frame_ids'][tgt_sess] == frame_id)[0][0]
                src_frame = np.where(meta['frame_ids'][cfg['ref_sess']] == frame_id)[0][0]
            elif cfg['match_mode'] == 'frame':
                tgt_frame = src_frame = frame_id
            else:
                raise ValueError("match_mode for lra_keypt_poses should be one of" +
                                 f" [frame, meta], got '{cfg['match_mode']}'.")
            
            for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                curr_ax = ax[2*i_body + row_ofs, i_frame]
                dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                # reference
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    all_keypts[src_slc][src_frame].reshape([14, 3]),
                    xaxis, yaxis,
                    scatter_kw = {'color': '.6'},
                    line_kw = {'color': '.6'},
                    label = cfg['ref_sess'])
                
                if tgt_sess != cfg['ref_sess']:
                    # dataset
                    viz.diagram_plots.plot_mouse(
                        curr_ax,
                        all_keypts[tgt_slc][tgt_frame].reshape([14, 3]),
                        xaxis, yaxis,
                        scatter_kw = {'color': pal[body], 's': 2},
                        line_kw = {'color': pal[body], 'lw': 1},
                        label = f'{tgt_sess}')
                
                if dolabel:
                    curr_ax.legend(
                        loc = 'center left',
                        bbox_to_anchor = (1, 0.5,),
                        frameon = False)
                sns.despine(ax = curr_ax)
    fig.tight_layout()
    
    return {plot_name: fig}


defaults = dict(
    ref_sess = 'subj0',
    colorby = 'body',
    origin_keypt = 'hips',
    match_mode = 'frame'
)