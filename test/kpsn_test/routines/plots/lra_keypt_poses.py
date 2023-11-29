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
    all_keypts = alignment.sagittal_align_insert_redundant_subspace(
        dataset['keypts'], cfg['origin_keypt'], skeleton.default_armature)
    params = init.morph


    frames = viz.diagram_plots.pose_gallery_ixs(
        all_keypts[meta['session_slice'][cfg['ref_sess']]],
        skeleton.default_armature)

    bodies, body_groups = keypt_io.get_groups_dict(meta[cfg['colorby']])
    pal = viz.defaults.age_pal(meta[cfg['colorby']])

    fig, ax = plt.subplots(
        nrows = 2*len(meta[cfg['colorby']]), ncols = len(frames),
        figsize = (3 * len(frames), 4 * len(meta['session_slice'])), sharex = 'col')

    for i_frame, (frame_name, frame) in enumerate(frames.items()):
        for i_body, (body, body_group) in enumerate(zip(bodies, body_groups)):

            tgt_sess = body_group[0]
            tgt_slc = meta['session_slice'][tgt_sess]
            src_slc = meta['session_slice'][cfg['ref_sess']]
            
            for row_ofs, xaxis, yaxis in [(0, 0, 1), (1, 0, 2)]:
                
                curr_ax = ax[2*i_body + row_ofs, i_frame]
                dolabel = (row_ofs == 0) and (i_frame == len(frames) - 1)

                # reference
                viz.diagram_plots.plot_mouse(
                    curr_ax,
                    all_keypts[src_slc][frame].reshape([14, 3]),
                    xaxis, yaxis,
                    scatter_kw = {'color': '.6'},
                    line_kw = {'color': '.6'},
                    label = cfg['ref_sess'])
                
                if tgt_sess != cfg['ref_sess']:
                    # dataset
                    viz.diagram_plots.plot_mouse(
                        curr_ax,
                        all_keypts[tgt_slc][frame].reshape([14, 3]),
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
    origin_keypt = 'hips'
)