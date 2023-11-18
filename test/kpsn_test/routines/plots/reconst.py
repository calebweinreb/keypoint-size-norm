
from kpsn.models import pose
from kpsn.models.morph import affine_mode as afm
from kpsn.util import keypt_io, skeleton
import kpsn_test.visualize as viz

import matplotlib.pyplot as plt
import numpy as np

def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs,
    ):

    metadata = dataset['metadata']

    def single_plot(params, pctile):

        group_keys, groups = keypt_io.get_groups_dict(metadata[cfg['groupby']])
        plot_compare_subset = [group[0] for group in groups]

        skel = skeleton.default_armature
        head_heights = dataset['keypts'].reshape([-1, skel.n_kpts, 3])[:, skel.keypt_by_name['head'], 2]
        plot_ref_frame = np.argsort(abs(head_heights - np.quantile(head_heights, pctile)))[0]

        ref_sess_name, fig, ax = viz.affine_mode.mode_reconstruction_diagrams(
            plot_ref_frame,
            dataset['keypts'],
            params.morph,
            metadata[cfg['groupby']],
            metadata[cfg['subj_id']],
            metadata['session_ix'],
            dataset['subject_ids'],
            metadata['session_slice'],
            0, 2,
            plot_compare_subset
        )
        fig.suptitle(f"Reference: m{metadata[cfg['subj_id']][ref_sess_name]} | " + 
                     f"{metadata[cfg['groupby']][ref_sess_name]}wk")

        return fig

    return {
        f'{plot_name}-init-high': single_plot(init, 0.9),
        f'{plot_name}-init-low':  single_plot(init, 0.2),
        f'{plot_name}-fit-high':  single_plot(fit['fit_params'], 0.9),
        f'{plot_name}-fit-low':   single_plot(fit['fit_params'], 0.2),
    }
    
    
defaults = dict(
    groupby = 'age',
    subj_id = 'id'
)