
from kpsn.models import pose
from kpsn.models.morph import affine_mode as afm
from kpsn.util import keypt_io, alignment
import kpsn_test.visualize as viz

import matplotlib.pyplot as plt

def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):
    
    morph_model = afm.AffineModeMorph
    # gt_obs = pose.Observations(dataset['keypts'], dataset['subject_ids'])
    metadata = dataset['metadata']
    to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])

    group_keys, groups = keypt_io.get_groups_dict(metadata[cfg['groupby']])
    plot_subset = [group[0] for group in groups]

    def plot_for_params(params, ttl):
        fig, ax = plt.subplots((params.morph.L * 2 + 2), len(plot_subset),
        figsize = (2 * len(plot_subset), 1.5 * (params.morph.L * 2 + 2)),
        sharex = 'row', sharey = 'row')

        age_pal = viz.defaults.age_pal(metadata[cfg['groupby']])

        all_poses = morph_model.inverse_transform(
                params.morph,
                dataset['keypts'],
                dataset['subject_ids'])
        quantiles = viz.affine_mode.mode_quantiles(params.morph, all_poses, 0.9)

        viz.affine_mode.mode_body_diagrams(
            params.morph, quantiles,
            metadata[cfg['groupby']], None,
            metadata['session_ix'],
            0, 1, plot_subset, age_pal,
            keypt_conv = to_kpt,
            ax = ax[0::2])

        viz.affine_mode.mode_body_diagrams(
            params.morph, quantiles,
            metadata[cfg['groupby']], None,
            metadata['session_ix'],
            0, 2, plot_subset, age_pal,
            keypt_conv = to_kpt,
            ax = ax[1::2], titles = False, label_suff = None)

        fig.suptitle(ttl)
        fig.tight_layout()
        return fig

    return {
        f'{plot_name}-init': plot_for_params(init, "Model: init"),
        f'{plot_name}-fit': plot_for_params(fit['fit_params'], "Model: fit")
    }


defaults = dict(
    groupby = 'age',
    origin_keypt = 'hips'
)