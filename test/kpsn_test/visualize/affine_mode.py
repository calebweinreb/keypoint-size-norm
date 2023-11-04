from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

from kpsn.models.morph import affine_mode as afm

from .diagram_plots import plot_mouse

def mode_quantiles(
    params: afm.AffineModeMorphAllParameters,
    poses: Float[Array, "*#K n_samples M"],
    quantile: float
    ) -> Float[Array, "*#K L"]:
    coords, components, complement = afm.mode_components(params, poses)
    return jnp.quantile(coords, quantile, axis = -2)


def mode_body_diagrams(
    params: afm.AffineModeMorphAllParameters,
    mode_magnitudes: Float[Array, "L"],
    ages: list, # indexed by vid_id
    subj_ids: list, # indexed by vid_id,
    xaxis: int,
    yaxis: int,
    plot_subset: Tuple[int],
    age_pal: dict,
    ):

    age_diagram_kws = {
        age: dict(
            scatter_kw = dict(color = age_pal[age]),
            line_kw = dict(linestyle='-', color = age_pal[age], lw = 0.5))
        for age in age_pal}
    ref_diagram_kws = dict(
        scatter_kw = dict(color = '.8'),
        line_kw = dict(linestyle='-', color = '.8', lw = 0.5))

    fig, ax = plt.subplots(params.L + 1, len(plot_subset),
    figsize = (2 * len(plot_subset), 2 * (params.L + 1)),
        sharex = 'row', sharey = 'row')
    
    mode_display = params.modes() * mode_magnitudes[None] # (M, L)

    for i_subj, vid_id in enumerate(plot_subset):

        for a in ax[1:, i_subj].ravel():
            plot_mouse(
                keypt_frame = params.offsets()[vid_id].reshape(14, 3),
                xaxis = xaxis, yaxis = yaxis, ax = a,
                **ref_diagram_kws)

        curr_age = ages[vid_id]
        plot_mouse(
            keypt_frame = params.offsets()[vid_id].reshape(14, 3),
            xaxis = xaxis, yaxis = yaxis, ax = ax[0, i_subj],
            **age_diagram_kws[curr_age]
        )
        ax[0, i_subj].set_title(
            f"m{subj_ids[vid_id]} | {ages[vid_id]}wk",
            fontsize = 7)
        ax[0, 0].set_ylabel("Centroid pose", fontsize = 6)

        for i_mode in range(params.L):
            mode_pose = afm.AffineModeMorph.transform(
                params, mode_display[:, i_mode], vid_id)
            plot_mouse(
                keypt_frame = mode_pose.reshape(14, 3),
                xaxis = xaxis, yaxis = yaxis, ax = ax[i_mode + 1, i_subj],
                **age_diagram_kws[curr_age]
            )
            ax[i_mode + 1, 0].set_ylabel(f"Mode {i_mode}\nadjustment", fontsize = 6)


    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    sns.despine(left = True, bottom = True)

    return fig, ax