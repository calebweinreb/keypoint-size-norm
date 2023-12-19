from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

from kpsn.models.morph import affine_mode as afm


from .diagram_plots import plot_mouse
from . import defaults

def mode_quantiles(
    params: afm.AFMParameters,
    poses: Float[Array, "*#K n_samples M"],
    quantile: float
    ) -> Float[Array, "*#K L"]:
    coords, components, complement = afm.mode_components(params, poses)
    return jnp.quantile(coords, quantile, axis = -2)


def reconst_feat_with_params(ref_feats, ref_ix, params, N, subj_ixs = None, return_subj_ids = False):
    """
    Transform `ref_feats` to bodies of `N` animals under given params."""

    pose = afm.inverse_transform(
        params, ref_feats, ref_ix)
    
    copied_poses = jnp.concatenate([pose for _ in range(N)])
    n_frame = len(ref_feats) 
    subj_ids = jnp.broadcast_to(jnp.arange(N)[:, None], [N, n_frame]).ravel()
    if subj_ixs is not None:
        subj_ids = jnp.array(subj_ixs)[subj_ids]

    ret = afm.transform(params, copied_poses, subj_ids)
    if return_subj_ids: ret = (ret, subj_ids)
    return ret



def mode_body_diagrams(
    params: afm.AFMParameters,
    mode_magnitudes: Float[Array, "L"],
    ages: list, # indexed by vid_id
    subj_ids: list, # indexed by vid_id,
    session_ixs: dict, # vid_id (str) -> vid_ix(int)
    xaxis: int,
    yaxis: int,
    plot_subset: Tuple[int],
    age_pal: dict,
    titles = True,
    label_suff = '',
    keypt_conv = lambda arr: arr,
    ax = None,
    ):

    age_diagram_kws = {
        age: dict(
            scatter_kw = dict(color = age_pal[age]),
            line_kw = dict(linestyle='-', color = age_pal[age], lw = 0.5))
        for age in age_pal}
    ref_diagram_kws = dict(
        scatter_kw = dict(color = '.8'),
        line_kw = dict(linestyle='-', color = '.8', lw = 0.5))

    ax_was_none = False
    if ax is None:
        ax_was_none = True
        fig, ax = plt.subplots(params.L + 1, len(plot_subset),
            figsize = (2 * len(plot_subset), 2 * (params.L + 1)),
            sharex = 'row', sharey = 'row')
    
    mode_display = params.modes * mode_magnitudes[None] + params.offset[:, None] # (M, L)

    for i_subj, vid_id in enumerate(plot_subset):
        vid_ix = session_ixs[vid_id]

        offset = (params.offset + params.offset_updates[vid_ix])
        for a in ax[1:, i_subj].ravel():
            plot_mouse(
                keypt_frame = keypt_conv(offset).reshape(14, 3),
                xaxis = xaxis, yaxis = yaxis, ax = a,
                **ref_diagram_kws)

        curr_age = ages[vid_id]
        plot_mouse(
            keypt_frame = keypt_conv(offset).reshape(14, 3),
            xaxis = xaxis, yaxis = yaxis, ax = ax[0, i_subj],
            **age_diagram_kws[curr_age]
        )
        if titles:
            ax[0, i_subj].set_title(
                (f"m{subj_ids[vid_id]} | {ages[vid_id]}wk"
                 if subj_ids is not None else f"{ages[vid_id]}wk"),
                fontsize = 7)
        if label_suff is not None:
            ax[0, 0].set_ylabel(f"Centroid pose{label_suff}", fontsize = 6)

        for i_mode in range(params.L):
            mode_pose = afm.AffineModeMorph.transform(
                params, mode_display[:, i_mode], jnp.array([vid_ix]))
            plot_mouse(
                keypt_frame = keypt_conv(mode_pose).reshape(14, 3),
                xaxis = xaxis, yaxis = yaxis, ax = ax[i_mode + 1, i_subj],
                **age_diagram_kws[curr_age]
            )
            if label_suff is not None:
                ax[i_mode + 1, 0].set_ylabel(f"Mode {i_mode}{label_suff}", fontsize = 6)


    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
        sns.despine(left = True, bottom = True)

    if ax_was_none:
        return fig, ax
    

def mode_reconstruction_diagrams(
    plot_ref_frame: int,
    keypts: Float[Array, "Nt M"],
    params: afm.AFMParameters,
    ages: list, # indexed by vid_id
    subj_ids: list, # indexed by vid_id,
    session_ixs: dict, # vid_id (str) -> vid_ix(int)
    session_ids: Float[Array, "Nt"],
    slices: dict,
    xaxis: int,
    yaxis: int,
    plot_subset: Tuple[int],
    ):

    age_pal = defaults.age_pal(ages)
    age_diagram_kws = {
        age: dict(
            scatter_kw = dict(color = age_pal[age]),
            line_kw = dict(linestyle='-', color = age_pal[age], lw = 0.5))
        for age in age_pal}
    ref_diagram_kws = dict(
        scatter_kw = dict(color = '.8'),
        line_kw = dict(linestyle='-', color = '.8', lw = 0.5))

    n_rows = len(plot_subset)
    n_cols = params.L + 2
    fig, ax = plt.subplots(
        n_rows, n_cols,
        sharex = True, sharey = True,
        figsize = (2 * n_cols, 2 * n_rows))

    ref_frame_kpts = keypts[plot_ref_frame]
    ref_subj_ix = session_ids[plot_ref_frame]
    ref_subj_name = session_ixs.inverse[ref_subj_ix]

    ref_frame_pose = afm.inverse_transform(
        params, ref_frame_kpts, ref_subj_ix)

    # unstacked_kpts = {sess_name: keypts[slices]}

    ref_diagram_plot = lambda kpts, ax: plot_mouse(
        keypt_frame = kpts.reshape(14, 3), ax = ax,
        xaxis = xaxis, yaxis = yaxis,  **ref_diagram_kws)
    age_diagram_plot = lambda kpts, ax, age: plot_mouse(
        keypt_frame = kpts.reshape(14, 3), ax = ax,
        xaxis = xaxis, yaxis = yaxis,  **age_diagram_kws[age])

    for i_subj, subj_name in enumerate(plot_subset):
        subj_age = ages[subj_name]

        # offset only
        offset_params = afm.as_offset_only(params)
        offset_kpts = afm.transform(
            offset_params,
            jnp.array(ref_frame_pose),
            jnp.array(session_ixs[subj_name])
            ).reshape(14, 3)
        ref_diagram_plot(ref_frame_kpts.reshape(14, 3), ax[i_subj, 0])
        age_diagram_plot(offset_kpts, ax[i_subj, 0], subj_age)
        ref_diagram_plot(offset_kpts, ax[i_subj, 1])

        # building up number of modes
        for l in range(1, params.L+1):
            limited_mode_params = afm.with_sliced_modes(
                params, slice(0, l))
            curr_kpts = afm.transform(
                    limited_mode_params,
                    jnp.array(ref_frame_pose),
                    jnp.array(session_ixs[subj_name]))
            age_diagram_plot(curr_kpts.reshape(14, 3), ax[i_subj, l], subj_age)
            ref_diagram_plot(curr_kpts.reshape(14, 3), ax[i_subj, l+1])

        # closest true set of keypoints to morphed
        subj_kpts = keypts[slices[subj_name]]
        closest_kpts = subj_kpts[jnp.argmin(
            ((subj_kpts - curr_kpts) ** 2).mean(axis = 1)
            )]
        age_diagram_plot(closest_kpts, ax[i_subj, -1], subj_age)

        ax[i_subj, 0].set_ylabel(f"m{subj_ids[subj_name]} | {subj_age}w", fontsize = 7)
        
    ax[0, 0].set_title("Offset-only", fontsize = 7)
    for l in range(params.L): ax[0, l+1].set_title(f"{l+1} mode(s)", fontsize = 7)
    ax[0, -1].set_title("Nearest frame", fontsize = 7)
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    sns.despine(left = True, bottom = True)

    return ref_subj_name, fig, ax