from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
import jax.numpy.linalg as jla
import numpy as np

import seaborn as sns

from kpsn.models.morph import affine_mode as afm

def plot_subjectwise_scalar_comparison(
    ax,
    scalars: Tuple[Float[Array, "n_subject"]],
    labels: Tuple[str],
    pal: Tuple = ('.5', '.3'),
    subject_whitelist = None
    ):
   
    if subject_whitelist is None:
        subject_whitelist = np.arange(len(scalars[0]))
    n_subject = len(subject_whitelist)
    n_scalars = len(scalars)
    bar_width = 0.1
    bar_ofs = 0.1
    first_bar_ofs = (n_scalars - 1) * bar_width
    for i_bar in range(n_scalars):
        curr_scalars = np.array(scalars[i_bar])[subject_whitelist]
        ax.bar(
            np.arange(n_subject) - first_bar_ofs + bar_ofs * i_bar,
            jnp.maximum(curr_scalars - 1, 0) - jnp.minimum(curr_scalars - 1, 0),
            bottom = jnp.minimum(curr_scalars - 1, 0) + 1,
            color = pal[i_bar % len(pal)], zorder = 2, width = bar_width,
            label = labels[i_bar]
        )
    ax.set_xlim([-0.5, n_subject - 1 + 0.5])
    ax.set_ylim([
        np.array(scalars).min() - 0.05,
        np.array(scalars).max() + 0.05])
    for tick in ax.get_yticks()[1:-1]:
        ax.axhline(tick, ls = '--', lw = 0.5, color = '.7', zorder = 1)
    ax.legend(loc = "upper right", fancybox = False, edgecolor = 'w', framealpha = 0.9)
    
    ax.set_xticks(np.arange(n_subject))
    ax.set_xticklabels([f"Subject {subj_id}" for subj_id in subject_whitelist])
    ax.set_ylabel("Subject scale")
    sns.despine(ax = ax)
        


def plot_vector(
    ax,
    dims: Float[Array, "2"],
    origins: Float[Array, "2"] = np.zeros([2, 2]),
    arrow_kws = {},
    origin_kws = {},
    display_scale = 1,
    ):
    vec = display_scale * (dims - origins)
    ax.arrow(
        origins[0], origins[1], 
        vec[0], vec[1],
        length_includes_head = True,
        head_length = 0.08 * jla.norm(vec),
        head_width = 0.08 * jla.norm(vec),
        **arrow_kws)
    ax.plot(
        [origins[0]], [origins[1]],
        'ko', **origin_kws)


def _find_pc_display_scale(
    data: Float[Array, "n_pts n_dim"],
    pc: Float[Array, "n_dim"],
    center: Float[Array, "n_dim"],
    pct_scale = 0.5):
    """Find scale of a PC vector to be visible in data scale."""
    norm_pc = pc / jla.norm(pc)
    pc_coord = ((data - center[None]) * norm_pc[None]).sum(axis = -1)
    data_range = pc_coord[pc_coord > 0].max()
    desired_norm = pct_scale * data_range
    return desired_norm / jla.norm(pc)


def plot_morph_action(
    ax,
    morph_params: afm.AFMParameters,
    vectors: Float[Array, "morph_in_dim n_vector"],
    display_scale = 1,
    display_scale_dataset = Float[Array, "n_pts morph_in_dim"],
    subject_whitelist: Tuple[int] = None,
    ):
    """Display input-output mapping of a morph on given vectors."""
    
    if subject_whitelist is None:
        subject_whitelist = np.arange(morph_params.N)
    n_vector = vectors.shape[1]

    _, pop_offset, sess_offsets = afm.get_transform(morph_params)
    morphed = afm.transform(morph_params,
        vectors.T[None], # [1, n_vector, morph_in_dim])
        sess_ids = np.array(subject_whitelist)[:, None])
    
    if display_scale_dataset is not None:
        display_scale = max([_find_pc_display_scale(
            display_scale_dataset,
            vectors[:, i_vector] - pop_offset,
            pop_offset,
            pct_scale = 0.5 * display_scale)
            for i_vector in range(n_vector)])
    
    for i_subj, subj_id in enumerate(subject_whitelist):
        for i_vector, ls, col in zip(
                range(min(n_vector, 2)),
                ['-', '-'],
                ['.4', '.6']
                ):

            arrow_kws = dict(lw = 1, color = col, ls = ls)
            origin_kws = dict(ms = 3)

            plot_vector(
                ax[0, i_subj],
                morphed[subj_id, i_vector, :],
                sess_offsets[subj_id],
                arrow_kws, origin_kws, display_scale)
            plot_vector(
                ax[1, i_subj],
                vectors[:, i_vector],
                pop_offset,
                arrow_kws, origin_kws, display_scale)



def plot_morph_action_standard_basis(
    ax,
    morph_model,
    morph_hyperparams,
    morph_params,
    display_scale = 1,
    subject_whitelist = None
    ):
    plot_morph_action(ax, morph_model, morph_hyperparams, morph_params,
                      jnp.eye(2), display_scale, subject_whitelist)



def plot_morph_dimensions(
    ax,
    morph_params, 
    display_scale = 1,
    display_scale_dataset = None,
    subject_whitelist = None,
    ):
    """Display input-output mapping of a 2D affine-modal morph.
    
    Plots image of each mode under $C(\phi_n)$ for any morph model
    with a modes attribute. Columns of `ax` array show each subject,
    with the image plotted in the first row (keypoint space) and the
    morph mode plotted in the second row (pose space).
    """
    mode_poses = morph_params.modes + morph_params.offset[:, None]
    plot_morph_action(
        ax,
        morph_params,
        mode_poses,
        display_scale,
        display_scale_dataset,
        subject_whitelist)