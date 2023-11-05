from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
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
        


def plot_paired_vectors(
    ax,
    dims: Float[Array, "2 2"],
    origins: Float[Array, "2 2"] = np.zeros([2, 2]),
    artist_kws = {},
    display_scale = 1,
    ):
    for i in range(2):
        ax[i].arrow(
            origins[i, 0], origins[i, 1], 
            display_scale * (dims[i, 0] - origins[i, 0]), 
            display_scale * (dims[i, 1] - origins[i, 1]),
            length_includes_head = True,
            **artist_kws)


def plot_morph_action(
    ax,
    morph_params: afm.AFMParameters,
    vectors: Float[Array, "morph_in_dim n_vector"],
    display_scale = 1,
    subject_whitelist: Tuple[int] = None,
    ):
    """Display input-output mapping of a morph on given vectors."""
    if subject_whitelist is None:
        subject_whitelist = np.arange(morph_params.N)
    
    _, pop_offset, sess_offsets = afm.get_transform(morph_params)
    morphed = afm.transform(morph_params,
        vectors.T[None], # [1, n_vector, morph_in_dim])
        sess_ids = np.array(subject_whitelist)[:, None])
    print("morphed:", morphed.shape)
    n_vector = vectors.shape[1]
    N = len(subject_whitelist)
    for i_subj, subj_id in enumerate(subject_whitelist):
        for i_vector, ls, col in zip(
                range(min(n_vector, 2)),
                ['-', '-'],
                ['.4', '.6']
                ):
            plot_paired_vectors(
                ax[:, i_subj],
                jnp.stack([
                    morphed[subj_id, i_vector, :],
                    vectors[:, i_vector]
                ], axis = 0),
                origins = jnp.stack([
                    sess_offsets[subj_id],
                    pop_offset,
                ]),
                artist_kws = dict(
                    lw = 1,
                    head_width = 0.6 * np.sqrt(display_scale),
                    head_length = 0.6 * np.sqrt(display_scale),
                    color = col, ls = ls),
                display_scale = display_scale
            )
        ax[0, i_subj].plot(
            [sess_offsets[subj_id, 0]], [sess_offsets[subj_id, 1]],
            'ko', ms = 3)
        ax[1, i_subj].plot([pop_offset[0]], [pop_offset[1]], 'ko', ms = 3)


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
    subject_whitelist = None,
    ):
    """Display input-output mapping of a 2D affine-modal morph.
    
    Plots image of each mode under $C(\phi_n)$ for any morph model
    with a modes attribute. Columns of `ax` array show each subject,
    with the image plotted in the first row (keypoint space) and the
    morph mode plotted in the second row (pose space).
    """
    plot_morph_action(
        ax,
        morph_params,
        morph_params.modes,
        display_scale,
        subject_whitelist)