from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
import numpy as np

import seaborn as sns

def plot_subjectwise_scalar_comparison(
    ax,
    scalars: Float[Array, "n_subject 2"],
    labels: Tuple[str],
    pal: Tuple = ('.5', '.3'),
    ):
    n_subject = len(scalars)
    ax.bar(
        np.arange(n_subject) - 0.1,
        jnp.maximum(scalars[:, 0] - 1, 0) - jnp.minimum(scalars[:, 0] - 1, 0),
        bottom = jnp.minimum(scalars[:, 0] - 1, 0) + 1,
        color = pal[0], zorder = 2, width = 0.1,
        label = labels[0]
    )
    ax.bar(
        np.arange(n_subject) + 0.1,
        jnp.maximum(scalars[:, 1] - 1, 0) - jnp.minimum(scalars[:, 1] - 1, 0),
        bottom = jnp.minimum(scalars[:, 1] - 1, 0) + 1,
        color = pal[1], zorder = 2, width = 0.1,
        label = labels[1]
    )
    ax.set_xlim([-0.5, n_subject - 1 + 0.5])
    ax.set_ylim([scalars.min() - 0.05, scalars.max() + 0.05])
    for tick in ax.get_yticks()[1:-1]:
        ax.axhline(tick, ls = '--', lw = 0.5, color = '.7', zorder = 1)
    ax.legend(loc = "upper right", fancybox = False, edgecolor = 'w', framealpha = 0.9)
    
    ax.set_xticks(np.arange(n_subject))
    ax.set_xticklabels([f"Subject {i_subj}" for i_subj in range(n_subject)])
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
    morph_model,
    morph_hyperparams,
    morph_params,
    vectors: Float[Array, "morph_in_dim n_vector"],
    display_scale = 1):
    """Display input-output mapping of a morph on given vectors."""
    morph_matrix, morph_ofs = morph_model.get_transform(
        morph_params, morph_hyperparams)
    morphed = ( # (n_subj, morph_out_dim, n_vector)
        (morph_matrix @ # (n_subj, morph_out_dim, morph_in_dim)
         vectors[None]) # (1, morph_in_dim, n_vector)
       + morph_ofs[..., None] # (n_subj, morph_out_dim, 1)
    )
    n_vector = vectors.shape[1]
    for i_subj in range(morph_hyperparams.N):
        for i_vector, ls, col in zip(
                range(min(n_vector, 2)),
                ['-', '-'],
                ['.4', '.6']
                ):
            plot_paired_vectors(
                ax[:, i_subj],
                jnp.stack([
                    morphed[i_subj, :, i_vector],
                    vectors[:, i_vector]
                ], axis = 0),
                origins = jnp.stack([
                    morph_ofs[i_subj],
                    jnp.zeros(2),
                ]),
                artist_kws = dict(
                    lw = 1, head_width = 0.6, head_length = 0.6,
                     color = col, ls = ls),
                display_scale = display_scale
            )
        ax[0, i_subj].plot(
            [morph_ofs[i_subj, 0]], [morph_ofs[i_subj, 1]],
            'ko', ms = 3)
        ax[1, i_subj].plot([0], [0], 'ko', ms = 3)


def plot_morph_action_standard_basis(
    ax,
    morph_model,
    morph_hyperparams,
    morph_params,
    display_scale = 1,
    ):
    plot_morph_action(ax, morph_model, morph_hyperparams, morph_params,
                      jnp.eye(2), display_scale)



def plot_morph_dimensions(
    ax,
    morph_hyperparams,
    morph_model,
    morph_params, 
    scale = 1
    ):
    """Display input-output mapping of a 2D affine-modal morph.
    
    Plots image of each mode under $C(\phi_n)$ for any morph model
    with a modes attribute. Columns of `ax` array show each subject,
    with the image plotted in the first row (keypoint space) and the
    morph mode plotted in the second row (pose space).
    """
    morphed_modes = (
        morph_model.get_transform(morph_params, morph_hyperparams) @
        morph_params.modes[None]
    )
    for i_subj in range(morph_hyperparams.N):
        for i_dim, ls, col in zip(
                range(min(morph_params.modes.shape[1], 2)),
                ['-', '--'],
                ['.4', '.6']
                ):
            plot_paired_vectors(
                ax[:, i_subj],
                jnp.stack([
                    scale * morphed_modes[i_subj, :, i_dim],
                    scale * morph_params.modes[:, i_dim]
                ], axis = 0),
                dict(lw = 1, head_width = 0.6, head_length = 0.6,
                     color = col, ls = ls)
            )