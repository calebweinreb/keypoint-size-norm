from typing import Tuple
from jaxtyping import Float, Array
import jax.numpy as jnp
import numpy as np

import seaborn as sns

def plot_subjectwise_scalar_comparison(
    ax,
    scalars: Float[Array, "n_subject 2"],
    labels: Tuple[str],
    pal: Tuple = ('.5', '.3')
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
    artist_kws
    ):
    for i in range(2):
        ax[i].arrow(
            0, 0, dims[i, 0], dims[i, 1],
            length_includes_head = True,
            **artist_kws)

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