from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns

from jaxtyping import Float, Array
from typing import Tuple, Union
import jax.numpy as jnp
import numpy as np


def ellipse_from_cov(
    x: Float [Array, "2"],
    A: Float[Array, "2 2"],
    **artist_kwargs):

    artist_kwargs = {**dict(
        lw = 1,
        edgecolor = 'k',
        fill = False
    ), **artist_kwargs}

    eig = jnp.linalg.eigvals(A).real
    if A[0, 1] == 0:
        if A[0, 0] > A[1, 1]: theta = 0
        else: theta = jnp.pi/2
    else: theta = jnp.arctan2(eig[0] - A[0, 0], A[1, 1])
    
    return Ellipse(
        xy = (x[0], x[1]),
        width = jnp.sqrt(eig[0]),
        height = jnp.sqrt(eig[1]),
        angle = theta,
        **artist_kwargs
    )

def fading_cov_ellipses(
    x: Float[Array, "2"],
    A: Float[Array, "2 2"],
    std = [1, 2, 3],
    alpha_range = (1, 0.3),
    ax = None,
    **artist_kwargs
    ):

    alphas = np.linspace(*alpha_range, len(std))
    ret = []
    for s, a in zip(std, alphas):
        ellipse = ellipse_from_cov(x, (s**2) * A, **{**artist_kwargs, **{'alpha': a}})
        ret.append(ellipse)
        if ax is not None: ax.add_artist(ellipse)
    marginal_stds = np.sqrt(np.diag(A))
    vmin = x - (std[-1] + 1) * marginal_stds
    vmax = x + (std[-1] + 1) * marginal_stds
    return ret, np.array([vmin, vmax]).T


def cov_ellipse_vrng(
    x: Float[Array, "N 2"],
    A: Float[Array, "N 2 2"],
    ) -> Tuple[Float[Array, "N 2"]]:
    """
    Returns:
        :param vrng: Tuple of `xrng`, `yrng`"""
    vmin, vmax = np.zeros([2, 2])
    for n in range(len(A)):
        stds = np.sqrt(np.diagonal(A[n], axis1 = -2, axis2 = -1))
        vmin = np.minimum(vmin, x[n] - 4 * stds)
        vmax = np.maximum(vmax, x[n] + 4 * stds)
    return np.stack([vmin, vmax]).T


def combine_vrng(*vrngs):
    """
    Returns:
        :param vrng: Tuple of `xrng`, `yrng`"""
    vmins, vmaxs = zip(*(np.array(vrng).T for vrng in vrngs))
    vmin = np.minimum(*vmins)
    vmax = np.maximum(*vmaxs)
    return np.array([vmin, vmax]).T

def apply_vrng(ax, vrng):
    ax.set_xlim(*vrng[0])
    ax.set_ylim(*vrng[1])
    ax.set_aspect(1)
    

def plot_many_fading_cov_ellipses(
    x: Float[Array, "N 2"],
    A: Float[Array, "N 2 2"],
    ax = None,
    pal = 'Set1',
    **artist_kwargs
    ):
    if isinstance(pal, str): pal = sns.color_palette(pal, n_colors = len(A))
    for n in range(len(A)):
        fading_cov_ellipses(
            x[n], A[n],
            ec = pal[n], fc = pal[n], ax = ax,
            **artist_kwargs)
    return cov_ellipse_vrng(x, A)

def plot_cov_ellipses_comparison(
    x1: Float[Array, "N 2"], A1, x2, A2,
    ax, pal = 'Set1',
    **artist_kwargs
    ):

    vrng1 = plot_many_fading_cov_ellipses(
        x1, A1,
        pal = np.zeros([len(x1), 3]), ax = ax,
        **{'lw': 0.5, **artist_kwargs})
    vrng2 = plot_many_fading_cov_ellipses(
        x2, A2,
        pal = pal,
        ax = ax,
        **{**artist_kwargs, 'fill': True, 'lw': 0})
    vrng = combine_vrng(vrng1, vrng2)

    return vrng
        




def dirichlet_blocks(
    p: Float[Array, "n"],
    pal = "Set1",
    ax = None,
    loc = 0,
    width = 0.3,
    hline = None,
    **artist_kwargs
    ):
    ret = []
    if isinstance(pal, str): pal = sns.color_palette(pal, n_colors = len(p))
    pts = jnp.concatenate([jnp.array([0]), jnp.cumsum(p)])
    for i, _ in enumerate(pts[:-1]):
        ret.append(Rectangle(
            xy = (loc - width / 2, pts[i]),
            width = width, height = p[i],
            **{**dict(fc = pal[i], lw = 0), **artist_kwargs}
        ))
        if ax is not None:
            ax.add_artist(ret[-1])
            if hline is not None and i != 0:
                ax.axhline(pts[i], **hline)
    return ret


def compare_dirichlet_blocks(
    p1: Float[Array, "n"],
    p2: Union[Float[Array, "n"], Float[Array, "k n"]],
    ax, labels, pal = "Set1",
    ):

    dirichlet_blocks(
        p1, pal = pal, ax = ax,
        loc = 0, width = 0.3,
        lw = 1, ec = 'w', alpha = 0.5,
        zorder = 1,
        hline = dict(ls = '--', lw = 0.5, color = '.7', zorder = -1))
    
    if jnp.array(p2).ndim == 1:
        p2 = [p2]
    for i, p in enumerate(p2):
        dirichlet_blocks(
            p, pal = pal, ax = ax,
            loc = i + 1, width = 0.3,
            lw = 1, ec = 'w',
            zorder = 1)
    
    ax.set_xlim(-0.5, len(p2) + 0.5)
    ax.set_xticks(jnp.arange(len(p2) + 1), labels, rotation = 75)
    ax.set_yticks([])
    sns.despine(ax = ax, bottom = True, left = True)

