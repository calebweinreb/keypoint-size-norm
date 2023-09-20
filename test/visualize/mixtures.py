from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns
from matplotlib import gridspec

from jaxtyping import Float, Array, Integer
from typing import Tuple, Union
import jax.numpy as jnp
import numpy as np

from .fitting import gaussian_mle
from kpsn import util
from kpsn.models import pose
from kpsn.models.pose import gmm

def ellipse_from_cov(
    x: Float [Array, "2"],
    A: Float[Array, "2 2"],
    **artist_kwargs):

    artist_kwargs = {**dict(
        lw = 1,
        edgecolor = 'k',
        fill = False
    ), **artist_kwargs}

    
    a = A[0, 0]; b = A[0, 1]; c = A[1, 1]
    eig = [
        (a + c) / 2 + jnp.sqrt((a - c) ** 2 / 4 + b ** 2),
        (a + c) / 2 - jnp.sqrt((a - c) ** 2 / 4 + b ** 2)
    ]
    if A[0, 1] == 0:
        if A[0, 0] > A[1, 1]: theta = 0
        else: theta = jnp.pi/2
    else: theta = jnp.arctan2(eig[0] - A[0, 0], A[0, 1])

    
    return Ellipse(
        xy = (x[0], x[1]),
        width = jnp.sqrt(eig[0]) * 2,
        height = jnp.sqrt(eig[1]) * 2,
        angle = 180 / jnp.pi * theta,
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
    ) -> Tuple[Float[Array, "2"]]:
    """
    Returns:
        :param vrng: Tuple of `xrng`, `yrng`"""
    vmin, vmax = np.zeros([2, 2])
    for n in range(len(A)):
        stds = np.sqrt(np.diagonal(A[n], axis1 = -2, axis2 = -1))
        vmin = np.minimum(vmin, x[n] - 4 * stds)
        vmax = np.maximum(vmax, x[n] + 4 * stds)
    return np.stack([vmin, vmax]).T

def scatter_vrng(
    data: Float[Array, "N 2"],
    margin: float = 0.05
    ) -> Tuple[Float[Array, "2"]]:
    """
    Returns:
        :param vrng: Tuple of `xrng`, `yrng`"""
    vrng = np.stack([data.min(axis = 0), data.max(axis = 0)]).T
    vrng += np.array([[-1, 1]]) * margin * (vrng[:, 1:] - vrng[:, :1])
    return vrng

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
    thickness = 0.3,
    axline = None,
    orientation = 'vertical',
    **artist_kwargs
    ):
    ret = []
    if isinstance(pal, str): pal = sns.color_palette(pal, n_colors = len(p))
    pts = jnp.concatenate([jnp.array([0]), jnp.cumsum(p)])
    for i, _ in enumerate(pts[:-1]):
        if orientation == 'vertical':
            kws = dict(xy = (loc - thickness / 2, pts[i]),
                       width = thickness, height = p[i])
        else:
            kws = dict(xy = (pts[i], loc - thickness / 2),
                       width = p[i], height = thickness)
        ret.append(Rectangle(
            **{**dict(fc = pal[i], lw = 0), **artist_kwargs, **kws}
        ))
        if ax is not None:
            ax.add_artist(ret[-1])
            if axline is not None and i != 0:
                if orientation == 'vertical':
                    ax.axhline(pts[i], **axline)
                else: 
                    ax.axvline(pts[i], **axline)
    return ret


def compare_dirichlet_blocks(
    p1: Float[Array, "n"],
    p2: Union[Float[Array, "n"], Float[Array, "k n"]],
    ax, labels, pal = "Set1", orientation = 'vertical',
    label_kwargs = {}
    ):

    dirichlet_blocks(
        p1, pal = pal, ax = ax,
        loc = 0, thickness = 0.3,
        orientation = orientation,
        lw = 1, ec = 'w', alpha = 0.5,
        zorder = 1,
        axline = dict(ls = '--', lw = 0.5, color = '.7', zorder = -1))
    
    if jnp.array(p2).ndim == 1:
        p2 = [p2]
    for i, p in enumerate(p2):
        dirichlet_blocks(
            p, pal = pal, ax = ax,
            orientation = orientation,
            loc = i + 1, thickness = 0.3,
            lw = 1, ec = 'w',
            zorder = 1)
    
    if orientation == 'vertical':
        ax.set_xlim(-0.5, len(p2) + 0.5)
        ax.set_xticks(jnp.arange(len(p2) + 1), labels,
            **{**dict(rotation = 75), **label_kwargs})
        ax.set_yticks([])
    else:
        ax.set_ylim(-0.5, len(p2) + 0.5)
        ax.set_yticks(jnp.arange(len(p2) + 1), labels,
            **{**dict(rotation = 75), **label_kwargs})
        ax.set_xticks([])
    sns.despine(ax = ax, bottom = True, left = True)


def subjectwise_mixture_plot(
    ax,
    n_components: int,
    data: Float[Array, "n_points 2"],
    component_ids: Float[Array, "n_points"],
    model_means: Float[Array, "n_components 2"],
    model_covs: Float[Array, "n_components 2 2"],
    compare_means: Float[Array, "n_components 2"],
    compare_covs: Float[Array, "n_components 2"],
    pal = 'Set1',
    ):
    """
    Colored data points by `component_ids` with model Gaussian and empirical
    MLE.
    """
    if isinstance(pal, str):
        pal = np.array(sns.color_palette('Set1', n_colors = n_components))

    ax.scatter(
        x = data[:, 0],
        y = data[:, 1],
        c = pal[component_ids],
        s = 1,
    )
    vrng = scatter_vrng(data)
    sns.despine(ax = ax)
    ax.set_aspect(1.)

    if model_means is not None:
        vrng = combine_vrng(vrng, plot_cov_ellipses_comparison(
            compare_means, compare_covs,
            model_means, model_covs,
            pal = pal, ax = ax,
            alpha_range = (0.2, 0.4)))
        
    return vrng


    

def sampled_mixture_plot(
    fig,
    pose_hyperparams: gmm.GMMHyperparams,
    pose_model: gmm.GMMParameters,
    latents: gmm.GMMPoseStates,
    obs: pose.Observations,
):
    gs = gridspec.GridSpec(
        4, pose_hyperparams.N ,
        height_ratios = [4, 4, 1.5, 1.5])
    ax = np.array([[
        fig.add_subplot(gs[r, c])
        for c in range(pose_hyperparams.N)]for r in range(3)])

    comp_pal = np.array(sns.color_palette('Set1', n_colors = pose_hyperparams.L))

    empirical_means, empirical_covs = gaussian_mle(util.computations.unstack(
        arr = latents.poses,
        ixs = latents.components,
        N = pose_hyperparams.L, axis = 0
    ))

    vrng = None
    for i_subj in range(pose_hyperparams.N):
        scatrng = subjectwise_mixture_plot(
            ax = ax[0, i_subj],
            n_components = pose_hyperparams.L,
            data = obs.unstack(obs.keypts)[i_subj],
            component_ids = obs.unstack(latents.components)[i_subj],
            model_means = None, model_covs = None,
            compare_means = None, compare_covs = None,
            pal = comp_pal
        )
        # ax[0, i_subj].scatter(
        #     x = scatterdat[:, 0],
        #     y = scatterdat[:, 1],
        #     c = comp_pal[obs.unstack(latents.components)[i_subj]],
        #     s = 1,
        # )
        # scatrng = scatter_vrng(scatterdat)
        # sns.despine(ax=ax[0, i_subj])
        # ax[0, i_subj].set_aspect(1.)
        vrng = scatrng if vrng is None else combine_vrng(scatrng, vrng)
        if i_subj != 0:
            ax[0, i_subj].sharex(ax[0, 0])
            ax[0, i_subj].sharey(ax[0, 0])

        vrng = combine_vrng(vrng, subjectwise_mixture_plot(
            ax = ax[1, i_subj],
            n_components = pose_hyperparams.L,
            data = obs.unstack(latents.poses)[i_subj],
            component_ids = obs.unstack(latents.components)[i_subj],
            model_means = pose_model.means,
            model_covs = pose_model.covariances(),
            compare_means = empirical_means,
            compare_covs = empirical_covs,
            pal = comp_pal
        ))

        # ax[1, i_subj].scatter(
        #     x = obs.unstack(latents.poses)[i_subj][:, 0],
        #     y = obs.unstack(latents.poses)[i_subj][:, 1],
        #     c = comp_pal[obs.unstack(latents.components)[i_subj]],
        #     s = 1,)
        # sns.despine(ax=ax[1, i_subj])
        # ax[1, i_subj].set_aspect(1.)
        ax[1, i_subj].sharex(ax[0, 0])
        ax[1, i_subj].sharey(ax[0, 0])

        # vrng = combine_vrng(vrng, plot_cov_ellipses_comparison(
        #     pose_model.means, pose_model.covariances(),
        #     empirical_means, empirical_covs,
        #     pal = comp_pal, ax = ax[1, i_subj],
        #     alpha_range = (0.2, 0.4)))

        empirical_latent_dist = jnp.histogram(
            obs.unstack(latents.components)[i_subj],
            jnp.arange(pose_hyperparams.L+1) - 0.5,
            density = True)[0]
        compare_dirichlet_blocks(
            pose_model.weights()[i_subj],
            [empirical_latent_dist],
            pal = comp_pal,
            ax = ax[2, i_subj],
            orientation = 'horizontal',
            labels = [r"Model", 'Data'],
            label_kwargs = dict(rotation = 90, verticalalignment = "center"))
        sns.despine(ax = ax[2, i_subj], bottom = True, left = True)

        ax[2, i_subj].set_xlabel(f"Subject {i_subj}")

    apply_vrng(ax[0, 0], vrng)

    ax[0, 0].set_ylabel("Keypoint space")
    ax[1, 0].set_ylabel("Pose space")
    ax[2, 0].set_ylabel("Component\nWeights")
    fig.tight_layout()

    return ax