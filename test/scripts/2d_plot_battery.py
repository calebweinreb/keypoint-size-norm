import joblib as jl
import argparse

from kpsn.models import pose
from kpsn.models import joint_model
from kpsn.models.morph import affine_mode as afm
from kpsn.models.morph import identity
from kpsn.models.pose import gmm

from kpsn_test import visualize as viz
from kpsn_test.routines.util import load_cfg, save_results

import matplotlib.pyplot as plt


# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("results_fmt", type = str,
    help = "path/to/results_{}.p")

parser.add_argument("output_fmt", type = str,
    help = "path/to/plot_{}.pdf")

parser.add_argument('--savefig_cfg', type = str,
    help = "Config for the savefig function")

parser.add_argument("--path", type = str, nargs = "*", default = (),
    help="Pairs of path overrides of form 'datset path/to/dataset-result.jl'")

args = parser.parse_args()
override_paths = {
    args.path[2*i]: args.path[2*i + 1]
    for i in range(len(args.path) // 2)}


# Load data / setup
# -----------------------------------------------------------------------------

load = lambda key: jl.load(
    args.results_fmt.format(key)
    if key not in override_paths else
    override_paths[key])

dataset = load('dataset')
init_params = load('init')
fit = load('fit')
print("Loaded data.")

gt_obs = pose.Observations(dataset['keypts'], dataset['subject_ids'])

model = joint_model.JointModel(
    # morph = identity.IdentityMorph,
    morph = afm.AffineModeMorph,
    posespace = gmm.GMMPoseSpaceModel
)

savefig_cfg = load_cfg(args.savefig_cfg)
savefig = lambda fmt, fig: save_results(
    args.output_fmt, fmt,
    lambda path: fig.savefig(path, **savefig_cfg))


# Data space plots
# -----------------------------------------------------------------------------


def dataspace_plot(params):

    fig = plt.figure(
        figsize = (4 * params.hyperparams.morph.N,
                   8 + 1.5))

    est_states = joint_model.latent_mle(model, gt_obs, params)

    ax = viz.mixtures.sampled_mixture_plot(
        fig, params.posespace, est_states, gt_obs)
    
    viz.morphs.plot_morph_dimensions(
        ax, params.morph, display_scale_dataset = est_states.poses)
    
    return fig


gt_obs.unstack(gt_obs.keypts)[0]


savefig('dataset-a-gen', dataspace_plot(dataset['metadata']['model_params']))
savefig('dataset-b-init', dataspace_plot(init_params))
savefig('dataset-c-fit', dataspace_plot(fit['fit_params']))


# Learning trajectory plots
# -----------------------------------------------------------------------------

def training_param_traces():
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    param_hist = fit['param_hist']
    mode_hist = np.stack([p.morph.mode_updates for p in param_hist])
    ofs_hist = np.stack([p.morph.offset_updates for p in param_hist])

    pop_weight_hist = np.stack([p.posespace.pop_weight_logits for p in param_hist])
    subj_weight_hist = np.stack([p.posespace.subj_weight_logits for p in param_hist])
    mean_hist = np.stack([p.posespace.means for p in param_hist])
    diag_mask = np.eye(init_params.morph.M).astype('bool')
    cov_hist = np.stack([
        p.posespace.with_hyperparams(init_params.hyperparams).covariances()
        for p in param_hist])
    cov_diag_hist    = np.stack([cov[:, diag_mask] for cov in cov_hist])
    cov_offdiag_hist = np.stack([cov[:, ~diag_mask] for cov in cov_hist])

    fig, ax = plt.subplots(3, init_params.morph.N, figsize = (6, 3), sharey = 'row')
    for subj_i in range(init_params.morph.N):
        ax[0, subj_i].plot(mode_hist [:, subj_i, :, 0], 'k-', lw = 1)
        ax[1, subj_i].plot(ofs_hist[:, subj_i], 'k-', lw = 1)
        ax[2, subj_i].plot(subj_weight_hist[:, subj_i], 'k-', lw = 1)
        for i in [0, 1, 2]:
            ax[i, subj_i].axhline(0, ls = '--', color = '.6', lw = 0.5)
    ax[0, 0].set_ylabel("Mode\nupdate")
    ax[1, 0].set_ylabel("Offset\nupdate")
    ax[2, 0].set_ylabel("Subject\nweights")
    sns.despine()
    fig.tight_layout()
    subjwise_fig = fig

    fig, ax = plt.subplots(3, init_params.posespace.L, figsize = (12, 4), sharey='row')
    for comp_i in range(init_params.posespace.L):
        ax[0, comp_i].plot(mean_hist[:, comp_i], 'k-', lw = 1)
        ax[1, comp_i].plot(cov_diag_hist[:, comp_i], 'k-', lw = 1)
        ax[1, comp_i].plot(cov_offdiag_hist[:, comp_i], '-', color = '.6', lw = 0.5)
        ax[2, comp_i].plot(pop_weight_hist[:, comp_i], 'k-', lw = 1)
        ax[0, comp_i].axhline(0, ls = '--', color = '.6', lw = 0.5)
        ax[1, comp_i].axhline(0, ls = '--', color = '.6', lw = 0.5)
        ax[2, comp_i].axhline(0, ls = '--', color = '.6', lw = 0.5)

    ax[0, 0].set_ylabel("Component\nmean")
    ax[1, 0].set_ylabel("Component cov\n(diag/off)")
    ax[2, 0].set_ylabel("Population\nlogits")
    sns.despine()
    fig.tight_layout()
    component_fig = fig

    return subjwise_fig, component_fig


savefig("fit-loss", viz.fitting.em_loss(fit['loss_hist'], fit['mstep_losses'], mstep_relative = False))

# subjwise_fig, component_fig = training_param_traces()
# savefig('param-subjwise', subjwise_fig)
# savefig('param-components', component_fig)

savefig('fit-reports', viz.fitting.report_plots(fit['reports']))


