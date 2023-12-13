
import jax.numpy as jnp
import joblib as jl
import argparse
import os.path, re

from kpsn.models import pose
from kpsn.util import pca

from kpsn_test import visualize as viz
from kpsn_test.routines.util import load_cfg, save_results, update, load_routine, find_file

import matplotlib.pyplot as plt


# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("results_fmt", type = str,
    help = "path/to/results_{}.p")

parser.add_argument("output_fmt", type = str,
    help = "path/to/plot_{}.pdf")

parser.add_argument("rtn_and_cfg", type = str, nargs = '+',
    help = "List of form '<plot_name> <routine_name> <routine_cfg> ...', where")

parser.add_argument('--savefig_cfg', type = str,
    help = "Config for the savefig function")

parser.add_argument("--path", type = str, nargs = "*", default = (),
    help="Pairs of path overrides of form 'datset path/to/dataset-result.jl'")

parser.add_argument("--do", type = str, nargs = "*", default = None,
    help="Whitelist of plot names to run, or - to perform all plots.")

parser.add_argument('--new_output', action = 'store_true')

args = parser.parse_args()

if len(args.path) == 1:
    override_paths = None
    fallback_fmt = args.path[0]
else:
    override_paths = {
        args.path[2*i]: args.path[2*i + 1]
        for i in range(len(args.path) // 2)}
    fallback_fmt = None
plot_specs = (
    args.rtn_and_cfg[3*i:3*i+3] 
    for i in range(len(args.rtn_and_cfg) // 3))

if args.do is not None and args.do[0] == '-':
    args.do = None

if not args.new_output:
    args.output_fmt = re.sub(
        r'\.\w+$', '.{ext}', args.output_fmt
        ).replace('{}', '{name}')


# Load data / setup
# -----------------------------------------------------------------------------

load = lambda key: jl.load(find_file(
    key, args.results_fmt, override_paths, fallback_fmt))


dataset = load('dataset')
init = load('init')
fit = load('fit')
print("Loaded data.")

gt_obs = pose.Observations(dataset['keypts'], dataset['subject_ids'])

results = dict(
    dataset = dataset,
    init = init,
    fit = fit)


# model = joint_model.JointModel(
#     # morph = identity.IdentityMorph,
#     morph = afm.AffineModeMorph,
#     posespace = gmm.GMMPoseSpaceModel
# )

savefig_cfg = update(dict(
    dpi = 300
    ), load_cfg(args.savefig_cfg))
savefig = lambda fmt, fig: save_results(
    args.output_fmt, fmt, 'png',
    lambda path: fig.savefig(path, **savefig_cfg))




# Run plots
# -----------------------------------------------------------------------------

for plot_name, rtn_name, cfg in plot_specs:

    if args.do is not None and plot_name not in args.do:
        continue

    plot_routine = load_routine(rtn_name, root = 'kpsn_test.routines.plots')
    figs = plot_routine.plot(
        plot_name = plot_name,
        cfg = update(plot_routine.defaults, load_cfg(cfg)),
        **results)
    for fig_name, fig in figs.items():
        savefig(fig_name, fig)

exit()


# --- fit pca
pc_subsample = 10
kp_pcs = pca.fit_with_center(gt_obs.keypts[::pc_subsample])

def dataspace_plot(
    data,
    pop_data = None,
    highlight_subj = None,
    ttl = None):

    coords = kp_pcs.whitened_coords(data)
    if pop_data is None: pop_coords = coords
    else: pop_coords = kp_pcs.whitened_coords(pop_data)
    xaxis = 0; yaxis = 1

    age_pal = viz.defaults.age_pal(metadata['tgt_age'])

    fig, axes, ax_iter, summ_col_iter = viz.struct.axes_by_age_and_id(
        metadata['tgt_age'], metadata['src-id'],
        summative_col = True,
        figsize = (2, 2))

    for ax, age, vids in summ_col_iter():
        ax.scatter(
            pop_coords[:, xaxis], pop_coords[:, yaxis],
            s = 0.005, marker = '.', color = '.9', rasterized = True)
        ax.set_ylabel(f"{age}wk")
        
    for ax, age, mouse_id, sess_name, summ_ax in ax_iter():
        bg_col = '.3' if sess_name == highlight_subj else '.9'
        fg_col = 'w' if sess_name == highlight_subj else age_pal[age]
        slc = metadata['session_slice'][sess_name]
        ax.scatter(
            pop_coords[:, xaxis], pop_coords[:, yaxis],
            s = 0.005, marker = '.', color = bg_col, rasterized = True)
        
        for a in [ax, summ_ax]:
            a.scatter(
                coords[slc][:, xaxis], coords[slc][:, yaxis],
                s = 0.01, marker = '.', color = fg_col, rasterized = True)

    if ttl is not None:
        fig.suptitle(ttl)
    fig.tight_layout()
    
    return fig



# --- keypoint space
# savefig('lowdim-dataset', dataspace_plot(gt_obs.keypts))

# --- pose space
poses = model.morph.inverse_transform(
    fit['fit_params'].morph, gt_obs.keypts, gt_obs.subject_ids)
# savefig('lowdim-posespace-fit', dataspace_plot(poses))

# --- match all distributions to a reference subject
converted_kps = model.morph.transform(
    fit['fit_params'].morph,
    poses,
    jnp.full([len(poses)], args.ref_subj))
# savefig('lowdim-match', dataspace_plot(
#     converted_kps,
#     pop_data = gt_obs.keypts,
#     highlight_subj = args.ref_subj))




# Learning trajectory plots
# -----------------------------------------------------------------------------


savefig("fit-loss", viz.fitting.em_loss(fit['loss_hist'], fit['mstep_losses']))
savefig('fit-reports', viz.fitting.report_plots(fit['reports']))