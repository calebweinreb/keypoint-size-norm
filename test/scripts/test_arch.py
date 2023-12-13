from kpsn_test.routines.util import load_routine, update, save_results, sort_cfg_list, load_cfg_lists, load_cfg

from kpsn.models import joint_model
from kpsn.fitting import em

import joblib as jl
import argparse
import os.path
from ruamel.yaml import YAML
import pathlib

import jax.numpy as jnp, sys
jnp.set_printoptions(threshold=sys.maxsize)

# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("output_fmt", type = str,
    help = "path/to/output_{}.p")

parser.add_argument("--ref_sess", type = str, required = True,
    help = "Name of reference session for initializing model.")

parser.add_argument("--data", type = str, required = True,
    help = "Dataset module")
parser.add_argument("--data_cfg", type = str, default = None,
    help = "Config for datset module")

parser.add_argument("--morph", type = str, required = True,
    help = "Morph module")
parser.add_argument("--morph_cfg", type = str, default = None,
    help = "Config for morph module")

parser.add_argument("--pose", type = str, required = True,
    help = "Pose space module")
parser.add_argument("--pose_cfg", type = str, default = None,
    help = "Config for pose space module")

parser.add_argument("--fit", type = str, required = True,
    help = "Default cfg iterate_em function")
parser.add_argument("--fit_cfg", type = str, default = None,
    help = "Config for fit procedure")

parser.add_argument("--cfg", type = str, nargs = '+', default = (),
    help = "Config for any of [d, m, p, f] specified as d!key=val")

parser.add_argument("--omit_save", type = str, nargs = "*", default = (),
    help="Any of dataset, init, or fit - do not save")

parser.add_argument("--quiet", action = 'store_true',
    help="Shh.")

parser.add_argument('--new_output', action = 'store_true')

args = parser.parse_args()


# Load modules / setup
# -----------------------------------------------------------------------------

cfgs = sort_cfg_list(args.cfg,
    shorthands = dict(
        d = 'data', m = 'morph', p = 'pose', f = 'fit'),
    base = dict(
        data = [load_cfg(args.data_cfg)],
        morph = [load_cfg(args.morph_cfg)],
        pose = [load_cfg(args.pose_cfg)],
        fit = [load_cfg(args.fit_cfg)]))
cfgs = load_cfg_lists(cfgs)

data_module = load_routine(args.data)
data_cfg = update(data_module.defaults, cfgs['data'], add = True)
morph_module = load_routine(args.morph)
morph_cfg = update(morph_module.defaults, cfgs['morph'], add = True)
pose_module = load_routine(args.pose)
pose_cfg = update(pose_module.defaults, cfgs['pose'], add = True)
fit_defaults = load_routine(args.fit).defaults
fit_cfg = update(fit_defaults, cfgs['fit'], add = True)

model = joint_model.JointModel(
    morph = morph_module.model,
    posespace = pose_module.model)


# ----- Supporting fns

if not args.new_output:
    args.output_fmt = args.output_fmt.replace(
        '{}', '{name}').replace('.jl', '.{ext}')

save_jl = lambda fmt, data: save_results(
    args.output_fmt, fmt, 'jl',
    lambda path: jl.dump(data, path),
    args.omit_save, not args.quiet)

save_yaml = lambda fmt, data: save_results(
    args.output_fmt, fmt, 'yml',
    lambda path: YAML().dump(data, pathlib.Path(path)),
    args.omit_save, not args.quiet)

save_yaml('cfg', cfgs)



# Generate dataset and fit
# -----------------------------------------------------------------------------

# ----- sample dataset using the given data module

if not args.quiet:
    print("Loading dataset")

(N, M), gt_obs, metadata = data_module.generate(
    cfg = data_cfg)

morph_cfg = morph_module.session_names_to_ixs(metadata, morph_cfg)

save_jl("dataset", dict(
    keypts = gt_obs.keypts,
    subject_ids = gt_obs.subject_ids,
    metadata = metadata,
    N = N,
    M = M    
))

if args.ref_sess not in metadata['session_ix']:
    raise ValueError(
        f"No such session '{args.ref_sess}' to be used as reference. " + 
        f"Options are: {list(metadata['session_ix'].keys())}")

# ----- initialize hyperparameters and parameters

if not args.quiet:
    print("Initializing model")

pose_hyperparams = pose_module.model.init_hyperparams(
    N = N, M = M,
    **pose_cfg['hyperparam'])

morph_hyperparams = morph_module.model.init_hyperparams(
    observations = gt_obs,
    N = N, M = M,
    reference_subject = metadata['session_ix'][args.ref_sess],
    **morph_cfg['hyperparam'])

hyperparams = joint_model.JointHyperparams(
    morph = morph_hyperparams,
    posespace = pose_hyperparams)

init_params = joint_model.init(
    model,
    hyperparams,
    gt_obs,
    reference_subject = metadata['session_ix'][args.ref_sess],
    morph_kws = morph_cfg['init'],
    posespace_kws = pose_cfg['init']
).with_hyperparams(hyperparams)

save_jl("init", init_params)



# ----- run EM

loss_hist, fit_params, mstep_losses, param_hist, reports = em.iterate_em(
    model = model,
    init_params = init_params,
    emissions = gt_obs,
    return_mstep_losses = True,
    return_reports = True,
    progress = False,
    **fit_cfg)

save_jl("fit", dict(
    loss_hist = loss_hist,
    fit_params = fit_params,
    mstep_losses = mstep_losses,
    param_hist = param_hist,
    reports = reports,
))

