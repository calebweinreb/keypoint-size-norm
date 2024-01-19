import numpy as np
from kpsn.models.morph import affine_mode as afm
from kpsn.models.morph import linear_skeletal as ls
from kpsn.util import skeleton, alignment, logging, keypt_io
from kpsn.util import simple_density as density
from kpsn_test import visualize as viz
import numpy as np
import jax.numpy as jnp
import tqdm
import joblib as jl
from ruamel.yaml import YAML
from kpsn_test.routines.util import load_routine, update

import matplotlib.pyplot as plt
import seaborn as sns
import jax.tree_util as pt


def plot(
    plot_name,
    dataset,
    init,
    fit,
    cfg,
    **kwargs
    ):

    ref_sessions = [cfg['ref_sess']] if isinstance(cfg['ref_sess'], str) else cfg['ref_sess']
    stepsize = cfg['stepsize']
    density_k = cfg['density_k']
    density_eps = cfg['density_eps']
    save = cfg['save']
    results_dir = cfg['results_dir']
    data_rtn = cfg['data_rtn']
    data_path = cfg['data_path']
    subsample = cfg['subsample']

    # ------------------------- setup: reload dataset with more samples

    data_module = load_routine(data_rtn)
    with open(f'{results_dir}/cfg.yml', 'r') as f:
        dataset_cfg = YAML().load(f)['data']
    dataset_cfg = update(data_module.defaults, dataset_cfg, add = True)
    dataset_cfg['subsample'] = subsample
    dataset_cfg['path'] = data_path
    (N, M), gt_obs, meta = data_module.generate(cfg = dataset_cfg)
    keypts = gt_obs.keypts

    # ------------------------- setup: organize data

    hyperparams = init.hyperparams
    sessions = list(meta['session_slice'].keys())
    slices = meta['session_slice']

    # to_kpt, _ = alignment.gen_kpt_func(dataset['keypts'], cfg['origin_keypt'])

    # if passed full m-step parameter traces: select last entry from m-step
    param_hist = fit['param_hist'].copy()
    mstep_lengths = np.array(viz.fitting.mstep_lengths(fit['mstep_losses']))
    if param_hist[0].posespace.means.ndim > 2:
        param_hist.map(lambda arr: arr[np.arange(len(arr)), mstep_lengths - 2])

    # nonref_sessions = [s for s in sessions if s not in ref_sessions]
    nonref_sessions = [s for s in sessions]
    nonref_feats = {s: keypts[slices[s]] for s in nonref_sessions}
    nonref_slices, all_nonref_feats = keypt_io.to_flat_array(nonref_feats)
    nonref_sess_ix, nonref_sess_ids = keypt_io.ids_from_slices(all_nonref_feats, nonref_slices)

    density_kw = dict(k = density_k)
    ref_clouds = {
        s: density.PointCloudDensity(**density_kw).fit(keypts[slices[s]])
        for s in ref_sessions}    

    # ------------------------------ compute: divergences to reference

    toref_base = np.array([
        viz.model_compare.distances_to_cloud(
            ref_clouds[ref_sess], all_nonref_feats,
            nonref_slices, density_kw)
        for ref_sess in ref_sessions])
    ref_to_ref = np.array([
        density.ball_cloud_js(
            ref_clouds[ref_sessions[i]], ref_clouds[ref_sessions[j]])
        for i in range(len(ref_sessions))
        for j in range(len(ref_sessions))
        if i < j])

    # distances to each reference session over training
    
    steps = np.arange(0, len(mstep_lengths), stepsize)
    toref_trace = logging.ReportTrace(n_steps = len(steps))
    reconsts = []

    for step_i, step_num in tqdm.tqdm(enumerate(steps), total = len(steps)):
        curr_divs = np.zeros([len(ref_sessions), len(nonref_sessions)])
        curr_reconsts = []
        for ref_i, ref_sess in enumerate(ref_sessions):
            
            step_params = param_hist[step_num].with_hyperparams(hyperparams).morph

            reconst = viz.affine_mode.to_refs(
                all_nonref_feats, ref_sess, step_params,
                meta['session_ix'], nonref_sess_ix, nonref_sess_ids)
            curr_reconsts.append([
                reconst[nonref_slices[s]] for s in nonref_sessions])
            
            curr_divs[ref_i] = viz.model_compare.distances_to_cloud(
                ref_clouds[ref_sess], reconst, nonref_slices, density_kw)
            
        toref_trace.record(curr_divs, step_i)
        reconsts.append(curr_reconsts)
    
    if save:
        jl.dump(dict(steps = steps,
                     ref_sessions = ref_sessions,
                     nonref_sessions = nonref_sessions,
                     jsd_base = toref_base,
                     jsd_trace = toref_trace,
                     ref_to_ref = ref_to_ref,
                     reconst = reconsts,
                     keypts = {s: keypts[slices[s]] for s in sessions}
                     ),
                f'{results_dir}/{plot_name}.jl')
        
    return {}
            

defaults = dict(
    ref_sess = None,
    data_rtn = None,
    results_dir = None,
    data_path = None,
    stepsize = 30,
    density_k = 15,
    density_eps = None,
    save = False,
    subsample = 2,
)
            

        

        
        