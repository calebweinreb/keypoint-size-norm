from kpsn.models.morph import affine_mode as afm
from kpsn.models import pose
from kpsn.util import keypt_io, alignment, skeleton

from kpsn_test.routines.util import update
from kpsn_test.routines.datasets import npy_keypts

import jax.numpy as jnp
import jax.random as jr

def generate(
    cfg: dict,
    ):

    (N, M), gt_obs, metadata = npy_keypts.generate(
        cfg = {**cfg, 'output_indep': True}
    )

    tgt_sessions = [cfg['src_sess'], *cfg['tgt_sessions']]
    new_sessions = [cfg['src_sess']] + [f'sim:{s}' for s in cfg['tgt_sessions']]
    names_tgts = list(zip(new_sessions, tgt_sessions))
    true_means = {n: gt_obs.keypts[metadata['session_slice'][s]].mean(axis = 0)
             for n, s in names_tgts}

    # ----- set up identical poses for each session
    src_keypts = gt_obs.keypts[metadata['session_slice'][cfg['src_sess']]]
    slices, gt_all_poses = keypt_io.to_flat_array({
        n: src_keypts for n, s in names_tgts
    })
    session_ix, session_ids = keypt_io.ids_from_slices(gt_all_poses, slices)

    # ------ sample parameters and apply to poses
    morph_hyperparams = afm.init_hyperparams(
        observations = pose.Observations(gt_all_poses, session_ids),
        N = len(new_sessions), M = M,
        reference_subject = session_ix[cfg['src_sess']],
        identity_sess = session_ix[cfg['src_sess']],
        upd_var_modes = 0, # prior variance params don't matter - not learning
        upd_var_ofs = 0,
        **cfg['hyperparam'])

    morph_params = afm.sample_parameters(
        rkey = jr.PRNGKey(cfg['seed']),
        hyperparams = morph_hyperparams,
        **cfg['param_sample'])
    morph_params = afm.AFMTrainedParameters(
        mode_updates = 0*morph_params.mode_updates,
        offset_updates = jnp.array([
            true_means[session_ix.inverse[sess_ix]] - morph_hyperparams.offset
            for sess_ix in range(len(new_sessions))
        ]))

    params = morph_params.with_hyperparams(morph_hyperparams)
    all_feats = afm.transform(params, gt_all_poses, session_ids)

    if not cfg['output_indep']:
        all_feats = alignment.sagittal_align_insert_redundant_subspace(
            all_feats, cfg['origin_keypt'], skeleton.default_armature)


    # ------ format new dataset and return
    new_obs = pose.Observations(all_feats, session_ids)

    new_metadata = dict(
        session_ix = session_ix,
        session_slice = slices,
        body = {n: i for i, n in enumerate(new_sessions)},
        **{k: {sess: v[cfg['src_sess']] for sess in slices}
        for k, v in metadata.items() if k not in ['session_ix']
        if k not in ['session_ix', 'session_slice']})
    
    return (len(new_sessions), M), new_obs, new_metadata
    
    

defaults = dict(
    src_sess = None,
    tgt_sessions=None,
    **npy_keypts.defaults,
    hyperparam = dict(
        L = 1,
    ),
    param_sample = dict(
        update_std = 0.03,
        offset_std = 0.5
    ),
    seed = 2
)