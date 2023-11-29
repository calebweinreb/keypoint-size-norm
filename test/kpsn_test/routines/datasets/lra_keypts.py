from kpsn.models.morph import affine_mode as afm
from kpsn.models import pose
from kpsn.util import keypt_io, alignment, skeleton

from kpsn_test.routines.util import update
from kpsn_test.routines.datasets import npy_keypts

import jax.random as jr

def generate(
    cfg: dict,
    ):

    (N, M), gt_obs, metadata = npy_keypts.generate(
        cfg = {**cfg, 'output_indep': True}
    )

    # ----- set up identical poses for each session
    src_keypts = gt_obs.keypts[metadata['session_slice'][cfg['src_sess']]]
    slices, gt_all_poses = keypt_io.to_flat_array({
        f'subj{i}': src_keypts for i in range(cfg['n_subj'])
    })
    session_ix, session_ids = keypt_io.ids_from_slices(gt_all_poses, slices)

    # ------ sample parameters and apply to poses
    morph_hyperparams = afm.init_hyperparams(
        observations = pose.Observations(gt_all_poses, session_ids),
        N = cfg['n_subj'], M = M,
        reference_subject = session_ix['subj0'],
        identity_sess = session_ix['subj0'],
        upd_var_modes = 0, # prior variance params don't matter - not learning
        upd_var_ofs = 0,
        **cfg['hyperparam'])

    morph_params = afm.sample_parameters(
        rkey = jr.PRNGKey(cfg['seed']),
        hyperparams = morph_hyperparams,
        **cfg['param_sample'])

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
        body = {f'subj{i}': i for i in range(cfg['n_subj'])},
        **{k: {sess: v[cfg['src_sess']] for sess in slices}
        for k, v in metadata.items() if k not in ['session_ix']
        if k not in ['session_ix', 'session_slice']})
    
    return (cfg['n_subj'], M), new_obs, new_metadata
    
    

defaults = dict(
    src_sess = None,
    **npy_keypts.defaults,
    n_subj = 2,
    hyperparam = dict(
        L = 1,
    ),
    param_sample = dict(
        update_std = 0.03,
        offset_std = 0.5
    ),
    seed = 2
)