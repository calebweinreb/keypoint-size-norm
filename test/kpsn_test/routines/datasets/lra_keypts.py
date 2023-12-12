from kpsn.models.morph import affine_mode as afm
from kpsn.models import pose
from kpsn.util import keypt_io, alignment, skeleton

from kpsn_test.routines.util import update
from kpsn_test.routines.datasets import npy_keypts

import jax.random as jr

def random_morph(gt_obs, M, n_subj, ref_subj_ix, id_subj_ix, rngk, hyper_kw, param_kw):

    morph_hyperparams = afm.init_hyperparams(
        observations = gt_obs,
        N = n_subj, M = M,
        reference_subject = ref_subj_ix,
        identity_sess = id_subj_ix,
        upd_var_modes = 0, # prior variance params don't matter - not learning
        upd_var_ofs = 0,
        **hyper_kw)

    morph_params = afm.sample_parameters(
        rkey = rngk,
        hyperparams = morph_hyperparams,
        **param_kw)

    return morph_params.with_hyperparams(morph_hyperparams)


def generate(
    cfg: dict,
    ):

    (N, M), gt_obs, metadata = npy_keypts.generate(
        cfg = cfg
    )

    # ----- set up identical poses for each session
    src_keypts = gt_obs.keypts[metadata['session_slice'][cfg['src_sess']]]
    slices, gt_all_poses = keypt_io.to_flat_array({
        f'subj{i}': src_keypts for i in range(cfg['n_subj'])
    })
    session_ix, session_ids = keypt_io.ids_from_slices(gt_all_poses, slices)

    # ------ sample parameters and apply to poses
    
    params = random_morph(
        pose.Observations(gt_all_poses, session_ids),
        M, cfg['n_subj'], session_ix['subj0'], session_ix['subj0'],
        jr.PRNGKey(cfg['seed']), cfg['hyperparam'], cfg['param_sample']
    )
    all_feats = afm.transform(params, gt_all_poses, session_ids)

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