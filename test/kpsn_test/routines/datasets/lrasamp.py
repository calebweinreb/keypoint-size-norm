from kpsn.models.morph import affine_mode as afm
from kpsn.models import pose
from kpsn.util import keypt_io, alignment, skeleton

from kpsn_test.routines.util import update
from kpsn_test.routines.datasets import resamp
from kpsn_test.routines.datasets import lra_keypts
import jax.random as jr
import jax.numpy as jnp

def random_morph(poses, rngk, hyper_kw, param_kw):
    session_ids = jnp.zeros(len(poses), dtype = int)
    params = lra_keypts.random_morph(
        gt_obs = pose.Observations(poses, session_ids),
        M = poses.shape[-1],
        n_subj = 1,
        ref_subj_ix = 0,
        id_subj_ix = None,
        rngk = rngk,
        hyper_kw = hyper_kw,
        param_kw = param_kw
    )
    return params
    

def apply_transform(params, poses):
    session_ids = jnp.zeros(len(poses), dtype = int)
    return afm.transform(params, poses, session_ids)


def generate(
    cfg: dict,
    ):

    # ----- create simulated animals with resampled behavior
    (N, M), gt_obs, metadata = resamp.generate(
        cfg = {**cfg, 'n_subj': cfg['n_bhv']}
    )

    samp_feats = {
        sess: gt_obs.keypts[slc]
        for sess, slc in metadata['session_slice'].items()}
    
    # ----- generate transforms and apply to each animal
    rands = jr.split(jr.PRNGKey(cfg['seed']), cfg['n_body'] + 1)
    meta_body = {}; meta_bhv = {}; new_feats = {}
    for sess in samp_feats:
        meta_body[f'{sess}_orig'] = 'orig'
        meta_bhv[f'{sess}_orig'] = sess
        new_feats[f'{sess}_orig'] = samp_feats[sess]
    
    for bod_i in range(cfg['n_body'] - 1):
        
        params = random_morph(
            gt_obs.keypts, rands[bod_i],
            cfg['hyperparam'], cfg['param_sample'])
        
        for sess in samp_feats:
            
            new_sess = f'{sess}_b{bod_i}'
            new_feats[new_sess] = apply_transform(
                params, samp_feats[sess])
        
            meta_body[new_sess] = f'b{bod_i}'
            meta_bhv[new_sess] = sess


    # ------ format new dataset and return
    slices, all_feats = keypt_io.to_flat_array(new_feats)
    session_ix, session_ids = keypt_io.ids_from_slices(all_feats, slices)
    new_obs = pose.Observations(all_feats, session_ids)

    new_metadata = dict(
        session_ix = session_ix,
        session_slice = slices,
        sess = {sess: sess for sess in slices},
        bhv = meta_bhv,
        body = meta_body,
        shared = metadata['shared'],
        **{k: {sess: v[meta_bhv[sess]] for sess in slices}
            for k, v in metadata.items()
            if k not in ['session_ix', 'session_slice',
                         'bhv', 'shared']
        })
    
    return (len(slices), M), new_obs, new_metadata
    
    

defaults = dict(
    **resamp.defaults,
    n_bhv = 2,
    n_body = 2,
    hyperparam = dict(
        L = 1,
    ),
    param_sample = dict(
        update_std = 0.03,
        offset_std = 0.5
    ),
    seed = 2
)