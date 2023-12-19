
from kpsn.util import alignment, skeleton
import jax.numpy as jnp

def keypt_errs(
    feats_a,
    feats_b,
    slices,
    skel = skeleton.default_armature,
    origin_keypt = 'hips',
    single_b = False,
    to_kpt = None,
    is_kpt = None):

    if not is_kpt:
        if to_kpt is None:
            to_kpt, _ = alignment.gen_kpt_func(feats_a, origin_keypt)
        kpt_a = to_kpt(feats_a).reshape([-1, skel.n_kpts, 3])
        kpt_b = to_kpt(feats_b).reshape([-1, skel.n_kpts, 3])
    else:
        kpt_a, kpt_b = feats_a, feats_b
    
    return {
        s: jnp.linalg.norm(kpt_a[slc] - (
            kpt_b if single_b else kpt_b[slc]
        ), axis = -1).mean(axis = 0)
        for s, slc in slices.items()}
