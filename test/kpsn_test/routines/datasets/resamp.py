from kpsn.models.morph import affine_mode as afm
from kpsn.models import pose
from kpsn.util import keypt_io, alignment, skeleton
from kpsn_test import clustering

from kpsn_test.routines.util import update
from kpsn_test.routines.datasets import npy_keypts

import jax.random as jr

def generate(
    cfg: dict,
    ):

    # ------ load source sesssio and resample
    (N, M), gt_obs, metadata = npy_keypts.generate(
        cfg = {**cfg, 'whitelist': [cfg['src_sess']]})\

    resamp = clustering.cluster_and_resample(
        gt_obs.keypts, cfg['n_clust'], clustering.methods[cfg['type']],
        cfg['temperature'], cfg['clust_seed'], cfg['samp_seed'])

    # ------ create matching metadata
    slices, all_keypts = keypt_io.to_flat_array(resamp['resampled'])
    id_by_name, session_ids = keypt_io.ids_from_slices(all_keypts, slices)

    new_metadata = dict(
        session_ix = id_by_name,
        session_slice = slices,
        bhv = {f'm{i}': i for i in range(cfg['n_clust'])},
        bhv_counts = resamp['counts'],
        src_sess = {sess: cfg['src_sess'] for sess in slices},
        shared = dict(clusters = resamp['clusters'].cluster_centers_,
                      kmeans = resamp['clusters']),
        **{k: {sess: v[cfg['src_sess']] for sess in slices}
            for k, v in metadata.items()
            if k not in ['session_ix', 'session_slice']},
        )
    

    # ------ format new dataset and return
    new_obs = pose.Observations(all_keypts, session_ids)

    return (cfg['n_clust'], M), new_obs, new_metadata


defaults = dict(
    src_sess = None,
    n_clust = 5,
    temperature = 0.5,
    clust_seed = 2,
    samp_seed = 2,
    type = 'max',
    **{k: v for k, v in npy_keypts.defaults.items()
       if k not in ['whitelist']},
)