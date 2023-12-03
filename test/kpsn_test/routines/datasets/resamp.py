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

    (N, M), gt_obs, metadata = npy_keypts.generate(
        cfg = {**cfg, 'whitelist': [cfg['src_sess']]})

    # ----- create clusters
    clusters = clustering.train_clusters(
        cfg['n_clust'],
        keypt_io.to_feats(gt_obs.keypts),
        seed = cfg['clust_seed'])

    # ------ measure counts and resample
    labels, counts, logits = clustering.masks_and_logits(
        clusters.labels_, cfg['n_clust'])
    samp_keypts, samp_counts, samp_logits = clustering.max_resamp(
        gt_obs.keypts, labels, logits,
        temperature = cfg['temperature'],
        seed = cfg['samp_seed'])

    # ------ create matching metadata
    slices, all_keypts = keypt_io.to_flat_array(samp_keypts)
    id_by_name, session_ids = keypt_io.ids_from_slices(all_keypts, slices)

    new_metadata = dict(
        session_ix = id_by_name,
        session_slice = slices,
        bhv = {f'm{i}': i for i in range(cfg['n_clust'])},
        bhv_counts = samp_counts,
        src_sess = {sess: cfg['src_sess'] for sess in slices},
        shared = dict(clusters = clusters.cluster_centers_),
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
    **{k: v for k, v in npy_keypts.defaults
       if k not in ['whitelist']},
)