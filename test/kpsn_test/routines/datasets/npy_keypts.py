from kpsn.util import keypt_io, skeleton, alignment
from kpsn.models import pose

import numpy as np

def generate(
    cfg: dict,
    ):

    metadata, keypts = keypt_io.npy_dataset(cfg['path'])
    if len(keypts) == 0:
        print(f"Warning: no keypoints found at {cfg['path']}")

    if cfg['age_blacklist'] is not None:
        whitelist = [ 
            i_subj for i_subj in range(len(keypts))
            if (not metadata['age'][i_subj] in cfg['age_blacklist'])
        ]
        metadata, keypts = keypt_io.select_subset(metadata, keypts, whitelist)
    if cfg['subsample'] is not None:
        keypts = keypt_io.subsample_time(keypts, cfg['subsample'])
    n_sess = len(keypts)
    
    align_result, centroids, rotations = alignment.sagittal_align(
        keypts, origin_keypt = cfg['origin_keypt'], return_inverse = True)
    if cfg['rescale']:
        scale_result, scales = alignment.scalar_align(
            align_result, return_inverse = True)
    else:
        scale_result = align_result
        scales = np.ones(len(align_result))

    # convert from list of sessions to dictionary
    sess_names = [
        f'{metadata["age"][sess_ix]}wk_m{metadata["id"][sess_ix]}'
        for sess_ix in range(n_sess)]
    by_name = lambda arr: {n: arr[i] for i, n in enumerate(sess_names)}

    metadata = {k: by_name(v) for k, v in metadata.items()}
    slices, all_keypts = keypt_io.to_flat_array(by_name(scale_result))
    centroids = by_name(centroids)
    rotations = by_name(rotations)
    scales = by_name(scales)

    id_by_name, session_ids = keypt_io.ids_from_slices(all_keypts, slices)

    all_feats = keypt_io.to_feats(all_keypts)
    if cfg['output_indep']:
        all_feats = alignment.sagittal_align_remove_redundant_subspace(
            all_feats,
            origin_keypt = cfg['origin_keypt'],
            skel = skeleton.default_armature)

    gt_obs = pose.Observations(all_feats, session_ids)
    return (n_sess, all_feats.shape[-1]), gt_obs, dict(
        session_slice = slices,
        session_ix = id_by_name,
        centroid = centroids,
        rotation = rotations,
        scale = scales,
        **metadata)


defaults = dict(
    path = None,
    age_blacklist = None,
    subsample = None,
    rescale = True,
    output_indep = False,
    origin_keypt = 'hips',
)