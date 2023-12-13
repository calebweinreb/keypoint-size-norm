from kpsn_test.routines.datasets import npy_keypts
from kpsn_test.routines.util import update

from kpsn.util import keypt_io, skeleton, alignment
from kpsn.models.morph import linear_skeletal as ls
from kpsn.models import pose
import numpy as np

def generate(
    cfg: dict
    ):

    skel = skeleton.default_armature
    if cfg['reroot'] is not None:
        skel = skeleton.reroot(skel, cfg['reroot'])
    ls_mat = ls.construct_transform(skel, skel.keypt_by_name[skel.root])

    (N, M), gt_obs, metadata = npy_keypts.generate(dict(
        rescale = False, 
        output_indep = False,
        **{k: v for k, v in cfg.items() if k not in ['output_indep', 'rescale']}))

    slices = metadata['session_slice']
    ages, age_groups = keypt_io.get_groups_dict(metadata['age'])
    if cfg['tgt_ages'] is None: tgt_ages = ages
    else: tgt_ages = cfg['tgt_ages']

    # ----- transform to bone space and measure lengths
    all_roots, all_bones = ls.transform(
        gt_obs.keypts.reshape([-1, skel.n_kpts, 3]), ls_mat)
    all_lengths = np.linalg.norm(all_bones, axis = -1)

    sess_lengths = {
        sess_name: np.mean(all_lengths[slc], axis = 0)
        for sess_name, slc in slices.items()}
    age_lengths = {
        age: np.mean([sess_lengths[s] for s in age_group], axis = 0)
        for age, age_group in zip(ages, age_groups)}
    
    # --- generate new dictionary of sessions with rescaled bones
    src_sess = cfg['src_sess']
    slc = slices[src_sess]
    src_age = metadata['age'][src_sess]
    
    remap_bones = {src_sess: all_bones[slc]}
    remap_roots = {src_sess: all_roots[slc]}
    remap_meta = {f'src-{k}': {src_sess: metadata[k][src_sess]} for k in metadata}
    remap_meta['tgt_age'] = {src_sess: src_age}
    
    for tgt_age in tgt_ages:
        if tgt_age == src_age: pass

        new_sess = f'{tgt_age}wk_m{metadata["id"][src_sess]}'
        length_ratios = (age_lengths[tgt_age] / age_lengths[src_age]) ** cfg['effect']
        remap_bones[new_sess] = all_bones[slc] * length_ratios[None, :, None]
        for k in metadata:
            remap_meta[f'src-{k}'][new_sess] = metadata[k][src_sess]
        remap_meta['tgt_age'][new_sess] = tgt_age
        remap_roots[new_sess] = all_roots[slc]

    # --- convert out of bone space and perform with scaling/alignment
    sessions = remap_bones.keys()
    remap_keypts = [
        ls.inverse_transform(
            remap_roots[sess_name], remap_bones[sess_name], ls_mat)
        for sess_name in remap_bones]
    
    if cfg['rescale']:
        remap_keypts, scales = alignment.scalar_align(
            remap_keypts, return_inverse = True)
    else:
        scales = np.ones([len(remap_keypts)])

    remap_slices, remap_all_keypts = keypt_io.to_flat_array(
        {s: k for s, k in zip(sessions, remap_keypts)})
    remap_all_feats = keypt_io.to_feats(remap_all_keypts)
    if cfg['output_indep']:
        remap_all_feats = alignment.sagittal_align_remove_redundant_subspace(
            remap_all_feats,
            origin_keypt = cfg['origin_keypt'],
            skel = skel)

    # --- format outputs and return
    remap_id_by_name, remap_sess_ids = keypt_io.ids_from_slices(
        remap_all_feats, remap_slices)
    
    remap_obs = pose.Observations(
        keypts = remap_all_feats,
        subject_ids = remap_sess_ids)

    return (len(ages), remap_all_feats.shape[-1]), remap_obs, dict(
        session_slice = remap_slices,
        session_ix = remap_id_by_name,
        centroid = {sess_name: remap_meta['src-centroid'] for sess_name in sessions},
        rotation = {sess_name: remap_meta['src-rotation'] for sess_name in sessions},
        scale = scales,
        armature = skel,
        **{k: v for k, v in remap_meta.items() if k not in [
            'src-session_slice', 'src-session_ix',
            'src-centroid', 'src-rotation', 'src-scale']}
    )


defaults = dict(
    src_sess = None,
    reroot = 'hips',
    tgt_ages = None,
    effect = 1.,
    **npy_keypts.defaults,
    resamp = dict(kind=None, temp=0.3)
)