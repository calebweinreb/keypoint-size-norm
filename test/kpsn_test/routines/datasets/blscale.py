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
    if isinstance(cfg['src_sess'], str):
        src_sessions = [cfg['src_sess']]
    else: src_sessions = cfg['src_sess']

    ages = metadata['age']
    length_ratios = {sess: {
        f'{tgt_age}wk': (age_lengths[tgt_age] / age_lengths[ages[sess]]) ** cfg['effect']
        for tgt_age in tgt_ages if tgt_age != ages[sess]}
        for sess in src_sessions}
    
    remap_meta, remap_roots, remap_bones = skeleton.apply_bone_scales(
        metadata,
        {s: all_roots[slices[s]] for s in src_sessions},
        {s: all_bones[slices[s]] for s in src_sessions},
        scales = length_ratios,
        scale_key = 'tgt_age'
    )

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

    return (len(remap_id_by_name), remap_all_feats.shape[-1]), remap_obs, dict(
        session_slice = remap_slices,
        session_ix = remap_id_by_name,
        centroid = {sess_name: remap_meta['src-centroid'] for sess_name in sessions},
        rotation = {sess_name: remap_meta['src-rotation'] for sess_name in sessions},
        scale = scales,
        armature = skel,
        sess = {sess: sess for sess in sessions},
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