from ...util.alignment import _optionally_apply_as_batch, _optionally_apply_as_zipped_batch
from ...util.keypt_io import keypt_names

import numpy as np


def construct_transform(skel, root_keypt):
    n_kpts = len(skel.bones) + 1
    u_to_x = np.zeros([n_kpts, n_kpts])
    x_to_u = np.zeros([n_kpts, n_kpts])
    u_to_x[root_keypt, root_keypt] = 1
    x_to_u[root_keypt, root_keypt] = 1
    # skel is topo sorted (thank youuuu)
    for child, parent in skel.bones:
        x_to_u[child, parent] = -1
        x_to_u[child, child] = 1
        u_to_x[child] = u_to_x[parent]
        u_to_x[child, child] = 1
    # x_to_u rows (bones) now ordered by ix of their child keypt
    # want ordered by bone index
    bone_order = np.concatenate([[root_keypt], skel.bones[:, 0]])
    x_to_u = x_to_u[bone_order]
    # same for cols of u_to_x
    # for rows of u_to_x, note that we cannot reconstruct the root
    # using u, so we drop it from u_to_x and leave it to be added
    # back with `join_root`
    u_to_x = u_to_x[:, bone_order]
    return {"u_to_x": u_to_x, "x_to_u": x_to_u,
            'root': root_keypt}


def transform(keypts, transform_data):
    root_and_bones = transform_data['x_to_u'] @ keypts
    return (root_and_bones[..., 0, :], root_and_bones[..., 1:, :])

def join_with_root(bones, roots, transform_data):
    return np.insert(
        bones, transform_data['root'], roots,
        axis = -2)  

def inverse_transform(roots, bones, transform_data):
    if roots is None:
        roots = np.zeros(bones.shape[:-2] + (bones.shape[-1],))
    root_and_bones = np.concatenate([roots[..., None, :], bones], axis = 1)
    keypts = transform_data['u_to_x'] @ root_and_bones
    return keypts


def bone_by_name(name, root_keypt: int, original_keypts = keypt_names):
    bone_names = np.delete(original_keypts, root_keypt)
    bones_by_name = {n: i for i, n in enumerate(bone_names)}
    return bones_by_name[name]