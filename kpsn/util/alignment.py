import numpy as np
import scipy.spatial.transform

from .keypt_io import keypt_by_name

anterior_pt_name = 'shldr'
anterior_pt = keypt_by_name[anterior_pt_name]
anterior_pts = [keypt_by_name['shldr'], keypt_by_name['head']]
posterior_pts = [keypt_by_name['hips'], keypt_by_name['back']]

def _optionally_apply_as_batch(func):
    def wrapped(keypts, *a, **kw):
        if (isinstance(keypts, list)
            or isinstance(keypts, tuple)
            or (isinstance(keypts, np.ndarray) and keypts.ndim != 3)
            ):
            return [func(k, *a, **kw) for k in keypts]
        else:
            return func(keypts, *a, **kw)
    return wrapped


def _optionally_apply_as_zipped_batch(func):
    def wrapped(keypts, *a, **kw):
        if (isinstance(keypts, list)
            or isinstance(keypts, tuple)
            ):
            return tuple(zip(*[func(k, *a, **kw) for k in keypts]))
        else:
            return func(keypts, *a, **kw)
    return wrapped


def _get_com(keypts, at_keypt):
    com = keypts[:, [keypt_by_name[at_keypt]]]
    com[..., 2] = 0
    return com


@_optionally_apply_as_zipped_batch
def sagittal_align(
    keypts,
    origin_keypt,
    return_inverse = False,):
    """
    keypts: shape (t, keypt, dim)
    """

    # center hips/back or origin-keypt on (0,0,0)
    com = _get_com(keypts, origin_keypt)
    centered = keypts - com

    # rotate shoulders/head to align with (1,1,0)
    centered_ant_com = centered[:, anterior_pt]
    theta = np.arctan2(centered_ant_com[:, 1], centered_ant_com[:, 0])
    # shape: (t, 3, 3)
    R = scipy.spatial.transform.Rotation.from_rotvec(
        (-theta[:, None]) * np.array([0, 0, 1])[None, :]
    ).as_matrix()
    rotated = (R[:, None] @ centered[..., None])[..., 0]
    
    if return_inverse:
        return rotated, com, theta
    else:
        return rotated
    

def inverse_saggital_align(kpts, centroid, theta):
    R = scipy.spatial.transform.Rotation.from_rotvec(
        (theta[:, None]) * np.array([0, 0, 1])[None, :]
    ).as_matrix()
    rotated = (R[:, None] @ kpts[..., None])[..., 0]
    uncentered = rotated + centroid
    return uncentered


def _redundancy_mask(skel, origin_keypt):

    mask = np.ones([skel.n_kpts, 3], dtype = bool)
    # (x, z) of origin keypt locked to zero - remove
    mask[skel.keypt_by_name[origin_keypt], :2] = 0
    # (y,) of anterior keypt locked to zero - remove
    mask[skel.keypt_by_name[anterior_pt_name], 1] = 0

    return mask.ravel()


def sagittal_align_remove_redundant_subspace(
    kpts,
    origin_keypt,
    skel,
    ):
    """
    Aligning introduces a dimensiona in keypoint space with no variation. This
    function maps keypoints to abstract features in a lower dimensional subspace
    without that redundancy.

    kpts : array, (...batch, n_keypts * n_dim)
        Array of keypoints flattened to features
    origin_keypt : str
        Name of keypoint used as origin in saggital_align
    skel : skeleton.Armature

    Returns
    -------
    feats : array, (..., batch, n_true_dim)
    """
    
    mask = _redundancy_mask(skel, origin_keypt)
    return kpts[..., mask]


def sagittal_align_insert_redundant_subspace(
    feats,
    origin_keypt,
    skel = None
    ):
    """Inverse to `sagittal_align_remove_redundant_subspace`.
    """

    mask = _redundancy_mask(skel, origin_keypt)
    ret = np.zeros(feats.shape[:-1] + (feats.shape[-1] + (~mask).sum(),))
    ret[..., mask] = feats
    return ret



def scalar_align(keypts, return_inverse = False):
    """
    Parameters
    ----------
    keypts : List[numpy array (frames, keypts, spatial)]"""
    
    absolute_scales = []
    for sess_i in range(len(keypts)):
        anterior_com = keypts[sess_i][:, anterior_pts].mean(axis = 1)
        posterior_com = keypts[sess_i][:, posterior_pts].mean(axis = 1)
        absolute_scales.append(np.median(
            np.linalg.norm(anterior_com - posterior_com, axis = -1),
        axis = 0))
    
    scales = np.array(absolute_scales) / np.mean(absolute_scales)
    scaled_keypts = [
        keypts[i] / scales[i]
        for i in range(len(keypts))]

    if return_inverse:
        return scaled_keypts, scales
    else: return scaled_keypts


def inverse_scalar_align(keypts, scales):
    return [
        keypts[i] * scales[i]
        for i in range(len(keypts))]