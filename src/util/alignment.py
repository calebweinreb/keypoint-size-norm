import numpy as np
import scipy.spatial.transform

from .keypt_io import keypt_by_name

frontal_pts = [keypt_by_name['shldr'], keypt_by_name['head']]
ventral_pts = [keypt_by_name['hips'], keypt_by_name['back']]

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


@_optionally_apply_as_batch
def sagittal_align(keypts):
    """
    keypts: shape (t, keypt, dim)
    """
    # center hips/back on (0,0,0)
    ventral_com = keypts[:, ventral_pts].mean(axis = 1, keepdims = True)
    frontal_com = keypts[:, frontal_pts].mean(axis = 1, keepdims = True)
    com = (ventral_com + frontal_com) / 2
    ventral_centered = keypts - com
    # rotate shoulders/head to align with (1,1,0)
    frontal_com = ventral_centered[:, frontal_pts].mean(axis = 1)
    theta = np.arctan2(frontal_com[:, 1], frontal_com[:, 0])
    # shape: (t, 3, 3)
    R = scipy.spatial.transform.Rotation.from_rotvec(
        (-theta[:, None]) * np.array([0, 0, 1])[None, :]
    ).as_matrix()
    frontal_rotated = (R[:, None] @ ventral_centered[..., None])[..., 0]
    return frontal_rotated
