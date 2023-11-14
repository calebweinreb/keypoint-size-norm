"""
Model loader for low rank affine (affine mode) morph
"""

from kpsn.models.morph import affine_mode as afm

model = afm.AffineModeMorph

defaults = dict(
    hyperparam = dict(
        L = 1,
        upd_var_modes = 1,
        upd_var_ofs = 1,
        identity_sess = None,
    ),
    init = dict()
)