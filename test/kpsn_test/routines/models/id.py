"""
Model loader for low rank affine (affine mode) morph
"""

from kpsn.models.morph import identity

model = identity.IdentityMorph

defaults = dict(
    hyperparam = dict(
        L = 1,
    ),
    init = dict()
)