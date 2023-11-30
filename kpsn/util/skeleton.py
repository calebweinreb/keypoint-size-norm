from .keypt_io import keypt_parents, keypt_names
from .alignment import _optionally_apply_as_batch

from typing import NamedTuple
import numpy as np
import jax.numpy as jnp


# Define armature
# -----------------------------------------------------------------------------

bones = np.array(
    [(child, parent) for child, parent in enumerate(keypt_parents)
     if child != parent])

# Description of the skeleton consists of the armature and a root bone
# armature is given as an array [n_bones, 3], with keypoint ids for bone i being
#     armature[i, 0], armature[i, 1]
# and the keypoint ids for the parent of bone i being given by
#     armature[i, 1], armature[i, 2]
# The root is bone 0. Note that armature[0, 2] is -1 because it has no parent
# Getting the bone id of the parent is a little tricky, so avoiding this until it
# becomes needed

# for definining an armature, make the root bone [0, 1] bidirectional
bonewise_keypt_parents = keypt_parents.copy()
bonewise_keypt_parents[0] = 1
root_bone = 0
# bones other than the root bone
child_bones = np.delete(bones, root_bone, axis = 0)
# add the parent keypoint of the parent end of each bone; this is the armature
armature = np.concatenate(
    [child_bones, bonewise_keypt_parents[child_bones[:, 1]][:, None]],
    axis = 1)


class Armature(NamedTuple):
    keypt_names: np.ndarray #<str>
    bones: np.ndarray #<int>
    root: str

    @property
    def keypt_by_name(self):
        return {n: i for i, n in enumerate(self.keypt_names)}
    
    @property
    def n_kpts(self): return len(self.keypt_names)

    def bone_name(self, i_bone, joiner = '-'):
        child_name = self.keypt_names[self.bones[i_bone, 0]]
        parent_name = self.keypt_names[self.bones[i_bone, 1]]
        return f'{child_name}{joiner}{parent_name}'
    

default_armature = Armature(
    keypt_names = keypt_names,
    bones = bones,
    root = 'shldr'
)

names_armature = Armature(
    keypt_names = np.array(['Shoulders', 'Back', 'Hips', 'Tail base', 'Head', 'L ear', 'R ear',
       'Nose', 'L rear knee', 'L rear paw', 'R rear knee', 'R rear paw', 'L front paw',
       'R front paw']),
    bones = bones,
    root = 'Shoulders'
)




# Skeleton traversal and manipulation
# -----------------------------------------------------------------------------

@_optionally_apply_as_batch
def apply_to_bones(keypts, func, skel, *a, **kw):
    """
    keypts: shape (frame, keypt, dim) or list of those
    func: child: (frame, dim), parent: (frame, dim) -> stat: (frame, bone, ...)
    """
    stat = []
    for i_bone, (child, parent) in enumerate(skel.bones):
        stat.append(func(keypts[:, child], keypts[:, parent]))
    return jnp.stack(stat, axis = 1)


@_optionally_apply_as_batch
def apply_to_bones_np(keypts, func, skel, *a, **kw):
    """
    keypts: shape (frame, kept, dim) or list of those
    func: child: (frame, dim), parent: (frame, dim) -> stat: (frame, bone, ...)
    """
    stat = []
    for i_bone, (child, parent) in enumerate(skel.bones):
        stat.append(func(keypts[:, child], keypts[:, parent]))
    return np.stack(stat, axis = 1)


def reroot(skel, new_root):
    new_bones = []

    def traverse_from(node):
        visited.add(node)
        for child in connected_to(node):
            if child in visited: continue
            new_bones.append((child, node))
            traverse_from(child)
    
    visited = set()
    def connected_to(i):
        return np.concatenate([
            skel.bones[skel.bones[:, 0] == i, 1],
            skel.bones[skel.bones[:, 1] == i, 0]])

    traverse_from(skel.keypt_by_name[new_root])
    return Armature(
        keypt_names = skel.keypt_names,
        bones = np.array(new_bones),
        root = new_root)


# Skeleton scalar summaries: bone lengths & joint angles
# -----------------------------------------------------------------------------


def bone_lengths(keypts):
    return apply_to_bones_np(keypts, lambda child, parent: (
        np.linalg.norm(child - parent, axis = -1)))


def bone_lengths_jax(keypts):
    return apply_to_bones(keypts, lambda child, parent: (
        jnp.linalg.norm(child - parent, axis = -1)))


def bone_name(i_bone, skel):
    child_name = skel.keypt_names[skel.bones[i_bone, 0]]
    parent_name = skel.keypt_names[skel.bones[i_bone, 1]]
    return f'{child_name}-{parent_name}'


def scalar_joint_angles(keypts, armature = armature):
    """
    keypts: jax.ndarray, shape (mouse, frame, keypt, dim) or (frame, keypt, dim)
    """
    # shape: (..., bone, dim)
    # from joint to end of child
    child_bone = keypts[..., armature[:, 0], :] - keypts[..., armature[:, 1], :]
    # from joint to end of parent
    parent_bone = keypts[..., armature[:, 2], :] - keypts[..., armature[:, 1], :]
    
    normed_child = child_bone / jnp.linalg.norm(child_bone, axis = -1, keepdims = True)
    normed_parent = parent_bone / jnp.linalg.norm(parent_bone, axis = -1, keepdims = True)
    
    # shape: (..., bone)
    cos_angles = (normed_child * normed_parent).sum(axis = -1)
    return cos_angles




