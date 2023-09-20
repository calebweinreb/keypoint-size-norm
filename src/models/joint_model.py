from typing import NamedTuple, Tuple, Dict
from jaxtyping import PRNGKeyArray as PRNGKey
import jax.random as jr

from ..models import pose
from ..models import morph

class JointModel(NamedTuple):
    posespace: pose.PoseSpaceModel
    morph: morph.MorphModel

class JointParameters(NamedTuple):
    posespace: pose.PoseSpaceParameters
    morph: morph.MorphParameters

class JointHyperparams(NamedTuple):
    posespace: pose.PoseSpaceHyperparams
    morph: morph.MorphHyperparams

def sample(
    rkey: PRNGKey,
    T: int,
    model: JointModel,
    hyperparams: JointHyperparams,
    morph_param_kwargs: Dict,
    posespace_param_kwargs: Dict
    ) -> Tuple[morph.MorphParameters, pose.PoseSpaceParameters,
               pose.PoseStates, pose.Observations]:
    
    rkey = jr.split(rkey, 3)
    
    morph_params = model.morph.sample_parameters(
        rkey[0],
        hyperparams.morph,
        **morph_param_kwargs
    )
    pose_params = model.posespace.sample_parameters(
        rkey[1],
        hyperparams.posespace,
        **posespace_param_kwargs
    )
    pose_latents, subject_ids = model.posespace.sample(
        rkey[2],
        pose_params,
        hyperparams.posespace,
        T
    )

    morph_matrix = model.morph.get_transform(morph_params, hyperparams.morph)
    obs = pose.Observations(
        keypts = (
            morph_matrix[subject_ids] @ pose_latents.poses[..., None]
        )[..., 0],
        subject_ids = subject_ids
    )

    return morph_params, pose_params, pose_latents, obs
    
def latent_mle(
    model: JointModel,
    observations: pose.Observations,
    params: JointParameters,
    hyperparams: JointHyperparams,
    ) -> pose.PoseStates:
    poses = model.morph.pose_mle(
        observations, params.morph, hyperparams.morph)
    return model.posespace.discrete_mle(
        poses, hyperparams.posespace, params.posespace)
