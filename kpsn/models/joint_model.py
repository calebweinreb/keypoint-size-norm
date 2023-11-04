from typing import NamedTuple, Tuple, Dict
from jaxtyping import PRNGKeyArray as PRNGKey
import jax.random as jr

from ..models import pose
from ..models import morph

class JointModel(NamedTuple):
    posespace: pose.PoseSpaceModel
    morph: morph.MorphModel

class JointHyperparams(NamedTuple):
    posespace: pose.PoseSpaceHyperparams
    morph: morph.MorphHyperparams

    def as_static_dynamic_parts(self):
        ps_static, ps_dynamic = self.posespace.as_static_dynamic_parts()
        m_static, m_dynamic = self.morph.as_static_dynamic_parts()
        return ((ps_static, m_static), (ps_dynamic, m_dynamic))
    
    @staticmethod
    def from_static_dynamic_parts(model: JointModel, static, dynamic):
        ps_hyper_class = model.posespace.ParameterClass.HyperparamClass
        m_hyper_class = model.morph.ParameterClass.HyperparamClass
        return JointHyperparams(
            posespace = ps_hyper_class.from_static_dynamic_parts(
                static[0], dynamic[0]),
            morph = m_hyper_class.from_static_dynamic_parts(
                static[1], dynamic[1])
        )

class JointParameters(NamedTuple):
    posespace: pose.PoseSpaceParameters
    morph: morph.MorphParameters

    def with_hyperparams(self, hyperparams: JointHyperparams):
        return JointAllParameters(
            posespace = self.posespace.with_hyperparams(
                hyperparams.posespace),
            morph = self.morph.with_hyperparams(
                hyperparams.morph))

class JointAllParameters(NamedTuple):
    posespace: pose.PoseSpaceAllParameters
    morph: morph.MorphAllParameters

    @property
    def params(self):
        return JointParameters(
            self.posespace.params, self.morph.params)

    @property
    def hyperparams(self):
        return JointHyperparams(
            self.posespace.hyperparams, self.morph.hyperparams)
    
    def with_morph(self, morph: morph.MorphAllParameters):
        return JointAllParameters(self.posespace, morph)

    def with_posespace(self, posespace: pose.PoseSpaceAllParameters):
        return JointAllParameters(posespace, self.morph)


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

    morph_matrix, morph_ofs = model.morph.get_transform(
        morph_params, hyperparams.morph)
    obs = pose.Observations(
        keypts = (
            morph_matrix[subject_ids] @  pose_latents.poses[..., None]
        )[..., 0] + morph_ofs[subject_ids],
        subject_ids = subject_ids
    )

    return morph_params, pose_params, pose_latents, obs
    

def latent_mle(
    model: JointModel,
    observations: pose.Observations,
    params: JointAllParameters,
    ) -> pose.PoseStates:
    poses = model.morph.pose_mle(
        observations, params.morph)
    return model.posespace.discrete_mle(
        poses, params.posespace)


def init(
    model: JointModel,
    hyperparams: JointHyperparams,
    observations: pose.Observations,
    reference_subject: int,
    seed: int = 0,
    morph_kws: dict = {},
    posespace_kws: dict = {}
    ) -> JointParameters:
    morph_params = model.morph.init(
        hyperparams.morph, observations,
        reference_subject, seed,
        **morph_kws)
    poses = model.morph.pose_mle(
        observations,
        morph_params.with_hyperparams(hyperparams.morph))
    posespace_params = model.posespace.init(
        hyperparams.posespace,
        observations,
        poses,
        reference_subject,
        seed,
        **posespace_kws
    )
    return JointParameters(posespace_params, morph_params)
    