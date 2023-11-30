from kpsn.models.morph import affine_mode as afm
from kpsn.models.pose import gmm
from kpsn.models import joint_model
from kpsn.models import pose

import jax.random as jr

def generate(cfg: dict):

    N = cfg['n_subj']
    M = cfg['n_dim']

    model = joint_model.JointModel(
        morph = afm.AffineModeMorph,
        posespace = gmm.GMMPoseSpaceModel)
    
    pose_hyperparams = model.posespace.ParameterClass.HyperparamClass(
        N = N, M = M,
        **cfg['pose']['hyperparam'])
    pose_params = model.posespace.sample_parameters(
        rkey = jr.PRNGKey(cfg['pose']['seed']),
        hyperparams = pose_hyperparams,
        **cfg['pose']['param_sample'])

    sampled_poses, sess_ids = model.posespace.sample_poses(
        rkey = jr.PRNGKey(cfg['pose']['seed']),
        params = pose_params.with_hyperparams(pose_hyperparams),
        T = cfg['n_frames'])

    morph_hyperparams = model.morph.init_hyperparams(
        observations = pose.Observations(sampled_poses.poses, sess_ids),
        N = N, M = M,
        **cfg['morph']['hyperparam'])

    morph_params = model.morph.sample_parameters(
        rkey = jr.PRNGKey(cfg['morph']['seed']),
        hyperparams = morph_hyperparams,
        **cfg['morph']['param_sample'])

    hyperparams = joint_model.JointHyperparams(
        morph = morph_hyperparams,
        posespace = pose_hyperparams)
    gt_params = joint_model.JointTrainedParams(
        morph = morph_params,
        posespace = pose_params
        ).with_hyperparams(hyperparams)
    gt_obs = pose.Observations(
        model.morph.transform(gt_params.morph, sampled_poses.poses, sess_ids),
        sess_ids)
    sess_ix = {str(i): i for i in range(N)}
    session_slice = {str(i): slice(i * cfg['n_frames'], (i + 1) * cfg['n_frames'])
                     for i in range(N)}
    
    return (N, M), gt_obs, dict(
        model_params = gt_params,
        session_ix = sess_ix,
        session_slice = session_slice
        )
    

defaults = dict(
    n_subj = 3,
    n_dim = 2,
    n_frames = 100,
    pose = dict(
        hyperparam = dict(
            L = 5,
            diag_eps = None,
            pop_weight_uniformity = 10,
            subj_weight_uniformity = 100,
        ),
        param_sample = dict(
            m_norm_center = 2,
            m_norm_spread = 0.4,
            q_var_center = 0.5,
            q_var_spread = 0.1
        ),
        seed = 1
    ),
    morph = dict(
        hyperparam = dict(
            L = 1,
            upd_var_modes = 1,
            upd_var_ofs = 1,
            reference_subject = 0,
            identity_sess = None,
        ),
        param_sample = dict(
            update_std = 0.3,
            offset_std = 0.6
        ),
        seed = 2
    )
)