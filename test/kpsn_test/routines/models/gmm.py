from kpsn.models.pose import gmm

model = gmm.GMMPoseSpaceModel

defaults = dict(
    hyperparam = dict(
        L = 5,
        diag_eps = None,
        pop_weight_uniformity = 10,
        subj_weight_uniformity = 100,
    ),
    init = dict(
        fit_to_all_subj = False
    )
)