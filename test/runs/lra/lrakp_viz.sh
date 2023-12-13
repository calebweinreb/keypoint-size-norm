#!/bin/bash
# clone of lrakp-default with bigger dataset variance

run_name=lrakp_viz

# with $HOME/projects/mph/generative/test/runs/env.sh
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:age_blacklist=[9,12]:subsample=15:output_indep"
src_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=4:learning_rate=1:return_param_hist=mstep"

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess subj0 \
    --data $RTN/datasets/lra_keypts.py \
    --data_cfg "$base_data:src_sess=$src_sess:n_subj=3:param_sample.offset_std=3:rescale=False" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --morph_cfg "hyperparam.identity_sess=subj0" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit"
# --pose_cfg "hyperparam.diag_eps=1e-4" \

fi

python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    dataset lra_keypt_poses "colorby=body:ref_sess=subj0" \
    fit em_fit "mstep_abs" \
    gt_compare toyset_compare "ref_sess=subj0" \
    lra lra_params "colorby=body" \
    gmm gmm_params "colorby=body:logdet=False:eigs" \
    match-poses matched_poses "colorby=body:ref_sess=subj0" \
    gmm-mean gmm_means - \
    gmm-use gmm_weights "colorby=body"
