#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:subsample=15:output_indep"
src_sess="3wk_m0"
id_sess="m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=200:learning_rate=8:return_param_hist=mstep"
run_name=resamp_boost

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $id_sess \
    --data $RTN/datasets/resamp.py \
    --data_cfg "$base_data:src_sess=$src_sess:n_clust=2:type='boost':temperature=0.3" \
    --pose $RTN/models/gmm.py \
    --pose_cfg "hyperparam.L=2:hyperparam.diag_eps=1e-1" \
    --morph $RTN/models/lra.py \
    --morph_cfg "hyperparam.identity_sess=$id_sess" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit:mstep_reinit_opt=False"

fi

python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    dataset lra_keypt_poses "colorby=bhv:ref_sess=$id_sess" \
    clusters cluster_means - \
    subj_means subj_means - \
    cluster_counts cluster_counts - \
    fit em_fit "mstep_abs" \
    gt_compare toyset_compare "match_mode=same:colorby=bhv:ref_sess=$id_sess" \
    lra lra_params "colorby=bhv" \
    gmm gmm_params "colorby=bhv:eigs" \
    match-poses matched_poses "colorby=bhv:ref_sess=$id_sess" \
    gmm-mean gmm_means - \
    gmm-use gmm_weights "colorby=bhv"
