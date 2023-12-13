#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
src_sess="3wk_m0"
id_sess="m0_orig"
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:subsample=15:output_indep"
base_data="$base_data:src_sess=$src_sess:n_clust=2:type='max':temperature=0.3"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=400:learning_rate=8:return_param_hist=mstep:mstep_reinit_opt=False"
run_name=lrasamp_base

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $id_sess \
    --data $RTN/datasets/lrasamp.py \
    --data_cfg "$base_data:param_sample.offset_std=3" \
    --pose $RTN/models/gmm.py \
    --pose_cfg "hyperparam.L=5" \
    --morph $RTN/models/lra.py \
    --morph_cfg "hyperparam.identity_sess=$id_sess" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit"

fi

colorby="colorby=sess"
python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.tiff \
    gt_compare toyset_compare "match_mode=meta:$colorby:ref_sess=$id_sess" \
    dataset lra_keypt_poses "$colorby:ref_sess=$id_sess" \
    clusters cluster_means - \
    subj_means subj_means - \
    cluster_counts cluster_counts - \
    fit em_fit "mstep_abs" \
    lra lra_params "$colorby" \
    gmm gmm_params "$colorby:eigs" \
    match-poses matched_poses "$colorby:ref_sess=$id_sess" \
    gmm-mean gmm_means - \
    gmm-use gmm_weights "$colorby"
