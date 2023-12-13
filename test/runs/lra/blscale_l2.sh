#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:subsample=15:output_indep"
src_sess="3wk_m0"
ref_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=100:learning_rate=8:return_param_hist=mstep"
run_name=blscale_l2

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $ref_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$base_data:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --morph_cfg "hyperparam.identity_sess=$ref_sess:hyperparam.L=2" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit"

fi

groupby="groupby=tgt_age"
colorby="colorby=tgt_age"
id="subj_id=src-id"
python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.jpg \
    reconst-err reconst_err "$colorby:ref_sess=$ref_sess" \
    # lengths bl_trace "ref_sess=$ref_sess:$groupby" \
    # dataset lra_keypt_poses "$colorby:ref_sess=$ref_sess" \
    # subj_means subj_means - \
    # fit em_fit "mstep_abs" \
    # gt_compare toyset_compare "match_mode=frame:$colorby:ref_sess=$ref_sess" \
    # match-poses matched_poses "$colorby:ref_sess=$ref_sess" \
    # lra lra_params "$colorby" \
    # gmm gmm_params "$colorby:eigs" \
    # gmm-mean gmm_means - \
    # gmm-use gmm_weights "$colorby" \
    # modes centroid_and_modes "$groupby"
