#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:age_blacklist=[9,12]:subsample=15:output_indep"
src_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=10:learning_rate=1e1"
run_name=blscale_only

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $src_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$base_data:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit"

fi

colorby="colorby=tgt_age"
ref_sess="ref_sess=$src_sess"
python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    fit em_fit "mstep_abs" \
    lengths bl_trace "stepsize=1:$ref_sess:groupby=tgt_age" \
    gt_compare toyset_compare "$ref_sess:$colorby" \
    match-poses matched_poses "$ref_sess:$colorby" \
    lra lra_params "$colorby" \
    gmm gmm_params "$colorby:logdet=False:eigs" \
    gmm-mean gmm_means - \
    gmm-use gmm_weights "$colorby"
