#!/bin/bash
#
# modify blscale_slow to add diagonal component to GMM covariances

# with $HOME/projects/mph/generative/test/runs/env.sh
keypt_dir="path=$HOME/projects/mph/data_explore/data"
age_blk="age_blacklist=[9,12]"
downsamp="subsample=15"
src_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
# 'ish' mode from blscale_slow
specific_fit="n_steps=10:learning_rate=5e-1"
run_name=blscale_slow_diag
# add diagonal component
pose_cfg="hyperparam.diag_eps=1e-4"

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $src_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$keypt_dir:$downsamp:$age_blk:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --pose_cfg $pose_cfg \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit:$specific_fit" \
    --omit_save dataset

fi

python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    fit em_fit "mstep_abs" \
    lengths bl_trace "groupby=tgt_age:ref_sess=$src_sess:$groupby" \
    --path $RESULT/lra/blscale_base/{}.jl

# gmm gmm_params "colorby=tgt_age:eigs" \
