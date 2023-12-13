#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
run_name=blscale_mstep_batch
keypt_dir="path=$HOME/projects/mph/data_explore/data"
age_blk="age_blacklist=[9,12]"
downsamp="subsample=15"
src_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:learning_rate=1e-4"
mstep_batch="mstep_batch_size=300:mstep_tol=None:mstep_n_steps=400"

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $src_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$keypt_dir:$downsamp:$age_blk:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "n_steps=200:$base_fit:$mstep_batch"

fi

groupby="groupby=tgt_age"
id="subj_id=src-id"
python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    fit em_fit - \
    lengths bl_trace "ref_sess=$src_sess:$groupby" \
    modes centroid_and_modes "$groupby" \
    reconst reconst "$groupby:$id"