#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
keypt_dir="path=$HOME/projects/mph/data_explore/data"
age_blk="age_blacklist=[9,12]"
downsamp="subsample=15"
src_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400:return_param_hist=mstep"

# "ish" mode: not *very* slow
if [ 'ish' = "$2" ]; then
specific_fit="n_steps=4:learning_rate=5e-1"
run_name=blscale_slowish
# normal mode: very slow
else
specific_fit="n_steps=20:learning_rate=1e-1"
run_name=blscale_slow
fi

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra/$run_name/{}.jl \
    --ref_sess $src_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$keypt_dir:$downsamp:$age_blk:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit:$specific_fit" \
    --omit_save dataset init

fi

python3 $SCRIPT/plot_battery.py \
    $RESULT/lra/$run_name/{}.jl \
    $PLOT/lra/$run_name/{}.png \
    fit em_fit "mstep_abs" \
    morph lra_params - \
    --path $RESULT/lra/blscale_base/{}.jl
