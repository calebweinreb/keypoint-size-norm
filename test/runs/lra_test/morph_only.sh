#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh

RUN_NAME=morph_only
blacklist="update_blacklist=('posespace/*',)"

python3 $SCRIPT/test_arch.py \
    $RESULT/lra_test/$RUN_NAME/{}.jl \
    -N 3 -M 2 --ref_subj 2 \
    --data $RTN/datasets/gmm_linear.py \
    --data_cfg "n_frames=300" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "learning_rate=1e-4:n_steps=100:$blacklist"

python3 $SCRIPT/2d_plot_battery.py \
    $RESULT/lra_test/$RUN_NAME/{}.jl \
    $PLOT/lra_test/$RUN_NAME/{}.pdf