#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh

# python3 $SCRIPT/test_arch.py \
#     $RESULT/lra_test/base/{}.jl \
#     --ref_sess 0 \
#     --data $RTN/datasets/gmm_linear.py \
#     --pose $RTN/models/gmm.py \
#     --morph $RTN/models/lra.py \
#     --fit $RTN/fitting/em_toy.py \
#     --fit_cfg "n_steps=50"

python3 $SCRIPT/2d_plot_battery.py \
    $RESULT/lra_test/base/{}.jl \
    $PLOT/lra_test/base/{}.pdf