#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
run_name=strong_priors
gmm_prior="hyperparam.subj_weight_uniformity=1e5:hyperparam.pop_weight_uniformity=1e5"
lra_prior="hyperparam.upd_var_modes=1e-5:hyperparam.upd_var_ofs=1e-5"

python3 $SCRIPT/test_arch.py \
    $RESULT/lra_test/$run_name/{}.jl \
    -N 3 -M 2 --ref_subj 0 \
    --data $RTN/datasets/gmm_linear.py \
    --pose $RTN/models/gmm.py \
    --pose_cfg $gmm_prior \
    --morph $RTN/models/lra.py \
    --morph_cfg $lra_prior \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "learning_rate=5e-4:n_steps=10:mstep_update_max=None"

python3 $SCRIPT/2d_plot_battery.py \
    $RESULT/lra_test/$run_name/{}.jl \
    $PLOT/lra_test/$run_name/{}.pdf