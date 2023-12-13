#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh
run_name=mstep_batch
no_step_batch="batch_size=None"
do_mstep_batch="mstep_batch_size=100:learning_rate=1e-4:mstep_tol=None:mstep_n_steps=400"

python3 $SCRIPT/test_arch.py \
    $RESULT/lra_test/$run_name/{}.jl \
    --ref_sess 0 \
    --data $RTN/datasets/gmm_linear.py \
    --data_cfg "n_frames=200" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "n_steps=10:$no_step_batch:$do_mstep_batch"

python3 $SCRIPT/2d_plot_battery.py \
    $RESULT/lra_test/$run_name/{}.jl \
    $PLOT/lra_test/$run_name/{}.pdf