#!/bin/bash

# with $HOME/projects/mph/generative/test/runs/env.sh

base_fit="log_every=1:batch_size=None:mstep_n_steps=400"
base_fit="$base_fit:n_steps=50:scale_lr=True:learning_rate=8:mstep_reinit_opt=False:return_param_hist=mstep"
ref_sess=0
run_name=weight_check_bad
dset="pose.hyperparam.L=2:n_frames=1000"
dset="$dset:pose.param.subj_log=[[1,1],[4,1],[1,4]]:pose.param.pop_log=[1,1]:pose.param.means=[[-1,0],[1,0]]:pose.param.var=[0.3,0.3]"
dset="$dset:morph.param.dmode=[[[0],[0]],[[0],[0]],[[0],[0]]]:morph.param.dofs=[[0,0],[0,0]]"

python3 $SCRIPT/test_arch.py \
    $RESULT/lra_test/$run_name/{}.jl \
    --ref_sess $ref_sess \
    --data $RTN/datasets/gmm_linear.py \
    --data_cfg $dset \
    --pose $RTN/models/gmm.py \
    --pose_cfg "hyperparam.L=2" \
    --morph $RTN/models/lra.py \
    --morph_cfg "hyperparam.identity_sess='$ref_sess':init.init_offsets=False" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg $base_fit

python3 $SCRIPT/2d_plot_battery.py \
    $RESULT/lra_test/$run_name/{}.jl \
    $PLOT/lra_test/$run_name/{}.png

python3 $SCRIPT/plot_battery.py \
    $RESULT/lra_test/$run_name/{}.jl \
    $PLOT/lra_test/$run_name/{}.png \
    lra lra_params "colorby=subjname" \
    gmm gmm_params "colorby=subjname:eigs" \