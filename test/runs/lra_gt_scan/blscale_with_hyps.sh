#!/bin/bash

run_key=$2
custom_args="${@:3:100}"
echo "running with cfg:" "${custom_args[@]}"

# with $HOME/projects/mph/generative/test/runs/env.sh
base_data="path=$HOME/projects/mph/data_explore/data"
base_data="$base_data:subsample=15:age_blacklist=[9,12]:output_indep"
src_sess="3wk_m0"
ref_sess="3wk_m0"
base_fit="log_every=1:batch_size=None:scale_lr=True:mstep_n_steps=400"
base_fit="$base_fit:n_steps=10:learning_rate=8:mstep_reinit_opt=False"
base_fit="$base_fit:update_blacklist=('posespace/means','posespace/cholesky')"
base_morph="hyperparam.identity_sess=$ref_sess:hyperparam.upd_var_ofs=1e1"
run_name=blscale_$run_key

if [ 'plot-only' != "$1" ]; then

python3 $SCRIPT/test_arch.py \
    $RESULT/lra_gt_scan/$run_name/{}.jl \
    --ref_sess $ref_sess \
    --data $RTN/datasets/blscale.py \
    --data_cfg "$base_data:src_sess=$src_sess" \
    --pose $RTN/models/gmm.py \
    --morph $RTN/models/lra.py \
    --morph_cfg "$base_morph" \
    --fit $RTN/fitting/em_toy.py \
    --fit_cfg "$base_fit" \
    --cfg "${custom_args[@]}"

fi

groupby="groupby=tgt_age"
colorby="colorby=tgt_age"
id="subj_id=src-id"
python3 $SCRIPT/plot_battery.py \
    $RESULT/lra_gt_scan/$run_name/{}.jl \
    $PLOT/lra_gt_scan/$run_name/{}.jpg \
    gmm gmm_params "$colorby:eigs" \
#     dataset lra_keypt_poses "$colorby:ref_sess=$ref_sess" \
#     subj_means subj_means - \
#     fit em_fit "mstep_abs" \
#     gt_compare toyset_compare "match_mode=frame:$colorby:ref_sess=$ref_sess" \
#     lra lra_params "$colorby" \
#     reconst-err reconst_err "$colorby:ref_sess=$ref_sess" \
#     lengths bl_trace "ref_sess=$ref_sess:$groupby" \
#     modes centroid_and_modes "$groupby" \
#     match-poses matched_poses "$colorby:ref_sess=$ref_sess" \
#     gmm-centroid gmm_means - \
#     gmm-use gmm_weights "$colorby" \
#     # --do ${2:--}
