#!/bin/bash
#
# usage: lscan.sh <run_script> <name> <cfg> <morphL> <poseL> <morphDefault> <poseDefault> 

run_script=$1
scan_name=$2
cfg=$3
morphL=($4)
poseL=($5)
morphDefault=$6
poseDefault=$7

L="hyperparam.L"
for mL in "${morphL[@]}"; do
	$run_script - "$scan_name-pl$poseDefault-ml$mL" "p@$L=$poseDefault # m@$L=$mL # $cfg"; done
for pL in "${poseL[@]}"; do
	$run_script - "$scan_name-pl$pL-ml$morphDefault" "p@$L=$pL # m@$L=$morphDefault # $cfg"; done

