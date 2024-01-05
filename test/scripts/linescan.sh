#!/bin/bash
#
# usage: lscan.sh <run_script> <name> <cfg> <param> <param_short> "1 2 3 4" 

run_script=$1
scan_name=$2
cfg=$3
param=$4
param_short=$5
vals=($6)

# for val in "${vals[@]}"; do
# 	$run_script - "$scan_name-$param_short$val" "$param$val # $cfg"
# done

for val in "${vals[@]}"; do
	$run_script plot-only "$scan_name-$param_short$val"
done
