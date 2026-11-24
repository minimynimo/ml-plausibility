#!/bin/bash
# HBV per-basin calibration + scenario runs. Dates, basin pool, seed, input (.mat) and
# output roots are read from config/experiment.py -- nothing is hard-coded here.
#
# RUN: nohup 05_hbv/parallel.sh [N_PROCS] > /dev/null 2>&1 &
# ENV: MATLAB_BIN   matlab executable                 (default: matlab on PATH)
#      MARRMOT_DIR  MARRMoT v2.1.2 with marrmot_patch/ (default: 05_hbv/MARRMoT)
#      PYTHON       python with the repo requirements  (default: python)
set -euo pipefail

HBV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HBV_DIR/.." && pwd)"
MATLAB_BIN="${MATLAB_BIN:-matlab}"
MARRMOT_DIR="${MARRMOT_DIR:-$HBV_DIR/MARRMoT}"
PY="${PYTHON:-python}"
CORES=${1:-40}   # concurrent single-threaded MATLAB processes
[ -d "$MARRMOT_DIR/Models" ] || { echo "MARRMoT not found in $MARRMOT_DIR (see README)"; exit 1; }

eval "$($PY - <<EOF
import sys; sys.path.insert(0, "$REPO")
from config import experiment as E
from core.io import load_basin_pool
out = E.OUT_ROOT / "runs" / "HBV" / "individual" / E.run_name("HBV", "individual", E.HBV_OPTIMISER_SEED) / "matlab"
out.mkdir(parents=True, exist_ok=True)
(out / "basin_list.txt").write_text("\n".join(load_basin_pool()) + "\n")
print(f'OUT_ROOT="{out}"')
print(f'DATA_START="{E.DATA_START:%Y-%m-%d}"')
print(f'CAL_START="{E.TRAIN[0]:%Y-%m-%d}"')
print(f'CAL_END="{E.TRAIN[1]:%Y-%m-%d}"')
print(f'SEED={E.HBV_OPTIMISER_SEED}')
print(f'N_BASINS={len(load_basin_pool())}')
print(f'MAT_ROOT="{E.HBV_MAT_ROOT}"')
EOF
)"

LOGS_DIR="${OUT_ROOT}/logs"
mkdir -p "$LOGS_DIR"
echo "HBV: $N_BASINS basins, data start $DATA_START, calibration $CAL_START..$CAL_END, seed $SEED -> $OUT_ROOT"



FILES_PER_GROUP=$(( (N_BASINS + CORES - 1) / CORES ))
mapfile -t BASINS < "$OUT_ROOT/basin_list.txt"

group_running() {  # $1=start $2=end
    [ "$(ps -eo args | grep "MATLAB" | grep -c "workflow_TEST_cal_parallel_HBV($1, $2, ")" -gt 0 ]
}
group_complete() {  # every basin of the group has a Q file or a failure record
    local i
    for ((i=$1; i<=$2; i++)); do
        local b=${BASINS[$((i-1))]}
        [ -f "$OUT_ROOT/Q/varssim_${b}_Q.csv" ] || [ -f "$OUT_ROOT/failed/varssim_${b}_failed.txt" ] || return 1
    done
    return 0
}
launch() {
    local LOG_FILE="${LOGS_DIR}/MARRMoT_workflow_log_$1_$2.log"
    nohup "$MATLAB_BIN" -singleCompThread -nodesktop -nodisplay -nosplash -r \
        "addpath(genpath('$MARRMOT_DIR')); addpath('$HBV_DIR'); workflow_TEST_cal_parallel_HBV($1, $2, '$OUT_ROOT/basin_list.txt', '$OUT_ROOT', '$DATA_START', '$CAL_START', '$CAL_END', $SEED, ${HBV_MAX_ITER:-[]}, '$MAT_ROOT'); exit;" \
        >> "$LOG_FILE" 2>&1 < /dev/null &
    echo "[$(date '+%F %T')] started MATLAB for basins $1..$2"
}

# Supervisor: (re)launch every group that is neither running nor complete, staggered so the MathWorks
# service host is not flooded (launching 64 at once failed with 'error 5001'). A group is retried at
# most MAX_TRIES times; basins inside a group that crash are recorded in failed/ by the workflow.
MAX_TRIES=5
declare -A TRIES
while true; do
    pending=0
    for ((gs=1; gs<=N_BASINS; gs+=FILES_PER_GROUP)); do
        ge=$((gs + FILES_PER_GROUP - 1)); [ $ge -gt $N_BASINS ] && ge=$N_BASINS
        group_complete $gs $ge && continue
        pending=$((pending + 1))
        group_running $gs $ge && continue
        t=${TRIES[$gs]:-0}
        if [ $t -ge $MAX_TRIES ]; then continue; fi
        TRIES[$gs]=$((t + 1))
        launch $gs $ge
        sleep 20
    done
    [ $pending -eq 0 ] && break
    # stop when nothing is running and every pending group has used up its retries
    if [ "$(ps -eo args | grep "MATLAB" | grep -c "workflow_TEST_cal_parallel_HBV(")" -eq 0 ]; then
        stuck=1
        for ((gs=1; gs<=N_BASINS; gs+=FILES_PER_GROUP)); do
            ge=$((gs + FILES_PER_GROUP - 1)); [ $ge -gt $N_BASINS ] && ge=$N_BASINS
            group_complete $gs $ge || [ ${TRIES[$gs]:-0} -ge $MAX_TRIES ] || stuck=0
        done
        [ $stuck -eq 1 ] && break
    fi
    sleep 300
done
echo "[$(date '+%F %T')] HBV supervisor finished: $(ls $OUT_ROOT/Q | wc -l) basins with output, $(ls $OUT_ROOT/failed | wc -l) failure records"
touch "$OUT_ROOT/_MATLAB_FINISHED"
