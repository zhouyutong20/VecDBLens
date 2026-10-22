#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_WORKFLOW="${FULL_WORKFLOW:-$SCRIPT_DIR/run_visualnews_full_workflow_clip_rerun_20260430.sh}"
RESULT_ROOT="${1:-/data3/hyhe/vlm-embed-compare/results_rerun/visualnews_clip_retrieval_rerun_20260430_1016}"
LOG_DIR="$RESULT_ROOT/_launcher_logs/cpu"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --output-root "$RESULT_ROOT"
  --models clip
  --skip-existing
)

launch_job() {
  local label="$1"
  local cores="$2"
  local steps="$3"
  local log_file="$LOG_DIR/${label}.log"
  echo "[$(date '+%F %T')] launch cpu label=$label cores=$cores steps=$steps log=$log_file"
  nohup env     OMP_THREADS=28     HNSW_THREADS=28     MKL_NUM_THREADS=28     OPENBLAS_NUM_THREADS=28     NUMEXPR_NUM_THREADS=28     taskset -c "$cores"     "$FULL_WORKFLOW"     "${COMMON_ARGS[@]}"     --steps "$steps"     >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG_DIR/${label}.pid"
  echo "[$(date '+%F %T')] started cpu label=$label pid=$pid"
}

launch_job flatip "1-27,84" "flatip"
launch_job hnsw "29-55,85" "hnsw_balanced,hnsw_highrecall"
launch_job ivfpq "57-83,86" "ivfpq_balanced,ivfpq_highrecall"

echo "$RESULT_ROOT"
