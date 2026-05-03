#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
FULL_WORKFLOW="/home/hyhe/vlm-embed-compare/src/run_visualnews_full_workflow_clip_rerun_20260430.sh"
RESULT_ROOT="${1:-/data3/hyhe/vlm-embed-compare/results_rerun/visualnews_clip_retrieval_rerun_20260430_1016}"
LOG_DIR="$RESULT_ROOT/_launcher_logs/gpu"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --output-root "$RESULT_ROOT"
  --models clip
  --cuda-device 0
  --skip-existing
)

launch_job() {
  local label="$1"
  local gpu_id="$2"
  local steps="$3"
  local log_file="$LOG_DIR/${label}.gpu${gpu_id}.log"
  echo "[$(date '+%F %T')] launch gpu label=$label gpu=$gpu_id steps=$steps log=$log_file"
  nohup env     CUDA_VISIBLE_DEVICES="$gpu_id"     "$FULL_WORKFLOW"     "${COMMON_ARGS[@]}"     --steps "$steps"     >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" > "$LOG_DIR/${label}.pid"
  echo "[$(date '+%F %T')] started gpu label=$label pid=$pid"
}

launch_job cuvs 0 "cuvs_ivfpq_balanced,cuvs_ivfpq_highrecall"
launch_job cagra 1 "cagra_balanced,cagra_highrecall"

echo "$RESULT_ROOT"
