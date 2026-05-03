#!/usr/bin/env bash
# Run CPU backends (flat/faiss_hnsw/faiss_ivfpq) for Flickr30K test5k pool.
# Launches 3 model workers in parallel, each pinned to 28 CPU cores by default.
#
# Usage:
#   ./run_cpu_faiss_flickr30k_test5k.sh
#   CPU_SETS="0-27;28-55;56-83" SKIP=1 REUSE=1 ./run_cpu_faiss_flickr30k_test5k.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

FAISS_PYTHON="${FAISS_PYTHON:-/home/ytzhou/anaconda3/envs/embedding_test/bin/python}"
MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/configs/benchmark_flickr30k_test5k_data3_3models.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/flickr30k_test5k_cpu_faiss_v1}"
LOG_ROOT="${LOG_ROOT:-$OUTPUT_ROOT/logs}"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_ROOT/.run.lock}"
CPU_SETS="${CPU_SETS:-0-27;28-55;56-83}"
THREADS="${THREADS:-28}"
BACKENDS=(flat faiss_hnsw faiss_ivfpq)

EXTRA=()
if [[ "${SKIP:-}" == "1" ]]; then EXTRA+=(--skip-existing); fi
if [[ "${REUSE:-}" == "1" ]]; then EXTRA+=(--reuse-index); fi
if [[ "${DRY:-}" == "1" ]]; then EXTRA+=(--dry-run); fi

mkdir -p "$LOG_ROOT"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another instance is already running (lock: $LOCK_FILE)" >&2
  exit 1
fi
IFS=';' read -r -a CPU_SET_ARR <<< "$CPU_SETS"
MODEL_NAMES=(CLIP SigLIP2 QwenVL3)

if [[ "${#CPU_SET_ARR[@]}" -lt "${#MODEL_NAMES[@]}" ]]; then
  echo "need at least ${#MODEL_NAMES[@]} CPU sets, got ${#CPU_SET_ARR[@]}" >&2
  exit 1
fi

run_model() {
  local model_name="$1"
  local cpu_set="$2"
  local backend
  local log_path
  local rc=0

  mapfile -t model_args < <(
    "$FAISS_PYTHON" - "$MODELS_CONFIG" "$model_name" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
target = sys.argv[2]
for m in cfg:
    if m["model_name"] == target:
        print(m["dataset_name"])
        print(m["base_emb"])
        print(m["query_emb"])
        print(m["gt_path"])
        print(m["text_max_length"])
        break
else:
    raise SystemExit(f"model not found: {target}")
PY
  )

  local dataset_name="${model_args[0]}"
  local base_emb="${model_args[1]}"
  local query_emb="${model_args[2]}"
  local gt_path="${model_args[3]}"
  local text_max_length="${model_args[4]}"

  echo "[$(date '+%F %T')] start model=$model_name cpuset=$cpu_set"
  for backend in "${BACKENDS[@]}"; do
    log_path="$LOG_ROOT/${model_name}_${backend}_$(date +%Y%m%d_%H%M%S).log"
    echo "[$(date '+%F %T')] launch model=$model_name backend=$backend cpuset=$cpu_set log=$log_path"
    OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" \
    VECLIB_MAXIMUM_THREADS="$THREADS" NUMEXPR_NUM_THREADS="$THREADS" FAISS_NUM_THREADS="$THREADS" \
    taskset -c "$cpu_set" \
      "$FAISS_PYTHON" "$ROOT/run_benchmark.py" \
        --backend "$backend" \
        --model-name "$model_name" \
        --dataset-name "$dataset_name" \
        --base-emb "$base_emb" \
        --query-emb "$query_emb" \
        --gt-path "$gt_path" \
        --text-max-length "$text_max_length" \
        --threads "$THREADS" \
        --gpu-id -1 \
        --output "$OUTPUT_ROOT/$model_name/$backend" \
        "${EXTRA[@]}" \
      2>&1 | tee -a "$log_path"
    cmd_rc=${PIPESTATUS[0]}
    if [[ "$cmd_rc" -ne 0 ]]; then
      rc=1
      echo "[$(date '+%F %T')] failed model=$model_name backend=$backend rc=$cmd_rc (continue)"
    else
      echo "[$(date '+%F %T')] done model=$model_name backend=$backend"
    fi
  done
  return "$rc"
}

pids=()
for idx in "${!MODEL_NAMES[@]}"; do
  run_model "${MODEL_NAMES[$idx]}" "${CPU_SET_ARR[$idx]}" &
  pids+=("$!")
done

final_rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    final_rc=1
  fi
done

if [[ "$final_rc" -ne 0 ]]; then
  echo "[$(date '+%F %T')] finished with failures"
  exit 1
fi

echo "[$(date '+%F %T')] all CPU model jobs finished successfully"
