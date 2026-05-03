#!/usr/bin/env bash
# GPU cuVS (IVF-PQ + CAGRA) for VisualNews 20k test_queries vs full corpus.
#
# Usage:
#   ./run_gpu_cuvs_visualnews_test20k.sh
#   MAX_JOBS=4 GPUS=0,1,2,3 ./run_gpu_cuvs_visualnews_test20k.sh
#   SKIP=1 REUSE=1 ./run_gpu_cuvs_visualnews_test20k.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CUVS_PYTHON="${CUVS_PYTHON:-/home/hyhe/embedding_test_full/bin/python}"
MAX_JOBS="${MAX_JOBS:-4}"
GPUS="${GPUS:-0,1,2,3}"
POLL_SEC="${POLL_SEC:-60}"
MAX_MEM_MIB="${MAX_MEM_MIB:-4096}"
MAX_UTIL="${MAX_UTIL:-20}"

MODELS_CONFIG="${MODELS_CONFIG:-$ROOT/configs/benchmark_visualnews_test20k_data3_3models.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/visualnews_test20k_gpu_cuvs_v1}"

EXTRA=()
if [[ "${SKIP:-}" == "1" ]]; then EXTRA+=(--skip-existing); fi
if [[ "${REUSE:-}" == "1" ]]; then EXTRA+=(--reuse-index); fi
if [[ "${DRY:-}" == "1" ]]; then EXTRA+=(--dry-run); fi

LAUNCH=(
  "$CUVS_PYTHON" "$ROOT/launch_jobs.py"
  --backends cuvs_ivf_pq cagra
  --faiss-python "$CUVS_PYTHON"
  --cuvs-python "$CUVS_PYTHON"
  --threads 28
  --max-jobs "$MAX_JOBS"
  --poll-sec "$POLL_SEC"
  --max-memory-used-mib "$MAX_MEM_MIB"
  --max-utilization "$MAX_UTIL"
  "${EXTRA[@]}"
)

LAUNCH+=(--gpus "$GPUS")

echo "== VisualNews silver test20k =="
echo "models-config: $MODELS_CONFIG"
echo "output-root:   $OUTPUT_ROOT"
echo "gpus:          $GPUS"
"${LAUNCH[@]}" \
  --models-config "$MODELS_CONFIG" \
  --output-root "$OUTPUT_ROOT"

echo "Done. Results under $OUTPUT_ROOT"
