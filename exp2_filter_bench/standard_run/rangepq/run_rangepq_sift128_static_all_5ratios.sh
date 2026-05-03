#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

BASE_DIR="${BASE_DIR:-/home/hyhe/datasets/sift128_filter_bench}"
CPUSET="${CPUSET:-0-27}"
THREADS="${THREADS:-28}"
LOG_ROOT="${LOG_ROOT:-/home/hyhe/RangePQ/experiments/logs/full_run_sift128_static}"
STRICT_DATASET_CHECK="${STRICT_DATASET_CHECK:-0}"

datasets=(
  sift128_n1m_f1_r1e-04_static
  sift128_n1m_f1_r1e-03_static
  sift128_n1m_f1_r1e-02_static
  sift128_n1m_f1_r1e-01_static
  sift128_n1m_f1_r5e-01_static
)

for stem in "${datasets[@]}"; do
  if [[ ! -f "${BASE_DIR}/${stem}/${stem}_base.fvecs" ]]; then
    if [[ "${STRICT_DATASET_CHECK}" == "1" ]]; then
      echo "Error: missing dataset ${BASE_DIR}/${stem}" >&2
      exit 1
    fi
    echo "Skip missing dataset: ${BASE_DIR}/${stem}" >&2
    continue
  fi

  echo "===== RangePQ ${stem} START $(date '+%F %T') ====="
  STEM="${stem}" \
  BASE_DIR="${BASE_DIR}" \
  CPUSET="${CPUSET}" \
  THREADS="${THREADS}" \
  LOG_ROOT="${LOG_ROOT}" \
  "${SCRIPT_DIR}/grid_static_target.sh"
  echo "===== RangePQ ${stem} END $(date '+%F %T') ====="
done
