#!/usr/bin/env bash
set -euo pipefail

# Single-thread Flat baseline:
# prefilter by label range, then exact L2 brute-force on candidates.
DATA_ROOT="${DATA_ROOT:-/HDD08/ytzhou/data10m_960}"
DATASET_PREFIX="${DATASET_PREFIX:-vec960}"
SIZE_TAG="${SIZE_TAG:-n990k}"
RATIOS="${RATIOS:-0p01pct,0p1pct,1pct,10pct,50pct}"
TOPK="${TOPK:-100}"
LIMIT_Q="${LIMIT_Q:-10000}"
RESULT_ROOT="${RESULT_ROOT:-/home/ytzhou/nips/result/flat1}"
RUN_TS="${RUN_TS:-$(date +'%Y%m%d_%H%M%S')}"
OUT_ROOT="${OUT_ROOT:-${RESULT_ROOT}/${DATASET_PREFIX}_${SIZE_TAG}_${RUN_TS}}"
PY="${PY:-/home/ytzhou/anaconda3/envs/nips311/bin/python}"

# Hard-limit to single-thread behavior.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${OUT_ROOT}"
echo "[run] FLAT1 DATA_ROOT=${DATA_ROOT} DATASET_PREFIX=${DATASET_PREFIX} SIZE_TAG=${SIZE_TAG} OUT_ROOT=${OUT_ROOT}"

"${PY}" /home/ytzhou/nips/filter_build_bench/run_flat_prefilter_exact.py \
  --data_root "${DATA_ROOT}" \
  --dataset_prefix "${DATASET_PREFIX}" \
  --size_tag "${SIZE_TAG}" \
  --ratios "${RATIOS}" \
  --topk "${TOPK}" \
  --limit_q "${LIMIT_Q}" \
  --out_root "${OUT_ROOT}"
