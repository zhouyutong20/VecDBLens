#!/usr/bin/env bash
# Static-only RangePQ grid for the 112-CPU target server.
# Each run is a 28-thread job bound to one physical socket core set by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="${RANGEPQ_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

THREADS="${THREADS:-28}"
CPUSET="${CPUSET:-0-27}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS}}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${THREADS}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${THREADS}}"

export PY="${PY:-/home/ytzhou/anaconda3/bin/python}"
export BASE_DIR="${BASE_DIR:-/home/hyhe/datasets/sift128_filter_bench}"
export STEM="${STEM:-sift128_n1m_f1_r1e-02_static}"
export TOPK="${TOPK:-100}"
export LIMIT_Q="${LIMIT_Q:-10000}"
export L_VALUES="${L_VALUES:-2500,4000,6000,8000,10000,12000,16000,20000,24000,32000}"
export RECALL_KS="${RECALL_KS:-1,5,10,100}"

export LOG_ROOT="${LOG_ROOT:-${ROOT}/experiments/logs}"
TS="$(date +'%Y%m%d_%H%M%S')"
BATCH_DIR="${LOG_ROOT}/grid_static_${STEM}_${TS}"
SUMMARY_DIR="${BATCH_DIR}/summaries"
mkdir -p "${BATCH_DIR}" "${SUMMARY_DIR}"

run_cfg() {
  local tag="$1"
  shift
  local log="${BATCH_DIR}/${tag}.log"
  echo "===== ${tag} START $(date '+%F %T') =====" | tee -a "${log}"
  echo "[env] PY=${PY} BASE_DIR=${BASE_DIR} STEM=${STEM} THREADS=${THREADS} CPUSET=${CPUSET}" | tee -a "${log}"
  taskset -c "${CPUSET}" "${PY}" -u "${SCRIPT_DIR}/run_dataset.py" \
    --base_dir "${BASE_DIR}" \
    --stem "${STEM}" \
    --nbits 8 \
    --iter 200 \
    --topk "${TOPK}" \
    --limit_q "${LIMIT_Q}" \
    --recall_ks "${RECALL_KS}" \
    --ntrain_pq 100000 \
    --L_values "${L_VALUES}" \
    "$@" \
    --out_dir "${SUMMARY_DIR}" \
    --summary_name "${tag}.json" \
    2>&1 | tee -a "${log}"
  echo "===== ${tag} END $(date '+%F %T') =====" | tee -a "${log}"
}

for m in 4 8 16 32; do
  for nlist in 512 1024; do
    for cz in 1024 2048; do
      tag="m${m}_nlist${nlist}_cz${cz}_nbits8_iter200"
      run_cfg "${tag}" --m "${m}" --nlist "${nlist}" --cz "${cz}"
    done
  done
done

echo "Done. Logs: ${BATCH_DIR}"
