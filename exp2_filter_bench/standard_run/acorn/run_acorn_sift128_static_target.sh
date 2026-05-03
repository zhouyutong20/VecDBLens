#!/usr/bin/env bash
# Static-only ACORN launcher for the 112-CPU target server.
# Runs one 28-thread ACORN process at a time, bound to CPU 0-27 by default.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build}"
DATA_ROOT="${DATA_ROOT:-/home/hyhe/datasets/sift128_filter_bench}"
CPUSET="${CPUSET:-0-27}"
THREADS="${THREADS:-28}"
N="${N:-990000}"
EFSEARCH_LIST="${EFSEARCH_LIST:-32,48,64,80,100,128,160,200,256,320,400,512,640,800,1000,1200,1600}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${THREADS}}"
export ACORN_OMP_THREADS="${ACORN_OMP_THREADS:-${THREADS}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${THREADS}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${THREADS}}"
export ACORN_EFSEARCH_LIST="${ACORN_EFSEARCH_LIST:-${EFSEARCH_LIST}}"

BIN="${BUILD_DIR}/demos/test_acorn"
if [[ ! -x "${BIN}" ]]; then
  echo "Error: executable not found: ${BIN}" >&2
  echo "Build first: cmake -S ${SCRIPT_DIR} -B ${BUILD_DIR} -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=OFF && cmake --build ${BUILD_DIR} --target test_acorn -j" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}/tmp" "${BUILD_DIR}/acorn_sift128_logs"
ln -sfn "${DATA_ROOT}" "${BUILD_DIR}/Datasets"

datasets=(
  sift128_n1m_f1_r1e-04_static
  sift128_n1m_f1_r1e-03_static
  sift128_n1m_f1_r1e-02_static
  sift128_n1m_f1_r1e-01_static
  sift128_n1m_f1_r5e-01_static
)
Ms=(32 64 128)
EFCs=(100 400)

gamma_list_by_dataset() {
  case "$1" in
    sift128_n1m_f1_r1e-04_static|sift128_n1m_f1_r1e-03_static)
      echo "200 400"
      ;;
    sift128_n1m_f1_r1e-02_static)
      echo "50 120"
      ;;
    sift128_n1m_f1_r1e-01_static)
      echo "8 20"
      ;;
    sift128_n1m_f1_r5e-01_static)
      echo "2 6"
      ;;
    *)
      echo "8 20"
      ;;
  esac
}

cd "${BUILD_DIR}"
echo "[env] DATA_ROOT=${DATA_ROOT} CPUSET=${CPUSET} THREADS=${THREADS} ACORN_EFSEARCH_LIST=${ACORN_EFSEARCH_LIST}"
for dataset in "${datasets[@]}"; do
  if [[ ! -f "Datasets/${dataset}/${dataset}_base.fvecs" ]]; then
    echo "Skip missing dataset: ${DATA_ROOT}/${dataset}" >&2
    continue
  fi
  mkdir -p "tmp/${dataset}"
  for M in "${Ms[@]}"; do
    Mb=$((2 * M))
    for efc in "${EFCs[@]}"; do
      for gamma in $(gamma_list_by_dataset "${dataset}"); do
        log="acorn_sift128_logs/${dataset}_N${N}_M${M}_efc${efc}_Mb${Mb}_gamma${gamma}.log"
        echo ">>> RUN dataset=${dataset} M=${M} efc=${efc} Mb=${Mb} gamma=${gamma} log=${log}"
        taskset -c "${CPUSET}" "${BIN}" "${N}" "${gamma}" "${dataset}" "${M}" "${Mb}" "${efc}" "${ACORN_EFSEARCH_LIST}" > "${log}" 2>&1
      done
    done
  done
done

echo "=== ACORN static-only done ==="
