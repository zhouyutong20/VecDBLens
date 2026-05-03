#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DIGRA run script aligned with ACORN sift128 settings.
# NOTE: This script is prepared only; do not start automatically.
# ============================================================

# Core settings aligned to ACORN
BASE_NUM=990000
QUERY_NUM=10000
DIM=128
TOPK=100
Ms=(32 64 128)
EFCs=(100 400)   # mapped to DIGRA's ef_con

# DIGRA-specific settings for sift128_n1m_f1_*
VDIM=1
RATIO=1

# Parallelism / thread fairness:
# - Each DIGRA process is forced to single-thread.
# - Up to PARALLEL_JOBS configs run concurrently.
# - This target-server benchmark is static-only.
PARALLEL_JOBS="${PARALLEL_JOBS:-11}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Data and logs. Override DATA_ROOT if the target server stores datasets elsewhere.
DATA_ROOT="${DATA_ROOT:-/home/hyhe/datasets/sift128_filter_bench}"
LOG_DIR="$(cd "$(dirname "$0")" && pwd)/logs_sift128_acorn_aligned"
mkdir -p "$LOG_DIR"

# Binary path
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="${DIGRA_BIN:-${SCRIPT_DIR}/build_target/Digra}"
if [[ ! -x "$BIN" ]]; then
  echo "Error: executable not found: $BIN"
  echo "Build first: cmake -S \"${SCRIPT_DIR}\" -B \"${SCRIPT_DIR}/build_target\" && cmake --build \"${SCRIPT_DIR}/build_target\" -j"
  exit 1
fi

# Discover static datasets only.
mapfile -t DATASETS_STATIC < <(
  python3 - <<'PY' "$DATA_ROOT"
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
if not root.is_dir():
    sys.exit(0)
pat = re.compile(r"^sift128_n1m_f1_r[0-9]+e-[0-9]+_static$")
wanted = {
    "sift128_n1m_f1_r1e-04_static",
    "sift128_n1m_f1_r1e-03_static",
    "sift128_n1m_f1_r1e-02_static",
    "sift128_n1m_f1_r1e-01_static",
    "sift128_n1m_f1_r5e-01_static",
}
items = [p.name for p in root.iterdir() if p.is_dir() and pat.match(p.name) and p.name in wanted]
order = ["sift128_n1m_f1_r1e-04_static", "sift128_n1m_f1_r1e-03_static",
         "sift128_n1m_f1_r1e-02_static", "sift128_n1m_f1_r1e-01_static",
         "sift128_n1m_f1_r5e-01_static"]
for x in [name for name in order if name in set(items)]:
    print(x)
PY
)

if [[ "${#DATASETS_STATIC[@]}" -eq 0 ]]; then
  echo "Error: no required static sift128 datasets found under ${DATA_ROOT}"
  exit 1
fi

run_one() {
  local dset="$1"
  local ef_con="$2"
  local m="$3"

  local base="${DATA_ROOT}/${dset}/${dset}_base.fvecs"
  local query="${DATA_ROOT}/${dset}/${dset}_query.fvecs"
  local labels="${DATA_ROOT}/${dset}/${dset}_labels.fvecs"
  local filters="${DATA_ROOT}/${dset}/${dset}_filters.fvecs"

  for f in "$base" "$query" "$labels" "$filters"; do
    if [[ ! -f "$f" ]]; then
      echo "Skip: missing file $f"
      return 0
    fi
  done

  local log_file="${LOG_DIR}/${dset}_N${BASE_NUM}_m${m}_efc${ef_con}.log"
  echo ">>> RUN dataset=${dset} m=${m} efc=${ef_con} log=${log_file} [pid=$$]"

  "$BIN" \
    "${BASE_NUM}" \
    "${QUERY_NUM}" \
    "${DIM}" \
    "${TOPK}" \
    "${ef_con}" \
    "${m}" \
    "${base}" \
    "${query}" \
    "${labels}" \
    "${filters}" \
    "${VDIM}" \
    "${RATIO}" \
    > "${log_file}" 2>&1
}

launch_one() {
  local dset="$1"
  local efc="$2"
  local m="$3"
  run_one "$dset" "$efc" "$m" &
}

run_phase() {
  local phase="$1"
  shift
  local datasets=("$@")
  local launched=0
  local total=0
  local dset m efc

  total=$(( ${#datasets[@]} * ${#Ms[@]} * ${#EFCs[@]} ))
  echo "[info] phase=${phase} datasets=${#datasets[@]} total_configs=${total} parallel=${PARALLEL_JOBS}"

  for dset in "${datasets[@]}"; do
    for m in "${Ms[@]}"; do
      for efc in "${EFCs[@]}"; do
        launch_one "$dset" "$efc" "$m"
        launched=$((launched + 1))
        if (( launched % PARALLEL_JOBS == 0 )); then
          wait -n
        fi
      done
    done
  done
  wait
  echo "[info] phase=${phase} done"
}

run_phase "static" "${DATASETS_STATIC[@]}"

echo "=== ALL DONE (static-only) ==="
