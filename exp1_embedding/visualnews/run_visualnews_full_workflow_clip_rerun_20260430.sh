#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-/home/hyhe/embedding_test_full_v2/bin/python}"
CPU_RUNNER="${CPU_RUNNER:-/home/hyhe/vlm-embed-compare/src/run_retrieval_benchmark.py}"
GPU_RUNNER="${GPU_RUNNER:-/home/hyhe/vlm-embed-compare/src/run_retrieval_benchmark_gpu.py}"
QRELS="/home/hyhe/vlm-embed-compare/gt/silver/qrels.tsv"
OUTPUT_ROOT="/home/hyhe/vlm-embed-compare/results_full_visualnews"
MODELS_CSV="clip,siglip2,qwen3"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OMP_THREADS="${OMP_THREADS:-8}"
HNSW_THREADS="${HNSW_THREADS:-8}"
TOPK=10
QUERY_LIMIT=0
CORPUS_LIMIT=0
SKIP_EXISTING=0
DRY_RUN=0
STEPS_CSV=""

usage() {
  cat <<'EOF'
usage: run_visualnews_full_workflow.sh [options]

Options:
  --output-root PATH      Result root. Default: /home/hyhe/vlm-embed-compare/results_full_visualnews
  --models CSV            Model order/filter. Default: clip,siglip2,qwen3
  --cuda-device N         GPU id for cuVS runs. Default: 0
  --topk N                TopK. Default: 10
  --query-limit N         Optional query subset for testing. Default: 0
  --corpus-limit N        Optional corpus subset for testing. Default: 0
  --steps CSV             Optional step filter. Names:
                          flatip,hnsw_balanced,hnsw_highrecall,
                          ivfpq_balanced,ivfpq_highrecall,
                          cuvs_ivfpq_balanced,cuvs_ivfpq_highrecall,
                          cagra_balanced,cagra_highrecall
  --skip-existing         Skip a step if its output JSON already exists
  --dry-run               Print commands only
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --models)
      MODELS_CSV="$2"
      shift 2
      ;;
    --cuda-device)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --topk)
      TOPK="$2"
      shift 2
      ;;
    --query-limit)
      QUERY_LIMIT="$2"
      shift 2
      ;;
    --corpus-limit)
      CORPUS_LIMIT="$2"
      shift 2
      ;;
    --steps)
      STEPS_CSV="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

declare -a MODEL_ORDER=()
IFS=',' read -r -a MODEL_ORDER <<< "$MODELS_CSV"
declare -a STEP_FILTER=()
if [[ -n "$STEPS_CSV" ]]; then
  IFS=',' read -r -a STEP_FILTER <<< "$STEPS_CSV"
fi

model_paths() {
  local model="$1"
  case "$model" in
    clip)
      Q_EMB="/home/hyhe/vlm-embed-compare/embeddings/clip/queries"
      C_EMB="/data3/hyhe/vlm-embed-compare/embeddings_rerun/visualnews_clip_corpus_rerun_20260429_173231/merged_corpus"
      IVFPQ_BAL_NLIST=8192
      IVFPQ_BAL_M=32
      IVFPQ_BAL_TRAIN=300000
      IVFPQ_HR_NLIST=16384
      IVFPQ_HR_M=64
      IVFPQ_HR_TRAIN=600000
      ;;
    siglip2)
      Q_EMB="/data2/hyhe/vlm-embed-compare/embeddings/siglip2/queries"
      C_EMB="/data2/hyhe/vlm-embed-compare/embeddings/siglip2/corpus"
      IVFPQ_BAL_NLIST=8192
      IVFPQ_BAL_M=48
      IVFPQ_BAL_TRAIN=300000
      IVFPQ_HR_NLIST=16384
      IVFPQ_HR_M=64
      IVFPQ_HR_TRAIN=600000
      ;;
    qwen3)
      Q_EMB="/home/hyhe/vlm-embed-compare/embeddings/qwen3/queries"
      C_EMB="/home/hyhe/vlm-embed-compare/embeddings/qwen3/corpus"
      IVFPQ_BAL_NLIST=8192
      IVFPQ_BAL_M=64
      IVFPQ_BAL_TRAIN=300000
      IVFPQ_HR_NLIST=16384
      IVFPQ_HR_M=128
      IVFPQ_HR_TRAIN=600000
      ;;
    *)
      echo "unsupported model: $model" >&2
      exit 2
      ;;
  esac
}

run_cmd() {
  local log_file="$1"
  shift
  echo
  echo "[$(date '+%F %T')] $*"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  "$@" 2>&1 | tee "$log_file"
}

should_run_step() {
  local step_name="$1"
  if [[ "${#STEP_FILTER[@]}" -eq 0 ]]; then
    return 0
  fi
  local step
  for step in "${STEP_FILTER[@]}"; do
    step="$(echo "$step" | xargs)"
    [[ -n "$step" ]] || continue
    if [[ "$step" == "$step_name" ]]; then
      return 0
    fi
  done
  return 1
}

maybe_run_step() {
  local expected_json="$1"
  local log_file="$2"
  shift 2
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$expected_json" ]]; then
    echo "[$(date '+%F %T')] skip existing: $expected_json"
    return 0
  fi
  run_cmd "$log_file" "$@"
}

mkdir -p "$OUTPUT_ROOT"

echo "[$(date '+%F %T')] workflow start"
echo "python_bin=$PYTHON_BIN"
echo "output_root=$OUTPUT_ROOT"
echo "models=$MODELS_CSV"
echo "steps=${STEPS_CSV:-all}"
echo "cuda_device=$CUDA_DEVICE"
echo "topk=$TOPK query_limit=$QUERY_LIMIT corpus_limit=$CORPUS_LIMIT"

for model in "${MODEL_ORDER[@]}"; do
  model="$(echo "$model" | xargs)"
  [[ -n "$model" ]] || continue

  model_paths "$model"
  MODEL_ROOT="$OUTPUT_ROOT/$model"
  LOG_DIR="$MODEL_ROOT/logs"
  mkdir -p "$MODEL_ROOT" "$LOG_DIR"

  echo
  echo "============================================================"
  echo "[$(date '+%F %T')] model=$model"
  echo "queries=$Q_EMB"
  echo "corpus=$C_EMB"
  echo "model_root=$MODEL_ROOT"
  echo "============================================================"

  if should_run_step "flatip"; then
    maybe_run_step \
      "$MODEL_ROOT/01_flatip/flatip.json" \
      "$LOG_DIR/01_flatip.log" \
      "$PYTHON_BIN" "$CPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/01_flatip" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --omp-num-threads "$OMP_THREADS" \
        --run-flat
  fi

  if should_run_step "hnsw_balanced"; then
    maybe_run_step \
      "$MODEL_ROOT/02_hnsw_balanced/hnsw.json" \
      "$LOG_DIR/02_hnsw_balanced.log" \
      "$PYTHON_BIN" "$CPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/02_hnsw_balanced" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --omp-num-threads "$OMP_THREADS" \
        --run-hnsw \
        --hnsw-m 32 \
        --hnsw-ef-construction 200 \
        --hnsw-ef-search 128 \
        --hnsw-num-threads "$HNSW_THREADS"
  fi

  if should_run_step "hnsw_highrecall"; then
    maybe_run_step \
      "$MODEL_ROOT/03_hnsw_highrecall/hnsw.json" \
      "$LOG_DIR/03_hnsw_highrecall.log" \
      "$PYTHON_BIN" "$CPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/03_hnsw_highrecall" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --omp-num-threads "$OMP_THREADS" \
        --run-hnsw \
        --hnsw-m 48 \
        --hnsw-ef-construction 400 \
        --hnsw-ef-search 256 \
        --hnsw-num-threads "$HNSW_THREADS"
  fi

  if should_run_step "ivfpq_balanced"; then
    maybe_run_step \
      "$MODEL_ROOT/04_ivfpq_balanced/ivfpq.json" \
      "$LOG_DIR/04_ivfpq_balanced.log" \
      "$PYTHON_BIN" "$CPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/04_ivfpq_balanced" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --omp-num-threads "$OMP_THREADS" \
        --run-ivfpq \
        --ivfpq-nlist "$IVFPQ_BAL_NLIST" \
        --ivfpq-m "$IVFPQ_BAL_M" \
        --ivfpq-nbits 8 \
        --ivfpq-nprobe 16 \
        --ivfpq-train-size "$IVFPQ_BAL_TRAIN"
  fi

  if should_run_step "ivfpq_highrecall"; then
    maybe_run_step \
      "$MODEL_ROOT/05_ivfpq_highrecall/ivfpq.json" \
      "$LOG_DIR/05_ivfpq_highrecall.log" \
      "$PYTHON_BIN" "$CPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/05_ivfpq_highrecall" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --omp-num-threads "$OMP_THREADS" \
        --run-ivfpq \
        --ivfpq-nlist "$IVFPQ_HR_NLIST" \
        --ivfpq-m "$IVFPQ_HR_M" \
        --ivfpq-nbits 8 \
        --ivfpq-nprobe 32 \
        --ivfpq-train-size "$IVFPQ_HR_TRAIN"
  fi

  if should_run_step "cuvs_ivfpq_balanced"; then
    maybe_run_step \
      "$MODEL_ROOT/06_cuvs_ivfpq_balanced/cuvs_ivfpq.json" \
      "$LOG_DIR/06_cuvs_ivfpq_balanced.log" \
      "$PYTHON_BIN" "$GPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/06_cuvs_ivfpq_balanced" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --cuda-device "$CUDA_DEVICE" \
        --run-cuvs-ivfpq \
        --metric inner_product \
        --cuvs-ivfpq-nlist "$IVFPQ_BAL_NLIST" \
        --cuvs-ivfpq-m "$IVFPQ_BAL_M" \
        --cuvs-ivfpq-nbits 8 \
        --cuvs-ivfpq-nprobe 16
  fi

  if should_run_step "cuvs_ivfpq_highrecall"; then
    maybe_run_step \
      "$MODEL_ROOT/07_cuvs_ivfpq_highrecall/cuvs_ivfpq.json" \
      "$LOG_DIR/07_cuvs_ivfpq_highrecall.log" \
      "$PYTHON_BIN" "$GPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/07_cuvs_ivfpq_highrecall" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --cuda-device "$CUDA_DEVICE" \
        --run-cuvs-ivfpq \
        --metric inner_product \
        --cuvs-ivfpq-nlist "$IVFPQ_HR_NLIST" \
        --cuvs-ivfpq-m "$IVFPQ_HR_M" \
        --cuvs-ivfpq-nbits 8 \
        --cuvs-ivfpq-nprobe 32
  fi

  if should_run_step "cagra_balanced"; then
    maybe_run_step \
      "$MODEL_ROOT/08_cagra_balanced/cagra.json" \
      "$LOG_DIR/08_cagra_balanced.log" \
      "$PYTHON_BIN" "$GPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/08_cagra_balanced" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --cuda-device "$CUDA_DEVICE" \
        --run-cagra \
        --metric inner_product \
        --cagra-graph-degree 64 \
        --cagra-intermediate-graph-degree 128 \
        --cagra-build-algo ivf_pq \
        --cagra-itopk-size 64 \
        --cagra-search-width 1
  fi

  if should_run_step "cagra_highrecall"; then
    maybe_run_step \
      "$MODEL_ROOT/09_cagra_highrecall/cagra.json" \
      "$LOG_DIR/09_cagra_highrecall.log" \
      "$PYTHON_BIN" "$GPU_RUNNER" \
        --model-name "$model" \
        --query-emb-dir "$Q_EMB" \
        --corpus-emb-dir "$C_EMB" \
        --qrels-tsv "$QRELS" \
        --output-dir "$MODEL_ROOT/09_cagra_highrecall" \
        --query-limit "$QUERY_LIMIT" \
        --corpus-limit "$CORPUS_LIMIT" \
        --topk "$TOPK" \
        --cuda-device "$CUDA_DEVICE" \
        --run-cagra \
        --metric inner_product \
        --cagra-graph-degree 64 \
        --cagra-intermediate-graph-degree 128 \
        --cagra-build-algo ivf_pq \
        --cagra-itopk-size 128 \
        --cagra-search-width 2
  fi
done

echo
echo "[$(date '+%F %T')] workflow done output_root=$OUTPUT_ROOT"
printf '%s\n' "$OUTPUT_ROOT"
