#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="$1"
GPU="$2"
VNENV="${VNENV:-/home/hyhe/vnemb}"
ROOT="${ROOT:-/home/hyhe/vlm-embed-compare}"
GT_DIR="${GT_DIR:-$ROOT/gt/silver}"
EMB_DIR="${EMB_DIR:-$ROOT/embeddings}"

mkdir -p "$ROOT/logs"

COMMON_EXTRA_ARGS=()
QUERY_EXTRA_ARGS=()
CORPUS_EXTRA_ARGS=()

if [[ "$MODEL" == "clip" ]]; then
  source "$VNENV/bin/activate"
  MODEL_PATH="/home/hyhe/clip-vit-base-patch32"
  QUERY_BATCH=128
  CORPUS_BATCH=64
  QUERY_SHARD=32768
  CORPUS_SHARD=32768
elif [[ "$MODEL" == "qwen3" ]]; then
  source "$VNENV/bin/activate"
  MODEL_PATH="/home/hyhe/Qwen3-VL-Embedding-8B"
  QUERY_BATCH=8
  CORPUS_BATCH=4
  QUERY_SHARD=4096
  CORPUS_SHARD=2048
  COMMON_EXTRA_ARGS+=(--qwen3-max-pixels 1048576)
elif [[ "$MODEL" == "siglip2" ]]; then
  source "$VNENV/bin/activate"
  MODEL_PATH="/home/hyhe/siglip2-base-patch16-224"
  QUERY_BATCH=256
  CORPUS_BATCH=128
  QUERY_SHARD=32768
  CORPUS_SHARD=32768
else
  echo "unknown model: $MODEL" >&2
  exit 2
fi

echo "[$(date '+%F %T')] start model=$MODEL gpu=$GPU"

QUERY_ARGS=(
  --model "$MODEL"
  --split queries
  --input-jsonl "$GT_DIR/queries_caption.jsonl"
  --output-dir "$EMB_DIR/$MODEL/queries"
  --model-path "$MODEL_PATH"
  --batch-size "$QUERY_BATCH"
  --shard-size "$QUERY_SHARD"
  --log-every 20
  "${COMMON_EXTRA_ARGS[@]}"
  "${QUERY_EXTRA_ARGS[@]}"
)
CORPUS_ARGS=(
  --model "$MODEL"
  --split corpus
  --input-jsonl "$GT_DIR/corpus_images.jsonl"
  --output-dir "$EMB_DIR/$MODEL/corpus"
  --model-path "$MODEL_PATH"
  --batch-size "$CORPUS_BATCH"
  --shard-size "$CORPUS_SHARD"
  --log-every 20
  "${COMMON_EXTRA_ARGS[@]}"
  "${CORPUS_EXTRA_ARGS[@]}"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  QUERY_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
  CORPUS_ARGS+=(--checkpoint-path "$CHECKPOINT_PATH")
fi

CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT_DIR/embed_visualnews.py" "${QUERY_ARGS[@]}"

CUDA_VISIBLE_DEVICES="$GPU" python "$SCRIPT_DIR/embed_visualnews.py" "${CORPUS_ARGS[@]}"

echo "[$(date '+%F %T')] done model=$MODEL gpu=$GPU"
