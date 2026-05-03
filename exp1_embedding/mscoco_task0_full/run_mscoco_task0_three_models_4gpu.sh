#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/hyhe/vnemb/bin/python}"
MBEIR_ROOT="${MBEIR_ROOT:-/data1/lnyan/projects/GENIUS-CVPR25/M-BEIR}"
DATA_DIR="${DATA_DIR:-/data3/hyhe/mscoco_task0_full}"
OUT_ROOT="${OUT_ROOT:-/data3/hyhe/embeddings/mscoco_task0_full}"
LOG_ROOT="${LOG_ROOT:-/data3/hyhe/logs/mscoco_task0_embedding_$(date +%Y%m%d_%H%M%S)}"

CLIP_MODEL="${CLIP_MODEL:-/data3/hyhe/models/clip-vit-base-patch32}"
SIGLIP2_MODEL="${SIGLIP2_MODEL:-/data3/hyhe/models/siglip2-base-patch16-224}"
QWEN3_MODEL="${QWEN3_MODEL:-/data3/hyhe/models/Qwen3-VL-Embedding-8B}"

QUERY_JSONL="${QUERY_JSONL:-$DATA_DIR/queries_caption.jsonl}"
CORPUS_JSONL="${CORPUS_JSONL:-$DATA_DIR/corpus_images.jsonl}"
QRELS_PATH="${QRELS_PATH:-$DATA_DIR/qrels.tsv}"

CLIP_GPU="${CLIP_GPU:-0}"
SIGLIP2_GPU="${SIGLIP2_GPU:-1}"
QWEN3_GPU0="${QWEN3_GPU0:-2}"
QWEN3_GPU1="${QWEN3_GPU1:-3}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$DATA_DIR" "$OUT_ROOT" "$LOG_ROOT"

if [[ ! -s "$QUERY_JSONL" || ! -s "$CORPUS_JSONL" || ! -s "$QRELS_PATH" ]]; then
  echo "[$(date '+%F %T')] building M-BEIR MSCOCO task0 full JSONL/qrels"
  "$PYTHON_BIN" "$ROOT/scripts/build_mbeir_mscoco_task0_jsonl.py" \
    --mbeir-root "$MBEIR_ROOT" \
    --output-dir "$DATA_DIR" | tee "$LOG_ROOT/build_jsonl.log"
fi

QUERY_COUNT="$(wc -l < "$QUERY_JSONL")"
CORPUS_COUNT="$(wc -l < "$CORPUS_JSONL")"
QWEN_QUERY_MID="$(( (QUERY_COUNT + 1) / 2 ))"
QWEN_CORPUS_MID="$(( (CORPUS_COUNT + 1) / 2 ))"

run_model_full() {
  local gpu="$1"
  local model="$2"
  local model_path="$3"
  local out_name="$4"
  local query_batch="$5"
  local corpus_batch="$6"
  local query_shard="$7"
  local corpus_shard="$8"
  local log_file="$LOG_ROOT/${out_name}.log"

  {
    echo "[$(date '+%F %T')] start $out_name on GPU $gpu"
    mkdir -p "$OUT_ROOT/$out_name"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/embed_flickr.py" \
      --model "$model" \
      --split queries \
      --input-jsonl "$QUERY_JSONL" \
      --output-dir "$OUT_ROOT/$out_name/queries" \
      --model-path "$model_path" \
      --batch-size "$query_batch" \
      --shard-size "$query_shard" \
      --log-every 20

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/embed_flickr.py" \
      --model "$model" \
      --split corpus \
      --input-jsonl "$CORPUS_JSONL" \
      --output-dir "$OUT_ROOT/$out_name/corpus" \
      --model-path "$model_path" \
      --image-root "$MBEIR_ROOT" \
      --batch-size "$corpus_batch" \
      --shard-size "$corpus_shard" \
      --log-every 20
    echo "[$(date '+%F %T')] done $out_name"
  } 2>&1 | tee "$log_file"
}

run_qwen_worker() {
  local gpu="$1"
  local worker="$2"
  local q_start="$3"
  local q_end="$4"
  local c_start="$5"
  local c_end="$6"
  local log_file="$LOG_ROOT/qwen3_${worker}.log"

  {
    echo "[$(date '+%F %T')] start qwen3 $worker on GPU $gpu query=$q_start:$q_end corpus=$c_start:$c_end"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/embed_flickr.py" \
      --model qwen3 \
      --split queries \
      --input-jsonl "$QUERY_JSONL" \
      --output-dir "$OUT_ROOT/mscoco_qwen3/queries.workers/$worker" \
      --model-path "$QWEN3_MODEL" \
      --batch-size 16 \
      --shard-size 2048 \
      --start-index "$q_start" \
      --end-index "$q_end" \
      --log-every 20

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "$ROOT/scripts/embed_flickr.py" \
      --model qwen3 \
      --split corpus \
      --input-jsonl "$CORPUS_JSONL" \
      --output-dir "$OUT_ROOT/mscoco_qwen3/corpus.workers/$worker" \
      --model-path "$QWEN3_MODEL" \
      --image-root "$MBEIR_ROOT" \
      --batch-size 4 \
      --shard-size 1024 \
      --start-index "$c_start" \
      --end-index "$c_end" \
      --log-every 20
    echo "[$(date '+%F %T')] done qwen3 $worker"
  } 2>&1 | tee "$log_file"
}

run_model_full "$CLIP_GPU" clip "$CLIP_MODEL" mscoco_clip 256 256 8192 8192 &
pid_clip=$!
run_model_full "$SIGLIP2_GPU" siglip2 "$SIGLIP2_MODEL" mscoco_siglip2 128 128 8192 8192 &
pid_siglip2=$!
run_qwen_worker "$QWEN3_GPU0" worker_00 0 "$QWEN_QUERY_MID" 0 "$QWEN_CORPUS_MID" &
pid_qwen0=$!
run_qwen_worker "$QWEN3_GPU1" worker_01 "$QWEN_QUERY_MID" "$QUERY_COUNT" "$QWEN_CORPUS_MID" "$CORPUS_COUNT" &
pid_qwen1=$!

wait "$pid_clip"
wait "$pid_siglip2"
wait "$pid_qwen0"
wait "$pid_qwen1"

"$PYTHON_BIN" "$ROOT/scripts/merge_embedding_worker_shards.py" \
  --workers-root "$OUT_ROOT/mscoco_qwen3/queries.workers" \
  --output-dir "$OUT_ROOT/mscoco_qwen3/queries" \
  --force | tee "$LOG_ROOT/qwen3_merge_queries.log"

"$PYTHON_BIN" "$ROOT/scripts/merge_embedding_worker_shards.py" \
  --workers-root "$OUT_ROOT/mscoco_qwen3/corpus.workers" \
  --output-dir "$OUT_ROOT/mscoco_qwen3/corpus" \
  --force | tee "$LOG_ROOT/qwen3_merge_corpus.log"

echo "[$(date '+%F %T')] all M-BEIR MSCOCO task0 embeddings complete: $OUT_ROOT"
