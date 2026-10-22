#!/usr/bin/env python3
"""Prepare VisualNews silver 20k-query benchmark artifacts (qrels subset + models JSON).

Corpus / query embeddings are *not* copied: use full-model ``corpus/`` shards plus
``test_queries/`` (20k) produced by ``make_test_queries.py``. This script only:

1. Reads query ids from ``source_mapping.jsonl`` (or ``picked_indices.npy`` + query id files).
2. Filters full silver ``qrels.tsv`` to those query ids.
3. Optionally writes a **filtered** ``qrels.tsv`` under ``--output-dataset-root`` (smaller copy).
4. Writes ``configs/benchmark_visualnews_test20k_data3_3models.json`` with ``gt_path`` set to the
   **full** silver qrels (default: ``/home/hyhe/vlm-embed-compare/gt/silver/qrels.tsv``), same as
   ``run_cpu_faiss_visualnews_test20k.sh`` -- evaluation uses 20k ``test_queries`` only; full qrels is fine.

Example::

  python prepare_visualnews_test20k.py \\
    --reference-mapping /data3/hyhe/vlm-embed-compare/embeddings/siglip2/test_queries/source_mapping.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent
DEFAULT_EMB_ROOT = Path("/data3/hyhe/vlm-embed-compare/embeddings")
DEFAULT_OUT_DATASET = Path("/data3/hyhe/visualnews_silver_test20k")
DEFAULT_REF = DEFAULT_EMB_ROOT / "siglip2/test_queries/source_mapping.jsonl"
DEFAULT_SOURCE_QRELS = Path("/home/hyhe/vlm-embed-compare/gt/silver/qrels.tsv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source-qrels",
        type=Path,
        default=DEFAULT_SOURCE_QRELS,
        help=f"Full VisualNews silver qrels.tsv (default: {DEFAULT_SOURCE_QRELS}).",
    )
    p.add_argument(
        "--reference-mapping",
        type=Path,
        default=DEFAULT_REF,
        help="source_mapping.jsonl from test_queries (uses field query_id).",
    )
    p.add_argument(
        "--output-dataset-root",
        type=Path,
        default=DEFAULT_OUT_DATASET,
        help="Directory for filtered qrels.tsv and stats.",
    )
    p.add_argument(
        "--embedding-root",
        type=Path,
        default=DEFAULT_EMB_ROOT,
        help="Root containing clip/, siglip2/, qwen3/ each with corpus/ and test_queries/.",
    )
    p.add_argument(
        "--output-models-config",
        type=Path,
        default=ROOT.parents[1] / "exp1_retrieval/configs/benchmark_visualnews_test20k_data3_3models.json",
        help="Benchmark JSON written for run_cpu/run_gpu scripts.",
    )
    p.add_argument(
        "--dataset-name",
        default="visualnews_test20k",
        help="dataset_name written into the models config (must match search_space prefix).",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_query_ids_from_mapping(path: Path) -> List[int]:
    ids: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.append(int(json.loads(line)["query_id"]))
    return ids


def filter_qrels_for_queries(qrels_path: Path, query_ids: Set[int]) -> List[str]:
    out: List[str] = []
    with qrels_path.open("r", encoding="utf-8") as f:
        for raw in f:
            row = raw.strip()
            if not row:
                continue
            parts = row.split()
            if parts[0].lower() in {"qid", "query-id", "query_id"}:
                continue
            if len(parts) < 2:
                continue
            qid = int(parts[0])
            if qid in query_ids:
                out.append(row)
    return out


def main() -> None:
    args = parse_args()
    if not args.reference_mapping.is_file():
        raise FileNotFoundError(f"reference mapping not found: {args.reference_mapping}")
    if not args.source_qrels.is_file():
        raise FileNotFoundError(f"source qrels not found: {args.source_qrels}")

    qids_list = load_query_ids_from_mapping(args.reference_mapping)
    query_ids: Set[int] = set(qids_list)
    if len(query_ids) != len(qids_list):
        raise ValueError(f"duplicate query_id in mapping: {len(qids_list)} lines, {len(query_ids)} unique")

    lines = filter_qrels_for_queries(args.source_qrels, query_ids)
    if not lines:
        raise SystemExit("filtered qrels is empty; check source qrels path and query_id overlap")

    emb = args.embedding_root.resolve()
    out_ds = args.output_dataset_root.resolve()
    qrels_out = out_ds / "qrels.tsv"
    stats = {
        "dataset_name": args.dataset_name,
        "num_query_ids_from_mapping": len(query_ids),
        "num_qrels_lines_kept": len(lines),
        "source_qrels": str(args.source_qrels.resolve()),
        "reference_mapping": str(args.reference_mapping.resolve()),
        "output_qrels": str(qrels_out),
    }

    models = [
        {
            "model_name": "CLIP",
            "dataset_name": args.dataset_name,
            "embedding_dim": 512,
            "text_max_length": 77,
            "base_emb": str(emb / "clip/corpus"),
            "query_emb": str(emb / "clip/test_queries"),
            "gt_path": str(args.source_qrels.resolve()),
        },
        {
            "model_name": "SigLIP2",
            "dataset_name": args.dataset_name,
            "embedding_dim": 768,
            "text_max_length": 64,
            "base_emb": str(emb / "siglip2/corpus"),
            "query_emb": str(emb / "siglip2/test_queries"),
            "gt_path": str(args.source_qrels.resolve()),
        },
        {
            "model_name": "QwenVL3",
            "dataset_name": args.dataset_name,
            "embedding_dim": 4096,
            "text_max_length": 512,
            "base_emb": str(emb / "qwen3/corpus"),
            "query_emb": str(emb / "qwen3/test_queries"),
            "gt_path": str(args.source_qrels.resolve()),
        },
    ]

    if args.dry_run:
        print(json.dumps({**stats, "models": models}, indent=2, ensure_ascii=False))
        return

    out_ds.mkdir(parents=True, exist_ok=True)
    qrels_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_ds / "prepare_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    args.output_models_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_models_config.write_text(
        json.dumps(models, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {qrels_out}")
    print(f"Wrote {args.output_models_config}")


if __name__ == "__main__":
    main()
