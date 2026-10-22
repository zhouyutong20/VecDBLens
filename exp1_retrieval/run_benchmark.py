#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np

from backends import BACKENDS
from search_space import get_param_groups
from utils.io import (
    append_csv,
    append_jsonl,
    as_float32_c_contiguous,
    load_embeddings,
    load_qrels,
    read_completed_keys,
)
from utils.metrics import compute_recall_at_k
from utils.system import gpu_memory_used_bytes, index_cache_path, rss_bytes, set_cpu_threads, write_json

FIXED_THREADS = 28


class PeakMemoryMonitor:
    def __init__(self, gpu_id: int, interval_sec: float = 0.2) -> None:
        self.gpu_id = gpu_id
        self.interval_sec = interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_peak_bytes: int = 0
        self.gpu_peak_bytes: int | None = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            cpu_now = rss_bytes()
            if cpu_now > self.cpu_peak_bytes:
                self.cpu_peak_bytes = cpu_now
            gpu_now = gpu_memory_used_bytes(self.gpu_id)
            if gpu_now is not None:
                if self.gpu_peak_bytes is None or gpu_now > self.gpu_peak_bytes:
                    self.gpu_peak_bytes = gpu_now
            self._stop_event.wait(self.interval_sec)

    def start(self) -> None:
        self.cpu_peak_bytes = rss_bytes()
        self.gpu_peak_bytes = gpu_memory_used_bytes(self.gpu_id)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 5))
            self._thread = None
        # Final sample to reduce undersampling risk.
        cpu_now = rss_bytes()
        if cpu_now > self.cpu_peak_bytes:
            self.cpu_peak_bytes = cpu_now
        gpu_now = gpu_memory_used_bytes(self.gpu_id)
        if gpu_now is not None:
            if self.gpu_peak_bytes is None or gpu_now > self.gpu_peak_bytes:
                self.gpu_peak_bytes = gpu_now


def peak_delta_bytes(baseline: int | None, peak: int | None) -> int | None:
    if baseline is None or peak is None:
        return None
    return max(0, peak - baseline)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANN retrieval benchmark on existing embeddings.")
    parser.add_argument("--backend", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--base-emb", required=True)
    parser.add_argument("--query-emb", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--text-max-length", type=int, required=True)
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument(
        "--threads",
        type=int,
        default=FIXED_THREADS,
        help=f"CPU thread count. The benchmark default is {FIXED_THREADS}.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path or output directory.")
    parser.add_argument("--metric", default="inner_product", choices=["inner_product"])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--search-batch-size", type=int, default=4096)
    parser.add_argument("--index-cache-dir", default=None)
    parser.add_argument("--reuse-index", action="store_true", help="Reuse serialized index if cache file exists.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip completed run_key rows in existing JSONL.")
    parser.add_argument("--max-base", type=int, default=0, help="Debug only: limit base vectors.")
    parser.add_argument("--max-query", type=int, default=0, help="Debug only: limit query vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned build/search groups without running.")
    return parser.parse_args()


def output_paths(output: str | Path) -> Tuple[Path, Path, Path]:
    output = Path(output)
    if output.suffix == ".jsonl":
        jsonl = output
        out_dir = output.parent
    else:
        out_dir = output
        jsonl = out_dir / "results.jsonl"
    csv_path = jsonl.with_suffix(".csv")
    return out_dir, jsonl, csv_path


def stable_hash(obj: Dict[str, Any]) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def make_run_key(args: argparse.Namespace, build_params: Dict[str, Any], search_params: Dict[str, Any]) -> str:
    return stable_hash(
        {
            "backend": args.backend,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "base_emb": str(Path(args.base_emb).resolve()),
            "query_emb": str(Path(args.query_emb).resolve()),
            "gt_path": str(Path(args.gt_path).resolve()),
            "build_params": build_params,
            "search_params": search_params,
            "topk": args.topk,
        }
    )


def make_index_key(args: argparse.Namespace, build_params: Dict[str, Any]) -> str:
    return stable_hash(
        {
            "backend": args.backend,
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "base_emb": str(Path(args.base_emb).resolve()),
            "build_params": build_params,
            "metric": args.metric,
        }
    )


def trim_debug_limit(x: np.ndarray, ids: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
    if limit and limit > 0:
        return x[:limit], ids[:limit]
    return x, ids


def result_template(
    args: argparse.Namespace,
    embedding_dim: int,
    num_base: int,
    num_query: int,
    build_params: Dict[str, Any],
    search_params: Dict[str, Any],
    run_key: str,
) -> Dict[str, Any]:
    return {
        "dataset": args.dataset_name,
        "model_name": args.model_name,
        "embedding_dim": int(embedding_dim),
        "text_max_length": int(args.text_max_length),
        "backend": args.backend,
        "build_params": build_params,
        "search_params": search_params,
        "num_base": int(num_base),
        "num_query": int(num_query),
        "run_key": run_key,
    }


def main() -> None:
    args = parse_args()
    # Respect an outer launcher-provided CUDA_VISIBLE_DEVICES mask. Only set it
    # here for standalone runs where no GPU mask has been established yet.
    if args.gpu_id >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_cpu_threads(args.threads)

    out_dir, jsonl_path, csv_path = output_paths(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_cache_dir = Path(args.index_cache_dir) if args.index_cache_dir else out_dir / "index_cache"

    base, base_ids = load_embeddings(args.base_emb)
    queries, query_ids = load_embeddings(args.query_emb)
    base, base_ids = trim_debug_limit(base, base_ids, args.max_base)
    queries, query_ids = trim_debug_limit(queries, query_ids, args.max_query)
    if base.ndim != 2 or queries.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got base={base.shape}, queries={queries.shape}")
    if base.shape[1] != queries.shape[1]:
        raise ValueError(f"Dim mismatch: base={base.shape}, queries={queries.shape}")
    embedding_dim = int(base.shape[1])

    qrels = load_qrels(args.gt_path)
    completed = read_completed_keys(jsonl_path) if args.skip_existing else set()
    plan = list(get_param_groups(args.backend, args.model_name, embedding_dim, args.dataset_name))

    run_meta = {
        "backend": args.backend,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "base_emb": args.base_emb,
        "query_emb": args.query_emb,
        "gt_path": args.gt_path,
        "embedding_dim": embedding_dim,
        "text_max_length": args.text_max_length,
        "num_base": int(base.shape[0]),
        "num_query": int(queries.shape[0]),
        "threads": args.threads,
        "gpu_id": args.gpu_id,
        "index_cache_dir": str(index_cache_dir),
        "num_build_groups": len(plan),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / "run_meta.json", run_meta)

    if args.dry_run:
        print(json.dumps({"meta": run_meta, "plan": plan}, ensure_ascii=False, indent=2))
        return

    Backend = BACKENDS[args.backend]
    if args.backend.startswith("faiss") or args.backend == "flat":
        base_for_backend = as_float32_c_contiguous(base)
        queries_for_backend = as_float32_c_contiguous(queries)
    else:
        base_for_backend = base
        queries_for_backend = queries

    for build_idx, (build_params, search_params_list) in enumerate(plan):
        index_key = make_index_key(args, build_params)
        backend = Backend(metric=args.metric, gpu_id=args.gpu_id, search_batch_size=args.search_batch_size)
        cache_path = index_cache_path(index_cache_dir, index_key, backend.index_suffix)

        rss_before = rss_bytes()
        gpu_before = gpu_memory_used_bytes(args.gpu_id)
        build_peak_monitor = PeakMemoryMonitor(args.gpu_id)
        build_peak_monitor.start()
        try:
            build_time, index_size_bytes = backend.build_or_load(
                base_for_backend,
                build_params,
                cache_path=cache_path,
                reuse_index=args.reuse_index,
            )
            build_error = None
        except Exception as exc:
            build_time, index_size_bytes = 0.0, 0
            build_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            build_peak_monitor.stop()

        rss_after = rss_bytes()
        gpu_after = gpu_memory_used_bytes(args.gpu_id)
        cpu_memory_delta = max(0, rss_after - rss_before)
        gpu_memory_delta = None if gpu_before is None or gpu_after is None else max(0, gpu_after - gpu_before)
        build_cpu_peak_delta = peak_delta_bytes(rss_before, build_peak_monitor.cpu_peak_bytes)
        build_gpu_peak_delta = peak_delta_bytes(gpu_before, build_peak_monitor.gpu_peak_bytes)

        if build_error:
            for search_params in search_params_list:
                run_key = make_run_key(args, build_params, search_params)
                if run_key in completed:
                    continue
                row = result_template(
                    args, embedding_dim, base.shape[0], queries.shape[0], build_params, search_params, run_key
                )
                row.update(
                    {
                        "build_time_sec": build_time,
                        "index_size_bytes": index_size_bytes,
                        "gpu_memory_bytes": gpu_memory_delta,
                        "cpu_memory_bytes": cpu_memory_delta,
                        "cpu_peak_memory_bytes": build_cpu_peak_delta,
                        "gpu_peak_memory_bytes": build_gpu_peak_delta,
                        "recall@1": None,
                        "recall@5": None,
                        "recall@10": None,
                        "qps": None,
                        "total_search_time_sec": None,
                        "avg_latency_ms": None,
                        "status": "build_error",
                        "error": build_error,
                    }
                )
                append_jsonl(jsonl_path, row)
                append_csv(csv_path, row)
            continue

        for search_params in search_params_list:
            run_key = make_run_key(args, build_params, search_params)
            if run_key in completed:
                print(f"skip existing run_key={run_key} build={build_params} search={search_params}", flush=True)
                continue

            row = result_template(args, embedding_dim, base.shape[0], queries.shape[0], build_params, search_params, run_key)
            search_rss_before = rss_bytes()
            search_gpu_before = gpu_memory_used_bytes(args.gpu_id)
            search_peak_monitor = PeakMemoryMonitor(args.gpu_id)
            search_peak_monitor.start()
            try:
                neighbors, search_time = backend.search(queries_for_backend, args.topk, search_params)
                recall = compute_recall_at_k(neighbors, query_ids, base_ids, qrels, ks=(1, 5, 10))
                qps = float(queries.shape[0] / search_time) if search_time > 0 else None
                search_peak_monitor.stop()
                search_cpu_peak_delta = peak_delta_bytes(search_rss_before, search_peak_monitor.cpu_peak_bytes)
                search_gpu_peak_delta = peak_delta_bytes(search_gpu_before, search_peak_monitor.gpu_peak_bytes)
                cpu_peak_delta = max(
                    x for x in [build_cpu_peak_delta, search_cpu_peak_delta, cpu_memory_delta] if x is not None
                )
                gpu_peak_candidates = [x for x in [build_gpu_peak_delta, search_gpu_peak_delta, gpu_memory_delta] if x is not None]
                gpu_peak_delta = max(gpu_peak_candidates) if gpu_peak_candidates else None
                row.update(
                    {
                        "build_time_sec": float(build_time),
                        "index_size_bytes": int(index_size_bytes),
                        "gpu_memory_bytes": gpu_memory_delta,
                        "cpu_memory_bytes": cpu_memory_delta,
                        "cpu_peak_memory_bytes": cpu_peak_delta,
                        "gpu_peak_memory_bytes": gpu_peak_delta,
                        **recall,
                        "qps": qps,
                        "total_search_time_sec": float(search_time),
                        "avg_latency_ms": float(search_time * 1000.0 / queries.shape[0]),
                        "status": "ok",
                        "error": None,
                    }
                )
            except Exception as exc:
                search_peak_monitor.stop()
                search_cpu_peak_delta = peak_delta_bytes(search_rss_before, search_peak_monitor.cpu_peak_bytes)
                search_gpu_peak_delta = peak_delta_bytes(search_gpu_before, search_peak_monitor.gpu_peak_bytes)
                cpu_peak_delta = max(
                    x for x in [build_cpu_peak_delta, search_cpu_peak_delta, cpu_memory_delta] if x is not None
                )
                gpu_peak_candidates = [x for x in [build_gpu_peak_delta, search_gpu_peak_delta, gpu_memory_delta] if x is not None]
                gpu_peak_delta = max(gpu_peak_candidates) if gpu_peak_candidates else None
                traceback.print_exc()
                row.update(
                    {
                        "build_time_sec": float(build_time),
                        "index_size_bytes": int(index_size_bytes),
                        "gpu_memory_bytes": gpu_memory_delta,
                        "cpu_memory_bytes": cpu_memory_delta,
                        "cpu_peak_memory_bytes": cpu_peak_delta,
                        "gpu_peak_memory_bytes": gpu_peak_delta,
                        "recall@1": None,
                        "recall@5": None,
                        "recall@10": None,
                        "qps": None,
                        "total_search_time_sec": None,
                        "avg_latency_ms": None,
                        "status": "search_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            append_jsonl(jsonl_path, row)
            append_csv(csv_path, row)
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

        print(f"finished build group {build_idx + 1}/{len(plan)}: {build_params}", flush=True)


if __name__ == "__main__":
    main()
