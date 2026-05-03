#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.scheduler import query_gpus, wait_for_free_gpu


DEFAULT_BACKENDS = ["flat", "faiss_hnsw", "faiss_ivfpq", "cuvs_ivf_pq", "cagra"]
FIXED_THREADS = 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch retrieval benchmark jobs on free GPUs.")
    parser.add_argument("--models-config", default=str(ROOT / "configs" / "visualnews_models.json"))
    parser.add_argument("--backends", nargs="+", default=DEFAULT_BACKENDS)
    parser.add_argument("--gpus", default="", help="Comma-separated GPU ids. Empty means all GPUs.")
    parser.add_argument(
        "--cpu-sets",
        default="",
        help='Semicolon-separated CPU sets (e.g. "0-27;28-55;56-83"). Empty means no taskset binding.',
    )
    parser.add_argument("--output-root", default="/data3/hyhe/retrieval_benchmark/results/visualnews")
    parser.add_argument("--python", default="", help="Force one Python for all backends. Empty means backend-specific default.")
    parser.add_argument("--faiss-python", default="/home/ytzhou/anaconda3/envs/embedding_test/bin/python")
    parser.add_argument("--cuvs-python", default="/home/hyhe/embedding_test_full/bin/python")
    parser.add_argument(
        "--threads",
        type=int,
        default=FIXED_THREADS,
        help=f"CPU thread count forwarded to run_benchmark.py. Default is {FIXED_THREADS}.",
    )
    parser.add_argument("--max-jobs", type=int, default=1, help="Total concurrent jobs.")
    parser.add_argument("--max-memory-used-mib", type=int, default=4096)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--poll-sec", type=int, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def allowed_gpus(gpus: str) -> List[int] | None:
    if not gpus.strip():
        return None
    return [int(x) for x in gpus.split(",") if x.strip()]


def allowed_cpu_sets(cpu_sets: str) -> List[str] | None:
    if not cpu_sets.strip():
        return None
    return [x.strip() for x in cpu_sets.split(";") if x.strip()]


def job_cmd(args: argparse.Namespace, model: Dict, backend: str, gpu_id: int) -> List[str]:
    output = Path(args.output_root) / model["model_name"] / backend
    python = args.python or (args.cuvs_python if backend in {"cuvs_ivf_pq", "cuvs_ivfpq", "cagra", "cuvs_cagra"} else args.faiss_python)
    cmd = [
        python,
        str(ROOT / "run_benchmark.py"),
        "--backend",
        backend,
        "--model-name",
        model["model_name"],
        "--dataset-name",
        model.get("dataset_name", "VisualNews"),
        "--base-emb",
        model["base_emb"],
        "--query-emb",
        model["query_emb"],
        "--gt-path",
        model["gt_path"],
        "--text-max-length",
        str(model["text_max_length"]),
        "--threads",
        str(args.threads),
        "--output",
        str(output),
        "--gpu-id",
        str(gpu_id),
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.reuse_index:
        cmd.append("--reuse-index")
    return cmd


def main() -> None:
    args = parse_args()
    models = json.loads(Path(args.models_config).read_text(encoding="utf-8"))
    gpu_pool = allowed_gpus(args.gpus)
    cpu_pool = allowed_cpu_sets(args.cpu_sets)
    jobs = [(model, backend) for model in models for backend in args.backends]

    if args.dry_run:
        for model, backend in jobs:
            print(json.dumps({"model": model["model_name"], "backend": backend}, ensure_ascii=False))
        return

    running: List[tuple[subprocess.Popen, int, str | None]] = []
    logs_dir = Path(args.output_root) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for model, backend in jobs:
        running = [(p, gpu, cpu_set) for p, gpu, cpu_set in running if p.poll() is None]
        while len(running) >= args.max_jobs:
            time.sleep(args.poll_sec)
            running = [(p, gpu, cpu_set) for p, gpu, cpu_set in running if p.poll() is None]

        while True:
            running = [(p, gpu, cpu_set) for p, gpu, cpu_set in running if p.poll() is None]
            busy_gpus = {gpu for _, gpu, _ in running}
            busy_cpu_sets = {cpu_set for _, _, cpu_set in running if cpu_set is not None}
            if gpu_pool is None:
                effective_pool = [gpu.gpu_id for gpu in query_gpus() if gpu.gpu_id not in busy_gpus]
            else:
                effective_pool = [gpu for gpu in gpu_pool if gpu not in busy_gpus]
            if cpu_pool is None:
                effective_cpu_sets = [None]
            else:
                effective_cpu_sets = [cpu_set for cpu_set in cpu_pool if cpu_set not in busy_cpu_sets]
            if effective_pool and effective_cpu_sets:
                break
            time.sleep(args.poll_sec)
        gpu_id = wait_for_free_gpu(
            effective_pool,
            max_memory_used_mib=args.max_memory_used_mib,
            max_utilization=args.max_utilization,
            poll_sec=args.poll_sec,
        )
        cmd = job_cmd(args, model, backend, gpu_id)
        cpu_set = effective_cpu_sets[0]
        launch_cmd = ["taskset", "-c", cpu_set, *cmd] if cpu_set is not None else cmd
        log_path = logs_dir / f"{model['model_name']}_{backend}_{int(time.time())}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("LAUNCH", " ".join(launch_cmd), "GPU", gpu_id, "CPUSET", cpu_set, "LOG", log_path, flush=True)
        log_f = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(launch_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
        running.append((proc, gpu_id, cpu_set))

    while any(p.poll() is None for p, _, _ in running):
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    main()
