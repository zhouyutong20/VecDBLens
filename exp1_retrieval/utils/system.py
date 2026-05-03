from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Optional

import psutil


def set_cpu_threads(threads: int) -> None:
    value = str(int(threads))
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "FAISS_NUM_THREADS",
    ]:
        os.environ[key] = value
    try:
        import faiss

        faiss.omp_set_num_threads(int(threads))
    except Exception:
        pass


def rss_bytes() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


def file_size(path: str | Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def gpu_memory_used_bytes(gpu_id: Optional[int]) -> Optional[int]:
    if gpu_id is None or gpu_id < 0:
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(out.splitlines()[0]) * 1024 * 1024
    except Exception:
        return None


def index_cache_path(cache_dir: Path, run_key: str, suffix: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{run_key}.{suffix}"


def write_json(path: str | Path, obj: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
