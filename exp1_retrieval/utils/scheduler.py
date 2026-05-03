from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class GPUInfo:
    gpu_id: int
    memory_used_mib: int
    utilization: int


def query_gpus() -> List[GPUInfo]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx, mem, util = [x.strip() for x in line.split(",")]
        gpus.append(GPUInfo(int(idx), int(mem), int(util)))
    return gpus


def pick_free_gpu(
    allowed: Optional[Iterable[int]] = None,
    max_memory_used_mib: int = 4096,
    max_utilization: int = 20,
) -> Optional[int]:
    allowed_set = set(allowed) if allowed is not None else None
    candidates = [
        gpu
        for gpu in query_gpus()
        if (allowed_set is None or gpu.gpu_id in allowed_set)
        and gpu.memory_used_mib <= max_memory_used_mib
        and gpu.utilization <= max_utilization
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x.memory_used_mib, x.utilization, x.gpu_id))
    return candidates[0].gpu_id


def wait_for_free_gpu(
    allowed: Optional[Iterable[int]] = None,
    max_memory_used_mib: int = 4096,
    max_utilization: int = 20,
    poll_sec: int = 60,
) -> int:
    while True:
        gpu = pick_free_gpu(allowed, max_memory_used_mib, max_utilization)
        if gpu is not None:
            return gpu
        time.sleep(poll_sec)
