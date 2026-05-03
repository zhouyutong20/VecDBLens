from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backends.base import BaseBackend


class CuVSIVFPQBackend(BaseBackend):
    name = "cuvs_ivf_pq"
    index_suffix = "cuvs_ivfpq"

    def __init__(self, metric: str = "inner_product", gpu_id: Optional[int] = None, search_batch_size: int = 4096) -> None:
        super().__init__(metric, gpu_id, search_batch_size)
        if gpu_id is not None and gpu_id >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self._base_gpu = None

    def build(self, base: np.ndarray, build_params: Dict[str, Any]):
        import cupy as cp
        from cuvs.neighbors import ivf_pq

        base_gpu = cp.asarray(base, dtype=cp.float32)
        self._base_gpu = base_gpu
        params = ivf_pq.IndexParams(
            n_lists=int(build_params["n_lists"]),
            metric=self.metric,
            pq_dim=int(build_params["pq_dim"]),
            pq_bits=int(build_params["pq_bits"]),
        )
        return ivf_pq.build(params, base_gpu)

    def search(self, queries: np.ndarray, topk: int, search_params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        import cupy as cp
        from cuvs.neighbors import ivf_pq

        params = ivf_pq.SearchParams(n_probes=int(search_params["n_probes"]))
        chunks = []
        start = time.perf_counter()
        for beg in range(0, queries.shape[0], self.search_batch_size):
            q_gpu = cp.asarray(queries[beg : beg + self.search_batch_size], dtype=cp.float32)
            _, neighbors = ivf_pq.search(params, self.index, q_gpu, topk)
            cp.cuda.Stream.null.synchronize()
            chunks.append(cp.asarray(neighbors).get().astype(np.int64, copy=False))
        search_time = time.perf_counter() - start
        return np.concatenate(chunks, axis=0), search_time

    def save_index(self, path: Path) -> None:
        from cuvs.neighbors import ivf_pq

        ivf_pq.save(str(path), self.index)

    def load_index(self, path: Path):
        from cuvs.neighbors import ivf_pq

        return ivf_pq.load(str(path))
