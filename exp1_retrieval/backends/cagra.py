from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from backends.base import BaseBackend


class CAGRABackend(BaseBackend):
    name = "cagra"
    index_suffix = "cagra"

    def __init__(self, metric: str = "inner_product", gpu_id: Optional[int] = None, search_batch_size: int = 4096) -> None:
        super().__init__(metric, gpu_id, search_batch_size)
        if gpu_id is not None and gpu_id >= 0 and "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        self._base_gpu = None

    def build(self, base: np.ndarray, build_params: Dict[str, Any]):
        import cupy as cp
        from cuvs.neighbors import cagra

        base_gpu = cp.asarray(base, dtype=cp.float32)
        self._base_gpu = base_gpu
        build_algo = str(build_params["graph_build_algo"]).lower()
        params = cagra.IndexParams(
            metric=self.metric,
            graph_degree=int(build_params["graph_degree"]),
            intermediate_graph_degree=int(build_params["intermediate_graph_degree"]),
            build_algo=build_algo,
        )
        return cagra.build(params, base_gpu)

    def search(self, queries: np.ndarray, topk: int, search_params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        import cupy as cp
        from cuvs.neighbors import cagra

        params = cagra.SearchParams(
            itopk_size=int(search_params["itopk_size"]),
            search_width=int(search_params["search_width"]),
        )
        chunks = []
        start = time.perf_counter()
        for beg in range(0, queries.shape[0], self.search_batch_size):
            q_gpu = cp.asarray(queries[beg : beg + self.search_batch_size], dtype=cp.float32)
            _, neighbors = cagra.search(params, self.index, q_gpu, topk)
            cp.cuda.Stream.null.synchronize()
            chunks.append(cp.asarray(neighbors).get().astype(np.int64, copy=False))
        search_time = time.perf_counter() - start
        return np.concatenate(chunks, axis=0), search_time

    def save_index(self, path: Path) -> None:
        from cuvs.neighbors import cagra

        cagra.save(str(path), self.index)

    def load_index(self, path: Path):
        from cuvs.neighbors import cagra

        return cagra.load(str(path))
