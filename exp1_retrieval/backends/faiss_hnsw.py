from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from backends.base import BaseBackend
from utils.io import as_float32_c_contiguous


class FaissHNSWBackend(BaseBackend):
    name = "faiss_hnsw"
    index_suffix = "faiss"

    def build(self, base: np.ndarray, build_params: Dict[str, Any]):
        import faiss

        base = as_float32_c_contiguous(base)
        m = int(build_params["M"])
        ef_construction = int(build_params["efConstruction"])
        index = faiss.IndexHNSWFlat(base.shape[1], m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(base)
        return index

    def search(self, queries: np.ndarray, topk: int, search_params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        queries = as_float32_c_contiguous(queries)
        self.index.hnsw.efSearch = int(search_params["efSearch"])
        start = time.perf_counter()
        _, neighbors = self.index.search(queries, topk)
        search_time = time.perf_counter() - start
        return neighbors.astype(np.int64, copy=False), search_time

    def save_index(self, path: Path) -> None:
        import faiss

        faiss.write_index(self.index, str(path))

    def load_index(self, path: Path):
        import faiss

        return faiss.read_index(str(path))
