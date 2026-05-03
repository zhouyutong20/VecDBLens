from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from backends.base import BaseBackend
from utils.io import as_float32_c_contiguous


class FaissIVFPQBackend(BaseBackend):
    name = "faiss_ivfpq"
    index_suffix = "faiss"

    def build(self, base: np.ndarray, build_params: Dict[str, Any]):
        import faiss

        base = as_float32_c_contiguous(base)
        d = base.shape[1]
        nlist = int(build_params["nlist"])
        m = int(build_params["M"])
        nbits = int(build_params["nbits"])
        if d % m != 0:
            raise ValueError(f"FAISS IVFPQ requires embedding_dim % M == 0, got d={d}, M={m}")
        if base.shape[0] < nlist:
            raise ValueError(f"nlist={nlist} is larger than num_base={base.shape[0]}")
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        index.train(base)
        index.add(base)
        return index

    def search(self, queries: np.ndarray, topk: int, search_params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        queries = as_float32_c_contiguous(queries)
        self.index.nprobe = int(search_params["nprobe"])
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
