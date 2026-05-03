from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


class BaseBackend:
    name = "base"
    index_suffix = "index"

    def __init__(self, metric: str = "inner_product", gpu_id: Optional[int] = None, search_batch_size: int = 4096) -> None:
        self.metric = metric
        self.gpu_id = gpu_id
        self.search_batch_size = search_batch_size
        self.index = None
        self.index_size_bytes = 0

    def build_or_load(
        self,
        base: np.ndarray,
        build_params: Dict[str, Any],
        cache_path: Optional[Path] = None,
        reuse_index: bool = True,
    ) -> Tuple[float, int]:
        if reuse_index and cache_path is not None and cache_path.exists():
            start = time.perf_counter()
            self.index = self.load_index(cache_path)
            build_time = time.perf_counter() - start
            self.index_size_bytes = cache_path.stat().st_size
            return build_time, self.index_size_bytes

        start = time.perf_counter()
        self.index = self.build(base, build_params)
        build_time = time.perf_counter() - start

        if cache_path is not None:
            self.save_index(cache_path)
            if cache_path.exists():
                self.index_size_bytes = cache_path.stat().st_size
        if not self.index_size_bytes:
            self.index_size_bytes = self.estimate_index_size(base)
        return build_time, self.index_size_bytes

    def build(self, base: np.ndarray, build_params: Dict[str, Any]):
        raise NotImplementedError

    def search(self, queries: np.ndarray, topk: int, search_params: Dict[str, Any]) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def save_index(self, path: Path) -> None:
        raise NotImplementedError(f"{self.name} does not support index serialization")

    def load_index(self, path: Path):
        raise NotImplementedError(f"{self.name} does not support index loading")

    def estimate_index_size(self, base: np.ndarray) -> int:
        return int(base.nbytes)
