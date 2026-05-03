from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def compute_recall_at_k(
    neighbor_indices: np.ndarray,
    query_ids: np.ndarray,
    base_ids: np.ndarray,
    qrels: Dict[int, set[int]],
    ks: Iterable[int] = (1, 5, 10),
) -> Dict[str, float]:
    max_k = int(max(ks))
    neighbor_indices = neighbor_indices[:, :max_k]
    retrieved_ids = base_ids[neighbor_indices]

    valid = 0
    hits = {int(k): 0 for k in ks}
    for row_idx, qid in enumerate(query_ids):
        rel = qrels.get(int(qid))
        if not rel:
            continue
        valid += 1
        for k in hits:
            if any(int(did) in rel for did in retrieved_ids[row_idx, :k]):
                hits[k] += 1
    if valid == 0:
        raise ValueError("No query ids overlap with qrels; cannot compute recall.")
    return {f"recall@{k}": hits[k] / valid for k in hits}
