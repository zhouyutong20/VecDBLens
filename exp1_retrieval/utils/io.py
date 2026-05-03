from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def load_embeddings(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings from .npy/.npz/.pt or an embedding shard directory."""
    path = Path(path)
    if path.is_dir():
        return _load_sharded_dir(path)
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode=None)
        ids = np.arange(arr.shape[0], dtype=np.int64)
        return arr, ids
    if path.suffix == ".npz":
        data = np.load(path)
        key = "embeddings" if "embeddings" in data else "vectors" if "vectors" in data else data.files[0]
        arr = data[key]
        ids = data["ids"].astype(np.int64) if "ids" in data else np.arange(arr.shape[0], dtype=np.int64)
        return arr, ids
    if path.suffix == ".pt":
        import torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            arr = obj.get("embeddings", obj.get("vectors", obj.get("x")))
            ids = obj.get("ids")
        else:
            arr, ids = obj, None
        if arr is None:
            raise ValueError(f"Cannot find embeddings tensor in {path}")
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        ids = ids.detach().cpu().numpy() if hasattr(ids, "detach") else ids
        if ids is None:
            ids = np.arange(arr.shape[0], dtype=np.int64)
        return np.asarray(arr), np.asarray(ids, dtype=np.int64)
    raise ValueError(f"Unsupported embedding path: {path}")


def _load_sharded_dir(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    shards = sorted(path.glob("shard_*.npy"))
    shards = [p for p in shards if not p.name.endswith(".ids.npy") and not p.name.endswith(".tmp")]
    if not shards:
        raise FileNotFoundError(f"No shard_*.npy found in {path}")

    arrays: List[np.ndarray] = []
    ids: List[np.ndarray] = []
    for shard in shards:
        stem = shard.name[:-4]
        ids_path = path / f"{stem}.ids.npy"
        arr = np.load(shard, mmap_mode=None)
        shard_ids = np.load(ids_path, mmap_mode=None).astype(np.int64) if ids_path.exists() else None
        if shard_ids is None:
            start = sum(a.shape[0] for a in arrays)
            shard_ids = np.arange(start, start + arr.shape[0], dtype=np.int64)
        arrays.append(arr)
        ids.append(shard_ids)

    return np.concatenate(arrays, axis=0), np.concatenate(ids, axis=0)


def as_float32_c_contiguous(x: np.ndarray) -> np.ndarray:
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def load_qrels(path: str | Path) -> Dict[int, set[int]]:
    qrels: Dict[int, set[int]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0].lower() in {"qid", "query-id", "query_id"}:
                continue
            if len(parts) >= 4:
                qid, did, rel = parts[0], parts[2], parts[3]
            elif len(parts) >= 3:
                qid, did, rel = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                qid, did, rel = parts[0], parts[1], "1"
            else:
                continue
            try:
                rel_val = float(rel)
            except ValueError:
                rel_val = 1.0
            if rel_val <= 0:
                continue
            qrels.setdefault(int(qid), set()).add(int(did))
    return qrels


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()}
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def read_completed_keys(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add(row.get("run_key", ""))
    return keys
