#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np


def read_fvecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0])
    data = raw.reshape(-1, dim + 1)[:, 1:].view(np.float32)
    return np.ascontiguousarray(data)


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    data = raw.reshape(-1, dim + 1)[:, 1:]
    return np.ascontiguousarray(data)


def read_neighbors_any(ds_dir: Path, stem: str) -> np.ndarray:
    cand = [
        ds_dir / f"{stem}_neighbors.ivecs",
        ds_dir / "neighbors.ivecs",
    ]
    for p in cand:
        if p.is_file():
            return read_ivecs(p)

    fvec_cand = [
        ds_dir / f"{stem}_neighbors.fvecs",
        ds_dir / "neighbors.fvecs",
    ]
    for p in fvec_cand:
        if p.is_file():
            return read_fvecs(p).astype(np.int32)

    raise FileNotFoundError(f"missing neighbors.ivecs/fvecs under {ds_dir}")


def dataset_paths(root: Path, stem: str):
    d = root / stem
    pref = {
        "base": d / f"{stem}_base.fvecs",
        "query": d / f"{stem}_query.fvecs",
        "labels": d / f"{stem}_labels.fvecs",
        "filters": d / f"{stem}_filters.fvecs",
        "neighbors": d / f"{stem}_neighbors.ivecs",
    }
    short = {
        "base": d / "base.fvecs",
        "query": d / "query.fvecs",
        "labels": d / "train_label.fvecs",
        "filters": d / "test_label.fvecs",
        "neighbors": d / "neighbors.ivecs",
    }
    out = {}
    for k in ("base", "query", "labels", "filters", "neighbors"):
        out[k] = pref[k] if pref[k].is_file() else short[k]
    return out


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    qn = min(pred.shape[0], gt.shape[0])
    if qn == 0:
        return 0.0
    vals = []
    for i in range(qn):
        p = [int(x) for x in pred[i, :k].tolist() if int(x) >= 0]
        t = [int(x) for x in gt[i, :k].tolist() if int(x) >= 0]
        if not t:
            vals.append(0.0)
            continue
        vals.append(len(set(p) & set(t)) / float(len(t)))
    return float(sum(vals) / len(vals))


def run_one(stem: str, root: Path, topk: int, limit_q: int, out_dir: Path):
    paths = dataset_paths(root, stem)
    for k, p in paths.items():
        if k == "neighbors":
            continue
        if not p.is_file():
            raise FileNotFoundError(f"missing file: {p}")

    xb = read_fvecs(paths["base"]).astype(np.float32, copy=False)
    xq = read_fvecs(paths["query"]).astype(np.float32, copy=False)
    labels = read_fvecs(paths["labels"]).astype(np.float32, copy=False)
    filters = read_fvecs(paths["filters"]).astype(np.float32, copy=False)
    gt = read_neighbors_any(root / stem, stem).astype(np.int32, copy=False)

    if labels.ndim != 2 or labels.shape[1] < 1:
        raise ValueError(f"labels shape unexpected: {labels.shape}")
    label_1d = labels[:, 0]

    qn = min(int(limit_q), xq.shape[0], filters.shape[0], gt.shape[0])
    pred = np.full((qn, topk), -1, dtype=np.int32)
    cand_counts = np.zeros(qn, dtype=np.int64)

    t0 = time.time()
    for i in range(qn):
        lo = float(filters[i, 0])
        hi = float(filters[i, 1])
        cand = np.where((label_1d >= lo) & (label_1d <= hi))[0]
        cand_counts[i] = cand.size
        if cand.size == 0:
            continue
        vecs = xb[cand]
        diff = vecs - xq[i]
        dists = np.einsum("ij,ij->i", diff, diff)
        kk = min(topk, cand.size)
        top = np.argpartition(dists, kk - 1)[:kk]
        top = top[np.argsort(dists[top])]
        pred[i, :kk] = cand[top]
    t1 = time.time()

    total_s = max(1e-12, t1 - t0)
    qps = float(qn / total_s)
    r1 = recall_at_k(pred, gt[:qn], 1)
    r5 = recall_at_k(pred, gt[:qn], 5)
    r10 = recall_at_k(pred, gt[:qn], 10)
    r100 = recall_at_k(pred, gt[:qn], min(100, topk))

    result = {
        "algo": "flat_prefilter_exact_l2",
        "stem": stem,
        "topk": topk,
        "limit_q": qn,
        "time_s": total_s,
        "qps": qps,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "recall@100": r100,
        "avg_candidates": float(cand_counts.mean()) if qn > 0 else 0.0,
        "p50_candidates": float(np.percentile(cand_counts, 50)) if qn > 0 else 0.0,
        "p90_candidates": float(np.percentile(cand_counts, 90)) if qn > 0 else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}_flat_prefilter_exact.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main():
    ap = argparse.ArgumentParser(description="Single-thread flat prefilter exact-L2 benchmark.")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--dataset_prefix", default="vec960")
    ap.add_argument("--size_tag", default="n990k")
    ap.add_argument("--ratios", default="0p01pct,0p1pct,1pct,10pct,50pct")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--limit_q", type=int, default=10000)
    ap.add_argument("--out_root", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    out_root = Path(args.out_root)
    ratios = [x.strip() for x in args.ratios.split(",") if x.strip()]
    stems = [f"{args.dataset_prefix}_{args.size_tag}_f1_r{r}_static" for r in ratios]

    rows = []
    for stem in stems:
        print(f"[run] {stem}")
        row = run_one(stem, root, args.topk, args.limit_q, out_root)
        rows.append(row)
        print(
            f"[done] {stem} qps={row['qps']:.2f} "
            f"R@1/5/10/100={row['recall@1']:.4f}/{row['recall@5']:.4f}/{row['recall@10']:.4f}/{row['recall@100']:.4f}"
        )

    csv_path = out_root / "flat_prefilter_exact_summary.csv"
    cols = [
        "algo",
        "stem",
        "topk",
        "limit_q",
        "time_s",
        "qps",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@100",
        "avg_candidates",
        "p50_candidates",
        "p90_candidates",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"[done] summary csv: {csv_path}")


if __name__ == "__main__":
    main()
