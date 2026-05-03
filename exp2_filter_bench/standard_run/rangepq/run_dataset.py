#!/usr/bin/env python3
"""
Range-filtered ANN benchmark: official segr + nanopq .

Data layout:
  {stem}_base.fvecs, {stem}_query.fvecs, {stem}_labels.fvecs, {stem}_filters.fvecs
Optional: {stem}_neighbors.fvecs / .ivecs

Single scalar filter (label dim = 1): no Python post-filter; C++ range is sufficient.
Multi-dim labels: use --post_filter to intersect other dimensions in Python (no ratio knob).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import resource
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nanopq  # noqa: E402
from segr import Segr  # noqa: E402


def read_fvecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    mv = memoryview(data)
    offset, vecs, d_first, n = 0, [], None, 0
    while offset < len(data):
        d = struct.unpack_from("i", mv, offset)[0]
        offset += 4
        if d_first is None:
            d_first = d
        elif d != d_first:
            raise ValueError(f"Inconsistent dims in {path}")
        need = 4 * d
        vec = struct.unpack_from("f" * d, mv, offset)
        offset += need
        vecs.append(vec)
        n += 1
    arr = np.asarray(vecs, dtype=np.float32)
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_fvecs_as_int_matrix(path: str) -> np.ndarray:
    arr = read_fvecs(path)
    return np.ascontiguousarray(np.rint(arr).astype(np.int64), dtype=np.int64)


def read_ivecs(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    mv = memoryview(data)
    offset, rows, n = 0, [], 0
    while offset < len(data):
        d = struct.unpack_from("<i", mv, offset)[0]
        offset += 4
        need = 4 * d
        vec = struct.unpack_from("<" + "i" * d, mv, offset)
        offset += need
        rows.append(vec)
        n += 1
    arr = np.asarray(rows, dtype=np.int32)
    return np.ascontiguousarray(arr, dtype=np.int32)


def resolve_data_paths(base_dir: str, stem: str) -> Dict[str, str]:
    def pack_with_prefix(dir_: str, prefix: str) -> Dict[str, str]:
        return {
            "base": os.path.join(dir_, f"{prefix}_base.fvecs"),
            "query": os.path.join(dir_, f"{prefix}_query.fvecs"),
            "labels": os.path.join(dir_, f"{prefix}_labels.fvecs"),
            "filters": os.path.join(dir_, f"{prefix}_filters.fvecs"),
            "neighbors_f": os.path.join(dir_, f"{prefix}_neighbors.fvecs"),
            "neighbors_i": os.path.join(dir_, f"{prefix}_neighbors.ivecs"),
            "dir": dir_,
        }

    def pack_generic(dir_: str) -> Dict[str, str]:
        return {
            "base": os.path.join(dir_, "base.fvecs"),
            "query": os.path.join(dir_, "query.fvecs"),
            "labels": os.path.join(dir_, "train_label.fvecs"),
            "filters": os.path.join(dir_, "test_label.fvecs"),
            "neighbors_f": os.path.join(dir_, "neighbors.fvecs"),
            "neighbors_i": os.path.join(dir_, "neighbors.ivecs"),
            "dir": dir_,
        }

    required = ("base", "query", "labels", "filters")
    candidates = [
        pack_with_prefix(base_dir, stem),
        pack_with_prefix(os.path.join(base_dir, stem), stem),
        pack_generic(base_dir),
        pack_generic(os.path.join(base_dir, stem)),
    ]

    def score(d: Dict[str, str]) -> Tuple[int, int]:
        required_hits = sum(os.path.isfile(d[k]) for k in required)
        total_hits = sum(os.path.isfile(v) for k, v in d.items() if k != "dir")
        return required_hits, total_hits

    chosen = max(candidates, key=score)
    for k in required:
        if not os.path.isfile(chosen[k]):
            raise FileNotFoundError(f"Missing required file: {chosen[k]}")
    return chosen


def _normalize_bounds_int(low: float, high: float) -> Tuple[int, int]:
    if low > high:
        low, high = high, low
    return int(math.floor(low)), int(math.ceil(high))


def _row_passes_all_dims(
    label_row: np.ndarray, query_bounds: np.ndarray, l: int
) -> bool:
    for i in range(l):
        lo_i, hi_i = _normalize_bounds_int(
            float(query_bounds[2 * i]), float(query_bounds[2 * i + 1])
        )
        v = int(label_row[i])
        if v < lo_i or v > hi_i:
            return False
    return True


def query_range(
    engine: Segr,
    q: np.ndarray,
    qf: np.ndarray,
    labels_int: np.ndarray,
    topk: int,
    primary_dim: int,
    label_min: int,
    label_max: int,
    L_eff: int,
    post_filter: bool,
    l_dim: int,
) -> List[int]:
    lo_i = int(
        math.floor(
            min(float(qf[2 * primary_dim]), float(qf[2 * primary_dim + 1]))
        )
    )
    hi_i = int(
        math.ceil(max(float(qf[2 * primary_dim]), float(qf[2 * primary_dim + 1])))
    )
    lo_i = max(lo_i, label_min)
    hi_i = min(hi_i, label_max)
    if lo_i > hi_i:
        return []

    N = int(labels_int.shape[0])
    L_use = min(int(L_eff), N)
    tk = min(int(topk), L_use, N)

    ids, _dists = engine.query(
        np.ascontiguousarray(q, dtype=np.float32),
        topk=tk,
        L=L_use,
        range_l=int(lo_i),
        range_r=int(hi_i),
        method="segr",
    )

    valid_ids: List[int] = []
    for nid in np.asarray(ids).ravel():
        ni = int(nid)
        if 0 <= ni < N:
            valid_ids.append(ni)

    if not post_filter:
        return valid_ids[: int(topk)]

    collected: List[int] = []
    for nid in valid_ids:
        if _row_passes_all_dims(labels_int[nid], qf, l_dim):
            collected.append(nid)
            if len(collected) >= int(topk):
                break
    return collected[: int(topk)]


def eval_recall_at_k(
    pred_topk: List[List[int]], truth: List[List[int]], ks: Tuple[int, ...]
) -> Dict[str, float]:
    qn = min(len(pred_topk), len(truth))
    if qn == 0:
        return {f"Recall@{k}": 0.0 for k in ks}
    out: Dict[str, float] = {}
    for k in ks:
        per: List[float] = []
        for i in range(qn):
            t = truth[i][:k]
            if not t:
                per.append(0.0)
                continue
            p = pred_topk[i][:k]
            inter = len(set(p) & set(t))
            per.append(inter / float(len(t)))
        out[f"Recall@{k}"] = sum(per) / len(per) if per else 0.0
    return out


def maybe_load_neighbors(paths: Dict[str, str], n_queries: int) -> Optional[List[List[int]]]:
    del n_queries  # reserved
    for key in ("neighbors_f", "neighbors_i"):
        p = paths.get(key, "")
        if p and os.path.isfile(p):
            if p.endswith(".fvecs"):
                arr = read_fvecs_as_int_matrix(p)
            else:
                arr = read_ivecs(p).astype(np.int64, copy=False)
            if arr.ndim == 2:
                return [list(row) for row in arr.astype(np.int64, copy=False)]
    return None


def parse_L_values(s: str) -> List[int]:
    out = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not out:
        raise SystemExit("Invalid --L_values (need comma-separated positive integers)")
    return out


def parse_recall_ks(s: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise SystemExit("Invalid --recall_ks (need comma-separated positive integers)")
    for k in vals:
        if k <= 0:
            raise SystemExit("Invalid --recall_ks: all k must be > 0")
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=os.path.expanduser("~/db/data"))
    p.add_argument("--stem", type=str, required=True)
    p.add_argument("--m", type=int, default=8, help="PQ subspaces (D must be divisible by m)")
    p.add_argument(
        "--nbits",
        type=int,
        default=None,
        help="If set, Ks = 2^nbits (e.g. nbits=8 -> Ks=256). Overrides --Ks.",
    )
    p.add_argument("--Ks", type=int, default=256)
    p.add_argument("--nlist", type=int, default=1000)
    p.add_argument("--iter", type=int, default=5, help="pqk-means iterations in reconfigure")
    p.add_argument("--cz", type=int, default=20000, help="chunk size (official reconfigure)")
    p.add_argument("--ntrain_pq", type=int, default=100_000, help="vectors used to fit nanopq")
    p.add_argument("--primary_dim", type=int, default=0)
    p.add_argument("--topk", type=int, default=100)
    p.add_argument(
        "--L_values",
        type=str,
        default="4000,8000,12000,16000,20000",
        help="Comma-separated L list; one index build, then sweep queries (efSearch analog)",
    )
    p.add_argument(
        "--post_filter",
        action="store_true",
        help="If labels have multiple dims, filter non-primary bounds in Python (default off)",
    )
    p.add_argument("--start_q", type=int, default=0)
    p.add_argument("--limit_q", type=int, default=10_000)
    p.add_argument(
        "--recall_ks",
        type=str,
        default="1,5,10,100",
        help="Comma-separated k values for recall report, e.g. 1,5,10,100",
    )
    p.add_argument("--out_dir", type=str, default="")
    p.add_argument("--summary_name", type=str, default="summary.json")
    return p.parse_args()


def _rss_mb() -> float:
    # Current resident set size in MB.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # kB -> MB
                    return float(parts[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def _status_kb(key: str) -> float:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith(key):
                    parts = line.split()
                    return float(parts[1])
    except Exception:
        pass
    return float("nan")


class _BuildMemWatcher:
    """Poll VmRSS/VmHWM during index build to get phase peak."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._running = False
        self._th: Optional[threading.Thread] = None
        self.max_rss_kb = 0.0
        self.max_hwm_kb = 0.0

    def start(self) -> None:
        self.max_rss_kb = 0.0
        self.max_hwm_kb = 0.0
        self._running = True

        def _loop() -> None:
            while self._running:
                rss = _status_kb("VmRSS:")
                hwm = _status_kb("VmHWM:")
                if not math.isnan(rss) and rss > self.max_rss_kb:
                    self.max_rss_kb = rss
                if not math.isnan(hwm) and hwm > self.max_hwm_kb:
                    self.max_hwm_kb = hwm
                time.sleep(self.interval_s)

        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()

    def stop(self) -> Tuple[float, float]:
        self._running = False
        if self._th is not None:
            self._th.join()
        return self.max_rss_kb / 1024.0, self.max_hwm_kb / 1024.0


def _peak_rss_mb() -> float:
    # Linux ru_maxrss unit is kB.
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _effective_l_estimates_for_dataset(
    primary_values: np.ndarray,
    query_filters: np.ndarray,
    primary_dim: int,
    L_list: List[int],
) -> Dict[str, Dict[str, float]]:
    """
    Estimate effective L behavior in C++:
      L' = max(int(L * selectivity), 1000)
    Here we use exact selectivity by binary search over sorted labels (1D primary dim),
    instead of C++'s random 0.1% sampling.
    """
    vals = np.sort(np.asarray(primary_values, dtype=np.int64))
    N = int(vals.size)
    if N == 0:
        return {}

    lo_arr = np.floor(
        np.minimum(
            query_filters[:, 2 * primary_dim], query_filters[:, 2 * primary_dim + 1]
        )
    ).astype(np.int64)
    hi_arr = np.ceil(
        np.maximum(
            query_filters[:, 2 * primary_dim], query_filters[:, 2 * primary_dim + 1]
        )
    ).astype(np.int64)

    # exact counts by searchsorted
    left = np.searchsorted(vals, lo_arr, side="left")
    right = np.searchsorted(vals, hi_arr, side="right")
    cnt = np.maximum(right - left, 0)
    sel = cnt.astype(np.float64) / float(N)

    out: Dict[str, Dict[str, float]] = {}
    for L in L_list:
        eff = np.maximum((L * sel).astype(np.int64), 1000)
        out[str(L)] = {
            "mean": float(np.mean(eff)),
            "p50": float(np.percentile(eff, 50)),
            "p90": float(np.percentile(eff, 90)),
            "min": float(np.min(eff)),
            "max": float(np.max(eff)),
            "floor_ratio_at_1000": float(np.mean(eff == 1000)),
        }
    return out


def main() -> None:
    args = parse_args()
    base_dir = os.path.expanduser(args.base_dir)
    paths = resolve_data_paths(base_dir, args.stem)
    L_list = parse_L_values(args.L_values)
    recall_ks = parse_recall_ks(args.recall_ks)

    t0 = time.time()
    xb = read_fvecs(paths["base"])
    xq = read_fvecs(paths["query"])
    lb = read_fvecs(paths["labels"])
    fq = read_fvecs(paths["filters"])
    assert fq.shape[1] == 2 * lb.shape[1]
    labels_int = np.rint(lb).astype(np.int32, copy=False)
    l_dim = int(labels_int.shape[1])
    t_load = time.time() - t0
    print(
        f"[load] {time.time() - t0:.3f}s  xb={xb.shape} xq={xq.shape} labels={labels_int.shape}"
    )

    if l_dim > 1 and not args.post_filter:
        print(
            "[warn] label dim > 1 but --post_filter not set; "
            "only primary_dim is enforced in C++. Add --post_filter if needed."
        )

    post_filter = bool(args.post_filter)
    if l_dim == 1:
        post_filter = False

    d = xb.shape[1]
    if d % args.m != 0:
        raise SystemExit(f"D={d} not divisible by m={args.m}")

    ks = int(args.Ks)
    if args.nbits is not None:
        ks = 2 ** int(args.nbits)
    if ks > 256:
        raise SystemExit("nanopq/segr expect Ks<=256 (uint8 codes)")

    t0 = time.time()
    pq = nanopq.PQ(M=args.m, Ks=ks, verbose=True)
    n_train = min(int(args.ntrain_pq), xb.shape[0])
    rs = np.random.RandomState(0)
    if xb.shape[0] <= n_train:
        xt = xb
    else:
        xt = xb[rs.choice(xb.shape[0], n_train, replace=False)]
    pq.fit(xt)
    t_pq_fit = time.time() - t0
    print(f"[nanopq] fit {t_pq_fit:.3f}s on {xt.shape[0]} vectors")

    rss_before_build_mb = _rss_mb()
    watcher = _BuildMemWatcher(interval_s=0.05)
    watcher.start()
    t0 = time.time()
    engine = Segr(pq)
    keys = np.arange(xb.shape[0], dtype=np.uint64)
    vals = np.ascontiguousarray(labels_int[:, args.primary_dim], dtype=np.int32)
    engine.add_configure(
        xb, keys, vals, nlist=int(args.nlist), iter=int(args.iter), cz=int(args.cz)
    )
    t_build_index = time.time() - t0
    peak_rss_build_mb, peak_hwm_build_mb = watcher.stop()
    rss_after_build_mb = _rss_mb()
    peak_rss_ru_maxrss_mb = _peak_rss_mb()
    build_mem_delta_mb = (
        rss_after_build_mb - rss_before_build_mb
        if not (math.isnan(rss_before_build_mb) or math.isnan(rss_after_build_mb))
        else float("nan")
    )
    print(
        f"[index] add_configure+reconfigure {t_build_index:.3f}s  "
        f"nlist={engine.nlist} N={engine.N}"
    )

    label_min = int(vals.min())
    label_max = int(vals.max())

    lo = max(0, int(args.start_q))
    hi = min(xq.shape[0], lo + int(args.limit_q))
    truth = maybe_load_neighbors(paths, xq.shape[0])

    summary: Dict[str, Any] = {
        "stem": args.stem,
        "N": int(xb.shape[0]),
        "Q": hi - lo,
        "m": args.m,
        "Ks": ks,
        "nbits": int(args.nbits) if args.nbits is not None else None,
        "nlist": int(args.nlist),
        "cz": int(args.cz),
        "iter": int(args.iter),
        "topk": int(args.topk),
        "L_values": L_list,
        "recall_ks": list(recall_ks),
        "post_filter": post_filter,
        "timing_s": {
            "load_data": float(t_load),
            "fit_pq": float(t_pq_fit),
            "build_index": float(t_build_index),
        },
        "memory_mb": {
            "rss_before_build": float(rss_before_build_mb),
            "rss_after_build": float(rss_after_build_mb),
            "rss_delta_build": float(build_mem_delta_mb),
            # Phase peak sampled during build (aligned with ACORN-style reporting).
            "peak_rss_build": float(peak_rss_build_mb),
            "peak_hwm_build": float(peak_hwm_build_mb),
            # Keep legacy field for backward compatibility.
            "peak_rss": float(peak_rss_ru_maxrss_mb),
            "peak_rss_ru_maxrss": float(peak_rss_ru_maxrss_mb),
        },
        "effective_L_estimate": _effective_l_estimates_for_dataset(
            primary_values=vals,
            query_filters=fq[lo:hi],
            primary_dim=int(args.primary_dim),
            L_list=L_list,
        ),
        "per_L": {},
    }

    eval_ks = sorted(set(int(x) for x in recall_ks if int(x) > 0))
    eval_ks = [kk for kk in eval_ks if kk <= int(args.topk)]
    if not eval_ks:
        eval_ks = [int(args.topk)]

    for L_eff in L_list:
        if int(args.topk) > int(L_eff):
            print(
                f"[warn] L={L_eff} < topk={args.topk}: official query needs topk<=L; "
                f"using topk_call=min(topk,L)={min(int(args.topk), int(L_eff))}"
            )

        Q = hi - lo
        block: Dict[str, Any] = {"per_k": {}}
        topk_eval = int(args.topk)
        bench_times: List[float] = []
        t_bench0 = time.time()
        preds_topk: List[List[int]] = []

        for qi in range(lo, hi):
            if (qi - lo) % 1000 == 0 and qi > lo:
                print(f"[bench] L={L_eff} query {qi - lo}/{hi - lo}")

            q = np.ascontiguousarray(xq[qi], dtype=np.float32)
            qfv = np.ascontiguousarray(fq[qi], dtype=np.float32)

            t1 = time.time()
            ids_list = query_range(
                engine,
                q,
                qfv,
                labels_int,
                topk=topk_eval,
                primary_dim=int(args.primary_dim),
                label_min=label_min,
                label_max=label_max,
                L_eff=int(L_eff),
                post_filter=post_filter,
                l_dim=l_dim,
            )
            bench_times.append(time.time() - t1)
            preds_topk.append(ids_list)

        dt = time.time() - t_bench0
        arr = np.asarray(bench_times, dtype=np.float64)
        qps_val = float(Q / dt if dt > 0 else 0.0)

        for k_eval in eval_ks:
            per_k_block: Dict[str, Any] = {
                "avg_s": float(arr.mean()),
                "p50_s": float(np.percentile(arr, 50)),
                "p90_s": float(np.percentile(arr, 90)),
            }
            if int(k_eval) == topk_eval:
                per_k_block["QPS@100"] = qps_val

            if truth is not None:
                preds_at_k = [row[: int(k_eval)] for row in preds_topk]
                rec_single = eval_recall_at_k(preds_at_k, truth[lo:hi], (int(k_eval),))
                per_k_block["recall"] = rec_single
                print(
                    f"[bench] L={L_eff} k={k_eval} Q={Q} "
                    f"recall={rec_single}"
                )
            else:
                print(
                    f"[bench] L={L_eff} k={k_eval} Q={Q} recall=skipped"
                )

            block["per_k"][str(k_eval)] = per_k_block

        print(
            f"[bench] L={L_eff} k={topk_eval} Q={Q} total={dt:.3f}s "
            f"avg={arr.mean():.6f}s p50={np.percentile(arr, 50):.6f}s QPS@100={qps_val:.2f}"
        )

        # Backward-compatible top-level fields: use k=topk if available, else largest evaluated k.
        k_legacy = str(int(args.topk)) if str(int(args.topk)) in block["per_k"] else str(max(eval_ks))
        legacy = block["per_k"][k_legacy]
        block["avg_s"] = legacy["avg_s"]
        block["p50_s"] = legacy["p50_s"]
        block["p90_s"] = legacy["p90_s"]
        block["QPS@100"] = legacy["QPS@100"]
        if truth is not None:
            rec_map: Dict[str, float] = {}
            for kk in eval_ks:
                kk_s = str(kk)
                rec_k = block["per_k"][kk_s].get("recall", {})
                rec_map[f"Recall@{kk}"] = float(rec_k.get(f"Recall@{kk}", 0.0))
            block["recall"] = rec_map

        summary["per_L"][str(L_eff)] = block

    if not truth:
        print("[recall] no neighbors file; skipped")

    if args.out_dir:
        out = Path(os.path.expanduser(args.out_dir))
        out.mkdir(parents=True, exist_ok=True)
        with open(out / args.summary_name, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[out] wrote {out / args.summary_name}")


if __name__ == "__main__":
    main()
