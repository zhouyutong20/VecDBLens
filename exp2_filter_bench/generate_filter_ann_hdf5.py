# -*- coding: utf-8 -*-
# Standalone filtered ANN HDF5 generator.
# This script does not depend on private project modules.

import argparse
import json
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def metric_mapping(metric_name: str) -> str:
    metric = metric_name.lower()
    if metric == "euclidean":
        return "euclidean"
    if metric == "angular":
        return "cosine"
    raise ValueError(f"Unsupported metric: {metric_name}")


def ratio_tag(ratio: float) -> str:
    return f"{ratio:.0e}".replace("+", "")


def sample_ratio_beta(
    mean_ratio: float,
    min_ratio: float,
    max_ratio: float,
    shape: float,
    rng: np.random.Generator,
) -> float:
    eps = 1e-12
    if not (0.0 < min_ratio < max_ratio <= 1.0):
        raise ValueError("min_ratio/max_ratio must satisfy 0 < min_ratio < max_ratio <= 1")
    if shape <= 0:
        raise ValueError("beta_shape must be > 0")

    mean_ratio = float(np.clip(mean_ratio, min_ratio + eps, max_ratio - eps))
    mean01 = (mean_ratio - min_ratio) / (max_ratio - min_ratio)
    mean01 = float(np.clip(mean01, eps, 1.0 - eps))

    alpha = float(shape)
    beta = alpha * (1.0 - mean01) / mean01

    x = rng.gamma(shape=alpha, scale=1.0)
    y = rng.gamma(shape=beta, scale=1.0)
    return float(min_ratio + (max_ratio - min_ratio) * (x / (x + y)))


def dynamic_range_for_ratio(ratio: float) -> Dict[str, float]:
    if ratio >= 1.0:
        return {
            "mean_ratio": 1.0,
            "min_ratio": 0.8,
            "max_ratio": 1.0,
            "beta_shape": 2.0,
        }

    low = max(1e-6, ratio * 0.5)
    high = min(1.0, ratio * 1.5)
    if high <= low:
        high = min(1.0, low + 1e-6)

    return {
        "mean_ratio": ratio,
        "min_ratio": low,
        "max_ratio": high,
        "beta_shape": 2.0,
    }


def generate_test_ranges(
    n_queries: int,
    n_filters: int,
    max_labels: int,
    ratio_request: float,
    mode: str,
    mean_ratio: float,
    min_ratio: float,
    max_ratio: float,
    beta_shape: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_filter_dims = int(n_filters)

    if n_filter_dims <= 0:
        raise ValueError("n_filters must be >= 1")

    total_span = float(max_labels + 1)
    out = np.zeros((n_queries, n_filter_dims, 2), dtype=np.int32)

    for i in tqdm(range(n_queries), desc=f"ranges-{mode}"):
        if mode == "dynamic_beta":
            ratio = sample_ratio_beta(mean_ratio, min_ratio, max_ratio, beta_shape, rng)
        elif mode == "static":
            ratio = ratio_request
        else:
            raise ValueError("mode must be static or dynamic_beta")

        per_dim_ratio = ratio if n_filter_dims == 1 else ratio ** (1.0 / n_filter_dims)
        sub_len = max(1.0, per_dim_ratio * (total_span - 1))

        for dim_id in range(n_filter_dims):
            start_max = max_labels - sub_len
            if start_max <= 0:
                left = 0.0
            else:
                left = float(rng.uniform(0, start_max))

            right = left + sub_len
            left_i = int(max(0, min(max_labels, left)))
            right_i = int(max(left_i + 1, min(max_labels, right)))

            out[i, dim_id, 0] = left_i
            out[i, dim_id, 1] = right_i

    return out


def exact_filtered_knn(
    distance: str,
    train_vec: np.ndarray,
    test_vec: np.ndarray,
    train_label: np.ndarray,
    test_label: np.ndarray,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbors_out = np.full((len(test_vec), topk), -1, dtype=np.int32)
    distances_out = np.full((len(test_vec), topk), -1.0, dtype=np.float32)
    sklearn_metric = metric_mapping(distance)

    for query_id, query_vec in tqdm(enumerate(test_vec), total=len(test_vec), desc="filtered-knn"):
        flags = np.ones((train_label.shape[0],), dtype=bool)

        for label_id in range(test_label.shape[1]):
            left = test_label[query_id, label_id, 0]
            right = test_label[query_id, label_id, 1]
            flags &= (left <= train_label[:, label_id]) & (train_label[:, label_id] <= right)

        candidate_ids = np.where(flags)[0]
        if candidate_ids.size == 0:
            continue

        candidate_vec = train_vec[candidate_ids]
        n_neighbors = min(topk, candidate_vec.shape[0])

        nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=sklearn_metric,
            n_jobs=-1,
            algorithm="brute",
        )
        nn.fit(candidate_vec)

        distances, indices = nn.kneighbors(np.array([query_vec]))
        neighbors_out[query_id, :n_neighbors] = candidate_ids[indices[0]]
        distances_out[query_id, :n_neighbors] = distances[0].astype(np.float32)

    return neighbors_out, distances_out


def build_filter_expr_func(n_filters: int) -> str:
    if n_filters > 1:
        args = ",".join([f"label_l_{i}, label_r_{i}" for i in range(n_filters)])
        clauses = ", ".join(
            [
                f"f\"label_{i} >= {{label_l_{i}}} and label_{i} <= {{label_r_{i}}}\""
                for i in range(n_filters)
            ]
        )
        return (
            "def filter_expr("
            + args
            + "):\n    return \" and \".join(["
            + clauses
            + "])"
        )

    return (
        "def filter_expr(label_l_0, label_r_0):\n"
        "    return f\"label_0 >= {label_l_0} and label_0 <= {label_r_0}\""
    )


def create_filter_dataset(cfg: Dict[str, object]) -> None:
    input_path = str(cfg["input_path"])
    output_path = str(cfg["output_path"])

    n_dims = int(cfg["dim"])
    n_samples = int(cfg["n_samples"])
    n_test = int(cfg["n_test"])
    n_filters = int(cfg["n_filters"])
    max_labels = int(cfg["max_labels"])
    ratio_request = float(cfg["ratio_request"])
    distance = str(cfg["distance"])
    topk = int(cfg["topk"])
    mode = str(cfg["mode"])

    vector_seed = int(cfg["vector_seed"])
    label_seed = int(cfg["label_seed"])
    range_seed = int(cfg["range_seed"])

    mean_ratio = float(cfg.get("mean_ratio", 0.03))
    min_ratio = float(cfg.get("min_ratio", 0.001))
    max_ratio = float(cfg.get("max_ratio", 0.25))
    beta_shape = float(cfg.get("beta_shape", 2.0))

    with h5py.File(input_path, "r") as source_file:
        train_dset = source_file["train"]
        test_dset = source_file["test"]

        if train_dset.shape[1] != n_dims:
            raise ValueError(f"train dim mismatch: {train_dset.shape[1]} vs {n_dims}")
        if test_dset.shape[1] != n_dims:
            raise ValueError(f"test dim mismatch: {test_dset.shape[1]} vs {n_dims}")

        rng = np.random.default_rng(vector_seed)
        target_train = min(n_samples, int(train_dset.shape[0]))
        target_test = min(n_test, int(test_dset.shape[0]))

        train_idx = np.sort(rng.choice(train_dset.shape[0], size=target_train, replace=False))
        test_idx = np.sort(rng.choice(test_dset.shape[0], size=target_test, replace=False))

        train_vec = train_dset[train_idx].astype(np.float32)
        test_vec = test_dset[test_idx].astype(np.float32)

    rng_label = np.random.default_rng(label_seed)
    label_cols = []
    for _ in range(n_filters):
        label_cols.append(
            rng_label.integers(
                1,
                max_labels,
                size=train_vec.shape[0],
                dtype=np.int32,
            )
        )
    train_label = np.stack(label_cols, axis=1)

    test_label = generate_test_ranges(
        n_queries=test_vec.shape[0],
        n_filters=n_filters,
        max_labels=max_labels,
        ratio_request=ratio_request,
        mode=mode,
        mean_ratio=mean_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        beta_shape=beta_shape,
        seed=range_seed,
    )

    neighbors, distances = exact_filtered_knn(
        distance=distance,
        train_vec=train_vec,
        test_vec=test_vec,
        train_label=train_label,
        test_label=test_label,
        topk=topk,
    )

    train_label_names = [f"label_{i}" for i in range(n_filters)]
    train_label_types = ["int32" for _ in range(n_filters)]
    test_label_range_names = [[f"label_l_{i}", f"label_r_{i}"] for i in range(n_filters)]
    test_label_range_types = [["int32", "int32"] for _ in range(n_filters)]

    output_file = Path(output_path).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(output_file), "w") as out_file:
        out_file.attrs["type"] = "filter-ann"
        out_file.attrs["source_dataset"] = str(cfg["dataset_name"])
        out_file.attrs["distance"] = distance
        out_file.attrs["dimension"] = n_dims
        out_file.attrs["point_type"] = "float"
        out_file.attrs["label_names"] = train_label_names
        out_file.attrs["label_types"] = train_label_types
        out_file.attrs["label_ranges"] = test_label_range_names
        out_file.attrs["label_range_types"] = test_label_range_types
        out_file.attrs["filter_expr_func"] = build_filter_expr_func(n_filters)
        out_file.attrs["range_mode"] = mode
        out_file.attrs["range_ratio_request"] = ratio_request
        out_file.attrs["range_mean_ratio"] = mean_ratio
        out_file.attrs["range_min_ratio"] = min_ratio
        out_file.attrs["range_max_ratio"] = max_ratio
        out_file.attrs["range_beta_shape"] = beta_shape
        out_file.attrs["vector_seed"] = vector_seed
        out_file.attrs["label_seed"] = label_seed
        out_file.attrs["range_seed"] = range_seed

        out_file.create_dataset("train_vec", data=train_vec, dtype=np.float32)
        out_file.create_dataset("test_vec", data=test_vec, dtype=np.float32)
        out_file.create_dataset("train_label", data=train_label, dtype=np.int32)
        out_file.create_dataset("test_label", data=test_label, dtype=np.int32)
        out_file.create_dataset("neighbors", data=neighbors, dtype=np.int32)
        out_file.create_dataset("distances", data=distances, dtype=np.float32)


def build_configs(args: argparse.Namespace) -> List[Dict[str, object]]:
    modes = [args.mode]
    if args.mode == "both":
        modes = ["static", "dynamic_beta"]

    configs: List[Dict[str, object]] = []

    for ratio in args.ratios:
        for mode in modes:
            tag = ratio_tag(float(ratio))
            desc = f"{args.dataset_name}_n{args.n_samples}_f{args.n_filters}_r{tag}_{mode}"
            output_path = Path(args.output_root) / f"{desc}.hdf5"

            cfg: Dict[str, object] = {
                "desc": desc,
                "input_path": args.input,
                "output_path": str(output_path),
                "dataset_name": args.dataset_name,
                "dim": args.dim,
                "n_samples": args.n_samples,
                "n_test": args.n_test,
                "n_filters": args.n_filters,
                "max_labels": args.max_labels,
                "ratio_request": float(ratio),
                "distance": args.distance,
                "topk": args.topk,
                "mode": mode,
                "vector_seed": args.vector_seed,
                "label_seed": args.label_seed,
                "range_seed": args.range_seed,
            }

            if mode == "dynamic_beta":
                cfg.update(dynamic_range_for_ratio(float(ratio)))
            else:
                cfg.update(
                    {
                        "mean_ratio": float(ratio),
                        "min_ratio": float(ratio),
                        "max_ratio": float(ratio),
                        "beta_shape": 2.0,
                    }
                )

            configs.append(cfg)

    return configs


def run_one(cfg: Dict[str, object], log_dir: Path) -> str:
    desc = str(cfg.get("desc", "job"))
    mode = str(cfg.get("mode", "unknown"))
    log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"{now_ts()}_{desc}.log"

    summary_keys = [
        "desc",
        "input_path",
        "output_path",
        "dataset_name",
        "dim",
        "n_samples",
        "n_test",
        "n_filters",
        "max_labels",
        "ratio_request",
        "distance",
        "topk",
        "mode",
        "mean_ratio",
        "min_ratio",
        "max_ratio",
        "beta_shape",
        "vector_seed",
        "label_seed",
        "range_seed",
    ]

    summary = {key: cfg[key] for key in summary_keys if key in cfg}

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)

        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("=" * 80)
            print(f"[start] {desc} ({mode})")
            print("[config]\n" + json.dumps(summary, ensure_ascii=False, indent=2))
            start_time = time.time()

            try:
                create_filter_dataset(cfg)
                print(f"[ok] elapsed={time.time() - start_time:.2f}s")
            except Exception as err:
                print(f"[error] {err}")
                traceback.print_exc()

            print(f"[log] {log_path}")
            print("=" * 80)

    return str(log_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate filtered ANN HDF5 files from public ANN benchmark HDF5 files."
    )

    parser.add_argument("--input", required=True, help="Path to the source HDF5 file.")
    parser.add_argument("--output-root", required=True, help="Directory for generated HDF5 files.")
    parser.add_argument("--dataset-name", required=True, help="Dataset name used in output filenames.")
    parser.add_argument("--dim", type=int, required=True, help="Vector dimensionality.")
    parser.add_argument("--n-samples", type=int, default=990000, help="Number of base vectors to sample.")
    parser.add_argument("--n-test", type=int, required=True, help="Number of query vectors to sample.")
    parser.add_argument("--n-filters", type=int, default=1, help="Number of synthetic filter dimensions.")
    parser.add_argument("--max-labels", type=int, default=10000, help="Maximum synthetic label value.")
    parser.add_argument("--distance", default="euclidean", choices=["euclidean", "angular"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[1e-4, 1e-3, 1e-2, 1e-1, 5e-1],
        help="Target filter ratios.",
    )
    parser.add_argument(
        "--mode",
        default="static",
        choices=["static", "dynamic_beta", "both"],
        help="Filter range generation mode.",
    )
    parser.add_argument("--vector-seed", type=int, default=42)
    parser.add_argument("--label-seed", type=int, default=123)
    parser.add_argument("--range-seed", type=int, default=42)
    parser.add_argument("--log-dir", default="./logs_filter_generation")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_configs(args)
    log_dir = Path(args.log_dir)

    print(f"[info] jobs={len(configs)}")
    print(f"[info] output_root={args.output_root}")
    print(f"[info] log_dir={log_dir.resolve()}")

    all_logs: List[str] = []

    for idx, cfg in enumerate(configs, 1):
        print(f"\n>>> ({idx}/{len(configs)}) {cfg.get('desc')} ({cfg.get('mode')})")
        all_logs.append(run_one(cfg, log_dir))

    print("\n[done] all jobs finished")
    for log_path in all_logs:
        print(" -", log_path)


if __name__ == "__main__":
    main()