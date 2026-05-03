#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge worker shard directories into one shard directory.")
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def shard_indices(worker_dir: Path) -> list[int]:
    out = []
    for path in worker_dir.glob("shard_*.npy"):
        stem = path.stem
        if stem.endswith(".ids"):
            continue
        try:
            out.append(int(stem.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def copy_file(src: Path, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def main() -> None:
    args = parse_args()
    workers = sorted(p for p in args.workers_root.glob("worker_*") if p.is_dir())
    if not workers:
        raise FileNotFoundError(f"No worker_* directories under {args.workers_root}")

    if args.output_dir.exists():
        existing = list(args.output_dir.glob("shard_*.npy"))
        if existing and not args.force:
            raise FileExistsError(f"{args.output_dir} already has shards. Use --force to overwrite.")
        if args.force:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    next_idx = 0
    total = 0
    sources = []
    for worker in workers:
        for shard_idx in shard_indices(worker):
            src_vec = worker / f"shard_{shard_idx:05d}.npy"
            src_ids = worker / f"shard_{shard_idx:05d}.ids.npy"
            src_meta = worker / f"shard_{shard_idx:05d}.meta.json"
            if not (src_vec.exists() and src_ids.exists() and src_meta.exists()):
                raise FileNotFoundError(f"Incomplete shard {shard_idx} in {worker}")
            dst_vec = args.output_dir / f"shard_{next_idx:05d}.npy"
            dst_ids = args.output_dir / f"shard_{next_idx:05d}.ids.npy"
            dst_meta = args.output_dir / f"shard_{next_idx:05d}.meta.json"
            copy_file(src_vec, dst_vec)
            copy_file(src_ids, dst_ids)
            meta = json.loads(src_meta.read_text(encoding="utf-8"))
            meta["merged_from_worker"] = worker.name
            meta["source_shard_idx"] = shard_idx
            meta["shard_idx"] = next_idx
            dst_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            total += int(meta.get("count", 0))
            sources.append(str(src_vec))
            next_idx += 1

    summary = {
        "workers_root": str(args.workers_root),
        "output_dir": str(args.output_dir),
        "num_shards": next_idx,
        "num_vectors": total,
        "sources": sources,
    }
    (args.output_dir / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
