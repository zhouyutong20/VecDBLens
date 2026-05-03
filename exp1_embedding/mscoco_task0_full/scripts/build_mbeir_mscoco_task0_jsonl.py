#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_MBEIR_ROOT = Path("/data1/lnyan/projects/GENIUS-CVPR25/M-BEIR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full M-BEIR MSCOCO task0 text-to-image query/corpus/qrels files."
    )
    parser.add_argument("--mbeir-root", type=Path, default=DEFAULT_MBEIR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("/data3/hyhe/mscoco_task0_full"))
    parser.add_argument("--strict-images", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def default_query_files(root: Path) -> List[tuple[str, Path]]:
    return [
        ("train", root / "query/train/mbeir_mscoco_train.jsonl"),
        ("val", root / "query/val/mbeir_mscoco_task0_val.jsonl"),
        ("test", root / "query/test/mbeir_mscoco_task0_test.jsonl"),
    ]


def default_cand_files(root: Path) -> List[tuple[str, Path]]:
    return [
        ("train", root / "cand_pool/local/mbeir_mscoco_train_cand_pool.jsonl"),
        ("val", root / "cand_pool/local/mbeir_mscoco_task0_val_cand_pool.jsonl"),
        ("test", root / "cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl"),
    ]


def main() -> None:
    args = parse_args()
    root = args.mbeir_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    corpus_path = out / "corpus_images.jsonl"
    queries_path = out / "queries_caption.jsonl"
    qrels_path = out / "qrels.tsv"
    mapping_path = out / "id_mappings.json"
    stats_path = out / "stats.json"

    source_did_to_int: Dict[str, int] = {}
    source_qid_to_int: Dict[str, int] = {}
    missing_images: List[str] = []
    split_counts: Dict[str, Dict[str, int]] = {}

    next_did = 0
    with corpus_path.open("w", encoding="utf-8") as fc:
        for split, cand_file in default_cand_files(root):
            if not cand_file.exists():
                raise FileNotFoundError(cand_file)
            split_stats = split_counts.setdefault(split, {"corpus": 0, "queries": 0, "qrels": 0})
            for obj in read_jsonl(cand_file):
                if obj.get("modality") != "image":
                    continue
                source_did = str(obj["did"])
                if source_did in source_did_to_int:
                    continue
                image_path = str(obj["img_path"])
                abs_image_path = root / image_path
                if not abs_image_path.exists():
                    missing_images.append(str(abs_image_path))
                did = next_did
                next_did += 1
                source_did_to_int[source_did] = did
                fc.write(
                    json.dumps(
                        {
                            "id": did,
                            "did": did,
                            "source_did": source_did,
                            "image_path": image_path,
                            "split": split,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                split_stats["corpus"] += 1

    next_qid = 0
    missing_pos = 0
    with queries_path.open("w", encoding="utf-8") as fq, qrels_path.open("w", encoding="utf-8") as fr:
        fr.write("qid\tdid\tscore\n")
        for split, query_file in default_query_files(root):
            if not query_file.exists():
                raise FileNotFoundError(query_file)
            split_stats = split_counts.setdefault(split, {"corpus": 0, "queries": 0, "qrels": 0})
            for obj in read_jsonl(query_file):
                if obj.get("query_modality") != "text":
                    continue
                if int(obj.get("task_id", 0)) != 0:
                    continue
                text = obj.get("query_txt")
                if not text:
                    continue
                source_qid = f"{split}:{obj['qid']}"
                qid = next_qid
                next_qid += 1
                source_qid_to_int[source_qid] = qid
                fq.write(
                    json.dumps(
                        {
                            "id": qid,
                            "qid": qid,
                            "source_qid": source_qid,
                            "text": text,
                            "caption": text,
                            "split": split,
                            "pos_source_dids": [str(x) for x in obj.get("pos_cand_list", [])],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                split_stats["queries"] += 1
                for source_did in obj.get("pos_cand_list", []):
                    source_did = str(source_did)
                    did = source_did_to_int.get(source_did)
                    if did is None:
                        missing_pos += 1
                        continue
                    fr.write(f"{qid}\t{did}\t1\n")
                    split_stats["qrels"] += 1

    mappings = {
        "source_did_to_int": source_did_to_int,
        "source_qid_to_int": source_qid_to_int,
    }
    mapping_path.write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "mbeir_root": str(root),
        "output_dir": str(out),
        "queries_path": str(queries_path),
        "corpus_path": str(corpus_path),
        "qrels_path": str(qrels_path),
        "num_corpus": next_did,
        "num_queries": next_qid,
        "num_qrels": sum(v["qrels"] for v in split_counts.values()),
        "split_counts": split_counts,
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "missing_positive_candidates": missing_pos,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.strict_images and missing_images:
        raise FileNotFoundError(f"{len(missing_images)} missing images. First: {missing_images[0]}")
    if missing_pos:
        raise RuntimeError(f"{missing_pos} positive candidates were not found in image corpus")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
