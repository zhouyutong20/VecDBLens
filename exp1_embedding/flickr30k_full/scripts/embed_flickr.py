#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Visual News embeddings with resume support.")
    parser.add_argument("--model", choices=["clip", "qwen3", "siglip2"], required=True)
    parser.add_argument("--split", choices=["corpus", "queries"], required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=8192)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--qwen3-min-pixels", type=int, default=0)
    parser.add_argument("--qwen3-max-pixels", type=int, default=0)
    return parser.parse_args()


def stable_record_id(obj: Dict) -> object:
    return obj.get("id", obj.get("qid", obj.get("did")))


def resolve_image_path(obj: Dict, image_root: Path | None) -> str:
    if image_root is not None:
        return str(image_root / str(obj["image_path"]).lstrip("./"))
    return str(obj["abs_image_path"])


def ensure_model_files(model: str, model_path: Path) -> None:
    required = {
        "clip": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "pytorch_model.bin",
        ],
        "qwen3": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ],
        "siglip2": [
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "model.safetensors",
        ],
    }
    missing = [name for name in required[model] if not (model_path / name).exists()]
    if missing:
        raise FileNotFoundError(f"Model {model} is missing files in {model_path}: {missing}")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def completed_shards(output_dir: Path) -> int:
    shard = 0
    while True:
        vec_path = output_dir / f"shard_{shard:05d}.npy"
        ids_path = output_dir / f"shard_{shard:05d}.ids.npy"
        meta_path = output_dir / f"shard_{shard:05d}.meta.json"
        if vec_path.exists() and ids_path.exists() and meta_path.exists():
            shard += 1
            continue
        return shard


def skip_items(path: Path, start: int, end: int) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if idx >= end:
                break
            yield json.loads(line)


def open_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


class ClipEmbedder:
    def __init__(self, model_path: Path) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model.eval()
        self.dim = int(self.model.config.projection_dim)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float16).numpy()

    @torch.inference_mode()
    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        images = [open_rgb_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float16).numpy()


class Qwen3Embedder:
    def __init__(self, model_path: Path, min_pixels: int = 0, max_pixels: int = 0) -> None:
        scripts_dir = model_path / "scripts"
        if scripts_dir.exists():
            sys.path.insert(0, str(scripts_dir))
            from qwen3_vl_embedding import Qwen3VLEmbedder
        else:
            sys.path.insert(0, str(ROOT / "third_party" / "Qwen3-VL-Embedding"))
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {
            "model_name_or_path": str(model_path),
            "dtype": torch.float16,
        }
        if min_pixels > 0:
            kwargs["min_pixels"] = min_pixels
        if max_pixels > 0:
            kwargs["max_pixels"] = max_pixels
        self.embedder = Qwen3VLEmbedder(**kwargs)
        config = self.embedder.model.config
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None and getattr(config, "text_config", None) is not None:
            hidden_size = getattr(config.text_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Unable to determine Qwen3 embedding dimension from model config")
        self.dim = int(hidden_size)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        items = [{"text": text} for text in texts]
        feats = self.embedder.process(items)
        return feats.detach().cpu().to(torch.float16).numpy()

    @torch.inference_mode()
    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        items = [{"image": path} for path in image_paths]
        feats = self.embedder.process(items)
        return feats.detach().cpu().to(torch.float16).numpy()


class SigLIP2Embedder:
    def __init__(self, model_path: Path) -> None:
        from transformers import AutoModel, AutoProcessor

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        kwargs = {"torch_dtype": torch.float16} if self.device.type == "cuda" else {}
        self.model = AutoModel.from_pretrained(str(model_path), **kwargs).to(self.device)
        self.processor = AutoProcessor.from_pretrained(str(model_path))
        self.model.eval()
        hidden = getattr(getattr(self.model.config, "text_config", None), "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            raise AttributeError("Unable to determine SigLIP2 embedding dimension from model config")
        self.dim = int(hidden)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float16).numpy()

    @torch.inference_mode()
    def encode_images(self, image_paths: Sequence[str]) -> np.ndarray:
        images = [open_rgb_image(p) for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu().to(torch.float16).numpy()


def build_embedder(args: argparse.Namespace):
    model = args.model
    model_path = args.model_path
    if model == "clip":
        return ClipEmbedder(model_path)
    if model == "qwen3":
        return Qwen3Embedder(
            model_path,
            min_pixels=args.qwen3_min_pixels,
            max_pixels=args.qwen3_max_pixels,
        )
    if model == "siglip2":
        return SigLIP2Embedder(model_path)
    raise ValueError(model)


def write_shard(
    output_dir: Path,
    shard_idx: int,
    vectors: np.ndarray,
    ids: np.ndarray,
    meta: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_vec = output_dir / f"shard_{shard_idx:05d}.npy.tmp"
    tmp_ids = output_dir / f"shard_{shard_idx:05d}.ids.npy.tmp"
    tmp_meta = output_dir / f"shard_{shard_idx:05d}.meta.json.tmp"
    with tmp_vec.open("wb") as f:
        np.save(f, vectors.astype(np.float16, copy=False))
    with tmp_ids.open("wb") as f:
        np.save(f, ids.astype(np.int64, copy=False))
    tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_vec.replace(output_dir / f"shard_{shard_idx:05d}.npy")
    tmp_ids.replace(output_dir / f"shard_{shard_idx:05d}.ids.npy")
    tmp_meta.replace(output_dir / f"shard_{shard_idx:05d}.meta.json")


def main() -> None:
    args = parse_args()
    ensure_model_files(args.model, args.model_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_shard = completed_shards(output_dir)
    resume_items = resume_shard * args.shard_size
    input_total_items = count_lines(args.input_jsonl)
    start_index = max(args.start_index, 0)
    end_index = input_total_items if args.end_index < 0 else min(args.end_index, input_total_items)
    if end_index < start_index:
        raise ValueError(f"Invalid range: start_index={start_index}, end_index={end_index}")
    range_total_items = end_index - start_index
    if args.max_items > 0:
        range_total_items = min(range_total_items, args.max_items)
        end_index = start_index + range_total_items
    total_items = range_total_items

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    embedder = build_embedder(args)
    summary = {
        "model": args.model,
        "split": args.split,
        "model_path": str(args.model_path),
        "checkpoint_path": None,
        "input_jsonl": str(args.input_jsonl),
        "output_dir": str(output_dir),
        "input_total_items": input_total_items,
        "start_index": start_index,
        "end_index": end_index,
        "range_total_items": range_total_items,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "qwen3_min_pixels": args.qwen3_min_pixels,
        "qwen3_max_pixels": args.qwen3_max_pixels,
        "resume_shard": resume_shard,
        "resume_items": resume_items,
        "embedding_dim": embedder.dim,
        "device": str(getattr(embedder, "device", "unknown")),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if resume_items >= total_items:
        print(json.dumps({"status": "already_complete", **summary}, ensure_ascii=False, indent=2))
        return

    iterator = skip_items(args.input_jsonl, start_index + resume_items, end_index)

    shard_ids: List[int] = []
    shard_vectors: List[np.ndarray] = []
    batch_ids: List[int] = []
    batch_payload: List[str] = []
    processed = resume_items
    shard_idx = resume_shard
    batches = 0
    start = time.time()

    pbar = tqdm(total=total_items, initial=resume_items, dynamic_ncols=True)
    try:
        for obj in iterator:
            rec_id = stable_record_id(obj)
            payload = resolve_image_path(obj, args.image_root) if args.split == "corpus" else obj["text"]
            batch_ids.append(rec_id)
            batch_payload.append(payload)

            if len(batch_payload) < args.batch_size:
                continue

            vectors = (
                embedder.encode_images(batch_payload)
                if args.split == "corpus"
                else embedder.encode_texts(batch_payload)
            )
            shard_ids.extend(batch_ids)
            shard_vectors.append(vectors)
            processed += len(batch_ids)
            batches += 1
            pbar.update(len(batch_ids))
            if batches % args.log_every == 0:
                elapsed = max(time.time() - start, 1e-6)
                rate = processed / elapsed
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "processed": processed,
                            "total": total_items,
                            "rate_items_per_sec": round(rate, 2),
                            "current_shard": shard_idx,
                            "batches": batches,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            batch_ids = []
            batch_payload = []

            if len(shard_ids) >= args.shard_size:
                vectors_out = np.concatenate(shard_vectors, axis=0)[: args.shard_size]
                ids_out = np.asarray(shard_ids[: args.shard_size])
                meta = {
                    "model": args.model,
                    "split": args.split,
                    "shard_idx": shard_idx,
                    "count": int(len(ids_out)),
                    "embedding_dim": int(vectors_out.shape[1]),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                write_shard(output_dir, shard_idx, vectors_out, ids_out, meta)
                print(json.dumps({"event": "shard_written", **meta}, ensure_ascii=False), flush=True)
                shard_idx += 1
                shard_ids = shard_ids[args.shard_size :]
                if shard_vectors:
                    keep = 0 if len(shard_ids) == 0 else len(shard_ids)
                    tail = np.concatenate(shard_vectors, axis=0)[args.shard_size :]
                    shard_vectors = [tail] if keep else []

        if batch_payload:
            vectors = (
                embedder.encode_images(batch_payload)
                if args.split == "corpus"
                else embedder.encode_texts(batch_payload)
            )
            shard_ids.extend(batch_ids)
            shard_vectors.append(vectors)
            processed += len(batch_ids)
            pbar.update(len(batch_ids))

        if shard_ids:
            vectors_out = np.concatenate(shard_vectors, axis=0)
            ids_out = np.asarray(shard_ids)
            meta = {
                "model": args.model,
                "split": args.split,
                "shard_idx": shard_idx,
                "count": int(len(ids_out)),
                "embedding_dim": int(vectors_out.shape[1]),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "is_final_partial": True,
            }
            write_shard(output_dir, shard_idx, vectors_out, ids_out, meta)
            print(json.dumps({"event": "shard_written", **meta}, ensure_ascii=False), flush=True)

    finally:
        pbar.close()

    elapsed = time.time() - start
    done = {
        **summary,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "processed_total": processed,
        "elapsed_sec": round(elapsed, 2),
        "avg_items_per_sec": round(processed / max(elapsed, 1e-6), 2),
    }
    (output_dir / "summary.json").write_text(json.dumps(done, indent=2), encoding="utf-8")
    print(json.dumps({"event": "done", **done}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
