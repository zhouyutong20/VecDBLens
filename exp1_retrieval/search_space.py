from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple


VISUALNEWS_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "faiss_hnsw": {
        "M": [32, 48, 64],
        "efConstruction": [400, 800],
        "efSearch": [128, 256, 512, 1024, 2048],
    },
    "faiss_ivfpq": {
        "nlist": [4096, 8192, 16384],
        "M_by_model": {
            "CLIP_d512": [64, 128],
            "SigLIP2_d768": [64, 96, 128],
            "QwenVL3_d4096": [128, 256, 512],
            "VLM2VecV2_d1536": [64, 96, 192],
        },
        "nbits": [8],
        "nprobe": [32, 64, 128, 256, 512],
    },
    "cuvs_ivf_pq": {
        "n_lists": [4096, 8192, 16384],
        "pq_dim_by_model": {
            "CLIP_d512": [64, 128],
            "SigLIP2_d768": [64, 96, 128],
            "QwenVL3_d4096": [128, 256, 512],
            "VLM2VecV2_d1536": [64, 96, 192],
        },
        "pq_bits": [8],
        "n_probes": [32, 64, 128, 256, 512],
    },
    "cagra": {
        "graph_build_algo": ["IVF_PQ"],
        "graph_degree": [64, 96, 128],
        # cuVS requires intermediate_graph_degree < 256; use 255 instead of 256 in grids.
        "intermediate_graph_degree": [128, 255],
        "itopk_size": [128, 256, 512, 1024],
        "search_width": [1, 2, 4, 8],
    },
    "flat": {
        "metric": ["inner_product"],
    },
}


MSCOCO_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "faiss_hnsw": {
        "M": [16, 32, 64],
        "efConstruction": [200, 400],
        "efSearch": [64, 128, 256, 512],
    },
    "faiss_ivfpq": {
        "nlist": [1024, 2048, 4096],
        "M_by_model": {
            "CLIP_d512": [32, 64],
            "SigLIP2_d768": [32, 64, 96],
            "QwenVL3_d4096": [64, 128, 256, 512],
            "VLM2VecV2_d1536": [32, 64, 96, 192],
        },
        "nbits": [8],
        "nprobe": [8, 16, 32, 64, 128],
    },
    "cuvs_ivf_pq": {
        "n_lists": [1024, 2048, 4096],
        "pq_dim_by_model": {
            "CLIP_d512": [32, 64],
            "SigLIP2_d768": [32, 64, 96],
            "QwenVL3_d4096": [64, 128, 256, 512],
            "VLM2VecV2_d1536": [32, 64, 96, 192],
        },
        "pq_bits": [8],
        "n_probes": [8, 16, 32, 64, 128],
    },
    "cagra": {
        "graph_build_algo": ["IVF_PQ"],
        "graph_degree": [32, 64, 128],
        # cuVS currently requires intermediate graph degree to be < 256.
        "intermediate_graph_degree": [64, 128, 255],
        "itopk_size": [64, 128, 256, 512],
        "search_width": [1, 2, 4],
    },
    "flat": {
        "metric": ["inner_product"],
    },
}


FLICKR30K_SEARCH_SPACE: Dict[str, Dict[str, Any]] = {
    "faiss_hnsw": {
        "M": [16, 32],
        "efConstruction": [128, 256],
        "efSearch": [32, 64, 128, 256],
    },
    "faiss_ivfpq": {
        "nlist": [256, 512, 1024],
        "M_by_model": {
            "CLIP_d512": [32, 64],
            "SigLIP2_d768": [32, 64, 96],
            "QwenVL3_d4096": [64, 128, 256, 512],
            "VLM2VecV2_d1536": [32, 64, 96, 192],
        },
        "nbits": [8],
        "nprobe": [4, 8, 16, 32],
    },
    "cuvs_ivf_pq": {
        "n_lists": [256, 512, 1024],
        "pq_dim_by_model": {
            "CLIP_d512": [32, 64],
            "SigLIP2_d768": [32, 64, 96],
            "QwenVL3_d4096": [64, 128, 256, 512],
            "VLM2VecV2_d1536": [32, 64, 96, 192],
        },
        "pq_bits": [8],
        # n_probes=1 can produce invalid sentinel neighbor ids on this setup.
        "n_probes": [4, 8, 16, 32],
    },
    "cagra": {
        "graph_build_algo": ["IVF_PQ"],
        "graph_degree": [32, 64],
        "intermediate_graph_degree": [64, 128],
        "itopk_size": [32, 64, 128, 256],
        "search_width": [1, 2],
    },
    "flat": {
        "metric": ["inner_product"],
    },
}


DATASET_SEARCH_SPACE: Dict[str, Dict[str, Dict[str, Any]]] = {
    "visualnews": VISUALNEWS_SEARCH_SPACE,
    "flickr30k": FLICKR30K_SEARCH_SPACE,
    "mscoco": MSCOCO_SEARCH_SPACE,
}


def model_key(model_name: str, embedding_dim: int) -> str:
    name = model_name.lower()
    if "siglip" in name:
        return f"SigLIP2_d{embedding_dim}"
    if "qwen" in name:
        return f"QwenVL3_d{embedding_dim}"
    if "vlm2vec" in name or "vlmvec" in name:
        return f"VLM2VecV2_d{embedding_dim}"
    if "clip" in name:
        return f"CLIP_d{embedding_dim}"
    return f"{model_name}_d{embedding_dim}"


def _model_specific_values(space: Dict[str, Any], key: str, model_name: str, embedding_dim: int) -> List[int]:
    by_model = space[key]
    preferred_key = model_key(model_name, embedding_dim)
    if preferred_key in by_model:
        return by_model[preferred_key]
    dim_suffix = f"_d{embedding_dim}"
    for candidate_key, values in by_model.items():
        if candidate_key.endswith(dim_suffix):
            return values
    raise ValueError(
        f"No parameter list for model_name={model_name!r}, embedding_dim={embedding_dim}. "
        f"Expected one of: {sorted(by_model)}"
    )


def search_space_for_dataset(dataset_name: str | None) -> Dict[str, Dict[str, Any]]:
    if not dataset_name:
        return VISUALNEWS_SEARCH_SPACE
    normalized = dataset_name.lower().replace("-", "").replace("_", "")
    if normalized == "visualnews" or normalized.startswith("visualnews"):
        return VISUALNEWS_SEARCH_SPACE
    if normalized == "flickr30k" or normalized.startswith("flickr30k"):
        return FLICKR30K_SEARCH_SPACE
    if normalized == "mscoco" or normalized.startswith("mscoco"):
        return MSCOCO_SEARCH_SPACE
    raise ValueError(
        f"Unknown dataset_name={dataset_name!r}. Available dataset search spaces: {sorted(DATASET_SEARCH_SPACE)}"
    )


def get_param_groups(
    backend: str,
    model_name: str,
    embedding_dim: int,
    dataset_name: str | None = None,
) -> Iterable[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    """Yield (build_params, search_params_list), avoiding duplicate index builds."""
    backend = "cuvs_ivf_pq" if backend == "cuvs_ivfpq" else backend
    backend = "cagra" if backend == "cuvs_cagra" else backend
    search_space = search_space_for_dataset(dataset_name)
    if backend not in search_space:
        raise ValueError(f"Unknown backend {backend}. Available: {sorted(search_space)}")
    space = search_space[backend]

    if backend == "flat":
        yield {}, [{"metric": space["metric"][0]}]
        return

    if backend == "faiss_hnsw":
        for m, efc in product(space["M"], space["efConstruction"]):
            yield {"M": m, "efConstruction": efc}, [{"efSearch": efs} for efs in space["efSearch"]]
        return

    if backend == "faiss_ivfpq":
        m_values = _model_specific_values(space, "M_by_model", model_name, embedding_dim)
        for nlist, m, nbits in product(space["nlist"], m_values, space["nbits"]):
            if embedding_dim % m != 0:
                continue
            yield {"nlist": nlist, "M": m, "nbits": nbits}, [{"nprobe": nprobe} for nprobe in space["nprobe"]]
        return

    if backend == "cuvs_ivf_pq":
        pq_dim_values = _model_specific_values(space, "pq_dim_by_model", model_name, embedding_dim)
        for n_lists, pq_dim, pq_bits in product(space["n_lists"], pq_dim_values, space["pq_bits"]):
            if embedding_dim % pq_dim != 0:
                continue
            yield {"n_lists": n_lists, "pq_dim": pq_dim, "pq_bits": pq_bits}, [
                {"n_probes": n_probes} for n_probes in space["n_probes"]
            ]
        return

    if backend == "cagra":
        for algo, graph_degree, intermediate_graph_degree in product(
            space["graph_build_algo"],
            space["graph_degree"],
            space["intermediate_graph_degree"],
        ):
            yield {
                "graph_build_algo": algo,
                "graph_degree": graph_degree,
                "intermediate_graph_degree": intermediate_graph_degree,
            }, [
                {"itopk_size": itopk_size, "search_width": search_width}
                for itopk_size, search_width in product(space["itopk_size"], space["search_width"])
            ]
        return

    raise AssertionError(f"Unhandled backend: {backend}")
