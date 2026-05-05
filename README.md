# VecDBLens Artifact

This repository provides the anonymized artifact for reproducing the main experimental scripts used in the paper.

The artifact covers two experimental groups:

- **Experiment 1:** multimodal retrieval evaluation over embedding, indexing, and search stages.
- **Experiment 2:** filtered ANN evaluation under different dimensionalities and filter ratios.

This repository contains runnable scripts, launchers, configuration templates, and data generation scripts. Large datasets, pretrained model checkpoints, compiled third-party binaries, machine-local configuration files, generated indexes, and raw result directories are not included.

## Anonymous review note

This repository is prepared for anonymous review. Author-identifying information, private machine paths, local configuration files, intermediate logs, hostnames, and unused analysis artifacts have been removed.

The code is intended to document and reproduce the reported experimental pipeline. Reproducers need to prepare datasets, model checkpoints, external algorithm builds, and local path configurations before running the scripts.

## Repository layout

```text
.
|-- exp1_embedding/
|   |-- flickr30k_full/
|   |-- mscoco_task0_full/
|   `-- visualnews/
|-- exp1_retrieval/
|   |-- run_benchmark.py
|   |-- launch_jobs.py
|   |-- search_space.py
|   `-- run_*.sh
`-- exp2_filter_bench/
    |-- data_generation/
    |   `-- generate_filter_ann_hdf5.py
    |-- standard_run/
    |   |-- acorn/
    |   |-- digra/
    |   `-- rangepq/
    `-- vec960_n990k/
        |-- acorn/
        |-- digra/
        |-- rangepq/
        `-- flat/
```

Only the main run paths are retained. Analysis scratch files, plotting scratch files, result aggregation notes, audit notes, machine-local configuration files, and empty result directories are intentionally omitted.

## Experiment 1: Multimodal retrieval

Experiment 1 evaluates the retrieval pipeline under different embedding models and ANN backends.

### Datasets

The retained scripts cover the following datasets:

- `flickr30k_test5k`
- `mscoco_test5k`
- `visualnews_test20k`

The datasets are not included in this repository. Reproducers should prepare the dataset roots, metadata files, query files, and qrels files in their local environment.

### Embedding scripts

The embedding launchers are stored under `exp1_embedding/`.

Main launchers:

```text
exp1_embedding/flickr30k_full/run_flickr_full_three_models_4gpu_server.sh
exp1_embedding/mscoco_task0_full/run_mscoco_task0_three_models_4gpu.sh
exp1_embedding/visualnews/run_embedding_pair.sh
exp1_embedding/visualnews/run_visualnews_full_workflow_clip_rerun_20260430.sh
```

Supporting Python scripts required by these launchers are kept alongside the corresponding shell scripts.

The retained embedding models are:

- `clip`
- `siglip2`
- `qwen3`

Model checkpoints or model identifiers should be configured locally before execution.

### Retrieval scripts

The retrieval benchmark scripts are stored under `exp1_retrieval/`.

Main components:

```text
exp1_retrieval/run_benchmark.py
exp1_retrieval/launch_jobs.py
exp1_retrieval/search_space.py
```

CPU FAISS launchers:

```text
exp1_retrieval/run_cpu_faiss_flickr30k_test5k.sh
exp1_retrieval/run_cpu_faiss_mscoco_test5k.sh
exp1_retrieval/run_cpu_faiss_visualnews_test20k.sh
```

GPU cuVS launchers:

```text
exp1_retrieval/run_gpu_cuvs_flickr30k_test5k.sh
exp1_retrieval/run_gpu_cuvs_mscoco_test5k.sh
exp1_retrieval/run_gpu_cuvs_visualnews_test20k.sh
```

The retained retrieval backends are:

- `flat`
- `faiss_hnsw`
- `faiss_ivfpq`
- `cuvs_ivfpq`
- `cagra`

### What to prepare for Experiment 1

Before running Experiment 1, the reproducer should prepare:

- dataset roots
- embedding output roots
- qrels and benchmark metadata paths
- Python environment paths
- model checkpoint paths or model identifiers
- writable output directories
- local configuration files required by the launchers

Machine-specific paths should be replaced through local configuration variables before execution.

### Minimal Experiment 1 flow

First generate embeddings, then run retrieval benchmarks.

Generate Flickr30K embeddings:

```bash
cd exp1_embedding/flickr30k_full
./run_flickr_full_three_models_4gpu_server.sh
```

Generate MS-COCO embeddings:

```bash
cd exp1_embedding/mscoco_task0_full
./run_mscoco_task0_three_models_4gpu.sh
```

Generate VisualNews embeddings:

```bash
cd exp1_embedding/visualnews
./run_embedding_pair.sh clip 0
./run_embedding_pair.sh siglip2 1
./run_embedding_pair.sh qwen3 2
```

Run retrieval benchmarks:

```bash
cd exp1_retrieval
SKIP=1 REUSE=1 ./run_cpu_faiss_flickr30k_test5k.sh
SKIP=1 REUSE=1 ./run_gpu_cuvs_flickr30k_test5k.sh
```

The `SKIP=1` and `REUSE=1` flags are used when precomputed embeddings or intermediate artifacts are already available.

## Experiment 2: Filtered ANN benchmark

Experiment 2 evaluates static filtered ANN search under different dimensionalities and filter ratios.

### Data sources and filtered-label generation

The filtered ANN datasets used in Experiment 2 are derived from two public ANN benchmark datasets:

- `SIFT`, used for the 128-dimensional setting.
- `GIST`, used for the 960-dimensional setting.

The original vector values are not modified. We construct filtered-search workloads by adding synthetic integer labels to the base vectors and generating query-specific label ranges with fixed random seeds. The filtered top-k ground truth is then recomputed under the corresponding label predicates.

The data generation script is provided at:

```text
exp2_filter_bench/data_generation/generate_filter_ann_hdf5.py
```

Large generated HDF5 files are not included in this anonymized artifact because of their size. They can be regenerated from the public SIFT and GIST HDF5 files using the provided script.

### Generated HDF5 format

Each generated filtered ANN HDF5 file contains:

- `train_vec`: base vectors used for index construction.
- `test_vec`: query vectors used for evaluation.
- `train_label`: synthetic integer labels assigned to base vectors.
- `test_label`: query-specific filter ranges.
- `neighbors`: filtered top-k ground-truth neighbor ids.
- `distances`: distances of the filtered top-k ground-truth neighbors.

The file attributes include the distance metric, dimension, label names, label types, filter expression template, filter ratio, and generation mode.

### Reproducibility of data generation

The default random seeds are fixed:

- vector sampling seed: `42`
- query range generation seed: `42`
- label generation seed: `123`

The source vectors are sampled from the original `train` and `test` arrays in the input HDF5 file. Synthetic labels are generated independently from the vector values.

### Data generation examples

Generate the 128-dimensional SIFT-derived filtered datasets:

```bash
python exp2_filter_bench/data_generation/generate_filter_ann_hdf5.py \
  --input /path/to/sift-128-euclidean.hdf5 \
  --output-root /path/to/output \
  --dataset-name sift128 \
  --dim 128 \
  --n-samples 990000 \
  --n-test 10000 \
  --n-filters 1 \
  --max-labels 10000 \
  --distance euclidean \
  --topk 100 \
  --ratios 1e-4 1e-3 1e-2 1e-1 5e-1 \
  --mode static
```

Generate the 960-dimensional GIST-derived filtered datasets:

```bash
python exp2_filter_bench/data_generation/generate_filter_ann_hdf5.py \
  --input /path/to/gist-960-euclidean.hdf5 \
  --output-root /path/to/output \
  --dataset-name gist960 \
  --dim 960 \
  --n-samples 990000 \
  --n-test 1000 \
  --n-filters 1 \
  --max-labels 10000 \
  --distance euclidean \
  --topk 100 \
  --ratios 1e-4 1e-3 1e-2 1e-1 5e-1 \
  --mode static
```

The generated files should then be used as the data roots for the filtered ANN launchers.

### Evaluation settings

The retained scripts cover:

- dimension `128`, `N=990000`
- dimension `960`, `N=990000`
- filter ratios `0.01%`, `0.1%`, `1%`, `10%`, `50%`

The retained Experiment 2 scripts are organized as:

```text
exp2_filter_bench/standard_run/
exp2_filter_bench/vec960_n990k/
```

### Included algorithms

The retained launchers cover:

- `ACORN`
- `DIGRA`
- `RangePQ`
- `Flat`

### External algorithm repositories

Experiment 2 relies on external algorithm implementations. The corresponding public repositories are:

- `ACORN`: `https://github.com/guestrin-lab/ACORN`
- `DIGRA`: `https://github.com/CUHK-DBGroup/DIGRA`
- `RangePQ`: `https://github.com/CUHK-DBGroup/RangePQ`

These upstream source trees, compiled binaries, and runtime environments are not bundled in this repository. Please prepare the corresponding builds locally and update the launcher paths before running the scripts.

### Standard 128D run path

`exp2_filter_bench/standard_run/` contains the retained `sift128` benchmark launchers.

ACORN:

```text
exp2_filter_bench/standard_run/acorn/run_acorn_sift128_static_target.sh
```

DIGRA:

```text
exp2_filter_bench/standard_run/digra/run_sift128_acorn_aligned.sh
```

RangePQ:

```text
exp2_filter_bench/standard_run/rangepq/run_rangepq_sift128_static_all_5ratios.sh
exp2_filter_bench/standard_run/rangepq/grid_static_target.sh
exp2_filter_bench/standard_run/rangepq/run_dataset.py
```

### 960D n990k run path

`exp2_filter_bench/vec960_n990k/` contains the retained `960D, N=990000` benchmark launchers.

ACORN:

```text
exp2_filter_bench/vec960_n990k/acorn/run_acorn_vec960_n990k_static.sh
exp2_filter_bench/vec960_n990k/acorn/run_acorn1_vec960_n990k_static.sh
```

DIGRA:

```text
exp2_filter_bench/vec960_n990k/digra/run_digra_vec960_n990k_static.sh
```

RangePQ:

```text
exp2_filter_bench/vec960_n990k/rangepq/run_rangepq_vec960_n990k_static.sh
```

Flat exact pre-filtering:

```text
exp2_filter_bench/vec960_n990k/flat/run_flat1_vec960_n990k_static.sh
exp2_filter_bench/vec960_n990k/flat/run_flat_prefilter_exact.py
```

### What to prepare for Experiment 2

Before running Experiment 2, the reproducer should prepare:

- public SIFT and GIST HDF5 source files
- generated filtered ANN HDF5 files
- a working `ACORN` build
- a working `DIGRA` build
- a working `RangePQ` runtime environment
- writable log and output directories
- local path settings required by each launcher

Machine-specific paths should be replaced in the corresponding launcher scripts or moved into local configuration files before execution.

### Minimal Experiment 2 flow

Generate the filtered datasets first, then run the algorithm-specific launchers.

Run ACORN on the standard 128D benchmark:

```bash
cd exp2_filter_bench/standard_run/acorn
./run_acorn_sift128_static_target.sh
```

Run DIGRA on the standard 128D benchmark:

```bash
cd exp2_filter_bench/standard_run/digra
./run_sift128_acorn_aligned.sh
```

Run RangePQ on the standard 128D benchmark:

```bash
cd exp2_filter_bench/standard_run/rangepq
./run_rangepq_sift128_static_all_5ratios.sh
```

Run ACORN on the 960D benchmark:

```bash
cd exp2_filter_bench/vec960_n990k/acorn
./run_acorn_vec960_n990k_static.sh
./run_acorn1_vec960_n990k_static.sh
```

Run DIGRA on the 960D benchmark:

```bash
cd exp2_filter_bench/vec960_n990k/digra
./run_digra_vec960_n990k_static.sh
```

Run RangePQ on the 960D benchmark:

```bash
cd exp2_filter_bench/vec960_n990k/rangepq
./run_rangepq_vec960_n990k_static.sh
```

Run Flat exact pre-filtering on the 960D benchmark:

```bash
cd exp2_filter_bench/vec960_n990k/flat
./run_flat1_vec960_n990k_static.sh
```

## Environment notes

The scripts were organized to preserve the reported experimental workflow. Since datasets, model checkpoints, external algorithm builds, and machine-local paths are environment-dependent, reproducing the full results requires local preparation.

Recommended preparation steps:

1. Create the required Python environments.
2. Download or prepare the datasets.
3. Prepare model checkpoints or model identifiers.
4. Generate the filtered ANN HDF5 files for Experiment 2.
5. Build external algorithm repositories when needed.
6. Update local paths in launcher scripts or local config files.
7. Create writable output and log directories.
8. Run embedding generation before retrieval benchmarks for Experiment 1.
9. Run algorithm-specific launchers for Experiment 2.

## Expected outputs

The scripts produce benchmark outputs such as:

- embedding files
- generated filtered ANN HDF5 files
- retrieval result files
- recall metrics
- QPS or latency logs
- algorithm-specific runtime logs
- filtered ANN benchmark summaries

Exact output locations depend on the local path settings configured by the reproducer.

## Notes on omitted files

The following files are intentionally omitted from the anonymized artifact:

- raw datasets
- pretrained model checkpoints
- generated embeddings
- generated indexes
- generated filtered ANN HDF5 files
- compiled third-party binaries
- raw logs
- intermediate result directories
- private configuration files
- machine-local path files
- plotting and post-processing scratch files

These omissions are intended to keep the artifact lightweight and anonymous while preserving the main experimental run paths and data generation procedure.

## Anonymous artifact checklist

Before sharing or submitting this repository, please check that the following information does not appear in the repository or rendered anonymous mirror:

- author names
- affiliations
- email addresses
- private GitHub usernames
- private server usernames
- absolute local paths
- hostnames
- internal project names unrelated to the paper
- notebook outputs containing local paths
- PDF or image metadata containing identifying information
