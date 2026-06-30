# AGENTS.md

## Cursor Cloud specific instructions

This repository is the **VecDBLens anonymized artifact**: a collection of runnable Python
scripts and shell launchers (no `package.json`/`requirements.txt`/test suite). See `README.md`
for the full experiment descriptions and command examples.

### Environment

- Python 3 with dependencies installed via the startup update script (numpy, h5py, scikit-learn,
  tqdm, faiss-cpu, psutil, pillow, nanopq, torch (CPU), transformers). Packages install to the
  per-user site (`pip install --user`); no virtualenv activation is needed.
- The VM is **CPU-only**. GPU code paths cannot run here: the `cuvs`/`cupy`/`cagra` retrieval
  backends (`exp1_retrieval/backends/{cuvs_ivfpq,cagra}.py`) and the `run_gpu_cuvs_*.sh`
  launchers require CUDA and are not exercisable.

### What is and isn't runnable

- **Exp2 data generation (`exp2_filter_bench/generate_filter_ann_hdf5.py`) is fully runnable**
  and is the best end-to-end smoke test. It only needs an input HDF5 with `train` and `test`
  float matrices; it samples vectors, adds synthetic labels, computes filtered top-k ground
  truth with scikit-learn, and writes an output HDF5. You can synthesize a tiny input HDF5
  (datasets `train`/`test`) to run it without downloading SIFT/GIST.
  - Note: the README references `exp2_filter_bench/data_generation/generate_filter_ann_hdf5.py`,
    but the file actually lives at `exp2_filter_bench/generate_filter_ann_hdf5.py`.
- **Exp1 retrieval (`exp1_retrieval/run_benchmark.py`) cannot be run as-is.** It imports
  `from configs.search_space import get_param_groups` and the CPU launchers read a
  `configs/benchmark_*.json`, but the `configs/` package was intentionally omitted from this
  anonymized artifact. The actual search-space module is `exp1_retrieval/search_space.py`. To
  run retrieval you must recreate the local `configs/` package (and prepare embedding/qrels
  files). The retrieval backend modules themselves import fine; the CPU `flat`/`faiss_*`
  backends work with `faiss-cpu`.
- **Exp1 embedding (`exp1_embedding/.../embed_*.py`)** and **Exp2 RangePQ
  (`exp2_filter_bench/.../rangepq/run_dataset.py`, which needs the external `segr` module from
  the RangePQ build)** require external datasets, model checkpoints, and/or compiled third-party
  builds that are not bundled. Their imports/syntax are fine but they cannot run end-to-end here.

### Lint / test

- There is no configured linter or test suite. Use `python3 -m compileall <dir>` as a syntax check.
  - Known caveat: `exp1_embedding/visualnews/prepare_visualnews_test20k.py` is **ISO-8859 (not
    UTF-8)** — its docstring contains a non-UTF-8 byte sequence (`\xa1\xaa`). It fails
    `py_compile`/`compileall` and `import` under the default UTF-8 codec. This is a pre-existing
    artifact quirk, not an environment problem.
