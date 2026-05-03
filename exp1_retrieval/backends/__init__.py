from backends.flat import FlatBackend
from backends.faiss_hnsw import FaissHNSWBackend
from backends.faiss_ivfpq import FaissIVFPQBackend
from backends.cuvs_ivfpq import CuVSIVFPQBackend
from backends.cagra import CAGRABackend


BACKENDS = {
    "flat": FlatBackend,
    "faiss_hnsw": FaissHNSWBackend,
    "faiss_ivfpq": FaissIVFPQBackend,
    "cuvs_ivf_pq": CuVSIVFPQBackend,
    "cuvs_ivfpq": CuVSIVFPQBackend,
    "cagra": CAGRABackend,
    "cuvs_cagra": CAGRABackend,
}
