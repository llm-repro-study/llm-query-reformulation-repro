from .genqr import GenQR
from .genqr_ensemble import GenQREnsemble
from .q2k import Query2Keyword
from .q2d import Query2DocZS, Query2DocFS, Query2DocCoT
from .qa_expand import QAExpand
from .mugi import MUGI
from .csqe import CSQE
from .lamer import LameR

METHOD_REGISTRY = {
    "genqr": GenQR,
    "genqr_ensemble": GenQREnsemble,
    "q2k": Query2Keyword,
    "q2d_zs": Query2DocZS,
    "q2d_fs": Query2DocFS,
    "q2d_cot": Query2DocCoT,
    "qa_expand": QAExpand,
    "mugi": MUGI,
    "csqe": CSQE,
    "lamer": LameR,
}


def get_method(name: str):
    """Return a reformulation method class by name."""
    if name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method '{name}'. Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[name]

