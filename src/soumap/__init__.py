"""
soumap — Self-Organizing UMAP.

A hybrid SOM / UMAP algorithm for jointly learning data prototypes
and their low-dimensional embedding.

Reference
---------
Taylor, J. & Offner, S. (2024). A Self-Organizing UMAP for Clustering.
International Workshop on Self-Organizing Maps, Learning Vector
Quantization & Beyond, pp. 63-73. Springer.
"""

from .soumap import SOUMAP, SOMParams, UMAPParams, CtrlParams
from .mpec import MPEClustering

__all__ = [
    "SOUMAP",
    "SOMParams",
    "UMAPParams",
    "CtrlParams",
    "MPEClustering",
]