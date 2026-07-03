# soumap

**Self-Organizing UMAP** — a hybrid SOM / UMAP algorithm for learning data prototypes and their low-dimensional embedding jointly.

## What it does

SOUMAP alternates between two steps:

1. **SOM step** — prototype vectors `W` are updated via batch SOM learning. The output-space topology governing neighbor updates is derived from Delaunay triangulation of the current embedding coordinates `Y`.
2. **UMAP step** — a UMAP similarity graph `UP` is constructed from the co-adjacency matrix (CONN) produced by vector quantization recall, and the UMAP layout optimizer updates `Y` warm-started from its current state. The new `Y` replaces the SOM's output topology for the next SOM step.

Unlike a standard SOM, the output lattice is not fixed — it evolves with the data structure. Unlike standard UMAP, the similarity graph is not user-prescribed (no `n_neighbors` parameter) — it is inferred from the learned VQ topology.

## Installation

```bash
pip install git+https://github.com/jt-ut/soumap.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/jt-ut/soumap.git
cd soumap
pip install -e .
```

## Dependencies

- [vqlp](https://github.com/jt-ut/vqlp) — vector quantization and recall
- [gtsom](https://github.com/jt-ut/gtsom) — general topology SOM learning
- [umap-learn](https://github.com/lmcinnes/umap) — UMAP layout optimizer
- numpy, scipy, scikit-learn

## Quick start

```python
from soumap import SOUMAP

model = SOUMAP(M=100)
model.fit(X, n_iters=50)

# 2D embedding coordinates of the prototypes
Y = model.Y

# Map new data to output space via BMU lookup
coords = model.transform(X_new)
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `M` | — | Number of prototypes |
| `rho_0` | `2.0` | Initial SOM neighborhood bandwidth |
| `rho_T` | `0.5` | Final SOM neighborhood bandwidth |
| `umap_update_freq` | `5` | SOM epochs between UMAP embedding updates |
| `umap_n_epochs` | `100` | UMAP optimizer epochs per update |
| `min_dist_0` | `0.01` | Initial UMAP min_dist |
| `min_dist_T` | `0.01` | Final UMAP min_dist |
| `spread_0` | `1.0` | Initial UMAP spread |
| `spread_T` | `0.5` | Final UMAP spread |
| `lrate_0` | `1.0` | Initial UMAP learning rate |
| `lrate_T` | `0.1` | Final UMAP learning rate |
| `age_anneal` | `100` | Epoch at which parameter annealing stops |
| `random_state` | `None` | Random seed |

## Citation

If you use SOUMAP in your work, please cite:

```bibtex
@inproceedings{taylor2024self,
  title={A Self-Organizing UMAP for Clustering},
  author={Taylor, Josh and Offner, Stella},
  booktitle={International Workshop on Self-Organizing Maps, Learning Vector Quantization \& Beyond},
  pages={63--73},
  year={2024},
  organization={Springer}
}
```

## Related packages

- [vqlp](https://github.com/jt-ut/vqlp) — vector quantization with arbitrary Lp metrics
- [gtsom](https://github.com/jt-ut/gtsom) — general topology self-organizing map
