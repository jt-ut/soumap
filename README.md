# soumap

**Self-Organizing UMAP** — a hybrid SOM / UMAP algorithm for jointly learning data prototypes and their low-dimensional embedding, with an optional multiview clustering stage (MPEC).

## What it does

SOUMAP alternates between two steps:

1. **SOM step** — prototype vectors `W` are updated via batch SOM learning. The output-space topology governing neighbor updates is derived from Delaunay triangulation (or Gabriel graph) of the current embedding coordinates.
2. **UMAP step** — a UMAP similarity graph is constructed from the co-adjacency matrix (CADJ) produced by vector quantization recall, and the UMAP layout optimizer updates the embedding warm-started from its current state. The new embedding replaces the SOM's output topology for the next SOM step.

Unlike a standard SOM, the output lattice is not fixed — it evolves with the data structure. Unlike standard UMAP, the similarity graph is not user-prescribed — it is inferred from the learned VQ topology.

Once a SOUMAP is trained, the optional **MPEC** (Multiview Prototype Embedding & Clustering) stage fuses three views of the prototype space — topology (CADJ), high-D geometry (W), and low-D geometry (embedding) — into a single weighted graph, then partitions it with the Walktrap community-detection algorithm.

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
- numpy, scipy, scikit-learn, h5py, igraph, plotnine

## Quick start

### SOUMAP

```python
from soumap import SOUMAP, SOMParams, UMAPParams, CtrlParams

som_params = SOMParams(
    rho_0=2.0,                  # initial neighborhood bandwidth (overwritten by tune_embed_scale)
    rho_f=0.5,                  # final neighborhood bandwidth
    nbr_topo_alpha_0=0.5,
    nbr_topo_alpha_f=1.0,
    coord_topo='gabriel',
    proto_topo='STK_CADJ',
)

umap_params = UMAPParams(
    update_freq=5,              # SOM epochs between UMAP updates
    n_epochs=100,
    min_dist_0=0.1,             # overwritten by tune_embed_scale
    min_dist_f=0.1,
    spread_0=2.0,               # overwritten by tune_embed_scale
    spread_f=2.0,
    lrate_0=1.0,
    lrate_f=0.5,
    negative_sample_rate=5,
    gamma=1.0,
)

ctrl_params = CtrlParams(
    total_epochs=100,
    coord_init='pca',
    embedding_range=(-3.0, 3.0),
    n_jobs=1,
    random_state=42,
    verbose=True,
)

model = SOUMAP(M=100, som_params=som_params, umap_params=umap_params, ctrl_params=ctrl_params)

# Calibrate rho_0, min_dist, and spread from the null geometry of X
model.tune_embed_scale(rho0_scale=0.75, min_dist0_scale=1.0, min_distf_scale=0.1)

# Build architecture (initializes W and embedding)
model.build(X)

# Train
model.fit(X)

# 2D embedding coordinates of the prototypes
Y = model.coords

# Map new data to output space via BMU lookup
coords = model.transform(X_new)

# Save and reload
model.save('model.h5')
model = SOUMAP.load('model.h5')
```

### MPEC clustering

```python
from soumap import MPEClustering

# Extract SOUMAP outputs
W    = model.W                          # (M, D) high-D prototypes
Y    = model.coords                     # (M, 2) low-D embedding
CADJ = model.gtsom.recaller.CADJ        # (M, M) sparse co-adjacency matrix

mpec = MPEClustering(
    walktrap_n_steps="auto",    # find optimal step count automatically
    kernel_w_cadj=1.0,
    kernel_w_high_d=1.0,
    kernel_w_low_d=1.0,
    kernel_power=2.0,
    verbose=True,
)

mpec.fit(W=W, Y=Y, CADJ=CADJ)

# Cluster labels for each prototype (-1 = empty prototype)
print(mpec.labels_)
print(mpec.active_labels_)      # labels for non-empty prototypes only

# Diagnostic plots
mpec.plot_diagnostics(embed_coords=Y, show_graph=True, save_dir=".")

# Save and reload
mpec.save('mpec.h5')
mpec = MPEClustering.load('mpec.h5')
```

## SOUMAP parameters

| Parameter | Default | Description |
|---|---|---|
| `M` | — | Number of prototypes |

**`SOMParams`**

| Parameter | Default | Description |
|---|---|---|
| `rho_0` | `2.0` | Initial SOM neighborhood bandwidth (overwritten by `tune_embed_scale`) |
| `rho_f` | `0.5` | Final SOM neighborhood bandwidth |
| `nbr_topo_alpha_0` | `0.5` | Initial neighbor topology alpha |
| `nbr_topo_alpha_f` | `1.0` | Final neighbor topology alpha |
| `coord_topo` | `'gabriel'` | Output-space topology type (`'gabriel'`, `'delaunay'`) |
| `proto_topo` | `'STK_CADJ'` | Prototype topology type |
| `compute_dr_metrics` | `False` | Whether to compute dimensionality reduction metrics during training |

**`UMAPParams`**

| Parameter | Default | Description |
|---|---|---|
| `update_freq` | `5` | SOM epochs between UMAP embedding updates |
| `n_epochs` | `100` | UMAP optimizer epochs per update |
| `min_dist_0` | `0.1` | Initial UMAP min_dist (overwritten by `tune_embed_scale`) |
| `min_dist_f` | `0.1` | Final UMAP min_dist |
| `spread_0` | `2.0` | Initial UMAP spread (overwritten by `tune_embed_scale`) |
| `spread_f` | `2.0` | Final UMAP spread |
| `lrate_0` | `1.0` | Initial UMAP learning rate |
| `lrate_f` | `0.5` | Final UMAP learning rate |
| `negative_sample_rate` | `5` | Negative samples per positive edge per epoch |
| `gamma` | `1.0` | Repulsion strength |
| `use_standard_umap` | `False` | If True, use standard UMAP graph instead of CADJ-derived graph |

**`CtrlParams`**

| Parameter | Default | Description |
|---|---|---|
| `total_epochs` | `100` | Total SOM training epochs |
| `min_nhbs` | `3` | Minimum number of neighbors per prototype |
| `fill_val` | `1` | Fill value for sparse topology |
| `coord_init` | `'pca'` | Embedding initialization (`'pca'`, `'random'`) |
| `embedding_range` | `(-3.0, 3.0)` | Initial embedding coordinate range |
| `n_jobs` | `1` | Parallelism (`1` = sequential, reproducible) |
| `random_state` | `None` | Random seed |
| `verbose` | `False` | Print training progress |
| `plot_every` | `0` | Plot embedding every N epochs (`0` = never) |

## MPEC parameters

| Parameter | Default | Description |
|---|---|---|
| `walktrap_n_steps` | `"auto"` | Random-walk steps for Walktrap. `"auto"` finds the optimal count by analysing the tradeoff between Markov-chain mixing and cluster cohesion |
| `kernel_w_cadj` | `1.0` | Fusion weight for the CADJ topology view |
| `kernel_w_high_d` | `1.0` | Fusion weight for the high-D Euclidean view (on `W`) |
| `kernel_w_low_d` | `1.0` | Fusion weight for the low-D Euclidean view (on `Y`) |
| `kernel_power` | `2.0` | Exponent applied to each view before fusion. Values > 1 sharpen inter-view agreement and suppress disagreements |
| `kernel_min_value` | `0.01` | Hard lower bound on retained kernel values after sparsification |
| `remove_empty_protos` | `True` | Exclude prototypes with empty receptive fields; their `labels_` entry is set to -1 |
| `kernel_min_nhbs` | `3` | Minimum neighbors guaranteed per prototype in Euclidean kernel views |
| `kernel_support` | `"dense"` | Sparsity pattern for high-D and low-D kernel views. `"dense"` allows Euclidean views to introduce edges not present in CADJ; `"CADJ"` restricts to existing CADJ edges |
| `walktrap_max_steps` | `15` | Steps evaluated when `walktrap_n_steps="auto"` |
| `verbose` | `False` | Print fitting progress and warnings |

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
