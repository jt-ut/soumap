# soumap

**Self-Organizing UMAP** — a hybrid SOM / UMAP algorithm for jointly learning data prototypes and their low-dimensional embedding, with an optional multiview clustering stage (MPEC).

## How it works

SOUMAP alternates between two steps:

1. **SOM step** — prototype vectors `W` are updated via batch SOM learning. The output-space topology governing neighbor updates is derived from a Gabriel graph (or Delaunay triangulation) of the current embedding coordinates.
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

## Citation

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
