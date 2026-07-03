"""
test_mpec_digits.py — MPEC clustering test on sklearn digits data.

Staged test script:
  Stage 1 — Load digits data, train SOUMAP, inspect embedding.
  Stage 2 — Run MPEClustering on SOUMAP outputs (W, CADJ, coords).
  Stage 3 — Run MPEClustering diagnostic plots.

Each stage can be run independently by commenting out later stages.

Data
----
sklearn load_digits: 1797 samples, 64 features (8x8 images), 10 classes.
Real handwritten-digit data with genuine within-class sub-structure
(different writing styles) and overlapping class boundaries — a more
realistic test than synthetic blobs.

SOUMAP
------
M=100 prototypes. tune_embed_scale() is called before build() to calibrate
rho_0 and min_dist from the null geometry, matching real-world usage.

Usage
-----
    python test_mpec_digits.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; change to "TkAgg" etc.
                               # if you want interactive windows
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from soumap import SOUMAP, SOMParams, UMAPParams, CtrlParams, MPEClustering


# ---------------------------------------------------------------------------
# Stage 1a: Load data
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage 1: Data + SOUMAP")
print("=" * 60)

digits = load_digits()
X = digits.data.astype(float)      # (1797, 64)
y_true = digits.target.astype(int) # (1797,)

N = X.shape[0]
D = X.shape[1]
K = len(np.unique(y_true))         # 10 digit classes
M = 100                            # number of SOUMAP prototypes

print(f"Data: N={N}, D={D}, K={K} digit classes")
print(f"  X shape  : {X.shape}")
print(f"  y unique : {np.unique(y_true)}")


# ---------------------------------------------------------------------------
# Stage 1b: Initialize and train SOUMAP
# ---------------------------------------------------------------------------

som_params = SOMParams(
    rho_0=2.0,               # overwritten by tune_embed_scale()
    rho_f=0.5,
    nbr_topo_alpha_0=1.0,
    nbr_topo_alpha_f=1.0,
    coord_topo='gabriel',    # 'delaunay' or 'gabriel'
    proto_topo='STK_CADJ',   # 'CONN' or 'CONN_STK'
    compute_dr_metrics=True,
)

umap_params = UMAPParams(
    update_freq=5,
    n_epochs=200,
    min_dist_0=0.1,          # overwritten by tune_embed_scale()
    min_dist_f=0.1,          # overwritten by tune_embed_scale()
    spread_0=2.0,            # overwritten by tune_embed_scale()
    spread_f=2.0,            # overwritten by tune_embed_scale()
    lrate_0=1.0,
    lrate_f=0.5,
    negative_sample_rate=5,
    gamma=1.0,
    use_standard_umap=False,
)

ctrl_params = CtrlParams(
    total_epochs=200,
    min_nhbs=3,
    fill_val=0.5,
    coord_init='pca',        # 'pca', 'le', or 'random'
    embedding_range=(-3.0, 3.0),
    n_jobs=None,              # None = all cores; 1 = sequential + reproducible
    random_state=1234,
    verbose=True,
    plot_every=0,             # suppress in-training plots for speed
)

model = SOUMAP(
    M=M,
    som_params=som_params,
    umap_params=umap_params,
    ctrl_params=ctrl_params,
)

print("\nTuning embedding scale parameters...")
model.tune_embed_scale(rho0_scale=0.75, min_dist0_scale=1.0, min_distf_scale=0.1)
print(f"  rho_0      = {model.som_params.rho_0:.4f}")
print(f"  min_dist_0 = {model.umap_params.min_dist_0:.4f}")
print(f"  spread_0   = {model.umap_params.spread_0:.4f}")

print("\nBuilding SOUMAP architecture...")
model.build(X, labels=y_true)
print(f"  W shape     : {model.W.shape}")
print(f"  coords shape: {model.coords.shape}")

print("\nPlotting SOUMAP initial embedding (age=0, pre-training)...")
fig_init = model.gtsom.plot(
    color_by='labels',
    title='SOUMAP — Digits',
    subtitle=f'Initialization (age=0), N={N}, D={D}, K={K}, M={M}',
)
fig_init.save('digits_soumap_init.pdf', dpi=150)
print("  Saved: digits_soumap_init.pdf")

print("\nFitting SOUMAP...")
model.fit(X, labels=y_true)
print(f"\nDone. age={model.age}")

if model.learn_hist:
    last = model.learn_hist[-1]
    print(f"  Final MQE={last['mqe']:.4f}, delBMU={last['delBMU']:.4f}")


# ---------------------------------------------------------------------------
# Stage 1c: Plot SOUMAP final embedding
# ---------------------------------------------------------------------------

print("\nPlotting SOUMAP final embedding...")
fig_som = model.gtsom.plot(
    color_by='labels',
    title='SOUMAP — Digits',
    subtitle=f'N={N}, D={D}, K={K}, M={M}, age={model.age}',
)
fig_som.save('digits_soumap_final.pdf', dpi=150)
print("  Saved: digits_soumap_final.pdf")

# ---------------------------------------------------------------------------
# Stage 1d: Save SOUMAP and reload to verify roundtrip
# ---------------------------------------------------------------------------

print("\nSaving SOUMAP...")
model.save('digits_soumap.h5')
print("  Saved: digits_soumap.h5")

print("Reloading SOUMAP from disk...")
model = SOUMAP.load('digits_soumap.h5')
print(f"  Reloaded: age={model.age}, W.shape={model.W.shape}, "
      f"coords.shape={model.coords.shape}")

print("Plotting reloaded SOUMAP (roundtrip check)...")
fig_reload = model.gtsom.plot(
    color_by='labels',
    title='SOUMAP — Digits (reloaded)',
    subtitle=f'N={N}, D={D}, K={K}, M={M}, age={model.age}',
)
fig_reload.save('digits_soumap_reloaded.pdf', dpi=150)
print("  Saved: digits_soumap_reloaded.pdf")

print("\nStage 1 complete. Inspect digits_soumap_init.pdf, "
      "digits_soumap_final.pdf, and digits_soumap_reloaded.pdf before proceeding.")
print()


# ---------------------------------------------------------------------------
# Stage 2: MPEC clustering
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage 2: MPEC clustering")
print("=" * 60)

# Extract SOUMAP outputs needed by MPEC
W    = model.W                      # (M, D) high-D prototypes
Y    = model.coords                 # (M, 2) low-D embedding
CADJ = model.gtsom.recaller.CADJ    # (M, M) sparse CADJ matrix
# CADJ_nhbs is optional — derived internally from CADJ if not passed
CADJ_nhbs = model.gtsom.recaller.CADJ_nhbs

print(f"MPEC inputs:")
print(f"  W shape    : {W.shape}")
print(f"  Y shape    : {Y.shape}")
print(f"  CADJ shape : {CADJ.shape}, nnz={CADJ.nnz}")
print(f"  CADJ dtype : {CADJ.dtype}")

mpec = MPEClustering(
    walktrap_n_steps="auto",
    kernel_w_cadj=1.0,
    kernel_w_high_d=1.0,
    kernel_w_low_d=1.0,
    kernel_power=2.0,
    kernel_min_value=0.01,
    remove_empty_protos=True,
    kernel_min_nhbs=3,
    kernel_support="dense",
    walktrap_max_steps=15,
    verbose=True,
)

print(f"\n{mpec}")
print("\nFitting MPEC...")

mpec.fit(W=W, Y=Y, CADJ=CADJ, CADJ_nhbs=CADJ_nhbs)

print(f"\n{mpec}")
print(f"\nResults:")
print(f"  labels_         : {mpec.labels_}")
print(f"  unique labels   : {np.unique(mpec.labels_)}")
print(f"  active_indices_ : {len(mpec.active_indices_)} active prototypes")
print(f"  n_steps_        : {mpec.n_steps_}")
print(f"  fused_graph_    : {mpec.fused_graph_.shape}, nnz={mpec.fused_graph_.nnz}")

# Quick sanity check: are the number of MPEC clusters close to K=10?
n_clusters_found = len(np.unique(mpec.active_labels_))
print(f"\n  Ground-truth K  : {K}")
print(f"  MPEC n_clusters : {n_clusters_found}")
if abs(n_clusters_found - K) <= 2:
    print("  [OK] Cluster count is within 2 of ground truth.")
else:
    print("  [NOTE] Cluster count differs from ground truth by more than 2.")
    print("         This may be expected — digit sub-styles (e.g. open vs.")
    print("         closed 4s) can split a single ground-truth class into")
    print("         multiple sub-clusters, or merge visually similar digits")
    print("         (e.g. 4/9, 3/8) — consider tuning weights or kernel_power.")

# --- Save and reload MPEC ---------------------------------------------------
print("\nSaving MPEC...")
mpec.save('digits_mpec.h5')
print("  Saved: digits_mpec.h5")

print("Reloading MPEC from disk...")
mpec = MPEClustering.load('digits_mpec.h5')
print(f"  Reloaded: {mpec}")
print(f"  n_steps_        : {mpec.n_steps_}")
print(f"  unique labels   : {np.unique(mpec.labels_)}")
print(f"  fused_graph_    : {mpec.fused_graph_.shape}, nnz={mpec.fused_graph_.nnz}")
print(f"  walktrap_result_ available: {mpec.walktrap_result_ is not None}")
print(f"  walktrap_dendrogram_ available: {mpec.walktrap_dendrogram_ is not None}")

print("\nStage 2 complete. Inspect labels and cluster count before proceeding.")
print()


# ---------------------------------------------------------------------------
# Stage 3: Diagnostic plots
# ---------------------------------------------------------------------------

print("=" * 60)
print("Stage 3: MPEC diagnostic plots")
print("=" * 60)

print("Generating diagnostic panel (from reloaded MPEC object)...")
print("(This may take a moment for the modularity curve.)")

panel = mpec.plot_diagnostics(
    embed_coords=Y,            # (M, 2) SOUMAP embedding coords
    show_graph=True,           # overlay fused_graph_ edges on the embedding
    which="all",
    max_components=3,
    point_size=3.0,
    figsize=(6, 5),
    save_dir=".",              # plot_diagnostics always saves as mpec_<name>.pdf
    save_format="pdf",
)

# Rename to the digits_mpec_* convention used by this test script.
# (plot_diagnostics() always writes mpec_<name>.<format> since mpec.py
# is a general-purpose module and doesn't know about per-test naming.)
for name in panel.keys():
    src = f"mpec_{name}.pdf"
    dst = f"digits_mpec_{name}.pdf"
    if os.path.exists(src):
        os.replace(src, dst)
        print(f"  Saved: {dst}")

print("Stage 3 complete.")