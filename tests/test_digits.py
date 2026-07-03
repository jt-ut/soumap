"""
test_digits.py — End-to-end SOUMAP test on sklearn's digits dataset.

Trains a SOUMAP model on 1797 handwritten digit images (64 features,
10 classes). Parameters are tuned from a null embedding via
tune_embed_scale() before building the model.

Usage
-----
    python tests/test_digits.py
"""

import numpy as np
from sklearn.datasets import load_digits
from soumap import SOUMAP, SOMParams, UMAPParams, CtrlParams


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

print("Loading digits dataset...")
X, y = load_digits(return_X_y=True)
print(f"  X: {X.shape}, classes: {np.unique(y)}")


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

som_params = SOMParams(
    rho_0=2.0,               # will be overwritten by tune_embed_scale()
    rho_f=0.5,
    nbr_topo_alpha_0=1.0,
    nbr_topo_alpha_f=1.0,
    coord_topo='gabriel',   # 'delaunay' or 'gabriel'
    proto_topo='STK_CADJ',       # 'CONN' or 'CONN_STK'    
    compute_dr_metrics=True,
)

umap_params = UMAPParams(
    update_freq=5,
    n_epochs=200,
    min_dist_0=0.1,          # will be overwritten by tune_embed_scale()
    min_dist_f=0.1,          # will be overwritten by tune_embed_scale()
    spread_0=2.0,            # will be overwritten by tune_embed_scale()
    spread_f=2.0,            # will be overwritten by tune_embed_scale()
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
    coord_init='pca',         # 'pca', 'le', or 'random'
    embedding_range=(-3.0, 3.0),
    n_jobs=None,             # None = all cores; 1 = sequential + reproducible
    random_state=1234,
    verbose=True,
    plot_every=0,            # suppress in-training plots for speed
)

model = SOUMAP(
    M=100,
    som_params=som_params,
    umap_params=umap_params,
    ctrl_params=ctrl_params,
)


# ---------------------------------------------------------------------------
# Tune embedding scale parameters from null embedding (optional)
# ---------------------------------------------------------------------------

print("\nTuning embedding scale parameters from null embedding...")
model.tune_embed_scale(rho0_scale=0.75, min_dist0_scale=1.0, min_distf_scale=0.1)

print(f"\n  Tuned parameters:")
print(f"    rho_0      = {model.som_params.rho_0:.4f}")
print(f"    rho_f      = {model.som_params.rho_f:.4f}")
print(f"    min_dist_0 = {model.umap_params.min_dist_0:.4f}")
print(f"    min_dist_f = {model.umap_params.min_dist_f:.4f}")
print(f"    spread_0   = {model.umap_params.spread_0:.4f}")
print(f"    spread_f   = {model.umap_params.spread_f:.4f}")


# ---------------------------------------------------------------------------
# Build — construct SOM architecture from data
# ---------------------------------------------------------------------------

print("\nBuilding SOUMAP...")
model.build(X, labels=y)

print(f"  W shape     : {model.W.shape}")
print(f"  coords shape: {model.coords.shape}")


# ---------------------------------------------------------------------------
# Fit — learning loop
# ---------------------------------------------------------------------------

print("\nFitting SOUMAP...")
model.fit(X, labels=y)


# ---------------------------------------------------------------------------
# Inspect results
# ---------------------------------------------------------------------------

print("\nDone.")
print(f"  model.age    : {model.age}")
print(f"  learn_hist   : {len(model.learn_hist)} entries")

coords_X = model.transform(X)
print(f"  transform(X) : {coords_X.shape}")

if model.learn_hist:
    last = model.learn_hist[-1]
    print(f"\n  Final epoch snapshot:")
    print(f"    age    = {last['age']}")
    print(f"    mqe    = {last['mqe']:.4f}")
    print(f"    delBMU = {last['delBMU']:.4f}")


# ---------------------------------------------------------------------------
# Save plots
# ---------------------------------------------------------------------------

print("\nSaving plots...")
import matplotlib.pyplot as plt

init_fig = model.gtsom.learn_history_[0]['fig']
if init_fig is not None:
    init_fig.save('soumap_digits_init.png', dpi=150)
    print("  Saved: soumap_digits_init.png")

fig = model.gtsom.plot(
    color_by='labels',
    title='SOUMAP',
    subtitle=f'Digits (age={model.age})',
)
fig.save('soumap_digits_final.png', dpi=150)
print("  Saved: soumap_digits_final.png")
fig.draw()
plt.show()