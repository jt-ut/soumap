"""
layout.py — Wrapper around the UMAP layout optimizer.

Exposes one public function:

  optimize_layout(FSIM, coords_init, a, b, lrate, n_epochs,
                  negative_sample_rate, gamma, parallel,
                  random_state, age, verbose)
      Run the UMAP SGD layout optimizer on a precomputed similarity
      matrix FSIM, warm-started from coords_init.

a and b parameters are computed directly in the caller via:
    umap.umap_.find_ab_params(spread, min_dist)   # note argument order
"""

import numpy as np
import umap.umap_ as umap_


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def optimize_layout(
    FSIM,
    coords_init,
    a,
    b,
    lrate,
    n_epochs,
    negative_sample_rate,
    gamma,
    parallel,
    random_state,
    age,
):
    """
    Run the UMAP SGD layout optimizer on a precomputed similarity matrix.

    Wraps umap.umap_.simplicial_set_embedding, passing FSIM directly as
    the graph argument so UMAP never constructs its own similarity graph.
    The optimizer warm-starts from coords_init rather than reinitializing
    from scratch each call.

    Parameters
    ----------
    FSIM : scipy.sparse.csr_matrix, shape (M, M)
        Fuzzy similarity matrix. Used directly as the UMAP graph — no
        internal kNN graph construction occurs.
    coords_init : np.ndarray, shape (M, 2)
        Current embedding coordinates. Used as the warm start for the
        optimizer. Not mutated — a float32 copy is made internally.
    a : float
        UMAP curve parameter (from find_ab_params).
    b : float
        UMAP curve parameter (from find_ab_params).
    lrate : float
        Initial SGD learning rate (initial_alpha in umap-learn).
    n_epochs : int
        Number of SGD epochs to run.
    negative_sample_rate : int
        Number of negative samples per positive sample per epoch.
    gamma : float
        Repulsion strength applied to negative samples.
    parallel : bool
        If True, run optimizer in parallel and draw fresh random negative
        samples each call (non-reproducible but faster).
        If False, seed the optimizer with (random_state + age) for
        fully deterministic behavior (slower, requires random_state).
    random_state : int or None
        Base random seed. Only used when parallel=False. If parallel=False
        and random_state is None, falls back to parallel=True with a warning.
    age : int
        Current SOUMAP epoch count. Combined with random_state to produce
        a unique deterministic seed per call when parallel=False.

    Returns
    -------
    coords : np.ndarray, shape (M, 2), dtype float32
        Updated embedding coordinates after optimization.
    """
    # Resolve random_state and parallel based on user intent
    if not parallel and random_state is None:
        import warnings
        warnings.warn(
            "parallel=False requires random_state to be set for reproducibility. "
            "Falling back to parallel=True with a fresh random state.",
            UserWarning,
            stacklevel=2,
        )
        umap_seed = np.random.RandomState()
        run_parallel = True
    elif not parallel:
        # Unique deterministic seed per UMAP call: different at each age,
        # but identical across runs with the same random_state.
        umap_seed = np.random.RandomState(random_state + age)
        run_parallel = False
    else:
        # parallel=True: fresh random state each call for unbiased sampling
        umap_seed = np.random.RandomState()
        run_parallel = True

    # umap-learn expects init as float32
    init = coords_init.astype(np.float32)

    embedding, _ = umap_.simplicial_set_embedding(
        data=np.zeros((FSIM.shape[0], 1), dtype=np.float32),  # unused — warm start
        graph=FSIM,
        n_components=2,
        initial_alpha=lrate,
        a=a,
        b=b,
        gamma=gamma,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init,
        random_state=umap_seed,
        metric='euclidean',
        metric_kwds={},
        densmap=False,
        densmap_kwds={},
        output_dens=False,
        parallel=run_parallel,
        verbose=False,       # always silence umap-learn's internal prints;
                             # SOUMAP handles its own verbosity separately
        tqdm_kwds={'disable': True},
    )

    return embedding