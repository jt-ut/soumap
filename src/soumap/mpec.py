"""
mpec
====
Multiview Prototype Embedding & Clustering (MPEC).

Clusters prototypes from a vector quantisation by fusing multiple graph views
of the prototype space into a single weighted graph, then partitioning it with
the Walktrap community-detection algorithm (Pons & Latapy, 2005).

Three views are combined:

* **CADJ view** — the raw co-adjacency matrix, row-normalised, reflecting the
  topology learned during VQ training.
* **High-D Euclidean view** — a self-tuning kernel on the prototype matrix W,
  capturing geometric proximity in the original feature space.
* **Low-D Euclidean view** — a self-tuning kernel on the prototype embedding Y
  (e.g. a UMAP or PCA projection), capturing proximity in the visualisation
  space.

The number of Walktrap random-walk steps can be set manually or found
automatically by analysing a tradeoff between Markov-chain mixing (Hellinger
distance from stationary distribution) and community cohesion (Barrat weighted
clustering coefficient) as the chain is iterated.  The optimal step maximises
the geometric mean of these two quantities (the tradeoff score), following the
intuition of Pons & Latapy.

Classes
-------
MPEClustering
    Fit a multiview prototype clustering.  Call :meth:`fit` then read results
    from instance attributes (``labels_``, ``fused_graph_``, etc.).

References
----------
Pons, P. & Latapy, M. (2005). Computing communities in large networks using
    random walks. Computer and Information Sciences, ISCIS 2005, 284-293.
Zelnik-Manor, L. & Perona, P. (2004). Self-tuning spectral clustering.
    Advances in Neural Information Processing Systems, 17.
Tasdemir, K. & Merenyi, E. (2009). Exploiting data topology in visualization
    and clustering of self-organizing maps.
    IEEE Transactions on Neural Networks, 20(4), 549-562.
"""

from __future__ import annotations

__all__ = ["MPEClustering"]

import numpy as np
import scipy.sparse as sp
import igraph
import pandas as pd
from plotnine import (
    ggplot, aes,
    geom_col, geom_density, geom_hline, geom_line, geom_point,
    geom_text, geom_tile, geom_vline,
    scale_fill_manual, scale_fill_gradient,
    scale_color_manual, scale_color_identity,
    scale_x_continuous,
    facet_wrap,
    guides,
    labs,
    theme, element_blank,
)
import matplotlib.pyplot as plt

from vqlp.cadj_utils import CADJ_self_tuning_kernel
from gtsom.vis_tools import vis_embedding_discrete, build_ctab, theme_minimal_bold


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _find_communicating_classes(A: sp.spmatrix) -> list[np.ndarray]:
    """
    Return the weakly connected components of the graph defined by A.

    Uses igraph's C-level component finder, which is substantially faster
    than networkx for the graph sizes typical in prototype clustering.

    Parameters
    ----------
    A : scipy.sparse matrix, shape (M, M)
        Symmetric adjacency matrix.  Edge weights are ignored; only the
        nonzero sparsity pattern is used.

    Returns
    -------
    components : list of np.ndarray
        Each element is a sorted array of vertex indices belonging to one
        weakly connected component, ordered by decreasing component size.
    """
    M = A.shape[0]
    A_coo = sp.triu(A, k=1).tocoo()  # upper triangle only; igraph is undirected
    g = igraph.Graph(
        n=M,
        edges=list(zip(A_coo.row.tolist(), A_coo.col.tolist())),
        directed=False,
    )
    membership = g.clusters(mode="weak").membership
    membership = np.asarray(membership, dtype=np.intp)

    # Group indices by component, largest first
    comp_ids, counts = np.unique(membership, return_counts=True)
    order = np.argsort(-counts)  # descending size
    return [np.where(membership == comp_ids[k])[0] for k in order]


def _rw_tradeoff_at_step(
    P: np.ndarray,
    stationary: np.ndarray,
    row_weights: np.ndarray,
) -> tuple[float, float]:
    """
    Compute raw mixing and cohesion for one power of the transition matrix.

    Returns the raw (un-normalised) values of Hellinger distance from
    stationary (mixing) and Barrat weighted clustering coefficient (cohesion).
    Normalisation into a comparable tradeoff score is handled by the caller
    :func:`_find_walktrap_steps`, which has access to the step-1 baselines
    needed to compute relative quantities.

    Parameters
    ----------
    P : np.ndarray, shape (n, n)
        Row-stochastic transition matrix P^t for one communicating class.
    stationary : np.ndarray, shape (n,)
        Stationary distribution (approximated as column sums / total of P^1).
    row_weights : np.ndarray, shape (n,)
        Per-vertex weights (e.g. CADJ row sums) for weighted averages.

    Returns
    -------
    mixing : float
        Weighted average Hellinger distance from rows of P^t to stationary.
        In [0, 1]; decreases monotonically toward 0 as t → ∞.
    cohesion : float
        Weighted average Barrat clustering coefficient of the graph whose
        edge weights are given by P^t.  Non-monotonic: dips when the walk
        first crosses community boundaries, then recovers as intra-community
        structure dominates.  Computed via igraph's
        ``transitivity_local_undirected(weights=...)``.
    """
    wsum = row_weights.sum()

    # --- Mixing: weighted average Hellinger distance from stationary ---------
    # H(p, q) = sqrt(1 - sum_i sqrt(p_i * q_i))
    sqrt_stat = np.sqrt(stationary)                        # shape (n,)
    inner = np.sqrt(np.maximum(P, 0.0)) @ sqrt_stat       # shape (n,)
    inner = np.clip(inner, 0.0, 1.0)                      # guard float errors
    mixing_per_vertex = np.sqrt(1.0 - inner)              # shape (n,)
    mixing = float(np.dot(row_weights, mixing_per_vertex) / wsum)

    # --- Cohesion: weighted average Barrat clustering coefficient -----------
    n = P.shape[0]
    P_sparse = sp.csr_matrix(P)
    upper = sp.triu(P_sparse, k=1).tocoo()
    g = igraph.Graph(
        n=n,
        edges=list(zip(upper.row.tolist(), upper.col.tolist())),
        edge_attrs={"weight": upper.data.tolist()},
        directed=False,
    )
    # NaN for degree-0 or degree-1 vertices (no triangles possible) → 0
    cohesion_per_vertex = np.asarray(
        g.transitivity_local_undirected(vertices=None, weights="weight"),
        dtype=np.float64,
    )
    cohesion_per_vertex = np.nan_to_num(cohesion_per_vertex, nan=0.0)
    cohesion = float(np.dot(row_weights, cohesion_per_vertex) / wsum)

    return mixing, cohesion


def _find_walktrap_steps(
    A: sp.spmatrix,
    row_weights: np.ndarray | None = None,
    max_steps: int = 15,
    verbose: bool = False,
) -> tuple[int, list[dict]]:
    """
    Automatically find the optimal number of Walktrap random-walk steps.

    For each weakly connected component, iterates P^t = P^1 @ P^(t-1) from
    t=1 to ``max_steps`` and evaluates two normalised quantities:

    .. code-block:: none

        HD_norm(t) = HD(t) / HD(1)
            Fraction of original non-stationarity retained.
            Starts at 1, decreases monotonically toward 0.

        CC_norm(t) = max(CC(t) - CC(1), 0) / (1 - CC(1))
            Fraction of available cohesion headroom gained since step 1.
            Starts at 0, rises as intra-community structure is revealed.
            Clipped to 0 during the step-2 CC trough.

    The optimal step is the **first crossing point** — the first t where
    ``CC_norm(t) >= HD_norm(t)``.  At this step, the cohesion gain (relative
    to available headroom) has caught up with the non-stationarity loss
    (relative to starting level).  This is a natural, symmetric tradeoff:
    both quantities are normalised to [0, 1] relative to their step-1
    baselines and theoretical bounds.

    The full curve is always evaluated to ``max_steps`` so the plot shows
    the complete picture.  If CC(1) is very close to 1 (the component is
    already fully clique-like at step 1), the denominator ``1 - CC(1)`` is
    guarded against division by zero; if no crossing is found within
    ``max_steps``, ``best_step = 1`` is returned with a warning (verbose
    mode only).

    Parameters
    ----------
    A : scipy.sparse matrix, shape (M, M)
        Symmetric weighted adjacency matrix of the fused graph.
    row_weights : np.ndarray, shape (M,), optional
        Per-vertex weights.  Defaults to uniform weights.
    max_steps : int, default 15
        Number of steps to evaluate per component.  Always runs to this
        limit so the full tradeoff curve is available for plotting.

    Returns
    -------
    opt_steps : int
        Component-size-weighted average optimal step count.
    step_analysis : list of dict
        One dict per communicating class, each containing:

        ``indices`` : np.ndarray
            Vertex indices in this component.
        ``mixing`` : list of float
            Raw Hellinger distance at each evaluated step.
        ``cohesion`` : list of float
            Raw Barrat clustering coefficient at each evaluated step.
        ``hd_norm`` : list of float
            Normalised HD retention: HD(t) / HD(1).
        ``cc_norm`` : list of float
            Normalised CC gain: max(CC(t) - CC(1), 0) / (1 - CC(1)).
        ``best_step`` : int
            First crossing step (or 1 if degenerate).
        ``size`` : int
            Number of vertices in this component.
        ``degenerate`` : bool
            True if CC(1) was too close to 1 for analysis.
        ``walk_lengths`` : list of int
            Step indices (1, 2, 3, ...).
    """
    M = A.shape[0]
    if row_weights is None:
        row_weights = np.ones(M, dtype=np.float64)

    components = _find_communicating_classes(A)
    n_components = len(components)
    if verbose:
        sizes = [len(c) for c in components]
        print(f"[MPEClustering] Step analysis: {n_components} communicating "
              f"class(es), sizes: {sizes}")
        if any(s <= 2 for s in sizes):
            print(f"[MPEClustering] [WARNING] {sum(s <= 2 for s in sizes)} "
                  f"component(s) with <=2 vertices — step analysis skipped.")
    step_analysis = []

    for k, indices in enumerate(components):
        n = len(indices)
        record: dict = {
            "indices": indices, "size": n,
            "mixing": [], "cohesion": [],
            "hd_norm": [], "cc_norm": [],
            "walk_lengths": [],
        }

        if n <= 2:
            record["best_step"] = 1
            step_analysis.append(record)
            continue

        # Row-stochastic transition matrix for this component
        A_sub = np.asarray(A[np.ix_(indices, indices)].todense(), dtype=np.float64)
        row_sums = A_sub.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = A_sub / row_sums
        P_orig = P.copy()

        ss = P.sum(axis=0); ss /= ss.sum()
        w = row_weights[indices]

        # --- Step 1 baselines ------------------------------------------------
        mixing1, cohesion1 = _rw_tradeoff_at_step(P=P, stationary=ss, row_weights=w)
        # Guard against CC(1) = 1.0 exactly (would cause division by zero).
        # Small cc_denom is fine — cc_norm will be large if CC improves even
        # slightly, but the crossing criterion is still well-defined.
        cc_denom = max(1.0 - cohesion1, 1e-10)

        if verbose:
            print(f"[MPEClustering] Component {k} (size={n}): "
                  f"CC(1)={cohesion1:.4f}, HD(1)={mixing1:.4f}")

        record["mixing"].append(mixing1)
        record["cohesion"].append(cohesion1)
        record["hd_norm"].append(1.0)   # HD_norm(1) = HD(1)/HD(1) = 1
        record["cc_norm"].append(0.0)   # CC_norm(1) = 0 by definition
        record["walk_lengths"].append(1)

        # --- Steps 2 .. max_steps: always run the full curve ----------------
        P = P_orig @ P_orig  # P^2
        for step in range(2, max_steps + 1):
            mixing, cohesion = _rw_tradeoff_at_step(P=P, stationary=ss, row_weights=w)
            hd_norm = mixing / mixing1
            cc_norm = max((cohesion - cohesion1) / cc_denom, 0.0)
            record["mixing"].append(mixing)
            record["cohesion"].append(cohesion)
            record["hd_norm"].append(hd_norm)
            record["cc_norm"].append(cc_norm)
            record["walk_lengths"].append(step)
            P = P_orig @ P  # P^(t+1) = P^1 @ P^t

        # --- Find first crossing: first t where CC_norm(t) >= HD_norm(t) ---
        hd_arr = np.array(record["hd_norm"])
        cc_arr = np.array(record["cc_norm"])
        crossing = np.where(cc_arr >= hd_arr)[0]
        if len(crossing) > 0:
            best_step = int(crossing[0]) + 1  # +1 because step 1 is index 0
        else:
            best_step = 1  # no crossing — degenerate or flat
            if verbose:
                print(f"[MPEClustering] [WARNING] Component {k} (size={n}): "
                      f"no crossing found within max_steps={max_steps}. "
                      f"Using best_step=1. Consider increasing walktrap_max_steps.")

        record["best_step"] = best_step
        if verbose:
            print(f"[MPEClustering] Component {k} (size={n}): "
                  f"crossing at step={best_step} "
                  f"(cc_norm={record['cc_norm'][best_step-1]:.4f}, "
                  f"hd_norm={record['hd_norm'][best_step-1]:.4f})")
        step_analysis.append(record)

    # Component-size-weighted average, rounded to nearest integer
    total_size = sum(r["size"] for r in step_analysis)
    opt_steps = round(
        sum(r["best_step"] * r["size"] for r in step_analysis) / total_size
    )

    return opt_steps, step_analysis


# ---------------------------------------------------------------------------
# MPEClustering
# ---------------------------------------------------------------------------

class MPEClustering:
    """
    Multiview Prototype Embedding & Clustering.

    Fuses a CADJ-topology view, a high-D Euclidean view, and a low-D
    Euclidean view of prototype space into a single weighted graph, then
    partitions it with the Walktrap community-detection algorithm.

    Parameters
    ----------
    walktrap_n_steps : int or "auto", default "auto"
        Number of random-walk steps for the Walktrap algorithm.  If
        ``"auto"``, the optimal step count is found automatically by
        analysing the tradeoff between Markov-chain transience and cluster
        cohesion (see :func:`_find_walktrap_steps`).
    kernel_w_cadj : float, default 1.0
        Fusion weight applied to the CADJ view.
    kernel_w_high_d : float, default 1.0
        Fusion weight applied to the high-D Euclidean kernel view.
    kernel_w_low_d : float, default 1.0
        Fusion weight applied to the low-D Euclidean kernel view.
    kernel_power : float, default 2.0
        Exponent applied to each kernel view before fusion.  Values > 1
        sharpen inter-view agreement: pairs that are similar in all views
        receive disproportionately stronger edges, while disagreements are
        suppressed.  Motivated by the kernel combination results of Cortes,
        Mohri & Rostamizadeh (NeurIPS 2009).
    kernel_min_value : float, default 0.01
        Hard lower bound on kernel values retained after sparsification in
        :func:`CADJ_self_tuning_kernel`.  Passed through to both Euclidean
        kernel calls.
    remove_empty_protos : bool, default True
        If True, prototypes with empty receptive fields (no data mapped to
        them during VQ recall, i.e. zero CADJ row sums) are excluded from
        clustering.  Their entries in ``labels_`` are set to -1.
    kernel_min_nhbs : int, default 3
        Minimum number of neighbours guaranteed per prototype.  Does double
        duty: passed to :func:`CADJ_self_tuning_kernel` as ``min_nhbs``,
        where it controls (a) internal padding used only for sigma/
        connectivity computation in the Euclidean views, and (b) the
        per-row minimum kept after sparsification.  Does **not** pad
        ``kernel_cadj_`` itself — see ``kernel_support`` and the Step 2
        comment in :meth:`fit` for why CADJ's own density signal is left
        untouched.
    kernel_support : {"dense", "CADJ"}, default "dense"
        Sparsity pattern used when building the high-D and low-D Euclidean
        kernel views (``kernel_high_d_``, ``kernel_low_d_``) via
        :func:`CADJ_self_tuning_kernel`.  Does not affect ``kernel_cadj_``,
        which is always derived directly from (unpadded) CADJ.

        ``"dense"`` (default): kernel evaluated for all M×M prototype
        pairs.  Allows the Euclidean views to introduce edges that CADJ
        did not encode — the intended behavior for multi-view fusion,
        since each view should be able to contribute independent
        structural information.

        ``"CADJ"``: kernel evaluated only for pairs where CADJ[i,j] > 0.
        Cheaper for large M, but the Euclidean views become topologically
        redundant with the CADJ view (they reuse its sparsity pattern,
        only contributing different edge weights).  Useful for
        experimentation or when M is large enough that the dense pairwise
        computation becomes a bottleneck.
    walktrap_max_steps : int, default 15
        Number of steps evaluated when ``walktrap_n_steps="auto"``.
        The tradeoff curve is computed at every step from 1 to this value
        and the first crossing point is returned.
    verbose : bool, default False
        If True, print progress messages and warnings during :meth:`fit`.
        Useful for monitoring long-running fits or diagnosing unexpected
        results (e.g. many empty prototypes, fragmented graph, step count
        hitting the maximum).

    Attributes
    ----------
    labels_ : np.ndarray, shape (M,), dtype int
        Cluster membership for each prototype (0-indexed).  Empty prototypes
        (excluded when ``remove_empty_protos=True``) are assigned the sentinel
        value -1.  This follows the sklearn convention for excluded or
        unclustered points and keeps the array in integer dtype (numpy integer
        arrays have no NaN representation).  Use ``labels_[labels_ >= 0]`` or
        the ``active_labels_`` property to access only the clustered labels.
    active_labels_ : np.ndarray, shape (M_active,), dtype int
        Cluster labels for active (non-empty) prototypes only — equivalent to
        ``labels_[active_indices_]``.  No -1 sentinels.  Convenience accessor
        for downstream code that operates on the clustered subset.
    active_indices_ : np.ndarray, shape (M_active,), dtype int
        Indices into the full prototype array of the prototypes that were
        actually clustered (i.e. the non-empty ones).
    rf_sizes_ : np.ndarray, shape (M_active,), dtype float
        Receptive field sizes for active prototypes — CADJ row sums of the
        active sub-matrix.  Reflects how many data points each prototype
        attracted as first BMU.  Useful as point-size weights in visualisation.
    fused_graph_ : scipy.sparse.csr_matrix, shape (M_active, M_active)
        Symmetric weighted adjacency matrix of the fused graph, defined over
        the active (non-empty) prototypes only.
    n_steps_ : int
        Walktrap step count actually used.
    walktrap_result_ : igraph.VertexClustering
        Walktrap clustering result.  Gives access to modularity, membership,
        and cluster sizes.
    walktrap_dendrogram_ : igraph.VertexDendrogram
        Raw Walktrap dendrogram before the optimal cut.  Use
        ``walktrap_dendrogram_.as_clustering(k)`` to cut at any number of
        clusters k, or extract the full modularity curve across all k.
    step_analysis_ : list of dict or None
        Per-communicating-class tradeoff analysis from
        :func:`_find_walktrap_steps`.  Each dict contains ``mixing``,
        ``cohesion``, ``hd_norm``, ``cc_norm``, ``best_step``,
        ``size``, ``walk_lengths``, and ``indices`` keys.
        ``None`` if ``walktrap_n_steps`` was supplied manually.
    kernel_cadj_ : scipy.sparse.csr_matrix, shape (M_active, M_active)
        Row-max-normalised CADJ view kernel (before fusion and power).
        Values in [0, 1].
    kernel_high_d_ : scipy.sparse.csr_matrix, shape (M_active, M_active)
        Self-tuning Euclidean kernel on W (high-D view, before fusion and
        power).  Values in (0, 1].
    kernel_low_d_ : scipy.sparse.csr_matrix, shape (M_active, M_active)
        Self-tuning Euclidean kernel on Y (low-D view, before fusion and
        power).  Values in (0, 1].
    """

    def __init__(
        self,
        walktrap_n_steps: int | str = "auto",
        kernel_w_cadj: float = 1.0,
        kernel_w_high_d: float = 1.0,
        kernel_w_low_d: float = 1.0,
        kernel_power: float = 2.0,
        kernel_min_value: float = 0.01,
        remove_empty_protos: bool = True,
        kernel_min_nhbs: int = 3,
        kernel_support: str = "dense",
        walktrap_max_steps: int = 15,
        verbose: bool = False,
    ) -> None:
        if walktrap_n_steps != "auto" and (
            not isinstance(walktrap_n_steps, int) or walktrap_n_steps < 1
        ):
            raise ValueError(
                "walktrap_n_steps must be a positive integer or 'auto'; "
                f"got {walktrap_n_steps!r}"
            )
        if not (kernel_w_cadj >= 0 and kernel_w_high_d >= 0 and kernel_w_low_d >= 0):
            raise ValueError("Kernel fusion weights must be non-negative.")
        if (kernel_w_cadj + kernel_w_high_d + kernel_w_low_d) == 0:
            raise ValueError("At least one kernel fusion weight must be positive.")
        if kernel_support not in ("dense", "CADJ"):
            raise ValueError(
                f"kernel_support must be 'dense' or 'CADJ'; got {kernel_support!r}"
            )

        self.walktrap_n_steps = walktrap_n_steps
        self.kernel_w_cadj = kernel_w_cadj
        self.kernel_w_high_d = kernel_w_high_d
        self.kernel_w_low_d = kernel_w_low_d
        self.kernel_power = kernel_power
        self.kernel_min_value = kernel_min_value
        self.remove_empty_protos = remove_empty_protos
        self.kernel_min_nhbs = kernel_min_nhbs
        self.kernel_support = kernel_support
        self.walktrap_max_steps = walktrap_max_steps
        self.verbose = verbose

        # Results — populated by fit()
        # Clustering outputs
        self.labels_: np.ndarray | None = None
        self.active_indices_: np.ndarray | None = None
        self.rf_sizes_: np.ndarray | None = None
        # Graph and Walktrap
        self.fused_graph_: sp.csr_matrix | None = None
        self.n_steps_: int | None = None
        self.walktrap_result_: igraph.VertexClustering | None = None
        self.walktrap_dendrogram_: object | None = None  # igraph VertexDendrogram
        self.step_analysis_: list[dict] | None = None
        # Per-view kernels (before fusion; useful for diagnostics)
        self.kernel_cadj_: sp.csr_matrix | None = None
        self.kernel_high_d_: sp.csr_matrix | None = None
        self.kernel_low_d_: sp.csr_matrix | None = None

    def _vprint(self, msg: str, warning: bool = False) -> None:
        """Print msg if verbose=True.  Prefix warnings with [WARNING]."""
        if self.verbose:
            prefix = "[WARNING] " if warning else ""
            print(f"[MPEClustering] {prefix}{msg}")

    def fit(
        self,
        W: np.ndarray,
        Y: np.ndarray,
        CADJ: sp.spmatrix,
        CADJ_nhbs: list[list[int]] | None = None,
    ) -> "MPEClustering":
        """
        Fit the multiview prototype clustering.

        Parameters
        ----------
        W : np.ndarray, shape (M, d)
            Prototype matrix in the original (high-D) feature space.
        Y : np.ndarray, shape (M, d2)
            Low-D embedding of the prototypes (e.g. UMAP or PCA coordinates).
        CADJ : scipy.sparse matrix, shape (M, M)
            Asymmetric co-adjacency matrix from VQRecaller.
        CADJ_nhbs : list of list of int, length M, optional
            Precomputed CADJ neighbour index lists.  Derived from CADJ's CSR
            structure if not supplied.

        Returns
        -------
        self : MPEClustering
            The fitted instance.  Results are available as instance attributes.
        """
        M = W.shape[0]
        if Y.shape[0] != M or CADJ.shape != (M, M):
            raise ValueError(
                f"W, Y, and CADJ must all have M={M} prototypes; "
                f"got Y.shape={Y.shape}, CADJ.shape={CADJ.shape}"
            )

        self._vprint(f"Starting fit: M={M} prototypes, W.shape={W.shape}, Y.shape={Y.shape}")

        # --- Step 1: identify and optionally remove empty prototypes ----------
        # A prototype is "empty" if its CADJ row sum is zero — no data point
        # ever had it as first BMU, so its receptive field is empty.
        # rf_sizes[i] = CADJ row sum = total co-mapping events from prototype i.
        rf_sizes = np.asarray(CADJ.sum(axis=1)).ravel()  # shape (M,)
        is_empty = rf_sizes == 0
        n_empty = int(is_empty.sum())

        if n_empty > 0:
            self._vprint(
                f"Found {n_empty}/{M} empty prototypes "
                f"({100 * n_empty / M:.1f}%).",
                warning=n_empty > 0.1 * M,  # warn if >10% are empty
            )
        else:
            self._vprint("No empty prototypes found.")

        if self.remove_empty_protos and is_empty.any():
            active_indices = np.where(~is_empty)[0]
        else:
            active_indices = np.arange(M)

        M_active = len(active_indices)
        self._vprint(f"Clustering {M_active} active prototypes.")

        W_active = W[active_indices]
        Y_active = Y[active_indices]
        CADJ_active = CADJ[np.ix_(active_indices, active_indices)].tocsr()

        # Derive neighbour lists for the active sub-matrix
        CADJ_active_nhbs = [
            list(CADJ_active.indices[
                CADJ_active.indptr[i]:CADJ_active.indptr[i + 1]
            ])
            for i in range(M_active)
        ]
        CADJ_active_nhbs_size = np.diff(CADJ_active.indptr).astype(int)

        n_under = int((CADJ_active_nhbs_size < self.kernel_min_nhbs).sum())
        if n_under > 0:
            self._vprint(
                f"{n_under} prototype(s) have fewer than "
                f"{self.kernel_min_nhbs} CADJ neighbours after subsetting. "
                f"CADJ_self_tuning_kernel will pad internally for the "
                f"Euclidean views; kernel_cadj_ is left unpadded so genuine "
                f"low-density rows remain visible in the fused graph.",
                warning=n_under > 0.1 * M_active,
            )

        # --- Step 2: build kernel views ---------------------------------------
        # CADJ is passed unpadded here. CADJ_self_tuning_kernel pads internally
        # (as needed) to compute well-defined sigmas and guarantee minimum
        # connectivity for the Euclidean views — but that padding stays
        # internal to the kernel function and never touches kernel_cadj_
        # (built below from CADJ_active directly). This preserves CADJ's
        # density signal: a genuinely sparse/empty CADJ row honestly reflects
        # low data support for that prototype, and the Euclidean views already
        # guarantee baseline connectivity for the fused graph as a whole.
        #
        # kernel_support controls the sparsity pattern of the Euclidean
        # views: "dense" (default) lets them introduce edges that CADJ did
        # not encode, which is the intended behavior for multi-view fusion.
        # "CADJ" restricts them to CADJ's existing topology — cheaper but
        # more redundant with the CADJ view. See class docstring.
        self._vprint("Building high-D Euclidean kernel view...")
        K_W = CADJ_self_tuning_kernel(
            W=W_active,
            CADJ=CADJ_active,
            CADJ_nhbs=CADJ_active_nhbs,
            CADJ_nhbs_size=CADJ_active_nhbs_size,
            min_similarity=self.kernel_min_value,
            min_nhbs=self.kernel_min_nhbs,
            support=self.kernel_support,
        )
        self.kernel_high_d_ = K_W

        self._vprint("Building low-D Euclidean kernel view...")
        K_Y = CADJ_self_tuning_kernel(
            W=Y_active,
            CADJ=CADJ_active,
            CADJ_nhbs=CADJ_active_nhbs,
            CADJ_nhbs_size=CADJ_active_nhbs_size,
            min_similarity=self.kernel_min_value,
            min_nhbs=self.kernel_min_nhbs,
            support=self.kernel_support,
        )
        self.kernel_low_d_ = K_Y

        # --- Step 3: fuse views into a single graph -------------------------
        self._vprint("Fusing views into weighted graph...")
        # CADJ view: scale each row by its own max value, so the row's
        # strongest co-mapping becomes 1.0 and all others are in [0, 1).
        # This bounds CADJ_norm to the same [0, 1] range as the two
        # Euclidean kernels (K_W, K_Y) regardless of CADJ's raw count
        # magnitudes, so kernel_w_* weights are comparable across views.
        # Rows with zero density stay all-zero — no synthetic padding is
        # applied here (see Step 2 comment above).
        CADJ_float = CADJ_active.astype(np.float64)
        row_maxes = np.asarray(CADJ_float.max(axis=1).todense()).ravel()
        row_maxes[row_maxes == 0] = 1.0  # guard div-by-zero; row stays all-zero
        CADJ_norm = CADJ_float.multiply(1.0 / row_maxes[:, None])
        self.kernel_cadj_ = CADJ_norm.tocsr()

        A = (
            self.kernel_w_cadj   * CADJ_norm.power(self.kernel_power)
            + self.kernel_w_high_d * K_W.power(self.kernel_power)
            + self.kernel_w_low_d  * K_Y.power(self.kernel_power)
        )

        # Symmetrise: A + A^T makes the graph undirected
        A = A + A.T
        A = A.tocsr()
        self._vprint(
            f"Fused graph: {M_active} vertices, {A.nnz // 2} edges "
            f"(density={A.nnz / M_active**2:.3f})."
        )

        # --- Step 4: find optimal Walktrap steps if requested ---------------
        if self.walktrap_n_steps == "auto":
            self._vprint("Finding optimal Walktrap step count...")
            # Row sums of original (un-padded) CADJ as vertex weights
            cadj_row_sums = np.asarray(
                CADJ_active.sum(axis=1)
            ).ravel().astype(np.float64)
            n_steps, step_analysis = _find_walktrap_steps(
                A=A,
                row_weights=cadj_row_sums,
                max_steps=self.walktrap_max_steps,
                verbose=self.verbose,
            )
            self.step_analysis_ = step_analysis
            self._vprint(f"Selected n_steps={n_steps}.")
        else:
            n_steps = self.walktrap_n_steps
            self.step_analysis_ = None
            self._vprint(f"Using user-supplied n_steps={n_steps}.")

        # --- Step 6: build igraph object and run Walktrap -------------------
        self._vprint(f"Running Walktrap (steps={n_steps})...")
        A_coo = sp.triu(A, k=1).tocoo()
        g = igraph.Graph(
            n=M_active,
            edges=list(zip(A_coo.row.tolist(), A_coo.col.tolist())),
            edge_attrs={"weight": A_coo.data.tolist()},
            directed=False,
        )
        wt = g.community_walktrap(weights="weight", steps=n_steps)
        wt_clustering = wt.as_clustering()
        membership = np.asarray(wt_clustering.membership, dtype=int)
        n_clusters = int(membership.max()) + 1
        self._vprint(f"Walktrap complete: {n_clusters} clusters found.")

        # --- Step 7: store results ------------------------------------------
        labels = np.full(M, -1, dtype=int)  # -1 sentinel for empty protos
        labels[active_indices] = membership

        # Clustering outputs
        self.labels_ = labels
        self.active_indices_ = active_indices
        self.rf_sizes_ = np.asarray(CADJ_active.sum(axis=1)).ravel().astype(np.float64)

        # Graph and Walktrap
        self.fused_graph_ = A
        self.n_steps_ = n_steps
        self.walktrap_result_ = wt_clustering
        self.walktrap_dendrogram_ = wt  # raw VertexDendrogram for modularity curve

        self._vprint(
            f"Done. {n_clusters} clusters over {M_active} active prototypes "
            f"({n_empty} empty, labelled -1)."
        )
        return self

    def __repr__(self) -> str:
        if self.labels_ is not None:
            n_clusters = len(np.unique(self.active_labels_))
            status = (
                f"n_clusters={n_clusters}, "
                f"n_active={len(self.active_indices_)}, "
                f"n_steps={self.n_steps_}"
            )
        else:
            status = "not fitted"
        return (
            f"MPEClustering("
            f"walktrap_n_steps={self.walktrap_n_steps!r}, "
            f"kernel_w_cadj={self.kernel_w_cadj}, "
            f"kernel_w_high_d={self.kernel_w_high_d}, "
            f"kernel_w_low_d={self.kernel_w_low_d}, "
            f"kernel_power={self.kernel_power}, "
            f"kernel_support={self.kernel_support!r}"
            f") [{status}]"
        )

    @property
    def active_labels_(self) -> np.ndarray:
        """
        Cluster labels for active (non-empty) prototypes only.

        Equivalent to ``labels_[active_indices_]``.  Contains no -1 sentinels
        — every entry is a valid cluster index (0-indexed).  Raises
        ``RuntimeError`` if called before :meth:`fit`.
        """
        if self.labels_ is None:
            raise RuntimeError(
                "active_labels_ is not available before fit() is called."
            )
        return self.labels_[self.active_indices_]

    # -----------------------------------------------------------------------
    # Persistence — save / load
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save this fitted MPEClustering instance to an HDF5 file.

        All parameters, clustering results, kernel matrices, and step
        analysis are serialized.  igraph objects (walktrap_result_,
        walktrap_dendrogram_) are not stored but are recomputed identically
        from fused_graph_ and n_steps_ on :meth:`load` — Walktrap is
        deterministic.

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.

        Examples
        --------
        >>> mpec.fit(W, Y, CADJ)
        >>> mpec.save("mpec.h5")
        >>> mpec2 = MPEClustering.load("mpec.h5")
        >>> mpec2.plot_diagnostics(embed_coords=Y)
        """
        from soumap.io import MPECio
        MPECio(self).save(path)

    @classmethod
    def load(cls, path: str) -> "MPEClustering":
        """
        Load a MPEClustering from an HDF5 file created by :meth:`save`.

        All parameters and results are restored.  walktrap_result_ and
        walktrap_dendrogram_ are recomputed from the saved fused_graph_
        and n_steps_ (Walktrap is deterministic — results are identical).

        Parameters
        ----------
        path : str
            Path to an HDF5 file previously created by :meth:`save`.

        Returns
        -------
        MPEClustering
            Fully reconstructed instance with all plot and diagnostic
            functionality available immediately.

        Examples
        --------
        >>> mpec = MPEClustering.load("mpec.h5")
        >>> mpec.n_steps_
        4
        >>> mpec.plot_diagnostics(embed_coords=Y)
        """
        from soumap.io import MPECio
        return MPECio.load(path)

    def _recompute_walktrap(self) -> None:
        """
        Recompute walktrap_result_ and walktrap_dendrogram_ from
        fused_graph_ and n_steps_.

        Called automatically by :meth:`load` after deserializing the
        saved fused graph.  Also useful if you manually modify fused_graph_
        or n_steps_ and want to re-derive the igraph clustering objects.

        Walktrap is fully deterministic — given the same graph and step
        count, this always produces bit-for-bit identical results.

        Raises
        ------
        RuntimeError
            If fused_graph_ or n_steps_ are not available (i.e. fit()
            has not been called or the file was not saved after fitting).
        """
        if self.fused_graph_ is None or self.n_steps_ is None:
            raise RuntimeError(
                "_recompute_walktrap() requires fused_graph_ and n_steps_ "
                "to be set. Call fit() or load() first."
            )
        M_active = self.fused_graph_.shape[0]
        A_coo = sp.triu(self.fused_graph_, k=1).tocoo()
        g = igraph.Graph(
            n=M_active,
            edges=list(zip(A_coo.row.tolist(), A_coo.col.tolist())),
            edge_attrs={"weight": A_coo.data.tolist()},
            directed=False,
        )
        wt = g.community_walktrap(weights="weight", steps=self.n_steps_)
        self.walktrap_dendrogram_ = wt
        self.walktrap_result_     = wt.as_clustering()

    # -----------------------------------------------------------------------
    # Private plot builders
    # -----------------------------------------------------------------------

    def _check_fitted(self, method: str) -> None:
        """Raise RuntimeError if fit() has not been called yet."""
        if self.labels_ is None:
            raise RuntimeError(
                f"{method} cannot be called before fit()."
            )

    def _plot_step_tradeoff(self, max_components: int = 3):
        """
        Plot 1 — Transience / Cluster Cohesion vs. Walktrap step.

        One facet per communicating class (up to ``max_components`` largest).
        A vertical line marks the selected best_step (first crossing) per
        component.  Returns a placeholder ggplot if step analysis is not
        available (i.e. walktrap_n_steps was supplied manually).
        """
        if self.step_analysis_ is None:
            # Placeholder when steps were user-supplied
            df = pd.DataFrame({"x": [0.5], "y": [0.5],
                               "label": [f"No step analysis available\nUser-supplied steps = {self.n_steps_}"]})
            return (
                ggplot(df, aes(x="x", y="y"))
                + geom_text(aes(label="label"), size=9, color="#666666")
                + labs(title="Step tradeoff analysis")
                + theme_minimal_bold()
                + theme(
                    axis_text=element_blank(),
                    axis_title=element_blank(),
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),
                )
            )

        # Select up to max_components largest non-trivial components
        records = [r for r in self.step_analysis_ if r["size"] > 2]
        records = sorted(records, key=lambda r: r["size"], reverse=True)[:max_components]

        if not records:
            df = pd.DataFrame({"x": [0.5], "y": [0.5],
                               "label": ["No components with > 2 vertices"]})
            return (
                ggplot(df, aes(x="x", y="y"))
                + geom_text(aes(label="label"), size=9, color="#666666")
                + labs(title="Walktrap Step Analysis")
                + theme_minimal_bold()
                + theme(
                    axis_text=element_blank(),
                    axis_title=element_blank(),
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),
                )
            )

        rows = []
        vlines = []
        for k, rec in enumerate(records):
            label = f"Component {k} (n={rec['size']})"
            n_steps_rec = len(rec.get("hd_norm", []))
            walk_lengths = rec.get("walk_lengths", list(range(1, n_steps_rec + 1)))
            hd_vals = rec.get("hd_norm", [])
            cc_vals = rec.get("cc_norm", [])
            for step_idx in range(n_steps_rec):
                if step_idx < len(hd_vals):
                    rows.append({"component": label, "step": walk_lengths[step_idx],
                                 "metric": "Transience (TR)", "value": hd_vals[step_idx]})
                if step_idx < len(cc_vals):
                    rows.append({"component": label, "step": walk_lengths[step_idx],
                                 "metric": "Cluster Cohesion (CC)", "value": cc_vals[step_idx]})
            best_walk = walk_lengths[rec["best_step"] - 1]
            vlines.append({"component": label, "best_step": best_walk})

        df = pd.DataFrame(rows)
        vline_df = pd.DataFrame(vlines)

        metric_colors = {
            "Transience (TR)":        "#E05C5C",
            "Cluster Cohesion (CC)":  "#5C8AE0",
        }

        max_walk = max(rec.get("walk_lengths", [1])[-1] for rec in records)
        step_breaks = list(range(1, max_walk + 1))

        return (
            ggplot(df, aes(x="step", y="value", color="metric", group="metric"))
            + geom_line(size=0.8)
            + geom_point(size=1.5)
            + geom_vline(
                data=vline_df,
                mapping=aes(xintercept="best_step"),
                linetype="dashed", color="#666666", size=0.6,
            )
            + scale_color_manual(
                values=list(metric_colors.values()),
                breaks=list(metric_colors.keys()),
                name="Metric",
            )
            + scale_x_continuous(breaks=step_breaks)
            + facet_wrap("~component", ncol=1)
            + labs(
                title="Walktrap Step Analysis",
                subtitle=f"Selected step: {self.n_steps_}",
                caption="Dashed line = first crossing (CC >= TR)",
                x="Step", y="Normalized Value",
            )
            + theme_minimal_bold()
            + theme(
                legend_position="bottom",
                legend_direction="horizontal",
            )
        )

    def _plot_edge_distributions(self):
        """
        Plot 2 — Nonzero edge weight distributions for each kernel view.

        Overlaid density curves for kernel_cadj_, kernel_high_d_, and
        kernel_low_d_, showing what each view contributes before fusion.
        """
        view_map = {
            "CADJ": self.kernel_cadj_,
            "High-D": self.kernel_high_d_,
            "Low-D": self.kernel_low_d_,
        }
        view_colors = {"CADJ": "#E05C5C", "High-D": "#5C8AE0", "Low-D": "#F5A623"}

        frames = []
        for name, K in view_map.items():
            vals = sp.triu(K, k=1).data  # numpy array of nonzero upper-triangle weights
            frames.append(pd.DataFrame({
                "view": np.repeat(name, len(vals)),
                "weight": vals.astype(float),
            }))
        df = pd.concat(frames, ignore_index=True)

        return (
            ggplot(df, aes(x="weight", color="view", fill="view"))
            + geom_density(alpha=0.15, size=0.8)
            + scale_color_manual(values=view_colors, name="View")
            + scale_fill_manual(values=view_colors, name="View")
            + labs(
                title="Kernel edge weight distributions",
                x="Edge weight", y="Density",
            )
            + theme_minimal_bold()
        )

    def _plot_modularity_curve(self):
        """
        Plot 3 — Walktrap modularity vs. number of clusters.

        Extracts the modularity curve from the stored dendrogram by cutting
        at every valid k.  The valid range is [n_components, M_active] where
        n_components is the number of weakly connected components in the fused
        graph — Walktrap cannot merge vertices across components, so k < n_components
        is not reachable.  Marks the chosen partition with a vertical line and
        annotates the peak.
        """
        M_active = len(self.active_indices_)

        # Determine number of components from step_analysis if available,
        # otherwise infer from the dendrogram merge matrix size:
        # n_merges = M_active - n_components  =>  n_components = M_active - n_merges
        n_merges = len(self.walktrap_dendrogram_._merges)
        n_components = M_active - n_merges

        # Valid k range: n_components (one cluster per component) to M_active
        ks = list(range(n_components, M_active + 1))
        mods = [
            self.walktrap_dendrogram_.as_clustering(k).modularity
            for k in ks
        ]

        chosen_k = len(np.unique(self.active_labels_))
        peak_k = ks[int(np.argmax(mods))]

        df = pd.DataFrame({"k": ks, "modularity": mods})
        mod_range = max(mods) - min(mods) if max(mods) != min(mods) else 0.01
        nudge = mod_range * 0.04
        peak_df = pd.DataFrame({
            "k": [peak_k],
            "modularity": [max(mods)],
            "label_y": [max(mods) + nudge],
            "label": [str(peak_k)],
        })

        if peak_k == chosen_k:
            subtitle = f"k = {chosen_k}  |  valid range [{n_components}, {M_active}]"
        else:
            subtitle = (
                f"chosen k = {chosen_k}, peak k = {peak_k}"
                f"  |  valid range [{n_components}, {M_active}]"
            )

        return (
            ggplot(df, aes(x="k", y="modularity"))
            + geom_vline(
                xintercept=chosen_k,
                linetype="dashed", color="#666666", size=0.6,
            )
            + geom_line(color="#5C8AE0", size=0.8)
            + geom_point(data=peak_df, mapping=aes(x="k", y="modularity"),
                         color="#E05C5C", size=3)
            + geom_text(
                data=peak_df,
                mapping=aes(x="k", y="label_y", label="label"),
                size=8, color="#E05C5C",
            )
            + labs(
                title="Modularity curve",
                subtitle=subtitle,
                x="Number of clusters (k)", y="Modularity",
            )
            + theme_minimal_bold()
        )

    def _plot_cluster_sizes(self, ctab: dict):
        """
        Plot 4 — Cluster size bar chart, sorted descending.

        Bars are colored using the shared ``ctab`` so colors match the
        embedding scatter plot.  A horizontal line marks the mean cluster size.
        """
        active_labels = self.active_labels_
        unique, counts = np.unique(active_labels, return_counts=True)
        order = np.argsort(-counts)
        unique = unique[order]
        counts = counts[order]

        df = pd.DataFrame({
            "cluster": pd.Categorical(
                [str(u) for u in unique],
                categories=[str(u) for u in unique],  # preserve sort order
            ),
            "count": counts,
        })
        # Build fill values keyed by cluster label (natural ggplot syntax)
        fill_values = {str(u): ctab.get(str(u), "#888888") for u in unique}

        mean_size = float(counts.mean())

        return (
            ggplot(df, aes(x="cluster", y="count", fill="cluster"))
            + geom_col()
            + geom_hline(
                yintercept=mean_size,
                linetype="dashed", color="#666666", size=0.6,
            )
            + scale_fill_manual(values=fill_values)
            + guides(fill="none")
            + labs(
                title="Cluster sizes",
                subtitle=f"Dashed line = mean ({mean_size:.1f} prototypes)",
                x="Cluster", y="Prototypes",
            )
            + theme_minimal_bold()
        )

    def _plot_embedding(
        self,
        embed_coords: np.ndarray | None,
        ctab: dict,
        show_graph: bool,
        point_size: float,
    ):
        """
        Plot 5 — Low-D embedding scatter colored by cluster label ("MPEC Clustering").

        Delegates to ``vis_embedding_discrete`` from ``gtsom.vis_tools``.
        Empty prototypes (label = -1) are converted to None so they are
        dropped from the plot.  Point sizes are scaled by ``rf_sizes_``
        to reflect data density per prototype.

        Returns a placeholder if ``embed_coords`` is None.
        """
        if embed_coords is None:
            df = pd.DataFrame({"x": [0.5], "y": [0.5],
                               "label": ["No embedding coordinates supplied\nPass embed_coords to plot_diagnostics()"]})
            return (
                ggplot(df, aes(x="x", y="y"))
                + geom_text(aes(label="label"), size=9, color="#666666")
                + labs(title="MPEC Clustering")
                + theme_minimal_bold()
                + theme(
                    axis_text=element_blank(),
                    axis_title=element_blank(),
                    panel_grid_major=element_blank(),
                    panel_grid_minor=element_blank(),
                )
            )

        coords = np.asarray(embed_coords)
        if coords.ndim != 2 or coords.shape[1] < 2:
            raise ValueError(
                f"embed_coords must have shape (M, 2) or (M, d) with d >= 2; "
                f"got shape {coords.shape}."
            )
        if coords.shape[0] != (self.active_indices_.max() + 1):
            raise ValueError(
                f"embed_coords must cover all M prototypes (rows = M); "
                f"got {coords.shape[0]} rows but M inferred as "
                f"{self.active_indices_.max() + 1}.  "
                f"Pass the full (M, d) coordinate matrix; active subsetting "
                f"is handled internally."
            )
        coords2 = coords[:, :2]
        subtitle = "First 2 dimensions shown" if coords.shape[1] > 2 else None

        # Subset to active prototypes
        coords_active = coords2[self.active_indices_]

        # Convert -1 labels to None so vis_embedding_discrete drops them
        z = [
            int(lb) if lb >= 0 else None
            for lb in self.active_labels_
        ]

        graph_arg = self.fused_graph_ if show_graph else None

        return vis_embedding_discrete(
            x=coords_active[:, 0],
            y=coords_active[:, 1],
            z=z,
            ctab=ctab,
            point_size=point_size,
            point_size_wts=self.rf_sizes_,
            point_size_wts_trans="sqrt",
            graph=graph_arg,
            edge_color="#CDC0B0",  # matplotlib hex equiv. of R's "antiquewhite3"
                                    # (vis_tools.py's default isn't matplotlib-valid)
            title="MPEC Clustering",
            subtitle=subtitle,
            xlab="SOM$_1$", ylab="SOM$_2$",
            legend_pos="right",
            legend_title="Cluster",
        )

    def _plot_view_agreement(self):
        """
        Plot 6 — Frobenius cosine similarity between kernel views.

        For each pair of view kernels, computes the cosine similarity
        in the Frobenius inner product sense::

            sim(A, B) = <A, B>_F / (||A||_F * ||B||_F)
                      = sum(A * B) / sqrt(sum(A^2) * sum(B^2))

        Evaluated on the full (M_active, M_active) space including structural
        zeros, so all pairs are comparable regardless of differing sparsity
        patterns.  Efficient on sparse matrices: uses ``.multiply()`` for
        element-wise product and ``.power(2).sum()`` for squared norms.

        Displayed as a 3×3 symmetric heatmap with numeric annotations.
        """
        views = {
            "CADJ": self.kernel_cadj_,
            "High-D": self.kernel_high_d_,
            "Low-D": self.kernel_low_d_,
        }
        names = list(views.keys())
        kernels = list(views.values())

        def frob_cosine(A, B):
            dot = float(A.multiply(B).sum())
            norm_a = float(np.sqrt(A.power(2).sum()))
            norm_b = float(np.sqrt(B.power(2).sum()))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        rows = []
        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                sim = frob_cosine(kernels[i], kernels[j])
                rows.append({"view_x": ni, "view_y": nj, "similarity": sim})

        df = pd.DataFrame(rows)
        df["label"] = df["similarity"].map(lambda v: f"{v:.2f}")
        # Dark text on light tiles, white text on dark tiles
        df["text_color"] = df["similarity"].map(
            lambda v: "white" if v > 0.6 else "#333333"
        )
        # Fix axis ordering
        df["view_x"] = pd.Categorical(df["view_x"], categories=names)
        df["view_y"] = pd.Categorical(df["view_y"], categories=list(reversed(names)))

        return (
            ggplot(df, aes(x="view_x", y="view_y", fill="similarity"))
            + geom_tile(color="white", size=0.5)
            + geom_text(aes(label="label", color="text_color"), size=9)
            + scale_color_identity()
            + scale_fill_gradient(
                low="#D6E8F5",    # light blue (low similarity)
                high="#08306B",   # dark navy (high similarity, matches Blues cmap)
                limits=(0, 1),
                name="Similarity",
            )
            + labs(
                title="Kernel Agreement",
                subtitle="Frobenius cosine similarity",
                x="", y="",
            )
            + theme_minimal_bold()
            + theme(
                panel_grid_major=element_blank(),
                panel_grid_minor=element_blank(),
            )
        )

    # -----------------------------------------------------------------------
    # Public diagnostic plot
    # -----------------------------------------------------------------------

    def plot_diagnostics(
        self,
        embed_coords: np.ndarray | None = None,
        show_graph: bool = False,
        which: list | str = "all",
        max_components: int = 3,
        point_size: float = 2.0,
        figsize: tuple = (6, 5),
        save_dir: str | None = None,
        save_format: str = "pdf",
    ) -> dict:
        """
        Produce diagnostic plots for the fitted clustering.

        Each plot is rendered independently as a separate figure, displayed
        on screen, and optionally saved to disk.  This avoids any dependency
        on panel-assembly libraries (e.g. patchworklib) and works with any
        plotnine version.

        Parameters
        ----------
        embed_coords : np.ndarray, shape (M, d), optional
            Coordinates for the embedding scatter.  First 2 columns are used
            if d > 2.  If None, the embedding plot is skipped.
        show_graph : bool, default False
            If True, overlay ``fused_graph_`` edges on the embedding scatter.
        which : list of str or "all", default "all"
            Subset of plots to produce.  Valid names:

            ``"step_tradeoff"``    — mixing / cohesion / score vs. step.
            ``"edge_distributions"`` — kernel edge weight distributions.
            ``"modularity"``       — Walktrap modularity curve.
            ``"cluster_sizes"``    — cluster size bar chart.
            ``"embedding"``        — prototype embedding scatter.
            ``"view_agreement"``   — kernel view agreement heatmap.

        max_components : int, default 3
            Maximum number of communicating classes shown in the step
            tradeoff plot.
        point_size : float, default 2.0
            Base point size for the embedding scatter.
        figsize : tuple, default (6, 5)
            (width, height) in inches for each individual figure.
        save_dir : str or None
            Directory in which to save figures.  If None, figures are shown
            on screen only.  Files are named
            ``mpec_<plot_name>.<save_format>``.
        save_format : str, default "pdf"
            File format passed to plotnine's ``save()`` method.  Any format
            supported by matplotlib is valid (e.g. ``"pdf"``, ``"svg"``,
            ``"png"``).

        Returns
        -------
        plots : dict
            ``{name: ggplot}`` for every plot that was produced.  Allows
            the caller to further customise or save individual plots.
        """
        self._check_fitted("plot_diagnostics()")

        all_names = [
            "step_tradeoff", "edge_distributions", "modularity",
            "cluster_sizes", "embedding", "view_agreement",
        ]
        if which == "all":
            which = list(all_names)
        invalid = set(which) - set(all_names)
        if invalid:
            raise ValueError(
                f"Unknown plot name(s): {invalid}. "
                f"Valid options: {all_names}"
            )

        # Skip embedding if no coords supplied (rather than placeholder)
        if embed_coords is None and "embedding" in which:
            print("[MPEClustering] embed_coords not supplied — skipping embedding plot.")
            which = [n for n in which if n != "embedding"]

        # Build shared color table so all plots use consistent cluster colors
        active_labels_str = [str(lb) for lb in self.active_labels_]
        ctab = build_ctab(active_labels_str)

        plot_map = {
            "step_tradeoff":      lambda: self._plot_step_tradeoff(max_components),
            "edge_distributions": lambda: self._plot_edge_distributions(),
            "modularity":         lambda: self._plot_modularity_curve(),
            "cluster_sizes":      lambda: self._plot_cluster_sizes(ctab),
            "embedding":          lambda: self._plot_embedding(
                                      embed_coords, ctab, show_graph, point_size),
            "view_agreement":     lambda: self._plot_view_agreement(),
        }

        plots = {}
        for name in all_names:
            if name not in which:
                continue

            self._vprint(f"Rendering: {name}...")
            gg = plot_map[name]()
            plots[name] = gg

            if save_dir is not None:
                import os
                path = os.path.join(save_dir, f"mpec_{name}.{save_format}")
                gg.save(path, width=figsize[0], height=figsize[1], verbose=False)
                self._vprint(f"  Saved: {path}")

            # Show on screen
            gg.draw()
            plt.show()

        return plots