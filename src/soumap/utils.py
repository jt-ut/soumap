"""
utils.py — General-purpose utilities for the soumap package.
"""

import numpy as np


def scale_coords(coords, embedding_range):
    """
    Rescale embedding coordinates so each axis spans embedding_range.

    Each column of coords is independently rescaled via min-max
    normalization to fit within [embedding_range[0], embedding_range[1]].
    This anchors the coordinate system to a fixed scale after each UMAP
    update, giving min_dist and spread an interpretable reference frame.

    Parameters
    ----------
    coords : np.ndarray, shape (M, 2)
        Embedding coordinates to rescale.
    embedding_range : tuple of float
        Target (min, max) for each axis, e.g. (-3.0, 3.0).

    Returns
    -------
    scaled : np.ndarray, shape (M, 2)
        Rescaled coordinates. A copy is always returned — the input is
        never mutated.
    """
    lo, hi = embedding_range
    scaled = coords.copy().astype(float)
    for dim in range(scaled.shape[1]):
        col_min = scaled[:, dim].min()
        col_max = scaled[:, dim].max()
        col_range = col_max - col_min
        if col_range == 0:
            # All points collapsed to same coordinate on this axis —
            # place them at the midpoint of embedding_range.
            scaled[:, dim] = (lo + hi) / 2
        else:
            scaled[:, dim] = (scaled[:, dim] - col_min) / col_range
            scaled[:, dim] = scaled[:, dim] * (hi - lo) + lo
    return scaled


def pad_CADJ(CADJ, CADJ_nhbs, CADJ_nhbs_size, W, min_nhbs=3, fill_val=0.5):
    """
    Pad a CADJ matrix so that every row has at least min_nhbs nonzero entries.

    Takes the current CADJ, CADJ_nhbs, and CADJ_nhbs_size directly from
    the VQRecaller recall object (gtsom.recaller.CADJ, .CADJ_nhbs,
    .CADJ_nhbs_size), makes copies of all three, and modifies only the
    rows that fall below min_nhbs. Rows that already satisfy the threshold
    are never touched.

    For any prototype i with fewer than min_nhbs neighbors, finds the
    geometrically closest unconnected prototypes (by Euclidean distance
    in W-space) and adds synthetic directed edges with weight fill_val.
    Padding is one-directional — only row i is modified, preserving the
    asymmetric nature of CADJ.

    Parameters
    ----------
    CADJ : scipy.sparse matrix, shape (M, M)
        Asymmetric co-adjacency matrix from VQRecaller. A copy is made
        internally so the original is never mutated.
    CADJ_nhbs : list of list of int, length M
        Neighbor index lists for each row of CADJ, as stored on
        VQRecaller. A copy is made internally.
    CADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbors per prototype, as stored on VQRecaller.
        A copy is made internally.
    W : np.ndarray, shape (M, d)
        Current prototype matrix. Used to find geometrically nearest
        candidates for synthetic edges.
    min_nhbs : int, default 3
        Minimum number of nonzero entries required per row.
    fill_val : float, default 0.5
        Weight assigned to synthetic padding edges.

    Returns
    -------
    PCADJ : scipy.sparse.csr_matrix, shape (M, M)
        Padded CADJ matrix. Every row has at least min_nhbs nonzero
        entries.
    PCADJ_nhbs : list of list of int, length M
        Neighbor index lists for each row of PCADJ. No element is
        empty — every prototype has at least min_nhbs neighbors.
    PCADJ_nhbs_size : np.ndarray, shape (M,), dtype int
        Number of neighbors for each prototype in PCADJ.
    """
    PCADJ = CADJ.astype(float).copy().tolil()
    PCADJ_nhbs = [nhbs.copy() for nhbs in CADJ_nhbs]
    PCADJ_nhbs_size = CADJ_nhbs_size.copy()

    M = W.shape[0]

    for i in range(M):
        n_existing = PCADJ_nhbs_size[i]

        if n_existing >= min_nhbs:
            continue

        n_needed = min_nhbs - n_existing

        # Exclude self and already-connected prototypes from candidates
        existing_cols = set(PCADJ_nhbs[i])
        dists = np.linalg.norm(W - W[i], axis=1)
        dists[i] = np.inf
        dists[list(existing_cols)] = np.inf

        # argpartition is O(M) vs argsort's O(M log M)
        candidates = np.argpartition(dists, n_needed)[:n_needed]

        for j in candidates:
            if np.isfinite(dists[j]):
                PCADJ[i, j] = fill_val
                PCADJ_nhbs[i].append(j)
                PCADJ_nhbs_size[i] += 1

    PCADJ = PCADJ.tocsr()

    return PCADJ, PCADJ_nhbs, PCADJ_nhbs_size