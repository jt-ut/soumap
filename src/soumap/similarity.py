"""
similarity.py — Construction of the fuzzy similarity matrix FSIM.

FSIM is an (M, M) sparse symmetric matrix whose entry (i, j) represents
the fuzzy membership strength of the 1-simplex between prototypes i and j.
It is constructed from the padded co-adjacency matrix PCADJ (produced by
pad_CADJ in utils.py) and used as the input graph to the UMAP layout
optimizer.

Construction pipeline:
    1. Row-normalize PCADJ (local term) + globally normalize (global term)
    2. Add a self-tuning Euclidean kernel term, gated by PCADJ sparsity
    3. Row-normalize the combined matrix
    4. Symmetrize via fuzzy set union: S + S.T - S * S.T (elementwise)
    5. Zero the diagonal

PCADJ padding is handled upstream by pad_CADJ() in utils.py before
build_FSIM() is called, so this module assumes all rows of PCADJ already
have at least min_nhbs nonzero entries.
"""

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_FSIM(W, PCADJ, PCADJ_nhbs, q_thresh=0.95):
    """
    Build the fuzzy similarity matrix FSIM from prototype positions and
    the padded co-adjacency matrix PCADJ.

    PCADJ and PCADJ_nhbs must already be padded (via pad_CADJ in utils.py)
    so that every row has at least min_nhbs nonzero entries before this
    function is called.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
        Current prototype matrix.
    PCADJ : scipy.sparse matrix, shape (M, M)
        Padded asymmetric co-adjacency matrix. Every row must have at
        least one nonzero entry. Not mutated.
    PCADJ_nhbs : list of list of int, length M
        Neighbor index lists for each row of PCADJ. No element may be
        empty. Used to compute per-prototype bandwidths for the
        self-tuning Euclidean kernel.
    q_thresh : float, default 0.95
        Quantile of neighbor distances used as the local bandwidth in the
        self-tuning Euclidean kernel.

    Returns
    -------
    FSIM : scipy.sparse.csr_matrix, shape (M, M)
        Symmetric fuzzy similarity matrix with zeros on the diagonal.
        Suitable for direct input to the UMAP layout optimizer.
    """
    PCADJ_f = PCADJ.astype(float)

    # ------------------------------------------------------------------
    # Step 1: PCADJ-based similarity terms (local + global)
    # ------------------------------------------------------------------
    FSIM = _cadj_similarity(PCADJ_f)

    # ------------------------------------------------------------------
    # Step 2: Self-tuning Euclidean kernel, gated by PCADJ sparsity
    # ------------------------------------------------------------------
    FSIM = FSIM + _zelnik_kernel(W, PCADJ_f, PCADJ_nhbs, q_thresh=q_thresh)

    # ------------------------------------------------------------------
    # Step 3: Row-normalize combined matrix
    # ------------------------------------------------------------------
    FSIM = _row_normalize(FSIM)

    # ------------------------------------------------------------------
    # Step 4: Symmetrize via fuzzy set union: S + S.T - S * S.T
    # ------------------------------------------------------------------
    FSIM = _fuzzy_union(FSIM)

    # ------------------------------------------------------------------
    # Step 5: Zero diagonal
    # ------------------------------------------------------------------
    FSIM.setdiag(0)
    FSIM.eliminate_zeros()

    return FSIM.tocsr()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _cadj_similarity(PCADJ):
    """
    Compute local + global PCADJ-based similarity.

    Local term:  PCADJ / row_max  — each row scaled to [0, 1] independently
    Global term: PCADJ / global_max  — all entries scaled by the same factor

    Returns a sparse matrix of the same shape as PCADJ.
    """
    PCADJ_csr = PCADJ.tocsr()

    # Local: divide each row by its own maximum
    row_maxes = np.array(PCADJ_csr.max(axis=1).todense()).ravel()
    row_maxes[row_maxes == 0] = 1.0
    local = _scale_rows(PCADJ_csr, 1.0 / row_maxes)

    # Global: divide all entries by the single largest value
    global_max = PCADJ_csr.max()
    if global_max == 0:
        global_max = 1.0
    global_ = PCADJ_csr / global_max

    return local + global_


def _zelnik_kernel(W, PCADJ, PCADJ_nhbs, q_thresh=0.95):
    """
    Self-tuning Euclidean kernel (Zelnik-Manor & Perona 2004), gated by
    the PCADJ sparsity pattern.

    For each connected pair (i, j) in PCADJ, computes a Gaussian kernel
    value using a self-tuning bandwidth: the bandwidth for prototype i is
    the q_thresh-quantile of Euclidean distances from W[i] to its PCADJ
    neighbors (PCADJ_nhbs[i]). Only edges already present in PCADJ are
    computed — the kernel does not introduce new edges.

    Parameters
    ----------
    W : np.ndarray, shape (M, d)
    PCADJ : scipy.sparse matrix, shape (M, M)
    PCADJ_nhbs : list of list of int
        PCADJ_nhbs[i] = column indices of nonzero entries in row i.
        No element may be empty.
    q_thresh : float, default 0.95

    Returns
    -------
    K : scipy.sparse.csr_matrix, shape (M, M)
    """
    M = W.shape[0]

    # Compute per-prototype bandwidth from PCADJ neighborhood distances
    sigma = np.zeros(M)
    for i in range(M):
        nhbs = PCADJ_nhbs[i]
        dists = np.linalg.norm(W[nhbs] - W[i], axis=1)
        sigma[i] = np.quantile(dists, q_thresh) if len(dists) > 1 else dists[0]
        if sigma[i] == 0:
            sigma[i] = 1.0

    # Build kernel values for all nonzero PCADJ entries
    PCADJ_coo = PCADJ.tocoo()
    rows = PCADJ_coo.row
    cols = PCADJ_coo.col

    if len(rows) == 0:
        return csr_matrix((M, M), dtype=float)

    dists_ij = np.linalg.norm(W[rows] - W[cols], axis=1)
    # Self-tuning: bandwidth is geometric mean of the two endpoint bandwidths
    sigma_ij = sigma[rows] * sigma[cols]
    kernel_vals = np.exp(-dists_ij ** 2 / sigma_ij)

    return csr_matrix((kernel_vals, (rows, cols)), shape=(M, M))


def _row_normalize(S):
    """
    Divide each row of sparse matrix S by its row maximum.
    Rows with all-zero entries are left as zero.
    """
    S_csr = S.tocsr()
    row_maxes = np.array(S_csr.max(axis=1).todense()).ravel()
    row_maxes[row_maxes == 0] = 1.0
    return _scale_rows(S_csr, 1.0 / row_maxes)


def _fuzzy_union(S):
    """
    Symmetrize S via fuzzy set union: S + S.T - S * S.T (elementwise).

    This is the probabilistic OR operation used by UMAP to merge a
    directed similarity graph into a symmetric fuzzy simplicial set.
    An edge (i, j) is retained if either i considers j a neighbor OR
    j considers i a neighbor.

    Parameters
    ----------
    S : scipy.sparse matrix, shape (M, M)

    Returns
    -------
    scipy.sparse.csr_matrix, shape (M, M)
    """
    S = S.tocsr()
    ST = S.T.tocsr()
    return (S + ST - S.multiply(ST)).tocsr()


def _scale_rows(S_csr, scale_vec):
    """
    Multiply each row i of sparse CSR matrix S by scale_vec[i].
    Returns a new csr_matrix.
    """
    S_coo = S_csr.tocoo()
    scaled_data = S_coo.data * scale_vec[S_coo.row]
    return csr_matrix(
        (scaled_data, (S_coo.row, S_coo.col)),
        shape=S_csr.shape
    )