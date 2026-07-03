"""
soumap.py — Self-Organizing UMAP (SOUMAP).

SOUMAP alternates between two learning steps:

  SOM step  — prototype vectors W are updated via batch SOM learning.
              The output-space topology governing neighbour updates is
              derived from Delaunay triangulation of the current
              embedding coordinates.

  UMAP step — a fuzzy similarity matrix FSIM is constructed from the
              padded co-adjacency matrix PCADJ, and the UMAP layout
              optimizer updates the embedding coordinates warm-started
              from their current state. The new coordinates replace the
              SOM topology for the next SOM step.

The SOM machinery is delegated entirely to a GTSOM instance. SOUMAP
drives GTSOM by calling gtsom.fit() in chunks of umap_params.update_freq
epochs, injecting new embedding coordinates between chunks via
gtsom.update_embedding().

References
----------
Taylor, J. & Offner, S. (2024). A Self-Organizing UMAP for Clustering.
International Workshop on Self-Organizing Maps, Learning Vector
Quantization & Beyond, pp. 63-73. Springer.
"""

import numpy as np
from dataclasses import dataclass
import umap.umap_ as umap_
from gtsom import GTSOM
from gtsom.utils import ExponentialAnneal
from .utils import pad_CADJ, scale_coords
from .similarity import build_FSIM
from .layout import optimize_layout


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------

@dataclass
class SOMParams:
    """
    Parameters governing SOM prototype learning.

    Attributes
    ----------
    rho_0 : float
        Initial neighbourhood bandwidth.
    rho_f : float
        Final neighbourhood bandwidth.
    nbr_topo_alpha_0 : float
        Initial topology-blending weight passed to GTSOM. Controls the
        mix between Delaunay-lattice-based and CONN-based neighbourhood
        activations in the SOM update rule:

            nbr_W = nbr_topo_alpha * H_lat + (1 - nbr_topo_alpha) * H_CONN

        0.0 = pure CONN (Neural Gas style, data-space neighbours).
        1.0 = pure lattice (classical SOM).
        0.5 = equal blend (default).
    nbr_topo_alpha_f : float
        Final topology-blending weight. Annealing from a low value
        (CONN-dominant) toward a high value (lattice-dominant) reflects
        the intuition that early learning should be guided by data-space
        topology (which is reliable from k-means initialization), while
        later refinement should respect the evolved embedding topology.
    coord_topo : str
        Adjacency graph used to define the SOM output-space topology.
        Passed to GTSOM.from_data() as coord_topo.

        ``'delaunay'`` (default)
            Delaunay triangulation. Always connected, denser edges,
            may include long edges across sparsely populated regions.
        ``'gabriel'``
            Gabriel graph (subgraph of Delaunay). Sparser, prunes long
            bridging edges, may be disconnected for non-uniform
            prototype distributions.
    proto_topo : str
        Form of the high-dimensional (data-space) neighbourhood used
        when blending CONN topology into the SOM update rule
        (i.e. when nbr_topo_alpha < 1). Passed to GTSOM.__init__().

        ``'CONN'`` (default)
            CONN graph geodesics. CONN is treated as an unweighted
            graph and shortest-path hop counts are fed through the
            standard exponential neighbourhood kernel.
        ``'CONN_STK'``
            CONN self-tuning kernel. CONN weights are used to compute
            a locally adaptive radius sigma_i for each prototype, then
            a Gaussian kernel is applied — no geodesic paths needed.
            More sensitive to local density variation in the prototype
            graph.
    compute_dr_metrics : bool
        If True, compute DR quality metrics (trustworthiness, continuity,
        and related topology-preservation criteria) at each GTSOM snapshot.
        Metrics are stored in learn_history_ and displayed as a caption
        on plots. Passed directly to GTSOM.compile(). Adds computational
        overhead — disable for speed when not needed. Default False.
    """
    rho_0: float = 2.0
    rho_f: float = 1.0
    nbr_topo_alpha_0: float = 0.5
    nbr_topo_alpha_f: float = 0.5
    coord_topo: str = 'delaunay'   # 'delaunay' or 'gabriel'
    proto_topo: str = 'CONN'       # 'CONN' or 'CONN_STK'
    compute_dr_metrics: bool = False


@dataclass
class UMAPParams:
    """
    Parameters governing UMAP embedding updates.

    Attributes
    ----------
    update_freq : int
        Number of SOM epochs between consecutive UMAP embedding updates.
    n_epochs : int
        Number of UMAP layout optimizer epochs per embedding update.
    min_dist_0 : float
        Initial UMAP min_dist parameter.
    min_dist_f : float
        Final UMAP min_dist parameter.
    spread_0 : float
        Initial UMAP spread parameter. Default is 2.0, derived from the
        default embedding_range (-3, 3) as (hi - lo) / 3 = 6 / 3 = 2.0.
        This ensures find_ab_params fits its curve over [0, diameter],
        covering the full range of inter-prototype distances. If you
        change embedding_range in CtrlParams, call tune_embed_scale()
        which sets this automatically.
    spread_f : float
        Final UMAP spread parameter. Defaults to spread_0 so spread
        does not anneal. Set asymmetrically only if you want spread
        to change over training.
    lrate_0 : float
        Initial UMAP layout optimizer learning rate.
    lrate_f : float
        Final UMAP layout optimizer learning rate.
    negative_sample_rate : int
        Number of negative samples per positive sample in the UMAP
        layout optimizer.
    gamma : float
        Repulsion strength in the UMAP layout loss. Higher values push
        non-neighbor prototypes further apart. Default 1.0 is the
        standard UMAP setting.
    use_standard_umap : bool
        Diagnostic flag. If True, bypasses FSIM entirely and runs a
        standard UMAP call on the current prototype matrix W at each
        embedding update step, using UMAP's own kNN similarity graph
        (default n_neighbors=15) warm-started from current coords.
        Useful for isolating whether cluster quality issues stem from
        FSIM construction or from the SOM step itself. Default False.
    """
    update_freq: int = 5
    n_epochs: int = 100
    min_dist_0: float = 0.01
    min_dist_f: float = 0.01
    spread_0: float = 2.0   # = (hi - lo) / 3 for default embedding_range (-3, 3)
    spread_f: float = 2.0   # kept in sync with spread_0; tune_embed_scale overwrites both
    lrate_0: float = 1.0
    lrate_f: float = 0.1
    negative_sample_rate: int = 5
    gamma: float = 1.0
    use_standard_umap: bool = False  # diagnostic: bypass FSIM, run standard UMAP on W


@dataclass
class CtrlParams:
    """
    Parameters governing training control and initialization.

    Attributes
    ----------
    total_epochs : int
        Total number of SOM epochs to run. Controls parameter annealing:
        all schedules decay from initial to final value over this many
        epochs (halflife at total_epochs / 2).
    min_nhbs : int
        Minimum number of neighbors required per row in PCADJ before
        similarity construction. Rows below this are padded with
        synthetic edges.
    fill_val : float
        Edge weight for synthetic PCADJ padding edges.
    coord_init : str
        Method for initializing embedding coordinates from prototypes W.
        Options: 'pca', 'le', 'random'. Passed to GTSOM.from_data().
    embedding_range : tuple of float
        Target coordinate range for each axis of the embedding. After
        each UMAP update, coordinates are rescaled to fit within this
        range via scale_coords(). Also used by tune_embed_scale() to
        derive spread = (hi - lo) / 3.
    n_jobs : int or None
        Number of parallel jobs. Passed to GTSOM (numba parallelism for
        prototype updates) and converted to a boolean for the UMAP layout
        optimizer (parallel=True if n_jobs != 1). Set to 1 for fully
        sequential execution (enables reproducibility). None uses all
        available cores.
    random_state : int or None
        Random seed for reproducibility. Note: parallel execution
        (n_jobs != 1) in the UMAP optimizer uses fresh random samples
        each call regardless of random_state. Set n_jobs=1 for fully
        deterministic UMAP steps.
    verbose : bool
        If True, print progress during fit().
    plot_every : int
        Plot the SOM embedding every this many UMAP update steps.
        Set to 0 to disable in-training plotting. The initialization
        state is always plotted if plot_every > 0.
    """
    total_epochs: int = 100
    min_nhbs: int = 3
    fill_val: float = 0.5
    coord_init: str = 'pca'
    embedding_range: tuple = (-3.0, 3.0)
    n_jobs: int = None
    random_state: int = None
    verbose: bool = True
    plot_every: int = 0


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SOUMAP:
    """
    Self-Organizing UMAP.

    Parameters
    ----------
    M : int
        Number of prototypes.
    som_params : SOMParams, optional
        SOM learning parameters. Defaults to SOMParams().
    umap_params : UMAPParams, optional
        UMAP embedding parameters. Defaults to UMAPParams().
    ctrl_params : CtrlParams, optional
        Training control parameters. Defaults to CtrlParams().

    Examples
    --------
    Default parameters:

    >>> model = SOUMAP(M=100)
    >>> model.fit(X)

    Custom parameters:

    >>> model = SOUMAP(
    ...     M=100,
    ...     som_params=SOMParams(rho_0=3.0, rho_f=0.3),
    ...     ctrl_params=CtrlParams(total_epochs=200, verbose=False),
    ... )
    """

    def __init__(
        self,
        M,
        som_params=None,
        umap_params=None,
        ctrl_params=None,
    ):
        self.M = M
        self.som_params  = som_params  or SOMParams()
        self.umap_params = umap_params or UMAPParams()
        self.ctrl_params = ctrl_params or CtrlParams()

        # --- annealing schedules ---
        # halflife = total_epochs / 2 so each parameter reaches its
        # final value at exactly total_epochs.
        # If initial == final, use a constant schedule (no annealing).
        # Note: rho and nbr_topo_alpha annealing are owned by GTSOM
        # via compile() — only UMAP-side parameters are scheduled here.
        halflife = self.ctrl_params.total_epochs / 2
        self._lrate_sched   = self._make_sched(
            self.umap_params.lrate_0,    self.umap_params.lrate_f,    halflife)
        self._mindist_sched = self._make_sched(
            self.umap_params.min_dist_0, self.umap_params.min_dist_f, halflife)
        self._spread_sched  = self._make_sched(
            self.umap_params.spread_0,   self.umap_params.spread_f,   halflife)

        # --- learned state ---
        self.FSIM            = None   # (M, M) csr  fuzzy similarity matrix
        self.PCADJ           = None   # (M, M) csr  padded CADJ
        self.PCADJ_nhbs      = None   # list[list]  neighbor indices per prototype
        self.PCADJ_nhbs_size = None   # (M,) int    neighbor counts per prototype
        self.age             = 0      # int          total SOM epochs completed

        # --- learning history ---
        self.learn_hist      = []     # list of dicts, one per UMAP update step

        # --- internal objects ---
        self.gtsom           = None   # GTSOM instance, set during build()
        self._is_fitted      = False

    # ------------------------------------------------------------------
    # Properties — lightweight accessors into gtsom internals
    # ------------------------------------------------------------------

    @property
    def W(self):
        """Prototype matrix, shape (M, d). Authoritative copy lives in gtsom."""
        return None if self.gtsom is None else self.gtsom.W

    @property
    def coords(self):
        """Embedding coordinates, shape (M, 2). Authoritative copy lives in gtsom.embed."""
        return None if self.gtsom is None else self.gtsom.embed.coords

    def save(self, path):
        """
        Save this SOUMAP instance to an HDF5 file.

        Serializes all parameter containers (SOMParams, UMAPParams,
        CtrlParams), training state, similarity matrices (FSIM, PCADJ),
        SOUMAP learning history, and the full GTSOM instance. The file
        can be reloaded with :meth:`load` to resume training or inspect
        results without retraining.

        Parameters
        ----------
        path : str
            Destination file path. Created or overwritten.

        Examples
        --------
        >>> model.fit(X, labels=y)
        >>> model.save("soumap.h5")
        >>> model2 = SOUMAP.load("soumap.h5")
        >>> model2.fit(X, labels=y)   # resume training
        """
        from .io import SOUMAPio
        SOUMAPio(self).save(path)

    @classmethod
    def load(cls, path):
        """
        Load a SOUMAP from an HDF5 file created by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to an HDF5 file previously created by :meth:`save`.

        Returns
        -------
        SOUMAP
            Fully reconstructed instance, ready for :meth:`transform`,
            :meth:`fit` (to resume training), or result inspection.

        Examples
        --------
        >>> model = SOUMAP.load("soumap.h5")
        >>> model.age
        200
        >>> model.gtsom.plot(color_by='labels')
        """
        from .io import SOUMAPio
        return SOUMAPio.load(path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, labels=None):
        """
        Run the SOUMAP learning loop.

        Alternates between SOM chunks (gtsom.fit) and UMAP embedding
        updates (build_FSIM + gtsom.update_embedding) for
        ctrl_params.total_epochs total SOM epochs.

        build() must be called before fit(). tune_embed_scale() must
        be called before build() if parameter calibration is desired.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        labels : array-like, shape (N,), optional
            Observation labels passed through to GTSOM for prototype-level
            label summaries. Not used by the SOUMAP learning algorithm.

        Returns
        -------
        self

        Raises
        ------
        RuntimeError
            If build() has not been called yet.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "build() must be called before fit(). "
                "Call model.build(X, labels) to initialize the SOM architecture."
            )

        n_chunks = self.ctrl_params.total_epochs // self.umap_params.update_freq
        remainder = self.ctrl_params.total_epochs % self.umap_params.update_freq

        for chunk in range(n_chunks):

            # --- SOM step ---
            self.gtsom.fit(
                X,
                n_epochs=self.umap_params.update_freq,
                labels=labels,
                verbose=self.ctrl_params.verbose,
                plot_every=self.ctrl_params.plot_every,
            )
            self.age = self.gtsom.age

            # --- UMAP step ---
            self.PCADJ, self.PCADJ_nhbs, self.PCADJ_nhbs_size = pad_CADJ(
                self.gtsom.recaller.CADJ,
                self.gtsom.recaller.CADJ_nhbs,
                self.gtsom.recaller.CADJ_nhbs_size,
                self.W,
                min_nhbs=self.ctrl_params.min_nhbs,
                fill_val=self.ctrl_params.fill_val,
            )

            if self.ctrl_params.verbose:
                pcadj_row_nnz = np.diff(self.PCADJ.indptr)
                print(
                    f"  PCADJ: min_nnz={pcadj_row_nnz.min()}, "
                    f"max_nnz={pcadj_row_nnz.max()}, "
                    f"mean_nnz={pcadj_row_nnz.mean():.1f}, "
                    f"empty_rows={(pcadj_row_nnz == 0).sum()}, "
                    f"below_min_nhbs={(pcadj_row_nnz < self.ctrl_params.min_nhbs).sum()}"
                )

            self.FSIM = build_FSIM(
                self.W,
                self.PCADJ,
                self.PCADJ_nhbs,
            )

            if self.ctrl_params.verbose:
                fsim_row_nnz = np.diff(self.FSIM.indptr)
                print(
                    f"  FSIM:  min_nnz={fsim_row_nnz.min()}, "
                    f"max_nnz={fsim_row_nnz.max()}, "
                    f"mean_nnz={fsim_row_nnz.mean():.1f}, "
                    f"val_range=[{self.FSIM.data.min():.3f}, {self.FSIM.data.max():.3f}]"
                )
            # note: umap-learn takes (spread, min_dist) — not (min_dist, spread)
            a, b = umap_.find_ab_params(
                self._cur_spread(), self._cur_min_dist()
            )
            if self.ctrl_params.verbose:
                print(
                    f"  UMAP update  "
                    f"age={self.age}  "
                    f"rho={self._cur_rho():.4f}  "
                    f"alpha={self._cur_alpha():.4f}  "
                    f"lrate={self._cur_lrate():.4f}  "
                    f"min_dist={self._cur_min_dist():.4f}"
                )

            if self.umap_params.use_standard_umap:
                # --- diagnostic path: standard UMAP on W ---
                # Bypasses FSIM entirely. UMAP builds its own kNN graph
                # from W in feature space (n_neighbors=15 by default).
                # Warm-started from current coords so the embedding
                # evolves incrementally rather than reinitializing.
                import umap as umap_pkg
                reducer = umap_pkg.UMAP(
                    n_components=2,
                    init=self.coords.astype(np.float32),
                    random_state=self.ctrl_params.random_state,
                    n_epochs=self.umap_params.n_epochs,
                )
                new_coords = reducer.fit_transform(self.W)
            else:
                # --- normal SOUMAP path: optimize FSIM-based layout ---
                new_coords = optimize_layout(
                    FSIM=self.FSIM,
                    coords_init=self.coords,
                    a=a,
                    b=b,
                    lrate=self._cur_lrate(),
                    n_epochs=self.umap_params.n_epochs,
                    negative_sample_rate=self.umap_params.negative_sample_rate,
                    gamma=self.umap_params.gamma,
                    parallel=self.ctrl_params.n_jobs != 1,
                    random_state=self.ctrl_params.random_state,
                    age=self.age,
                )
            new_coords = scale_coords(new_coords, self.ctrl_params.embedding_range)
            self.gtsom.update_embedding(new_coords)

            self._log_hist(
                n_new_epochs=self.umap_params.update_freq,
                umap_fields={
                    'rho'     : self._cur_rho(),
                    'alpha'   : self._cur_alpha(),
                    'lrate'   : self._cur_lrate(),
                    'min_dist': self._cur_min_dist(),
                    'spread'  : self._cur_spread(),
                    'a'       : a,
                    'b'       : b,
                }
            )

        # run any remaining epochs that don't fill a full chunk
        if remainder > 0:
            self.gtsom.fit(
                X,
                n_epochs=remainder,
                labels=labels,
                verbose=self.ctrl_params.verbose,
                plot_every=self.ctrl_params.plot_every,
            )
            self.age = self.gtsom.age
            self._log_hist(n_new_epochs=remainder, umap_fields=None)

        return self

    def transform(self, X):
        """
        Map observations to output space via BMU lookup.

        Each observation is assigned to its nearest prototype (BMU) and
        the embedding coordinates of that prototype are returned.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)

        Returns
        -------
        coords : np.ndarray, shape (N, 2)
        """
        return self.gtsom.transform(X)

    def fit_transform(self, X, labels=None):
        """
        Run the SOUMAP learning loop and return embedding coordinates for X.

        Equivalent to fit(X, labels).transform(X). build() must be called
        before fit_transform().

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
        labels : array-like, shape (N,), optional

        Returns
        -------
        coords : np.ndarray, shape (N, 2)
        """
        return self.fit(X, labels=labels).transform(X)

    def tune_embed_scale(
        self,
        rho0_scale=0.5,
        min_dist0_scale=0.5,
        min_distf_scale=0.5,
    ):
        """
        Calibrate rho_0, min_dist, and spread from a uniform random null
        embedding.

        Motivation
        ----------
        In a classical SOM with a fixed lattice, the output topology has a
        known, regular geometry — prototype spacing is uniform and predictable,
        so rho and min_dist can be set intuitively relative to the lattice
        step size (e.g. rho_0 = 2 lattice steps). In SOUMAP, the output
        topology is data-driven and changes during learning, so there is no
        fixed lattice to anchor parameter choices.

        This method recovers an equivalent reference scale by simulating what
        the embedding geometry would look like for M completely unstructured
        (uniform random) prototypes within the embedding space. This null
        embedding serves as a maximum-entropy reference — the geometry you
        would expect in the complete absence of cluster structure. Parameters
        are then set as fractions of the key geometric quantities in this null
        embedding, giving them a scale-free, interpretable meaning analogous
        to the fixed-lattice SOM case.

        Null embedding construction
        ---------------------------
        M points are sampled uniformly at random within embedding_range in
        2D. Three geometric summaries are computed:

        null_max_geo_dist : float
            Maximum geodesic distance across all connected prototype pairs
            in the null Delaunay graph, measured in **hop counts** (number
            of Delaunay edges traversed). This matches the convention used
            by GTSOM, which applies rho to unweighted shortest-path
            distances in embed.dist. Setting rho_0 to half the null hop
            diameter (rho0_scale=0.5) matches the standard SOM
            initialization convention of covering roughly half the map
            with the initial neighbourhood.

        null_mean_geo_dist : float
            Mean hop-count geodesic distance between connected prototype
            pairs. Computed alongside null_max_geo_dist but not used
            directly in calibration. Stored in the verbose output for
            reference.

        null_mean_edge_len : float
            Mean Euclidean length of edges in the null Delaunay
            triangulation — the typical distance between directly adjacent
            prototypes in output space. This is the reference scale for
            min_dist, which is a Euclidean distance threshold in the
            embedding. Computed directly from edge lengths, independently
            of the geodesic computation.

        Parameter calibration
        ---------------------
        rho_0      = null_max_geo_dist  * rho0_scale
        min_dist_0 = null_mean_edge_len * min_dist0_scale
        min_dist_f = null_mean_edge_len * min_distf_scale

        spread is set directly from embedding_range (not from the null
        geometry) because it parameterizes the x-axis scale of the curve
        fitted by find_ab_params, which must cover the full range of
        inter-prototype distances in the embedding:

            spread = (embedding_range[1] - embedding_range[0]) / 3

        The denominator 3 comes from find_ab_params internals: the curve
        is fitted over [0, spread * 3], so setting spread = diameter / 3
        ensures the fitting domain spans [0, diameter] — the full range
        of possible distances between prototypes. This is independent of
        the null geometry and is therefore not a scale parameter.

        rho_f is not calibrated here — it should remain close to 1.0
        (the default). Reducing rho_f toward 0 causes SOM learning to
        degenerate toward k-means, as prototype updates become independent
        of their neighbours.

        With default scale factors, min_dist_0 == min_dist_f so min_dist
        does not anneal. Annealing activates only if asymmetric scale
        factors are provided.

        Theoretical alternatives (not yet implemented)
        ----------------------------------------------
        For a uniform distribution of M points in a 2D square of side
        length L = embedding_range[1] - embedding_range[0], the following
        closed-form approximations from stochastic geometry could replace
        the Monte Carlo sampling:

        Mean Delaunay edge length (replaces null_mean_edge_len):
            E[edge_len] ≈ 0.9 * L / sqrt(M)
            Source: Miles (1970), random Delaunay tessellations.

        Mean nearest-neighbour distance:
            E[d_nn] ≈ L / (2 * sqrt(M))
            Source: Clark & Evans (1954).

        These would make calibration fully deterministic and free of
        sampling variability, at the cost of a slight approximation error
        (typically < 5% for M >= 20). null_mean_geo_dist does not have
        an equally clean closed form and would still require Monte Carlo
        or a cruder approximation (null_mean_edge_len * O(sqrt(M))).

        Parameters
        ----------
        rho0_scale : float, default 0.5
            rho_0 = null_max_geo_dist * rho0_scale. Controls how large
            the initial SOM neighbourhood is relative to the null lattice
            diameter. rho0_scale=0.5 covers half the null lattice,
            matching the standard SOM "half the map" initialization
            convention.
        min_dist0_scale : float, default 0.5
            min_dist_0 = null_mean_edge_len * min_dist0_scale.
        min_distf_scale : float, default 0.5
            min_dist_f = null_mean_edge_len * min_distf_scale. Equal to
            min_dist0_scale by default, so min_dist does not anneal.

        Returns
        -------
        self
            Returns self to allow chaining:
            model.tune_embed_scale().build(X).fit(X)

        Notes
        -----
        Annealing schedules for rho, min_dist, and spread are rebuilt
        after calibration to reflect the new parameter values. Call this
        method before fit(), not after, to ensure the schedules are
        correct before training begins.

        The Monte Carlo sample is seeded by ctrl_params.random_state for
        reproducibility. Two runs with the same random_state and M will
        produce identical calibrated values.
        """
        from scipy.spatial import Delaunay
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        if self.gtsom is not None:
            raise RuntimeError(
                "tune_embed_scale() must be called before build(). "
                "The GTSOM architecture has already been constructed with "
                "the current parameters."
            )

        rng = np.random.default_rng(self.ctrl_params.random_state)
        lo, hi = self.ctrl_params.embedding_range

        # --- sample M uniform random points in embedding_range ---
        null_coords = rng.uniform(lo, hi, size=(self.M, 2))

        # --- Delaunay triangulation of null embedding ---
        tri = Delaunay(null_coords)

        # Build sparse weighted adjacency from Delaunay simplices.
        # Edge weights are Euclidean distances between prototype pairs.
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))

        # Build binary adjacency matrix (edge weights = 1, not Euclidean).
        # Only hop-count geodesics are needed for rho calibration, since
        # GTSOM applies rho to unweighted shortest-path distances in embed.dist.
        rows, cols, data = [], [], []
        for a, b in edges:
            rows += [a, b]
            cols += [b, a]
            data += [1, 1]

        null_adj = csr_matrix((data, (rows, cols)), shape=(self.M, self.M))

        # Unweighted hop-count geodesics — matches GTSOM's embed.dist convention
        geo_dist = shortest_path(null_adj, method='D', directed=False, unweighted=True)

        # null_max_geo_dist: hop-count diameter of null lattice (used for rho_0)
        # null_mean_geo_dist: mean hop-count geodesic (for reference only)
        finite_geo_dists = geo_dist[(geo_dist > 0) & np.isfinite(geo_dist)]
        null_max_geo_dist  = float(finite_geo_dists.max())
        null_mean_geo_dist = float(finite_geo_dists.mean())

        # null_mean_edge_len: mean Euclidean edge length — computed directly
        # from the edge list, independent of shortest_path. Used for min_dist
        # calibration since min_dist is a Euclidean distance in output space.
        null_edges = np.array(list(edges))
        null_edge_lens = np.linalg.norm(
            null_coords[null_edges[:, 0]] - null_coords[null_edges[:, 1]], axis=1
        )
        null_mean_edge_len = float(null_edge_lens.mean())

        # --- calibrate parameters ---
        self.som_params.rho_0       = null_max_geo_dist  * rho0_scale
        self.umap_params.min_dist_0 = null_mean_edge_len * min_dist0_scale
        self.umap_params.min_dist_f = null_mean_edge_len * min_distf_scale

        # spread is derived from embedding_range, not the null geometry.
        # find_ab_params fits its curve over [0, spread * 3], so setting
        # spread = diameter / 3 ensures the domain covers [0, diameter].
        self.umap_params.spread_0 = (hi - lo) / 3
        self.umap_params.spread_f = (hi - lo) / 3

        # --- rebuild UMAP-side annealing schedules to reflect calibrated values ---
        # rho_0 is updated in som_params above and will be picked up by
        # GTSOM.__init__ when _initialize() runs inside fit(). No need to
        # rebuild a rho schedule here since SOUMAP no longer owns one.
        halflife = self.ctrl_params.total_epochs / 2
        self._mindist_sched = self._make_sched(
            self.umap_params.min_dist_0, self.umap_params.min_dist_f, halflife)
        self._spread_sched  = self._make_sched(
            self.umap_params.spread_0,   self.umap_params.spread_f,   halflife)

        if self.ctrl_params.verbose:
            print(
                f"tune_embed_scale: "
                f"null_max_geo_dist={null_max_geo_dist:.4f}, "
                f"null_mean_geo_dist={null_mean_geo_dist:.4f}, "
                f"null_mean_edge_len={null_mean_edge_len:.4f} | "
                f"rho_0={self.som_params.rho_0:.4f}, "
                f"min_dist_0={self.umap_params.min_dist_0:.4f}, "
                f"spread_0={self.umap_params.spread_0:.4f}"
            )

        return self

    # ------------------------------------------------------------------
    # Private — initialization
    # ------------------------------------------------------------------

    def build(self, X, labels=None):
        """
        Construct the SOUMAP architecture from data.

        This is the equivalent of GTSOM.from_data() — it constructs the
        GTSOM instance with the current parameter settings, initializes
        prototypes via k-means, reduces them to 2D, builds the initial
        output-space topology, runs the first recall, and takes the initial
        snapshot (including DR metrics if compute_dr_metrics=True).

        Call order:

            model = SOUMAP(M=..., som_params=..., umap_params=..., ctrl_params=...)
            model.tune_embed_scale()   # optional — override rho_0, min_dist
            model.build(X, labels=y)      # construct architecture
            model.gtsom.plot(...)         # optional — inspect initial state
            model.fit(X, labels=y)        # learning loop

        FSIM and PCADJ are not computed here — they are computed at the
        start of the first UMAP step in fit(), ensuring they reflect the
        prototypes and recall state at that point.

        Parameters
        ----------
        X : np.ndarray, shape (N, d)
            Data matrix. Used to fit k-means prototypes and run initial
            recall.
        labels : array-like, shape (N,), optional
            Observation labels passed through to GTSOM for prototype-level
            label summaries and DR metric computation.

        Raises
        ------
        RuntimeError
            If build() has already been called on this instance.
        """
        if self.gtsom is not None:
            raise RuntimeError(
                "build() has already been called. To reinitialize, "
                "create a new SOUMAP instance."
            )

        # Construct GTSOM with all learning parameters
        self.gtsom = GTSOM(
            rho_0=self.som_params.rho_0,
            rho_f=self.som_params.rho_f,
            halflife_epochs=self.ctrl_params.total_epochs / 2,
            nbr_topo_alpha_0=self.som_params.nbr_topo_alpha_0,
            nbr_topo_alpha_f=self.som_params.nbr_topo_alpha_f,
            proto_topo=self.som_params.proto_topo,
            compute_dr_metrics=self.som_params.compute_dr_metrics,
            random_state=self.ctrl_params.random_state,
            n_jobs=self.ctrl_params.n_jobs,
        )

        # Initialise topology and prototypes from data
        self.gtsom.from_data(
            X,
            M=self.M,
            coord_init=self.ctrl_params.coord_init,
            coord_topo=self.som_params.coord_topo,
            labels=labels,
        )

        # Scale initial coords (from PCA/LE) into embedding_range.
        # This anchors the coordinate system before any learning begins,
        # giving min_dist and spread an interpretable scale and ensuring
        # the first UMAP warm-start operates in a consistent regime.
        initial_coords = scale_coords(
            self.coords, self.ctrl_params.embedding_range
        )
        self.gtsom.update_embedding(initial_coords)

        self._is_fitted = True
        self.age = 0

        # Regenerate the initial snapshot figure with a SOUMAP-specific
        # title and the post-rescale coordinates. The figure GTSOM auto-
        # generated inside from_data() used its own default title and the
        # pre-rescale coordinates, so it does not accurately represent
        # the true initial state SOUMAP uses going forward.
        color_by = 'labels' if labels is not None else 'rfsize'
        init_fig = self.gtsom.plot(
            color_by=color_by,
            title='SOUMAP',
            subtitle='Initialization (age=0)',
        )
        self.gtsom.learn_history_[0]['fig'] = init_fig

        # Visualize initial state if plotting is enabled.
        if self.ctrl_params.plot_every > 0:
            # plotnine ggplot: draw to matplotlib and display non-blocking
            import matplotlib.pyplot as plt
            init_fig.draw()
            plt.show(block=False)
            plt.pause(0.1)

        return self

    # ------------------------------------------------------------------
    # Private — annealed parameter accessors
    # ------------------------------------------------------------------

    @staticmethod
    def _make_sched(val_0, val_f, halflife):
        """
        Build an ExponentialAnneal schedule if val_0 > val_f, otherwise
        return a constant callable that always returns val_f.

        This allows parameters like min_dist and spread to be set equal
        (no annealing) without ExponentialAnneal raising a ValueError.
        """
        if val_0 > val_f:
            return ExponentialAnneal.from_halflife(val_0, val_f, halflife)
        else:
            return lambda age: val_f

    def _cur_rho(self):
        """Current annealed SOM neighbourhood bandwidth (read from gtsom)."""
        return self.gtsom.rho_schedule(self.age)

    def _cur_alpha(self):
        """Current annealed topology-blending weight (read from gtsom)."""
        return self.gtsom.alpha_schedule(self.age)

    def _cur_lrate(self):
        """Current annealed UMAP learning rate."""
        return self._lrate_sched(self.age)

    def _cur_min_dist(self):
        """Current annealed UMAP min_dist."""
        return self._mindist_sched(self.age)

    def _cur_spread(self):
        """Current annealed UMAP spread."""
        return self._spread_sched(self.age)

    # ------------------------------------------------------------------
    # Private — utilities
    # ------------------------------------------------------------------

    def _log_hist(self, n_new_epochs, umap_fields=None):
        """
        Copy the most recent n_new_epochs GTSOM snapshots into learn_hist,
        appending UMAP parameter fields to each entry.

        Called after every gtsom.fit() chunk. For chunks that include a
        UMAP update, umap_fields carries the current annealed UMAP state.
        For the remainder chunk (if any), umap_fields is None and all UMAP
        fields are stored as None to indicate no UMAP step occurred.

        Parameters
        ----------
        n_new_epochs : int
            Number of new snapshots appended to gtsom.learn_history_ by
            the most recent gtsom.fit() call.
        umap_fields : dict or None
            Dict with keys 'rho', 'lrate', 'min_dist', 'spread', 'a', 'b'
            containing the UMAP optimizer state at this update step.
            Pass None for chunks where no UMAP step was performed.
        """
        null_umap = {
            'rho'     : None,
            'alpha'   : None,
            'lrate'   : None,
            'min_dist': None,
            'spread'  : None,
            'a'       : None,
            'b'       : None,
        }
        umap_entry = umap_fields if umap_fields is not None else null_umap

        new_snaps = self.gtsom.learn_history_[-n_new_epochs:]
        for snap in new_snaps:
            self.learn_hist.append({
                'age'    : snap['age'],
                'mqe'    : snap['mqe'],
                'delBMU' : snap['delBMU'],
                **umap_entry,
            })

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self):
        status = f"age={self.age}" if self._is_fitted else "unfitted"
        return (
            f"SOUMAP(M={self.M}, "
            f"total_epochs={self.ctrl_params.total_epochs}, "
            f"{status})"
        )