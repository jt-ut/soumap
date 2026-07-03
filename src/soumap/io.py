"""
io.py — HDF5 persistence layer for the soumap package.

Reuses the recursive _write_dict / _read_dict helpers from gtsom.io to
handle all type dispatch (dense arrays, sparse CSR matrices, ragged
list-of-lists, scalars, None, nested dicts).  SOUMAPio owns the SOUMAP
file layout and delegates GTSOM serialization to GTSOMio.  MPECio owns
the MPEClustering file layout.

SOUMAP file layout
------------------
::

    soumap.h5
    ├── meta/           version string and class tag
    ├── params/
    │   ├── som/        SOMParams fields  (via dataclasses.asdict)
    │   ├── umap/       UMAPParams fields
    │   └── ctrl/       CtrlParams fields
    ├── state/          M, age, is_fitted
    ├── similarity/     FSIM, PCADJ, PCADJ_nhbs, PCADJ_nhbs_size
    ├── learn_hist/     SOUMAP's own per-UMAP-update history entries
    └── gtsom/          full GTSOM saved via GTSOMio helpers

MPEC file layout
----------------
::

    mpec.h5
    ├── meta/           version string and class tag
    ├── params/         all MPEClustering __init__ parameters (scalars + strings)
    ├── results/
    │   ├── clustering/ labels_, active_indices_, rf_sizes_, n_steps_
    │   └── kernels/    fused_graph_, kernel_cadj_, kernel_high_d_, kernel_low_d_
    └── step_analysis/  one subgroup per communicating class, each containing
                        arrays for mixing, cohesion, hd_norm, cc_norm,
                        walk_lengths, plus scalars best_step, size

    Note: walktrap_result_ and walktrap_dendrogram_ are igraph objects and
    cannot be serialized.  Both are recomputed from fused_graph_ and
    n_steps_ on load — Walktrap is deterministic so results are identical.

Typical usage
-------------
Save a fitted SOUMAP::

    model.save("soumap.h5")

Load back::

    model = SOUMAP.load("soumap.h5")
    model.fit(X, labels=y)   # resume training

Save a fitted MPEClustering::

    mpec.save("mpec.h5")

Load back::

    mpec = MPEClustering.load("mpec.h5")
    mpec.plot_diagnostics(embed_coords=Y)
"""

from __future__ import annotations

import numpy as np
from dataclasses import asdict

__all__ = ["SOUMAPio", "MPECio"]


# ---------------------------------------------------------------------------
# SOUMAPio
# ---------------------------------------------------------------------------

class SOUMAPio:
    """
    HDF5 save/load manager for SOUMAP instances.

    Owns the entire HDF5 file structure for SOUMAP. Delegates GTSOM
    serialization to GTSOMio and type-dispatch to _write_dict / _read_dict
    from gtsom.io.

    Parameters
    ----------
    model : SOUMAP
        A built (or partially fitted) SOUMAP instance. Only required for
        :meth:`save`; use :meth:`load` as a classmethod to reconstruct.

    Examples
    --------
    Save::

        SOUMAPio(model).save("soumap.h5")

    Load::

        model = SOUMAPio.load("soumap.h5")
    """

    # HDF5 top-level group names
    _GRP_META       = "meta"
    _GRP_PARAMS     = "params"
    _GRP_PARAMS_SOM  = "som"
    _GRP_PARAMS_UMAP = "umap"
    _GRP_PARAMS_CTRL = "ctrl"
    _GRP_STATE      = "state"
    _GRP_SIM        = "similarity"
    _GRP_HIST       = "learn_hist"
    _GRP_GTSOM      = "gtsom"
    _ENTRY_FMT      = "entry_{:06d}"

    def __init__(self, model):
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path):
        """
        Save the SOUMAP instance to an HDF5 file at ``path``.

        Parameters
        ----------
        path : str
            Destination file path. Created or overwritten.
        """
        import h5py
        from gtsom.io import GTSOMio, _write_dict
        from importlib.metadata import version as _pkg_version

        model = self._model

        try:
            _version = _pkg_version("soumap")
        except Exception:
            _version = "unknown"

        with h5py.File(path, "w") as f:

            # --- meta -----------------------------------------------------
            mg = f.create_group(self._GRP_META)
            mg.attrs["soumap_version"] = _version
            mg.attrs["class"]          = "SOUMAP"

            # --- params ---------------------------------------------------
            pg = f.create_group(self._GRP_PARAMS)
            _write_dict(pg.create_group(self._GRP_PARAMS_SOM),
                        asdict(model.som_params))
            _write_dict(pg.create_group(self._GRP_PARAMS_UMAP),
                        asdict(model.umap_params))
            _write_dict(pg.create_group(self._GRP_PARAMS_CTRL),
                        asdict(model.ctrl_params))

            # --- state ----------------------------------------------------
            _write_dict(f.create_group(self._GRP_STATE), {
                "M":         model.M,
                "age":       model.age,
                "is_fitted": model._is_fitted,
            })

            # --- similarity -----------------------------------------------
            sg = f.create_group(self._GRP_SIM)
            sim_dict = {}
            if model.FSIM is not None:
                sim_dict["FSIM"] = model.FSIM
            if model.PCADJ is not None:
                sim_dict["PCADJ"] = model.PCADJ
            if model.PCADJ_nhbs is not None:
                sim_dict["PCADJ_nhbs"] = model.PCADJ_nhbs
            if model.PCADJ_nhbs_size is not None:
                sim_dict["PCADJ_nhbs_size"] = model.PCADJ_nhbs_size
            _write_dict(sg, sim_dict)

            # --- learn_hist -----------------------------------------------
            hg = f.create_group(self._GRP_HIST)
            hg.attrs["n_entries"] = len(model.learn_hist)
            for i, entry in enumerate(model.learn_hist):
                eg = hg.create_group(self._ENTRY_FMT.format(i))
                _write_dict(eg, self._build_hist_entry(entry))

            # --- gtsom ----------------------------------------------------
            if model.gtsom is not None:
                GTSOMio(model.gtsom)._write_to_group(
                    f.create_group(self._GRP_GTSOM)
                )

    @classmethod
    def load(cls, path):
        """
        Load a SOUMAP from an HDF5 file written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.

        Returns
        -------
        SOUMAP
            Fully reconstructed instance. If the model was fitted before
            saving, it is ready for :meth:`~SOUMAP.transform`,
            :meth:`~SOUMAP.fit` (to resume training), or inspection of
            ``learn_hist`` and ``gtsom.learn_history_``.
        """
        import h5py
        from gtsom.io import GTSOMio, _read_dict
        from soumap.soumap import SOUMAP, SOMParams, UMAPParams, CtrlParams

        with h5py.File(path, "r") as f:

            # --- params ---------------------------------------------------
            pg = f[cls._GRP_PARAMS]
            som_params  = SOMParams( **_read_dict(pg[cls._GRP_PARAMS_SOM]))
            umap_params = UMAPParams(**_read_dict(pg[cls._GRP_PARAMS_UMAP]))
            ctrl_params = CtrlParams(**_read_dict(pg[cls._GRP_PARAMS_CTRL]))

            # --- state ----------------------------------------------------
            state = _read_dict(f[cls._GRP_STATE])
            M          = state["M"]
            age        = state["age"]
            is_fitted  = state["is_fitted"]

            # --- reconstruct SOUMAP without calling __init__ --------------
            model = SOUMAP.__new__(SOUMAP)
            model.M           = M
            model.som_params  = som_params
            model.umap_params = umap_params
            model.ctrl_params = ctrl_params
            model.age         = age
            model._is_fitted  = is_fitted

            # Rebuild UMAP-side annealing schedules from loaded params
            from gtsom.utils import ExponentialAnneal
            halflife = ctrl_params.total_epochs / 2
            model._lrate_sched   = model._make_sched(
                umap_params.lrate_0,    umap_params.lrate_f,    halflife)
            model._mindist_sched = model._make_sched(
                umap_params.min_dist_0, umap_params.min_dist_f, halflife)
            model._spread_sched  = model._make_sched(
                umap_params.spread_0,   umap_params.spread_f,   halflife)

            # --- similarity -----------------------------------------------
            sim = _read_dict(f[cls._GRP_SIM])
            model.FSIM           = sim.get("FSIM")
            model.PCADJ          = sim.get("PCADJ")
            model.PCADJ_nhbs     = sim.get("PCADJ_nhbs")
            model.PCADJ_nhbs_size = sim.get("PCADJ_nhbs_size")

            # --- learn_hist -----------------------------------------------
            hg = f[cls._GRP_HIST]
            n  = int(hg.attrs.get("n_entries", 0))
            model.learn_hist = [
                cls._restore_hist_entry(_read_dict(hg[cls._ENTRY_FMT.format(i)]))
                for i in range(n)
            ]

            # --- gtsom ----------------------------------------------------
            model.gtsom = None
            if cls._GRP_GTSOM in f:
                model.gtsom = GTSOMio._load_from_group(f[cls._GRP_GTSOM])

        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hist_entry(entry):
        """
        Convert one learn_hist entry into a write-ready dict.
        All values are scalars or None — no arrays or sparse matrices.
        """
        return {
            "age"     : entry["age"],
            "mqe"     : entry["mqe"],
            "delBMU"  : entry["delBMU"],
            "rho"     : entry.get("rho"),
            "alpha"   : entry.get("alpha"),
            "lrate"   : entry.get("lrate"),
            "min_dist": entry.get("min_dist"),
            "spread"  : entry.get("spread"),
            "a"       : entry.get("a"),
            "b"       : entry.get("b"),
        }

    @staticmethod
    def _restore_hist_entry(d):
        """Restore one learn_hist entry from a raw dict."""
        return {
            "age"     : d["age"],
            "mqe"     : d["mqe"],
            "delBMU"  : d["delBMU"],
            "rho"     : d.get("rho"),
            "alpha"   : d.get("alpha"),
            "lrate"   : d.get("lrate"),
            "min_dist": d.get("min_dist"),
            "spread"  : d.get("spread"),
            "a"       : d.get("a"),
            "b"       : d.get("b"),
        }


# ---------------------------------------------------------------------------
# MPECio
# ---------------------------------------------------------------------------

class MPECio:
    """
    HDF5 save/load manager for MPEClustering instances.

    Owns the entire HDF5 file structure for MPEClustering. Delegates
    type-dispatch to _write_dict / _read_dict from gtsom.io.

    igraph objects (walktrap_result_, walktrap_dendrogram_) are not
    serialized — both are recomputed from fused_graph_ and n_steps_ on
    load.  Walktrap is deterministic so results are bit-for-bit identical.

    Parameters
    ----------
    model : MPEClustering
        A fitted MPEClustering instance.  Only required for :meth:`save`;
        use :meth:`load` as a classmethod to reconstruct.

    Examples
    --------
    Save::

        MPECio(mpec).save("mpec.h5")

    Load::

        mpec = MPECio.load("mpec.h5")
    """

    # HDF5 group names
    _GRP_META        = "meta"
    _GRP_PARAMS      = "params"
    _GRP_RESULTS     = "results"
    _GRP_CLUSTERING  = "clustering"
    _GRP_KERNELS     = "kernels"
    _GRP_STEP        = "step_analysis"
    _COMP_FMT        = "component_{:06d}"

    def __init__(self, model):
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save the MPEClustering instance to an HDF5 file at ``path``.

        Parameters
        ----------
        path : str
            Destination file path.  Created or overwritten.
        """
        import h5py
        from gtsom.io import _write_dict
        from importlib.metadata import version as _pkg_version

        m = self._model

        if m.labels_ is None:
            raise RuntimeError(
                "MPECio.save() requires a fitted MPEClustering instance. "
                "Call mpec.fit(W, Y, CADJ) before saving."
            )

        try:
            _version = _pkg_version("soumap")
        except Exception:
            _version = "unknown"

        with h5py.File(path, "w") as f:

            # --- meta -----------------------------------------------------
            mg = f.create_group(self._GRP_META)
            mg.attrs["soumap_version"] = _version
            mg.attrs["class"]          = "MPEClustering"

            # --- params ---------------------------------------------------
            # All 11 __init__ parameters. Strings stored as HDF5 attributes
            # (not datasets) since _write_dict handles them as scalars.
            _write_dict(f.create_group(self._GRP_PARAMS), {
                "walktrap_n_steps"  : m.walktrap_n_steps
                                      if isinstance(m.walktrap_n_steps, int)
                                      else -1,        # -1 sentinel for "auto"
                "walktrap_n_steps_auto": int(m.walktrap_n_steps == "auto"),
                "kernel_w_cadj"     : m.kernel_w_cadj,
                "kernel_w_high_d"   : m.kernel_w_high_d,
                "kernel_w_low_d"    : m.kernel_w_low_d,
                "kernel_power"      : m.kernel_power,
                "kernel_min_value"  : m.kernel_min_value,
                "remove_empty_protos": int(m.remove_empty_protos),
                "kernel_min_nhbs"   : m.kernel_min_nhbs,
                "kernel_support"    : m.kernel_support,  # "dense" or "CADJ"
                "walktrap_max_steps": m.walktrap_max_steps,
                "verbose"           : int(m.verbose),
            })

            # --- results --------------------------------------------------
            rg = f.create_group(self._GRP_RESULTS)

            # clustering outputs
            cg = rg.create_group(self._GRP_CLUSTERING)
            _write_dict(cg, {
                "labels_"        : m.labels_,
                "active_indices_": m.active_indices_,
                "rf_sizes_"      : m.rf_sizes_,
                "n_steps_"       : m.n_steps_,
            })

            # kernel/graph outputs — all sparse CSR, handled by _write_dict
            kg = rg.create_group(self._GRP_KERNELS)
            kg_dict = {"fused_graph_": m.fused_graph_}
            if m.kernel_cadj_ is not None:
                kg_dict["kernel_cadj_"] = m.kernel_cadj_
            if m.kernel_high_d_ is not None:
                kg_dict["kernel_high_d_"] = m.kernel_high_d_
            if m.kernel_low_d_ is not None:
                kg_dict["kernel_low_d_"] = m.kernel_low_d_
            _write_dict(kg, kg_dict)

            # --- step_analysis --------------------------------------------
            sg = f.create_group(self._GRP_STEP)
            if m.step_analysis_ is None:
                sg.attrs["available"] = 0
            else:
                sg.attrs["available"] = 1
                sg.attrs["n_components"] = len(m.step_analysis_)
                for i, rec in enumerate(m.step_analysis_):
                    cg_s = sg.create_group(self._COMP_FMT.format(i))
                    # Convert Python lists → numpy arrays for _write_dict
                    _write_dict(cg_s, {
                        "size"       : rec["size"],
                        "best_step"  : rec["best_step"],
                        "indices"    : np.asarray(rec["indices"]),
                        "mixing"     : np.asarray(rec["mixing"],      dtype=np.float64),
                        "cohesion"   : np.asarray(rec["cohesion"],    dtype=np.float64),
                        "hd_norm"    : np.asarray(rec["hd_norm"],     dtype=np.float64),
                        "cc_norm"    : np.asarray(rec["cc_norm"],     dtype=np.float64),
                        "walk_lengths": np.asarray(rec["walk_lengths"], dtype=np.int32),
                    })

    @classmethod
    def load(cls, path: str):
        """
        Load a MPEClustering from an HDF5 file written by :meth:`save`.

        Walktrap igraph objects (walktrap_result_, walktrap_dendrogram_)
        are recomputed from the saved fused_graph_ and n_steps_ — Walktrap
        is deterministic so results are identical to the original fit.

        Parameters
        ----------
        path : str
            Path to an HDF5 file previously created by :meth:`save`.

        Returns
        -------
        MPEClustering
            Fully reconstructed instance with all plot and diagnostic
            functionality available.
        """
        import h5py
        from gtsom.io import _read_dict
        from soumap.mpec import MPEClustering

        with h5py.File(path, "r") as f:

            # --- params ---------------------------------------------------
            p = _read_dict(f[cls._GRP_PARAMS])

            # Reconstruct walktrap_n_steps (int or "auto")
            n_steps_param = (
                "auto" if p["walktrap_n_steps_auto"]
                else int(p["walktrap_n_steps"])
            )

            mpec = MPEClustering(
                walktrap_n_steps   = n_steps_param,
                kernel_w_cadj      = float(p["kernel_w_cadj"]),
                kernel_w_high_d    = float(p["kernel_w_high_d"]),
                kernel_w_low_d     = float(p["kernel_w_low_d"]),
                kernel_power       = float(p["kernel_power"]),
                kernel_min_value   = float(p["kernel_min_value"]),
                remove_empty_protos= bool(p["remove_empty_protos"]),
                kernel_min_nhbs    = int(p["kernel_min_nhbs"]),
                kernel_support     = str(p["kernel_support"]),
                walktrap_max_steps = int(p["walktrap_max_steps"]),
                verbose            = bool(p["verbose"]),
            )

            # --- results — clustering ------------------------------------
            c = _read_dict(f[cls._GRP_RESULTS][cls._GRP_CLUSTERING])
            mpec.labels_         = c["labels_"]
            mpec.active_indices_ = c["active_indices_"]
            mpec.rf_sizes_       = c["rf_sizes_"]
            mpec.n_steps_        = int(c["n_steps_"])

            # --- results — kernels / graph --------------------------------
            k = _read_dict(f[cls._GRP_RESULTS][cls._GRP_KERNELS])
            mpec.fused_graph_    = k["fused_graph_"]
            mpec.kernel_cadj_    = k.get("kernel_cadj_")
            mpec.kernel_high_d_  = k.get("kernel_high_d_")
            mpec.kernel_low_d_   = k.get("kernel_low_d_")

            # --- step_analysis --------------------------------------------
            sg = f[cls._GRP_STEP]
            if not int(sg.attrs.get("available", 0)):
                mpec.step_analysis_ = None
            else:
                n_comp = int(sg.attrs["n_components"])
                step_analysis = []
                for i in range(n_comp):
                    d = _read_dict(sg[cls._COMP_FMT.format(i)])
                    step_analysis.append({
                        "size"        : int(d["size"]),
                        "best_step"   : int(d["best_step"]),
                        "indices"     : d["indices"],
                        "mixing"      : d["mixing"].tolist(),
                        "cohesion"    : d["cohesion"].tolist(),
                        "hd_norm"     : d["hd_norm"].tolist(),
                        "cc_norm"     : d["cc_norm"].tolist(),
                        "walk_lengths": d["walk_lengths"].tolist(),
                    })
                mpec.step_analysis_ = step_analysis

        # --- recompute igraph objects from fused_graph_ + n_steps_ -------
        # Walktrap is deterministic: identical graph + steps → identical
        # dendrogram and clustering, bit-for-bit.
        mpec._recompute_walktrap()

        return mpec