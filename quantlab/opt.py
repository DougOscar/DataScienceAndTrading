"""Optimisation machinery: search spaces, logged studies, CPCV, walk-forward, plateau selection.

DESIGN §4.2 (plateau gate, PBO inputs), §4.6 (the re-optimisation *schedule* is the
system), §7 (performance), §8 (ledger).  Contracts: ``contracts.Evaluator``,
``contracts.Outcome``, ``contracts.StudyResult`` and the daily-returns convention.

Architecture (rule-based systems)
---------------------------------
Each parameter set is evaluated **once** over the full dev window (signals at *t* only
use data ≤ *t*, so a column of the resulting daily-returns matrix restricted to any
window is exactly what that parameter set would have earned there).  Everything else —
the robust objective, plateau selection, CPCV paths, the simulated walk-forward
re-optimisation — operates on that ``T × M`` matrix (plus a same-shape matrix of
trades closed per day, for window-level minimum-trade checks).  Systems that must be
re-fitted per window (``Evaluator.requires_refit``, i.e. ML) are out of scope for
Phase 1 and raise ``NotImplementedError``.

Trials and the ledger
---------------------
Every configuration the search (see "Search methods" below) touches is a trial and is recorded with
``ledger.TrialRecorder`` — status one of:

* ``ok``          evaluated; passes the minimum-trade constraint on the full window;
* ``low_trades``  evaluated, returns recorded, but fewer trades than ``min_trades``
                  (objective = −inf; still a trial, still a column of the matrix);
* ``invalid``     rejected by ``SearchSpace.constraint`` — never evaluated, but counted
                  (it was part of the searched space; the count feeds deflation);
* ``error``       the evaluator raised (message kept in ``meta['errors']`` and in
                  ``opt_aux-*.parquet``);

(``pruned`` is reserved for early-stopped ML trials in a later phase.)  TPE duplicates
(the sampler re-proposing an already-evaluated configuration) are *not* new
configurations: they re-use the cached result and are not re-counted.  The ledger gets
``study_created`` (with the planned scheme), ``trials`` (raw count, incl. invalid/error)
and ``selection`` events; an exception mid-study still logs ``trials`` with
``status="aborted"`` so no evaluation is ever off-ledger.

Returns matrix alignment
------------------------
Columns ``t<trial_id>`` are aligned on the **union** of all trials' dates; a trial with
no row on a date gets ``0.0`` there (no position → no P&L).  For rule-based systems
evaluated on the same bars every trial has the same dates, so this only matters for
evaluators whose date sets differ.

Objective (``Objective``)
-------------------------
Default ``block_sharpe``: split the window's rows into ``n_blocks`` contiguous blocks,
annualised Sharpe per block, score = mean − ``lam``·std across blocks (net of costs —
the evaluator's returns already are).  Rewards edges that are *consistently* present
rather than one lucky sub-period.  A zero-variance block counts as Sharpe 0.
``min_trades`` applies to the full window and is scaled proportionally to the number
of rows for sub-windows (CPCV train sets, walk-forward training windows); failing it
gives −inf.

Plateau selection (``PlateauConfig``)
-------------------------------------
Neighbourhoods live in normalised parameter space:

* ``grid``: each parameter is indexed by its rank among the grid's distinct values;
  two configurations are neighbours if every index differs by ≤ ``radius``
  (Chebyshev ball — up to 3^d − 1 neighbours at radius 1) and unordered categoricals
  are equal;
* ``knn``: the ``k`` nearest configurations in unit coordinates (numeric parameters
  scaled to [0, 1], log-scaled when ``log=True``; unordered categoricals must match) —
  used for TPE samples, whose irregular layout would leave box neighbourhoods empty.

The objective is smoothed as the mean over ``{self} ∪ neighbours``; neighbours that
failed (low trades, error, non-finite) enter with the worst finite score of the window
(conservative, not −inf); ``invalid`` configurations are not systems and are excluded.
Only configurations whose *own* objective is at least the ``min_centre_quantile``
(default: median) of the eligible objectives may be picked as the centre — otherwise
smoothing can pick a hole ringed by good neighbours (seen on real SMA grids).
A configuration with fewer than ``min_neighbours`` neighbours is padded with that same
floor, so isolated points cannot win on their own value.  The winner is the argmax
of the smoothed surface among eligible (non-failed) configurations, ties → lowest
trial id.  ``plateau_score`` (DESIGN §4.2) = share of the winner's neighbours whose
**Sharpe** on the window is ≥ ``peak_fraction`` (50%) × the winner's own Sharpe
(0 when that Sharpe ≤ 0; failed neighbours count as outside).  The raw argmax and a
plateau score on the objective itself are reported alongside.

CPCV and purging for daily mark-to-market returns
-------------------------------------------------
``cpcv_splits``: the T trading days are cut into ``n_groups`` contiguous groups;
every combination of ``k_test`` groups is a test set (C(N, k) splits).  Daily MTM
returns have no "labels", but they still leak across a train/test boundary through
**open positions**: a trade opened before a test block and closed inside it books part
of one P&L on train days and the rest on test days (and the parameter set that looked
good on the train days right before the block is partly being rewarded for that same
trade).  Hence, measured in trading-day rows:

* **purge**: drop ``purge_days`` training rows immediately *before* each test block;
* **embargo**: drop ``embargo_days`` training rows immediately *after* each test block
  (positions opened in the test block carry over; serial correlation of MTM returns).

Both default to the maximum holding period seen among the ``ok`` trials
(``m_hold_days_max``, in rows of the date index between entry and exit — L4: a trade across
a data gap spans only the rows that exist, not the calendar time of the gap), floored at 1
and capped at a quarter of a group.
Selection on each split's training rows uses the SAME objective + plateau procedure as
the full-sample selection; the selected column's test-row returns are stored.  Paths
follow López de Prado: each group is in the test set of φ = C(N−1, k−1) splits; path
*p* takes group *g* from the *p*-th of those splits (split-id order), giving φ full-
length OOS paths → ``cpcv_paths`` (date, path_id, ret).

Walk-forward re-optimisation (``WFOConfig``, DESIGN §4.6)
-------------------------------------------------------
Refit dates start at ``first date + min_train`` and repeat every ``refit_every``.  At
refit date *d* the selection procedure sees only rows with date < *d* (anchored: from
the first date; rolling: from *d* − ``window_length``) and its pick trades rows
[*d*, next refit).  **This holds only if the candidate set (the columns of the matrix)
does not itself depend on the data** — ``grid``, ``sobol`` and ``random`` candidate sets
are fixed before any evaluation; a ``tpe`` candidate set is chosen with the full-dev
objective, so it already encodes the test folds / the future (red-team B3, probe p07:
+0.17 Sharpe of spurious "OOS" under a zero-edge null).  ``meta["candidate_set_data_
dependent"]`` flags this and the gates SKIP the OOS gates for such studies.  Every trial records
its ``source`` (grid / sobol / random / tpe); one TPE-sourced trial flags the study for good, and
a resume can never change the method recorded in the ledger (red-team N1).  The switch
is modelled as adopting the new configuration's book at *d* (its return on *d* includes
positions it opened earlier — a documented simplification; the rule-based columns are
evaluated continuously).  ``wfo_oos.n_trades`` = entries of the active configuration on
each OOS day (the procedure's own trade counts, input to the holdout band).  Drift
diagnostics: normalised Euclidean distance (unit coordinates / √d) between consecutive
selections.

Search methods (candidate sets)
-------------------------------
* ``grid``   every grid point (≤ 3 params unless ``allow_large_grid``);
* ``sobol``  ``n_trials`` distinct configurations from a scrambled Sobol sequence
             (``scipy.stats.qmc``, seeded), mapped onto the space: numeric grids /
             categoricals get equal-width cells per level, continuous floats linear or
             log-uniform, log-ints log-uniform then floored.  Data-independent;
* ``random`` same mapping with iid uniforms (seeded).  Data-independent;
* ``tpe``    Optuna TPE on the full-dev objective — exploration only (data-dependent);
* ``auto``   (default) ``grid`` for ≤ 3 all-discrete params, else ``sobol``.

For sobol/random, duplicate configurations (discrete spaces) are dropped and the
sequence continues until ``n_trials`` distinct configurations exist; invalid ones
(``space.constraint``) are logged as ``invalid`` trials and count toward ``n_trials``
(same convention as grid and TPE).

Worker processes (F1)
---------------------
``n_jobs > 1`` uses a **spawn** process pool whose children start with
``POLARS_MAX_THREADS=1`` (and Numba/OpenMP/BLAS/Rayon caps) in their environment, set
in the parent only for the instant each child is launched — a forked child inherits the
parent's already-initialised all-core Polars pool, which oversubscribed the box 8×16.
Consequence of spawn: a script that calls :func:`run_study` with ``n_jobs > 1`` needs an
``if __name__ == "__main__":`` guard, and the evaluator must be importable by reference.
An evaluator defined in an interactive ``__main__`` (notebook / REPL) cannot be
unpickled by a spawned child; for those the pool falls back to ``fork`` with a warning
(``QUANTLAB_MP_START=fork|spawn`` forces a method).
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import math
import multiprocessing
import multiprocessing.context
import os
import pickle
import sys
import threading
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from . import config, ledger
from .metrics import data_gaps as _data_gaps
from .contracts import StudyResult

__all__ = [
    "Param", "IntParam", "FloatParam", "CategoricalParam", "SearchSpace",
    "Objective", "PlateauConfig", "CPCVConfig", "WFOConfig", "CPCVSplit",
    "neighbourhoods", "plateau_select", "cpcv_splits", "cpcv_path_map", "cpcv_paths",
    "walk_forward", "run_study", "StudyError", "candidate_set", "METHODS", "DATA_INDEPENDENT_METHODS",
    "WORKER_THREAD_ENV", "PLATEAU_RADIUS_DEFAULT", "PLATEAU_RADIUS_MIN", "normalise_plateau_radius",
    "space_from_json",
]

log = logging.getLogger("quantlab.opt")

STATUSES = ("ok", "low_trades", "invalid", "error", "pruned")


class StudyError(RuntimeError):
    pass


# =========================================================================== search space
def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def _numeric_like(v: Any) -> bool:
    """A number, or a string that parses as one (``"50"``, ``" 1e3 "``) — red-team R3-3."""
    if _is_number(v):
        return True
    if isinstance(v, str):
        try:
            float(v.strip().replace("_", ""))
            return True
        except ValueError:
            return False
    return False


@dataclass(frozen=True)
class Param:
    """One searched parameter.  Build with :func:`IntParam` / :func:`FloatParam` /
    :func:`CategoricalParam`.

    Plateau scale (DESIGN §4.2, red-team R2-1) — every numeric (int / float) parameter must
    declare, when the space is built, how the judge-run plateau perturbs it:

    * ``plateau_scale="relative"``: ±r/2·|x| and ±r·|x| around the selected value x (r = the
      study's pre-registered radius, default 0.20).  Allowed only for strictly positive
      parameters (declared ``low > 0``) whose origin is economically meaningful (a lookback, a
      multiplier) — never for an offset or a threshold whose zero is arbitrary;
    * ``plateau_step=<float > 0>``: an absolute step in the parameter's own units; the judge
      perturbs by ±step/2 and ±step (the radius does not apply to it).

    There is no default: a numeric parameter without one of them raises ``ValueError``.  The
    choice is written into the space JSON (``to_json``), which is stored in the study's
    ``study_created`` ledger row, so it is pre-registered.  The S2 red-team reviews every
    declared scale against the hypothesis card (a scale much finer than the card's economic
    neighbourhood is a finding).

    The judge floors every numeric perturbation at ``gates.PLATEAU_MIN_RANGE_FRACTION`` (5 %) of the
    declared range (R3-3), so a scale finer than that has no effect.

    ``levels=`` (numeric params only): an explicit, uneven grid of values in [low, high] (e.g.
    holds 12 / 24 / 48) — the candidate set uses exactly these values, the plateau perturbs the
    number itself (between and beyond the levels) with the declared scale.

    Categoricals (R2-1 / R3-3): a categorical with any choice that is a number or a string that
    parses as a number is refused (use a numeric param, with ``levels=`` for an uneven grid) —
    ordered or not.  Ordered categoricals (non-numeric labels) are perturbed by ±1 and ±2 levels
    (up to 4 points; a level beyond either end does not exist and is not scored).  Unordered
    categoricals are never perturbed: the gate lists them as "not judged — review at S2", and
    the card must name and justify each one."""

    name: str
    kind: str                                   # "int" | "float" | "categorical"
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log: bool = False
    choices: tuple = ()
    ordered: bool = False                       # categorical only: is there a natural order?
    plateau_scale: Optional[str] = None         # numeric only: "relative" (see class docstring)
    plateau_step: Optional[float] = None        # numeric only: absolute perturbation step
    levels: tuple = ()                          # numeric only: explicit (uneven) grid of values

    def __post_init__(self) -> None:
        if self.kind not in ("int", "float", "categorical"):
            raise ValueError(f"{self.name}: unknown kind {self.kind!r}")
        if self.kind == "categorical":
            if not self.choices:
                raise ValueError(f"{self.name}: categorical needs choices")
            if self.plateau_scale is not None or self.plateau_step is not None or self.levels:
                raise ValueError(f"{self.name}: plateau_scale / plateau_step / levels apply to numeric parameters "
                                 f"only (ordered categoricals are perturbed by ±1 / ±2 levels)")
            num = [c for c in self.choices if _numeric_like(c)]
            if num:
                raise ValueError(
                    f"{self.name}: a categorical with numeric choices {num} is refused (red-team R2-1 / R3-3: a "
                    f"numeric knob as a categorical escapes the plateau scale and its floor); use an "
                    f"ordered/numeric param — IntParam / FloatParam with a plateau scale, and levels=(...) for "
                    f"an uneven grid")
        else:
            if self.low is None or self.high is None or self.high < self.low:
                raise ValueError(f"{self.name}: need low <= high")
            if self.log and self.low <= 0:
                raise ValueError(f"{self.name}: log scale needs low > 0")
            if self.step is not None and self.step <= 0:
                raise ValueError(f"{self.name}: step must be > 0")
            if self.levels:
                lv = tuple(int(v) if self.kind == "int" else float(v) for v in self.levels)
                if (self.kind == "int" and any(float(v) != float(o) for v, o in zip(lv, self.levels))) \
                        or list(lv) != sorted(set(lv)) or lv[0] < self.low or lv[-1] > self.high:
                    raise ValueError(f"{self.name}: levels must be distinct, increasing, within [low, high] "
                                     f"(and ints for an IntParam); got {self.levels!r}")
                if self.step not in (None, 1) and self.kind == "int" or (self.kind == "float" and self.step is not None):
                    raise ValueError(f"{self.name}: pass either step= or levels=, not both")
                object.__setattr__(self, "levels", lv)
            has_rel, has_step = self.plateau_scale is not None, self.plateau_step is not None
            if has_rel == has_step:
                raise ValueError(
                    f"{self.name}: a numeric parameter must declare exactly one plateau scale — "
                    f"plateau_scale='relative' (strictly positive params only) or plateau_step=<float> (an "
                    f"absolute step in param units); got plateau_scale={self.plateau_scale!r}, "
                    f"plateau_step={self.plateau_step!r} (DESIGN §4.2, red-team R2-1: no silent default)")
            if has_rel:
                if self.plateau_scale != "relative":
                    raise ValueError(f"{self.name}: plateau_scale must be 'relative' (or use plateau_step=), "
                                     f"got {self.plateau_scale!r}")
                if not self.low > 0:
                    raise ValueError(f"{self.name}: plateau_scale='relative' needs a strictly positive parameter "
                                     f"(declared low > 0, got {self.low}); declare an absolute plateau_step=")
            else:
                ps = self.plateau_step
                if isinstance(ps, bool) or not _is_number(ps) or not math.isfinite(float(ps)) or float(ps) <= 0:
                    raise ValueError(f"{self.name}: plateau_step must be a finite number > 0, got {ps!r}")
                object.__setattr__(self, "plateau_step", float(ps))

    @property
    def discrete(self) -> bool:
        return self.kind != "float" or self.step is not None or bool(self.levels)

    def grid_values(self) -> list:
        if self.kind == "categorical":
            return list(self.choices)
        if self.levels:
            return list(self.levels)
        if self.kind == "int":
            return list(range(int(self.low), int(self.high) + 1, int(self.step or 1)))
        if self.step is None:
            raise ValueError(f"{self.name}: a float parameter needs step= to be enumerated on a grid")
        n = int(math.floor((self.high - self.low) / self.step + 1e-9)) + 1
        return [round(self.low + i * self.step, 12) for i in range(n)]

    def unit(self, v: Any) -> float:
        """Normalised coordinate in [0, 1] (unordered categoricals: the choice index)."""
        if self.kind == "categorical":
            i = self.choices.index(v)
            if not self.ordered:
                return float(i)
            return i / (len(self.choices) - 1) if len(self.choices) > 1 else 0.0
        lo, hi, x = float(self.low), float(self.high), float(v)
        if hi == lo:
            return 0.0
        if self.log:
            return (math.log(x) - math.log(lo)) / (math.log(hi) - math.log(lo))
        return (x - lo) / (hi - lo)

    def from_unit(self, u: float) -> Any:
        """Map a uniform ``u`` in [0, 1) onto this parameter (Sobol / random candidate sets).

        Discrete levels (categoricals, ints, stepped floats) get equal-width cells; continuous
        floats are linear or log-uniform; log-ints are log-uniform on [low, high + 1) floored.
        """
        u = min(max(float(u), 0.0), 1.0 - 1e-12)
        if self.kind == "categorical":
            return self.choices[int(u * len(self.choices))]
        if self.levels:
            return self.levels[int(u * len(self.levels))]
        if self.log:
            lo, hi = float(self.low), float(self.high)
            if self.kind == "int":
                v = math.exp(math.log(lo) + u * (math.log(hi + 1.0) - math.log(lo)))
                return int(min(max(math.floor(v), int(lo)), int(hi)))
            return float(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))
        if self.discrete:
            vals = self.grid_values()
            return vals[int(u * len(vals))]
        return float(self.low + u * (self.high - self.low))

    def suggest(self, trial: Any) -> Any:
        if self.kind == "categorical":
            return trial.suggest_categorical(self.name, list(self.choices))
        if self.levels:
            return trial.suggest_categorical(self.name, list(self.levels))
        if self.kind == "int":
            if self.log:
                return trial.suggest_int(self.name, int(self.low), int(self.high), log=True)
            return trial.suggest_int(self.name, int(self.low), int(self.high), step=int(self.step or 1))
        if self.log:
            return trial.suggest_float(self.name, float(self.low), float(self.high), log=True)
        return trial.suggest_float(self.name, float(self.low), float(self.high), step=self.step)

    def distribution(self):
        import optuna.distributions as d
        if self.kind == "categorical":
            return d.CategoricalDistribution(list(self.choices))
        if self.levels:
            return d.CategoricalDistribution(list(self.levels))
        if self.kind == "int":
            return d.IntDistribution(int(self.low), int(self.high), log=self.log,
                                     step=1 if self.log else int(self.step or 1))
        return d.FloatDistribution(float(self.low), float(self.high), log=self.log,
                                   step=None if self.log else self.step)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        out = {"name": d["name"], "kind": d["kind"]}
        for k in ("low", "high", "step"):
            if d[k] is not None:
                out[k] = d[k]
        if self.log:
            out["log"] = True
        if self.kind == "categorical":
            out["choices"] = list(self.choices)
            out["ordered"] = self.ordered
        if self.plateau_scale is not None:
            out["plateau_scale"] = self.plateau_scale
        if self.plateau_step is not None:
            out["plateau_step"] = float(self.plateau_step)
        if self.levels:
            out["levels"] = list(self.levels)
        return out


def IntParam(name: str, low: int, high: int, step: int = 1, log: bool = False, *,
             plateau_scale: Optional[str] = None, plateau_step: Optional[float] = None,
             levels: Sequence[int] = ()) -> Param:
    """Integer parameter; declare ``plateau_scale="relative"`` or ``plateau_step=`` (see :class:`Param`);
    ``levels=`` for an explicit uneven grid."""
    return Param(name, "int", low, high, step, log, plateau_scale=plateau_scale, plateau_step=plateau_step,
                 levels=tuple(levels))


def FloatParam(name: str, low: float, high: float, step: Optional[float] = None, log: bool = False, *,
               plateau_scale: Optional[str] = None, plateau_step: Optional[float] = None,
               levels: Sequence[float] = ()) -> Param:
    """Float parameter; declare ``plateau_scale="relative"`` or ``plateau_step=`` (see :class:`Param`);
    ``levels=`` for an explicit uneven grid."""
    return Param(name, "float", low, high, step, log, plateau_scale=plateau_scale, plateau_step=plateau_step,
                 levels=tuple(levels))


def CategoricalParam(name: str, choices: Sequence[Any], ordered: bool = False) -> Param:
    return Param(name, "categorical", choices=tuple(choices), ordered=ordered)


@dataclass(frozen=True)
class SearchSpace:
    """Searched parameters + an optional validity constraint (``constraint(params) -> bool``).

    Configurations failing the constraint are never evaluated but are logged as
    ``invalid`` trials (they still count toward the trial total).
    """

    params: tuple
    constraint: Optional[Callable[[dict[str, Any]], bool]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(self.params))
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate parameter names: {names}")
        if not names:
            raise ValueError("empty search space")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    @property
    def is_discrete(self) -> bool:
        return all(p.discrete for p in self.params)

    def grid_size(self) -> int:
        return int(np.prod([len(p.grid_values()) for p in self.params]))

    def grid(self) -> list[dict[str, Any]]:
        """Every grid combination (valid or not), in lexicographic order of the params."""
        vals = [p.grid_values() for p in self.params]
        return [dict(zip(self.names, combo)) for combo in itertools.product(*vals)]

    def is_valid(self, params: dict[str, Any]) -> bool:
        return True if self.constraint is None else bool(self.constraint(params))

    def suggest(self, trial: Any) -> dict[str, Any]:
        return {p.name: p.suggest(trial) for p in self.params}

    def distributions(self) -> dict[str, Any]:
        return {p.name: p.distribution() for p in self.params}

    def from_unit(self, u: Sequence[float]) -> dict[str, Any]:
        """A configuration from one point of the unit hypercube (one coordinate per param)."""
        return {p.name: p.from_unit(x) for p, x in zip(self.params, u)}

    def unit_coords(self, params_list: Sequence[dict[str, Any]]) -> np.ndarray:
        return np.array([[p.unit(pp[p.name]) for p in self.params] for pp in params_list], dtype=float
                        ).reshape(len(params_list), len(self.params))

    def to_json(self) -> dict[str, Any]:
        c = self.constraint
        return {"params": [p.to_json() for p in self.params],
                "constraint": None if c is None else getattr(c, "__name__", repr(c))}


DATA_INDEPENDENT_METHODS = ("grid", "sobol", "random")
METHODS = DATA_INDEPENDENT_METHODS + ("tpe",)

# DESIGN §4.2 plateau neighbourhood: ±20 % economic radius, pre-registered on the hypothesis card
# and stored in the study's ``study_created`` ledger row (red-team N3).  The floor keeps a
# pre-registered radius from collapsing the neighbourhood onto the selected point.
PLATEAU_RADIUS_DEFAULT = 0.20
PLATEAU_RADIUS_MIN = 0.10


def normalise_plateau_radius(radius: Any, names: Optional[Sequence[str]] = None) -> Any:
    """Canonical (JSON-able) plateau radius: ``None`` → the DESIGN default 0.20; a number →
    ``float``; a per-parameter mapping → ``{name: float}`` sorted by name (parameters not listed
    use the default).  Every value must be finite and ≥ :data:`PLATEAU_RADIUS_MIN`; mapping keys
    must be parameter names of the space when ``names`` is given."""
    def _one(v: Any, what: str) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            raise TypeError(f"plateau_radius{what} must be a number, got {v!r}")
        v = float(v)
        if not math.isfinite(v) or v < PLATEAU_RADIUS_MIN:
            raise ValueError(f"plateau_radius{what} = {v!r} is below the floor {PLATEAU_RADIUS_MIN} "
                             f"(DESIGN §4.2: ±20 % economic neighbourhood)")
        return v
    if radius is None:
        return PLATEAU_RADIUS_DEFAULT
    if isinstance(radius, Mapping):
        out = {}
        for k in sorted(radius, key=str):
            if names is not None and k not in names:
                raise ValueError(f"plateau_radius names unknown parameter {k!r} (space: {list(names)})")
            out[str(k)] = _one(radius[k], f"[{k!r}]")
        return out
    return _one(radius, "")


def space_from_json(js: Mapping[str, Any]) -> "SearchSpace":
    """Rebuild a :class:`SearchSpace` from :meth:`SearchSpace.to_json` (the constraint callable is
    not serialisable and is lost).  A pre-R2-1 space whose numeric params carry no plateau scale
    raises ``ValueError`` (it cannot be judged)."""
    ps = []
    for p in js["params"]:
        kw = {k: p[k] for k in ("low", "high", "step") if k in p}
        if p["kind"] == "categorical":
            ps.append(Param(p["name"], "categorical", choices=tuple(p["choices"]), ordered=bool(p.get("ordered"))))
        else:
            ps.append(Param(p["name"], p["kind"], log=bool(p.get("log", False)),
                            plateau_scale=p.get("plateau_scale"), plateau_step=p.get("plateau_step"),
                            levels=tuple(p.get("levels") or ()), **kw))
    return SearchSpace(tuple(ps))


def _canon(x: Any) -> Any:
    """JSON round-trip (tuples → lists, keys sorted) so call values compare with ledger values."""
    return json.loads(json.dumps(_py(x), sort_keys=True, default=str))


def candidate_set(space: SearchSpace, n: int, *, method: str = "sobol", seed: int = 0,
                  max_draws: Optional[int] = None) -> list[dict[str, Any]]:
    """``n`` distinct configurations of ``space`` fixed before any evaluation (B3).

    ``method="sobol"``: scrambled Sobol points (``scipy.stats.qmc.Sobol``, seeded by ``seed``),
    drawn in power-of-two blocks (the sequence's balance property holds for every prefix of
    size 2^m); ``"random"``: iid uniforms from ``numpy.random.default_rng(seed)``.  Each point
    is mapped by :meth:`SearchSpace.from_unit`; duplicates (discrete spaces) are dropped and
    drawing continues until ``n`` distinct configurations exist or ``max_draws`` points were
    drawn (default ``max(64·n, 65536)``; a warning reports a short set, e.g. a discrete space
    smaller than ``n``).  Validity is *not* checked here (invalid configurations are part of
    the searched space and are logged as ``invalid`` trials).  The first *k* configurations do
    not depend on ``n`` (the sequence is the same), so a resumed study with a larger ``n``
    extends the same candidate set.
    """
    if method not in ("sobol", "random"):
        raise ValueError("candidate_set method must be 'sobol' or 'random'")
    n = int(n)
    if n < 1:
        raise ValueError("n must be >= 1")
    d = len(space.params)
    if method == "sobol":
        from scipy.stats import qmc
        eng = qmc.Sobol(d, scramble=True, rng=np.random.default_rng(seed))
        draw = eng.random
    else:
        rng = np.random.default_rng(seed)
        draw = lambda m: rng.random((m, d))   # noqa: E731
    block = 1 << max(0, (n - 1).bit_length())
    cap = int(max_draws) if max_draws is not None else max(64 * n, 1 << 16)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    drawn = 0
    while len(out) < n and drawn < cap:
        m = min(block, cap - drawn)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # qmc balance warning when the cap truncates
            U = draw(m)
        drawn += m
        for u in U:
            p = space.from_unit(u)
            k = _pjson(p)
            if k in seen:
                continue
            seen.add(k)
            out.append(p)
            if len(out) == n:
                break
        block = drawn                                # doubling: every total drawn is 2^m
    if len(out) < n:
        warnings.warn(f"{method} candidate set: only {len(out)} distinct configurations after {drawn} draws "
                      f"(requested {n}); the space may be smaller than n_trials", stacklevel=2)
    return out


def _resolve_method(method: Optional[str], space: SearchSpace) -> str:
    if method in (None, "auto"):
        return "grid" if len(space.params) <= 3 and space.is_discrete else "sobol"
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS + ('auto',)}, got {method!r}")
    return method


# =========================================================================== configs
@dataclass(frozen=True)
class Objective:
    """Robust, net-of-cost objective on a (T_window × M) block of daily returns."""

    kind: str = "block_sharpe"          # "block_sharpe" | "sharpe"
    n_blocks: int = 4
    lam: float = 0.5
    min_trades: float = 0.0             # on the full window; scaled by rows for sub-windows

    def __post_init__(self) -> None:
        if self.kind not in ("block_sharpe", "sharpe"):
            raise ValueError(f"unknown objective kind {self.kind!r}")
        if self.n_blocks < 1:
            raise ValueError("n_blocks must be >= 1")

    def describe(self) -> str:
        if self.kind == "sharpe":
            return f"sharpe(min_trades={self.min_trades:g})"
        return f"block_sharpe(K={self.n_blocks},lam={self.lam:g},min_trades={self.min_trades:g})"

    def raw(self, R: np.ndarray, ppy: float) -> np.ndarray:
        """Objective per column, ignoring the trade constraint (NaN → −inf)."""
        R = np.asarray(R, dtype=float)
        if R.ndim == 1:
            R = R[:, None]
        if self.kind == "sharpe" or self.n_blocks == 1:
            out = _col_sharpe(R, ppy)
        else:
            T = R.shape[0]
            if T < 2 * self.n_blocks:
                return np.full(R.shape[1], -np.inf)
            blocks = np.array_split(np.arange(T), self.n_blocks)
            S = np.vstack([_col_sharpe(R[b], ppy) for b in blocks])
            out = S.mean(axis=0) - self.lam * S.std(axis=0, ddof=1)
        return np.where(np.isfinite(out), out, -np.inf)

    def trade_ok(self, C: Optional[np.ndarray], window_rows: int, total_rows: int,
                 fallback: Optional[np.ndarray] = None) -> np.ndarray:
        if self.min_trades <= 0:
            n = C.shape[1] if C is not None else (len(fallback) if fallback is not None else 0)
            return np.ones(n, dtype=bool)
        need = self.min_trades * window_rows / max(total_rows, 1)
        if C is None:
            return np.asarray(fallback, dtype=bool)
        return C.sum(axis=0) >= need - 1e-9

    def score(self, R: np.ndarray, C: Optional[np.ndarray], ppy: float, *, total_rows: int,
              fallback_ok: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        raw = self.raw(R, ppy)
        ok = self.trade_ok(C, R.shape[0], total_rows, fallback_ok)
        return np.where(ok, raw, -np.inf), raw


def _col_sharpe(R: np.ndarray, ppy: float) -> np.ndarray:
    if R.shape[0] < 2:
        return np.zeros(R.shape[1])
    mu = R.mean(axis=0)
    sd = R.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(sd > 0, mu / sd * math.sqrt(ppy), 0.0)
    return s


@dataclass(frozen=True)
class PlateauConfig:
    neighbourhood: str = "auto"         # "grid" | "knn" | "auto"
    radius: int = 1                     # grid: Chebyshev radius in grid steps
    knn: Optional[int] = None           # knn: k (default max(4, 2·d))
    peak_fraction: float = 0.5          # DESIGN §4.2: neighbours within 50% of the peak
    min_neighbours: int = 1             # pad sparser neighbourhoods with the floor value
    min_centre_quantile: float = 0.5    # the centre's own objective must be >= this quantile

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 10
    k_test: int = 2
    embargo_days: Optional[int] = None  # None → max holding period of ok trials (trading days)
    purge_days: Optional[int] = None    # None → same as embargo

    def describe(self, embargo: Any = "auto", purge: Any = "auto") -> str:
        return f"CPCV(n={self.n_groups},k={self.k_test},purge={purge}d,embargo={embargo}d)"


@dataclass(frozen=True)
class WFOConfig:
    refit_every: str = "3mo"
    window: str = "anchored"            # "anchored" | "rolling"
    window_length: str = "3y"           # rolling only
    min_train: str = "2y"

    def __post_init__(self) -> None:
        if self.window not in ("anchored", "rolling"):
            raise ValueError("window must be 'anchored' or 'rolling'")

    def describe(self) -> str:
        w = self.window if self.window == "anchored" else f"rolling:{self.window_length}"
        return f"WFO(refit={self.refit_every},window={w},min_train={self.min_train})"


# =========================================================================== neighbourhoods / plateau
def _resolve_mode(cfg: PlateauConfig, space: SearchSpace, params_list: Sequence[dict[str, Any]],
                  method: Optional[str] = None) -> str:
    if cfg.neighbourhood in ("grid", "knn"):
        return cfg.neighbourhood
    if method is not None:
        return "grid" if method == "grid" else "knn"
    if space.is_discrete and len(params_list) >= 0.5 * space.grid_size():
        return "grid"
    return "knn"


def neighbourhoods(space: SearchSpace, params_list: Sequence[dict[str, Any]],
                   cfg: PlateauConfig = PlateauConfig(), *, method: Optional[str] = None) -> np.ndarray:
    """Boolean ``M × M`` neighbour matrix (diagonal False) — see the module docstring."""
    M, d = len(params_list), len(space.params)
    if M == 0:
        return np.zeros((0, 0), dtype=bool)
    mode = _resolve_mode(cfg, space, params_list, method)
    unordered = [i for i, p in enumerate(space.params) if p.kind == "categorical" and not p.ordered]
    same_cat = np.ones((M, M), dtype=bool)
    for i in unordered:
        p = space.params[i]
        v = np.array([p.choices.index(pp[p.name]) for pp in params_list])
        same_cat &= v[:, None] == v[None, :]
    if mode == "grid":
        idx = np.empty((M, d))
        for i, p in enumerate(space.params):
            try:
                gv = p.grid_values()
            except ValueError:
                gv = sorted({pp[p.name] for pp in params_list})
            pos = {v: k for k, v in enumerate(gv)}
            idx[:, i] = [pos[pp[p.name]] if pp[p.name] in pos else
                         int(np.argmin([abs(float(pp[p.name]) - float(g)) for g in gv]))
                         for pp in params_list]
        num = [i for i in range(d) if i not in unordered]
        if num:
            cheb = np.max(np.abs(idx[:, None, num] - idx[None, :, num]), axis=2)
            nb = (cheb <= cfg.radius) & same_cat
        else:
            nb = same_cat.copy()
    else:
        U = space.unit_coords(params_list)
        num = [i for i in range(d) if i not in unordered]
        D = np.sqrt(((U[:, None, num] - U[None, :, num]) ** 2).sum(axis=2)) if num else np.zeros((M, M))
        D = np.where(same_cat, D, np.inf)
        np.fill_diagonal(D, np.inf)
        k = min(cfg.knn or max(4, 2 * d), M - 1)
        nb = np.zeros((M, M), dtype=bool)
        if k > 0:
            order = np.argsort(D, axis=1, kind="stable")[:, :k]
            rows = np.repeat(np.arange(M), k)
            cols = order.ravel()
            keep = np.isfinite(D[rows, cols])
            nb[rows[keep], cols[keep]] = True
    np.fill_diagonal(nb, False)
    return nb


def plateau_select(scores: np.ndarray, nbr: np.ndarray, *, eligible: Optional[np.ndarray] = None,
                   gate: Optional[np.ndarray] = None, cfg: PlateauConfig = PlateauConfig()) -> dict[str, Any]:
    """Pick the centre of the best broad region of ``scores`` (see module docstring).

    ``scores``: objective per configuration (−inf / NaN = failed).  ``nbr``: neighbour
    matrix from :func:`neighbourhoods`.  ``gate``: metric for the DESIGN §4.2 plateau
    score (window Sharpe); defaults to ``scores``.  Returns positional indices.
    """
    s = np.asarray(scores, dtype=float)
    M = s.size
    fin = np.isfinite(s)
    elig = fin if eligible is None else (np.asarray(eligible, dtype=bool) & fin)
    if not elig.any():
        return {"status": "no_eligible", "index": None, "raw_index": None}
    floor = float(s[fin].min())
    v = np.where(fin, s, floor)
    nb = np.asarray(nbr, dtype=bool)
    n_nb = nb.sum(axis=1)
    pad = np.maximum(cfg.min_neighbours - n_nb, 0)
    smoothed = (v + nb.astype(float) @ v + pad * floor) / (1 + n_nb + pad)
    cand = elig
    if cfg.min_centre_quantile > 0:
        cand = elig & (s >= np.quantile(s[elig], cfg.min_centre_quantile))
    sm = np.where(cand, smoothed, -np.inf)
    w = int(np.argmax(sm))                      # first max → lowest index on ties
    raw = int(np.argmax(np.where(elig, s, -np.inf)))
    g = s if gate is None else np.asarray(gate, dtype=float)

    def _pscore(metric: np.ndarray, i: int) -> float:
        nbi = np.flatnonzero(nb[i])
        if nbi.size == 0:
            return float("nan")
        peak = metric[i]
        if not np.isfinite(peak) or peak <= 0:
            return 0.0
        vals = metric[nbi]
        within = np.isfinite(vals) & fin[nbi] & (vals >= cfg.peak_fraction * peak)
        return float(within.mean())

    return {
        "status": "ok", "index": w, "raw_index": raw,
        "objective": float(s[w]), "smoothed": float(smoothed[w]),
        "raw_objective": float(s[raw]), "raw_smoothed": float(smoothed[raw]),
        "plateau_score": _pscore(g, w), "plateau_score_objective": _pscore(s, w),
        "raw_plateau_score": _pscore(g, raw),
        "n_neighbours": int(n_nb[w]), "gate_value": float(g[w]) if np.isfinite(g[w]) else float("nan"),
        "smoothed_all": smoothed,
    }


# =========================================================================== surface (matrix + trials)
@dataclass
class _Surface:
    trial_ids: np.ndarray           # (M,) trials that are real systems (status != invalid)
    status: np.ndarray              # (M,)
    params: list                    # (M,) dicts
    U: np.ndarray                   # (M, d) unit coords
    nbr: np.ndarray                 # (M, M)
    dates: np.ndarray               # (T,) datetime64[D]
    R: np.ndarray                   # (T, M) returns (0 where the trial has none / failed)
    C: Optional[np.ndarray]         # (T, M) trades closed per day, or None
    E: Optional[np.ndarray]         # (T, M) trades entered per day, or None
    has_E: np.ndarray               # (M,) bool: the trial has an entry-count column
    has_ret: np.ndarray             # (M,) bool
    full_ok: np.ndarray             # (M,) bool: status == "ok"
    mode: str


def _params_from_trials(trials: pl.DataFrame, space: SearchSpace) -> list[dict[str, Any]]:
    cols = [f"param_{n}" for n in space.names]
    if all(c in trials.columns for c in cols):
        return [dict(zip(space.names, row)) for row in trials.select(cols).iter_rows()]
    return [json.loads(p) for p in trials["params"]]


def _wide_to_np(df: Optional[pl.DataFrame], ids: np.ndarray, dates: Optional[np.ndarray] = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = df.sort("date")
    d = df["date"].cast(pl.Date).to_numpy().astype("datetime64[D]")
    X = np.zeros((d.size, ids.size))
    has = np.zeros(ids.size, dtype=bool)
    cols = set(df.columns)
    for j, t in enumerate(ids):
        c = f"t{int(t)}"
        if c in cols:
            X[:, j] = df[c].fill_null(0.0).fill_nan(0.0).to_numpy()
            has[j] = True
    if dates is not None and not np.array_equal(d, dates):
        Y = np.zeros((dates.size, ids.size))
        pos = np.searchsorted(dates, d)
        okp = (pos < dates.size) & (dates[np.clip(pos, 0, dates.size - 1)] == d)
        Y[pos[okp]] = X[okp]
        return dates, Y, has
    return d, X, has


def _build_surface(returns: pl.DataFrame, trials: pl.DataFrame, space: SearchSpace,
                   selection: PlateauConfig, trade_counts: Optional[pl.DataFrame] = None,
                   method: Optional[str] = None, entry_counts: Optional[pl.DataFrame] = None) -> _Surface:
    tr = trials.filter(pl.col("status") != "invalid").sort("trial_id")
    ids = tr["trial_id"].to_numpy()
    status = tr["status"].to_numpy().astype(str)
    params = _params_from_trials(tr, space)
    dates, R, has = _wide_to_np(returns, ids)
    C = None
    if trade_counts is not None and trade_counts.height:
        _, C, _ = _wide_to_np(trade_counts, ids, dates)
    E, has_E = None, np.zeros(ids.size, dtype=bool)
    if entry_counts is not None and entry_counts.height:
        _, E, has_E = _wide_to_np(entry_counts, ids, dates)
    ok = status == "ok"
    R[:, ~ok & ~np.isin(status, ["low_trades"])] = 0.0
    mode = _resolve_mode(selection, space, params, method)
    nbr = neighbourhoods(space, params, PlateauConfig(**{**asdict(selection), "neighbourhood": mode}))
    return _Surface(trial_ids=ids, status=status, params=params, U=space.unit_coords(params), nbr=nbr,
                    dates=dates, R=R, C=C, E=E, has_E=has_E, has_ret=has, full_ok=ok, mode=mode)


def _select_rows(S: _Surface, rows: np.ndarray, objective: Objective, cfg: PlateauConfig,
                 ppy: float) -> dict[str, Any]:
    """Objective + plateau selection using only ``rows`` of the matrix."""
    Rw = S.R[rows]
    Cw = S.C[rows] if S.C is not None else None
    evaluable = S.has_ret & np.isin(S.status, ["ok", "low_trades"])
    scores, raw = objective.score(Rw, Cw, ppy, total_rows=S.R.shape[0], fallback_ok=S.full_ok)
    scores = np.where(evaluable, scores, -np.inf)
    gate = np.where(evaluable, _col_sharpe(Rw, ppy), np.nan)
    sel = plateau_select(scores, S.nbr, eligible=evaluable, gate=gate, cfg=cfg)
    sel["scores"] = scores
    sel["gate"] = gate
    return sel


# =========================================================================== CPCV
@dataclass(frozen=True)
class CPCVSplit:
    split_id: int
    test_groups: tuple[int, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray


def cpcv_splits(dates: Sequence[Any] | int, n_groups: int = 10, k_test: int = 2, *,
                embargo_days: int = 5, purge_days: Optional[int] = None) -> tuple[list[CPCVSplit], list[np.ndarray]]:
    """Combinatorial purged splits over trading-day rows (see module docstring).

    Returns ``(splits, groups)``; ``groups[g]`` = row indices of group *g*.  ``purge_days``
    defaults to ``embargo_days``.  Both are counted in rows (trading days).
    """
    T = dates if isinstance(dates, int) else len(dates)
    if not 1 <= k_test < n_groups:
        raise ValueError("need 1 <= k_test < n_groups")
    if T < n_groups:
        raise ValueError(f"{T} rows cannot be cut into {n_groups} groups")
    purge = embargo_days if purge_days is None else purge_days
    groups = np.array_split(np.arange(T), n_groups)
    splits = []
    for sid, combo in enumerate(itertools.combinations(range(n_groups), k_test)):
        test = np.zeros(T, dtype=bool)
        for g in combo:
            test[groups[g]] = True
        drop = test.copy()
        # contiguous test blocks (adjacent groups merge into one block)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], test.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):       # block rows [a, b)
            drop[max(0, a - purge):a] = True
            drop[b:min(T, b + embargo_days)] = True
        splits.append(CPCVSplit(sid, tuple(combo), np.flatnonzero(~drop), np.flatnonzero(test)))
    return splits, groups


def cpcv_path_map(n_groups: int, k_test: int) -> np.ndarray:
    """``(n_paths, n_groups)`` array: split id supplying group *g* on path *p* (LdP)."""
    combos = list(itertools.combinations(range(n_groups), k_test))
    phi = math.comb(n_groups - 1, k_test - 1)
    out = np.full((phi, n_groups), -1, dtype=int)
    for g in range(n_groups):
        occ = [s for s, c in enumerate(combos) if g in c]
        assert len(occ) == phi
        out[:, g] = occ
    return out


def _resolve_embargo(cv: CPCVConfig, trials: pl.DataFrame, T: int) -> tuple[int, int, dict[str, Any]]:
    """``(embargo, purge, info)``; ``info`` = ``embargo_capped`` (the auto embargo — max holding
    period of the ok trials — exceeded a quarter of a CPCV group and was cut: long-hold
    systems are then under-purged, red-team m3), ``embargo_days_uncapped``, ``embargo_cap``."""
    cap = max(1, (T // cv.n_groups) // 4)
    capped = False
    if cv.embargo_days is not None:
        emb = uncapped = int(cv.embargo_days)
    else:
        hold = 0.0
        if "m_hold_days_max" in trials.columns:
            h = trials.filter(pl.col("status") == "ok")["m_hold_days_max"].drop_nulls().drop_nans()
            hold = float(h.max()) if h.len() else 0.0
        emb = uncapped = max(1, int(math.ceil(hold)))
        if emb > cap:
            warnings.warn(f"auto embargo {emb}d capped at {cap}d (a quarter of a CPCV group); the OOS "
                          f"gates will be SKIPPED — use fewer CPCV groups", stacklevel=3)
            emb, capped = cap, True
    purge = emb if cv.purge_days is None else int(cv.purge_days)
    return emb, purge, {"embargo_capped": capped, "embargo_days_uncapped": uncapped, "embargo_cap": cap}


def cpcv_paths(returns: pl.DataFrame, trials: pl.DataFrame, space: SearchSpace,
               cv: CPCVConfig = CPCVConfig(), *, objective: Objective = Objective(),
               selection: PlateauConfig = PlateauConfig(), trade_counts: Optional[pl.DataFrame] = None,
               periods_per_year: float = 260.0, method: Optional[str] = None,
               _surface: Optional[_Surface] = None) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """OOS paths of the *selection procedure* under CPCV.

    Returns ``(paths, splits, meta)``: ``paths`` (date, path_id, ret); ``splits`` one row
    per split (selected trial, train objective, plateau score, OOS Sharpe of the pick,
    and ``oos_rank_pct`` = share of evaluable trials whose OOS Sharpe on that split's
    test rows is below the pick's — a CPCV-PBO diagnostic).
    """
    S = _surface or _build_surface(returns, trials, space, selection, trade_counts, method)
    cfg = PlateauConfig(**{**asdict(selection), "neighbourhood": S.mode})
    T = S.R.shape[0]
    emb, purge, emb_info = _resolve_embargo(cv, trials, T)
    splits, groups = cpcv_splits(T, cv.n_groups, cv.k_test, embargo_days=emb, purge_days=purge)
    pmap = cpcv_path_map(cv.n_groups, cv.k_test)
    picks: list[Optional[int]] = []
    rows = []
    evaluable = S.has_ret & np.isin(S.status, ["ok", "low_trades"])
    for sp in splits:
        sel = _select_rows(S, sp.train_idx, objective, cfg, periods_per_year)
        w = sel["index"]
        picks.append(w)
        te = S.R[sp.test_idx]
        oos = np.where(evaluable, _col_sharpe(te, periods_per_year), np.nan)
        if w is None:
            rows.append({"split_id": sp.split_id, "test_groups": ",".join(map(str, sp.test_groups)),
                         "selected_trial": None, "train_objective": None, "train_smoothed": None,
                         "plateau_score": None, "oos_sharpe": None, "oos_rank_pct": None,
                         "n_train": int(sp.train_idx.size), "n_test": int(sp.test_idx.size)})
            continue
        others = np.flatnonzero(evaluable & np.isfinite(oos))
        others = others[others != w]
        rank = float(np.mean(oos[others] < oos[w])) if others.size else float("nan")
        rows.append({"split_id": sp.split_id, "test_groups": ",".join(map(str, sp.test_groups)),
                     "selected_trial": int(S.trial_ids[w]), "train_objective": sel["objective"],
                     "train_smoothed": sel["smoothed"], "plateau_score": sel["plateau_score"],
                     "oos_sharpe": float(oos[w]), "oos_rank_pct": rank,
                     "n_train": int(sp.train_idx.size), "n_test": int(sp.test_idx.size)})
    parts = []
    for p in range(pmap.shape[0]):
        ret = np.zeros(T)
        for g in range(cv.n_groups):
            w = picks[pmap[p, g]]
            if w is not None:
                ret[groups[g]] = S.R[groups[g], w]
        parts.append(pl.DataFrame({"date": S.dates, "path_id": np.full(T, p, dtype=np.int32), "ret": ret}))
    paths = pl.concat(parts).with_columns(pl.col("date").cast(pl.Date)) if parts else pl.DataFrame()
    meta = {"n_groups": cv.n_groups, "k_test": cv.k_test, "n_splits": len(splits), "n_paths": int(pmap.shape[0]),
            "embargo_days": emb, "purge_days": purge, "scheme": cv.describe(emb, purge), **emb_info,
            "n_no_selection": int(sum(p is None for p in picks))}
    split_df = pl.DataFrame(rows, schema={
        "split_id": pl.Int64, "test_groups": pl.Utf8, "selected_trial": pl.Int64, "train_objective": pl.Float64,
        "train_smoothed": pl.Float64, "plateau_score": pl.Float64, "oos_sharpe": pl.Float64,
        "oos_rank_pct": pl.Float64, "n_train": pl.Int64, "n_test": pl.Int64})
    return paths, split_df, meta


# =========================================================================== walk-forward
def _offset(d: date, by: str) -> date:
    return pl.Series([d]).cast(pl.Date).dt.offset_by(by).item()


def walk_forward(returns: pl.DataFrame, trials: pl.DataFrame, space: SearchSpace,
                 schedule: WFOConfig = WFOConfig(), *, objective: Objective = Objective(),
                 selection: PlateauConfig = PlateauConfig(), trade_counts: Optional[pl.DataFrame] = None,
                 periods_per_year: float = 260.0, method: Optional[str] = None,
                 entry_counts: Optional[pl.DataFrame] = None,
                 _surface: Optional[_Surface] = None) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any]]:
    """Simulate the re-optimisation schedule (DESIGN §4.6) on the returns matrix.

    Returns ``(wfo_oos, wfo_params, meta)``: ``wfo_oos`` (date, ret, refit_id, n_trades) —
    ``n_trades`` = trades the active configuration *entered* on that OOS day (from
    ``entry_counts``, wide ``date`` + ``t<id>``; null where unavailable, e.g. no
    ``entry_ts`` in the evaluator's trades or no counts passed);
    ``wfo_params`` (refit_id, refit_date, train_start, train_end, test_end, selected_trial,
    param_<name>…, train_objective, plateau_score); ``meta`` with drift diagnostics.
    Every refit sees only rows with date < refit_date (and, for a data-dependent candidate set
    such as TPE's, the columns themselves were chosen with the full sample — see module docstring).
    """
    S = _surface or _build_surface(returns, trials, space, selection, trade_counts, method, entry_counts)
    cfg = PlateauConfig(**{**asdict(selection), "neighbourhood": S.mode})
    dates = S.dates
    if dates.size == 0:
        raise StudyError("empty returns matrix")
    d_first = dates[0].astype(object)
    d_last = dates[-1].astype(object)
    first_refit = _offset(d_first, schedule.min_train)
    if first_refit > d_last:
        raise StudyError(f"min_train={schedule.min_train} leaves no out-of-sample period")
    refits = pl.date_range(first_refit, d_last, interval=schedule.refit_every, eager=True).to_list()
    refit_np = np.array(refits, dtype="datetime64[D]")
    oos_parts, prow, sel_units = [], [], []
    rid = 0
    for k, rd in enumerate(refit_np):
        nxt = refit_np[k + 1] if k + 1 < refit_np.size else dates[-1] + np.timedelta64(1, "D")
        test = np.flatnonzero((dates >= rd) & (dates < nxt))
        if test.size == 0:
            continue
        if schedule.window == "anchored":
            ts = dates[0]
        else:
            ts = max(dates[0], np.datetime64(_offset(rd.astype(object), "-" + schedule.window_length), "D"))
        train = np.flatnonzero((dates >= ts) & (dates < rd))
        if train.size < 2 * max(objective.n_blocks, 2):
            continue
        sel = _select_rows(S, train, objective, cfg, periods_per_year)
        w = sel["index"]
        ret = S.R[test, w] if w is not None else np.zeros(test.size)
        if w is None:
            ntr = pl.Series("n_trades", np.zeros(test.size), dtype=pl.Float64).cast(pl.Int64)
        elif S.E is not None and S.has_E[w]:
            ntr = pl.Series("n_trades", S.E[test, w]).round(0).cast(pl.Int64)
        else:
            ntr = pl.Series("n_trades", [None] * test.size, dtype=pl.Int64)
        oos_parts.append(pl.DataFrame({"date": dates[test], "ret": ret, "refit_id": np.full(test.size, rid, dtype=np.int32),
                                       "n_trades": ntr}))
        row = {"refit_id": rid, "refit_date": rd.astype(object), "train_start": dates[train[0]].astype(object),
               "train_end": dates[train[-1]].astype(object), "test_end": dates[test[-1]].astype(object),
               "selected_trial": int(S.trial_ids[w]) if w is not None else None}
        for n in space.names:
            row[f"param_{n}"] = S.params[w][n] if w is not None else None
        row["train_objective"] = sel.get("objective")
        row["plateau_score"] = sel.get("plateau_score")
        prow.append(row)
        sel_units.append(S.U[w] if w is not None else None)
        rid += 1
    wfo_oos = (pl.concat(oos_parts).with_columns(pl.col("date").cast(pl.Date)) if oos_parts
               else pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64, "refit_id": pl.Int32, "n_trades": pl.Int64}))
    wfo_params = pl.DataFrame(prow) if prow else pl.DataFrame()
    meta = {"schedule": asdict(schedule), "scheme": schedule.describe(), "n_refits": rid,
            **_drift(sel_units, space, [r["selected_trial"] for r in prow])}
    return wfo_oos, wfo_params, meta


def _drift(units: list, space: SearchSpace, picks: list) -> dict[str, Any]:
    d = len(space.params)
    unordered = [i for i, p in enumerate(space.params) if p.kind == "categorical" and not p.ordered]
    dist = []
    for a, b in zip(units, units[1:]):
        if a is None or b is None:
            continue
        diff = np.abs(a - b)
        for i in unordered:
            diff[i] = float(a[i] != b[i])
        dist.append(float(np.sqrt((diff ** 2).sum() / d)))
    arr = np.asarray(dist)
    return {
        "drift_distances": dist,
        "drift_mean": float(arr.mean()) if arr.size else 0.0,
        "drift_max": float(arr.max()) if arr.size else 0.0,
        "drift_share_jumps_gt_0.25": float((arr > 0.25).mean()) if arr.size else 0.0,
        "n_distinct_selections": len({p for p in picks if p is not None}),
    }


# =========================================================================== evaluation plumbing
_W: dict[str, Any] = {}

# F1: thread caps every worker process must start with (before Polars / Numba / BLAS
# initialise their pools — i.e. in the environment the child is *launched* with).
WORKER_THREAD_ENV: dict[str, str] = {
    "POLARS_MAX_THREADS": "1", "NUMBA_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "RAYON_NUM_THREADS": "1",
}
_ENV_LOCK = threading.Lock()


@contextlib.contextmanager
def _capped_env() -> Iterator[None]:
    """Set :data:`WORKER_THREAD_ENV` in this process's environment, restore on exit."""
    with _ENV_LOCK:
        saved = {k: os.environ.get(k) for k in WORKER_THREAD_ENV}
        os.environ.update(WORKER_THREAD_ENV)
        try:
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


class _CappedSpawnProcess(multiprocessing.context.SpawnProcess):
    # ProcessPoolExecutor launches spawn workers on demand (at submit, and to replace dead
    # workers), not at construction — so the caps are applied around *each* launch.
    def start(self) -> None:
        with _capped_env():
            super().start()


class _CappedSpawnContext(multiprocessing.context.SpawnContext):
    Process = _CappedSpawnProcess


def _references_interactive_main(obj: Any) -> bool:
    """True if pickling ``obj`` references ``__main__`` and ``__main__`` has no file (a notebook
    / REPL): a spawned child could not import those globals."""
    main = sys.modules.get("__main__")
    if main is not None and getattr(main, "__file__", None) and "ipykernel" not in sys.modules:
        return False
    try:
        blob = pickle.dumps(obj)
    except Exception:
        return False
    return b"__main__" in blob


def _pool_context(payload: Any) -> tuple[Any, str]:
    forced = os.environ.get("QUANTLAB_MP_START", "").strip().lower()
    if forced == "fork" or (not forced and _references_interactive_main(payload)):
        if not forced:
            warnings.warn("run_study: the evaluator references an interactive __main__ (notebook/REPL), which a "
                          "spawned worker cannot import — using a fork pool; worker thread caps (F1) cannot be "
                          "applied, so Polars may oversubscribe. Move the strategy into a module to fix.",
                          stacklevel=4)
        return multiprocessing.get_context("fork"), "fork"
    return _CappedSpawnContext(), "spawn"


def _worker_threads(_: Any = None) -> dict[str, Any]:
    """Diagnostic run inside a worker: its real Polars pool size and the cap env vars."""
    return {"pid": os.getpid(), "polars_threads": int(pl.thread_pool_size()),
            **{k: os.environ.get(k) for k in WORKER_THREAD_ENV}}


def _worker_init(evaluator: Any, cost: Any) -> None:
    # Receive the evaluator config once, load its data once per worker.  The thread caps
    # are already in this process's environment from launch (spawn, F1); setting them
    # here is only a fallback for fork pools and has no effect on an initialised Polars.
    for k, v in WORKER_THREAD_ENV.items():
        os.environ.setdefault(k, v)
    _W["ev"], _W["cost"], _W["init_error"] = evaluator, cost, None
    try:
        prep = getattr(evaluator, "prepare", None)
        if prep is not None:
            prep()
    except Exception as e:  # surfaced per task as an error trial rather than a broken pool
        _W["init_error"] = f"worker init failed: {type(e).__name__}: {e}"


def _worker_eval(params: dict[str, Any]) -> dict[str, Any]:
    if _W.get("init_error"):
        return {"ok": False, "error": _W["init_error"]}
    return _evaluate(_W["ev"], params, _W["cost"])


def _trade_counts(trades: pl.DataFrame, dates: np.ndarray, col: Optional[str] = None) -> np.ndarray:
    """Trades per row of ``dates`` by the server date of ``col`` (default: exit, i.e. trades
    closed per day; a date between rows maps to the next row).  Skipped trades excluded."""
    out = np.zeros(dates.size)
    if trades is None or trades.height == 0 or dates.size == 0:
        return out
    t = trades
    if "skipped" in t.columns:
        t = t.filter(~pl.col("skipped"))
    if col is None:
        col = "exit_ts" if "exit_ts" in t.columns else ("date" if "date" in t.columns else None)
    if col is None or col not in t.columns or t.height == 0:
        return out
    s = t[col]
    xd = (s.dt.date() if s.dtype != pl.Date else s).to_numpy().astype("datetime64[D]")
    pos = np.clip(np.searchsorted(dates, xd, side="left"), 0, dates.size - 1)
    np.add.at(out, pos, 1.0)
    return out


def _evaluate(ev: Any, params: dict[str, Any], cost: Any) -> dict[str, Any]:
    try:
        out = ev(params, cost=cost)
        daily = out.daily.sort("date")
        dates = daily["date"].cast(pl.Date).to_numpy().astype("datetime64[D]")
        ret = daily["ret"].cast(pl.Float64).fill_null(0.0).to_numpy()
        counts = _trade_counts(out.trades, dates)
        has_entry = out.trades is not None and "entry_ts" in out.trades.columns
        entries = _trade_counts(out.trades, dates, "entry_ts") if has_entry else None
        m: dict[str, float] = {}
        for k, v in (out.metrics or {}).items():
            try:
                m[k] = float(v)
            except (TypeError, ValueError):
                pass
        m.setdefault("n_trades", float(counts.sum()))
        return {"ok": True, "dates": dates, "ret": ret, "counts": counts, "entries": entries, "metrics": m}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=6)}


@dataclass
class _Rec:
    trial_id: int
    params: dict[str, Any]
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    dates: Optional[np.ndarray] = None
    ret: Optional[np.ndarray] = None
    counts: Optional[np.ndarray] = None
    error: Optional[str] = None
    entries: Optional[np.ndarray] = None
    source: str = ""                # how the configuration was proposed: grid / sobol / random / tpe (N1)


class _Runner:
    """Evaluates parameter lists in order, in-process or on a persistent process pool."""

    def __init__(self, evaluator: Any, cost: Any, n_jobs: int):
        self.ev, self.cost, self.n_jobs = evaluator, cost, n_jobs
        self.pool: Optional[ProcessPoolExecutor] = None
        self.start_method: Optional[str] = None
        if n_jobs > 1:
            ctx, self.start_method = _pool_context((evaluator, cost))
            self.pool = ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx, initializer=_worker_init,
                                            initargs=(evaluator, cost))

    def worker_threads(self) -> Optional[dict[str, Any]]:
        """F1 diagnostic from one worker (None in-process)."""
        if self.pool is None:
            return None
        return self.pool.submit(_worker_threads).result()

    def map(self, plist: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Results in submission order, *streamed*: the caller records each trial as soon as
        it is available, so an abort loses at most the in-flight evaluations."""
        if not plist:
            return iter(())
        if self.pool is None:
            return (_evaluate(self.ev, p, self.cost) for p in plist)
        cs = max(1, len(plist) // (self.n_jobs * 4))
        return self.pool.map(_worker_eval, plist, chunksize=cs)

    def close(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def _pjson(params: dict[str, Any]) -> str:
    return json.dumps(_py(params), sort_keys=True, default=str)


def _py(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def _matrix(recs: Sequence[_Rec], field_: str = "ret") -> pl.DataFrame:
    """Wide ``date`` + ``t<trial_id>`` frame over the union of dates (missing = 0.0)."""
    with_ret = [r for r in recs if getattr(r, field_) is not None]
    if not with_ret:
        return pl.DataFrame(schema={"date": pl.Date})
    dates = np.unique(np.concatenate([r.dates for r in with_ret]))
    cols: dict[str, Any] = {"date": dates}
    for r in with_ret:
        col = np.zeros(dates.size)
        col[np.searchsorted(dates, r.dates)] = getattr(r, field_)
        cols[f"t{r.trial_id}"] = col
    return pl.DataFrame(cols).with_columns(pl.col("date").cast(pl.Date))


def _evaluator_symbols(evaluator: Any, book: str) -> tuple[list[str], list[str]]:
    """(traded symbols, conversion legs) of a real evaluator, recorded in the ``study_created`` row
    (R2-2: they scope holdout access and the holdout horizon).  Legs = the series
    ``data.conversion_rate`` reads for the instrument's quote and swap currencies → account
    currency.  Synthetic evaluators (no instrument) → ([], [])."""
    from .evaluators import SyntheticEvaluator
    if (isinstance(evaluator, SyntheticEvaluator) or getattr(evaluator, "is_synthetic", False)
            or isinstance(getattr(evaluator, "inner", None), SyntheticEvaluator)):
        return [], []
    sym = getattr(evaluator, "symbol", None)
    if not sym:
        return [], []
    from . import data
    ccys: list[str] = []
    spec = getattr(evaluator, "spec", None)
    if spec is None:
        try:
            from .costs import load_instrument
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = load_instrument(sym, book=book)
        except Exception:  # noqa: BLE001 — the quote-currency leg below still applies
            spec = None
    if spec is not None:
        ccys = [c for c in (getattr(spec, "quote_ccy", None), getattr(spec, "swap_ccy", None)) if c]
    try:
        legs = data.conversion_legs([sym], book=book, currencies=ccys)
    except ValueError as e:
        raise StudyError(f"cannot resolve the conversion legs of {sym!r} ({e}); the study's holdout symbols "
                         f"must be registered at creation (R2-2)") from e
    return [str(sym)], legs


def evaluator_code_fingerprint(evaluator: Any) -> Optional[dict[str, Any]]:
    """SHA-256 of the source file(s) that define the evaluated system (red-team R3-7): the strategy
    class's module for an evaluator with ``strategy_cls`` (RuleEvaluator), else the evaluator
    class's own module unless it is part of the ``quantlab`` library (library evaluators are
    identified by ``describe()`` and the cost model version).  None when no source file is found.
    Recorded in ``study_created`` and compared by the gates: a strategy edited after its study
    was run cannot be gated as that study."""
    import hashlib
    import inspect
    objs = [getattr(evaluator, "strategy_cls", None)] if getattr(evaluator, "strategy_cls", None) is not None \
        else [type(evaluator)]
    lib = Path(__file__).resolve().parent
    files = []
    for o in objs:
        try:
            f = inspect.getsourcefile(o)
        except (TypeError, OSError):
            f = None
        if not f:
            continue
        fp = Path(f).resolve()
        if fp.is_relative_to(lib) and getattr(evaluator, "strategy_cls", None) is None:
            continue
        files.append(fp)
    if not files:
        return None
    h = hashlib.sha256()
    for fp in sorted(files):
        h.update(fp.read_bytes())
    rel = [str(fp.relative_to(config.ROOT)) if fp.is_relative_to(config.ROOT) else fp.name for fp in sorted(files)]
    return {"files": rel, "sha256": h.hexdigest()}


def _auto_jobs(n_jobs: Any) -> int:
    if n_jobs in (None, "auto"):
        return max(1, min(8, (os.cpu_count() or 2) // 2))
    return max(1, int(n_jobs))


# =========================================================================== the study
class _Book:
    """In-memory record list + checkpointed TrialRecorder (+ opt_aux sidecar parquet)."""

    def __init__(self, study_id: str, studies_dir: Optional[Path], space: SearchSpace, objective: Objective,
                 ppy: float, checkpoint_every: int, abort_after: int = 0):
        self.study_id = study_id
        self.abort_after = int(abort_after)   # L1: size of the first batch checked for a uniform error
        self.first_batch: list[Optional[str]] = []
        self.rec = ledger.TrialRecorder(study_id, studies_dir, flush_every=10 ** 9)
        self.dir = self.rec.dir
        self.space, self.objective, self.ppy = space, objective, ppy
        self.recs: list[_Rec] = []
        self.pending: list[_Rec] = []
        self.seen: dict[str, _Rec] = {}
        self.keys: list[str] = []
        self.checkpoint_every = checkpoint_every

    @property
    def next_id(self) -> int:
        return self.rec.n_trials + len(self.pending)

    def _classify(self, r: _Rec) -> None:
        if r.status != "evaluated":
            return
        raw = float(self.objective.raw(r.ret, self.ppy)[0]) if r.ret is not None else -np.inf
        n = r.metrics.get("n_trades", float(r.counts.sum()) if r.counts is not None else 0.0)
        ok = n >= self.objective.min_trades
        r.status = "ok" if ok else "low_trades"
        r.metrics["objective_raw"] = raw
        r.metrics["objective"] = raw if ok else -np.inf

    def add_result(self, params: dict[str, Any], res: Optional[dict[str, Any]], source: str) -> _Rec:
        tid = self.next_id
        if res is None:
            r = _Rec(tid, params, "invalid", source=source)
        elif not res["ok"]:
            r = _Rec(tid, params, "error", error=res["error"], source=source)
        else:
            r = _Rec(tid, params, "evaluated", dict(res["metrics"]), res["dates"], res["ret"], res["counts"],
                     entries=res.get("entries"), source=source)
        self._classify(r)
        self.pending.append(r)
        self.seen[_pjson(params)] = r
        if len(self.pending) >= self.checkpoint_every:
            self.flush()
        if res is not None and len(self.first_batch) < self.abort_after:
            self.first_batch.append(r.error if r.status == "error" else None)
            fb = self.first_batch
            if len(fb) == self.abort_after and fb[0] is not None and all(e == fb[0] for e in fb):
                # L1: every trial of the first batch failed with the same exception type and
                # message → a data / setup error, not a parameter-specific one.  Abort instead
                # of grinding through the whole candidate set (the failed trials are recorded:
                # run_study's `finally` flushes them and logs the `trials` event as aborted).
                raise StudyError(
                    f"study {self.study_id}: the first {len(fb)} evaluated trials all failed with the same "
                    f"error — aborting (data/setup error, not a parameter effect): {fb[0]}")
        return r

    def restore(self, recs: list[_Rec]) -> None:
        for r in recs:
            self.recs.append(r)
            self.seen[_pjson(r.params)] = r
            for k in r.metrics:
                if k not in self.keys:
                    self.keys.append(k)

    def flush(self) -> None:
        if not self.pending:
            return
        for r in self.pending:
            for k in r.metrics:
                if k not in self.keys:
                    self.keys.append(k)
        first = self.pending[0].trial_id
        for r in self.pending:
            # Every row carries the same metric keys (NaN-padded) so each parquet part has a
            # stable Float64 schema (TrialRecorder builds parts with pl.DataFrame(list_of_dicts)).
            m = {k: r.metrics.get(k, float("nan")) for k in self.keys}
            ret_df = (pl.DataFrame({"date": r.dates, "ret": r.ret}).with_columns(pl.col("date").cast(pl.Date))
                      if r.ret is not None else None)
            tid = self.rec.add(_py_params(r.params), m, status=r.status, returns=ret_df, source=r.source or None)
            if tid != r.trial_id:
                raise StudyError(f"trial id drift: recorder {tid} vs study {r.trial_id}")
        self.rec.flush()
        aux_c = [pl.DataFrame({"trial_id": np.full(int((r.counts > 0).sum()), r.trial_id, dtype=np.int64),
                               "date": r.dates[r.counts > 0], "n": r.counts[r.counts > 0]})
                 for r in self.pending if r.counts is not None and (r.counts > 0).any()]
        if aux_c:
            pl.concat(aux_c).with_columns(pl.col("date").cast(pl.Date)).write_parquet(
                self.dir / f"opt_aux-counts-{first:06d}.parquet")
        # entries: one row per (trial, day with entries) + a sentinel row (n = 0 on the first
        # date) so a trial with entry counts but zero entries is still known to have them
        aux_e = [pl.DataFrame({"trial_id": np.full(int((r.entries > 0).sum()) + 1, r.trial_id, dtype=np.int64),
                               "date": np.concatenate([r.dates[:1], r.dates[r.entries > 0]]),
                               "n": np.concatenate([[0.0], r.entries[r.entries > 0]])})
                 for r in self.pending if r.entries is not None and r.dates is not None and r.dates.size]
        if aux_e:
            pl.concat(aux_e).with_columns(pl.col("date").cast(pl.Date)).write_parquet(
                self.dir / f"opt_aux-entries-{first:06d}.parquet")
        errs = [{"trial_id": r.trial_id, "error": r.error} for r in self.pending if r.error]
        if errs:
            pl.DataFrame(errs).write_parquet(self.dir / f"opt_aux-errors-{first:06d}.parquet")
        self.recs.extend(self.pending)
        self.pending = []

    def all(self) -> list[_Rec]:
        return self.recs + self.pending


def _py_params(p: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.item() if isinstance(v, np.generic) else v) for k, v in p.items()}


def _load_existing(study_id: str, studies_dir: Optional[Path], default_source: str = "") -> list[_Rec]:
    """Recorded trials of a study.  Trials recorded before sources were tracked get
    ``default_source`` (the method of the study's ledger ``study_created`` row)."""
    trials = ledger.load_trials(study_id, studies_dir)
    if trials.height == 0:
        return []
    rets = ledger.load_trial_returns(study_id, studies_dir)
    d = (Path(studies_dir) if studies_dir else config.STUDIES_DIR) / study_id
    cparts = sorted(d.glob("opt_aux-counts-*.parquet"))
    counts = pl.concat([pl.read_parquet(p) for p in cparts]) if cparts else None
    nparts = sorted(d.glob("opt_aux-entries-*.parquet"))
    entries = pl.concat([pl.read_parquet(p) for p in nparts]) if nparts else None
    eparts = sorted(d.glob("opt_aux-errors-*.parquet"))
    errors = dict(pl.concat([pl.read_parquet(p) for p in eparts]).iter_rows()) if eparts else {}
    mcols = [c for c in trials.columns if c.startswith("m_")]
    out = []
    for row in trials.sort("trial_id").iter_rows(named=True):
        tid = row["trial_id"]
        r = _Rec(tid, json.loads(row["params"]), row["status"],
                 {c[2:]: row[c] for c in mcols if row[c] is not None}, error=errors.get(tid),
                 source=row.get("source") or default_source)
        col = f"t{tid}"
        if rets.height and col in rets.columns:
            sub = rets.select("date", col).drop_nulls(col)
            r.dates = sub["date"].to_numpy().astype("datetime64[D]")
            r.ret = sub[col].to_numpy()
            r.counts = np.zeros(r.dates.size)
            if counts is not None:
                c = counts.filter(pl.col("trial_id") == tid)
                if c.height:
                    pos = np.searchsorted(r.dates, c["date"].to_numpy().astype("datetime64[D]"))
                    np.add.at(r.counts, np.clip(pos, 0, r.dates.size - 1), c["n"].to_numpy())
            if entries is not None:
                e = entries.filter(pl.col("trial_id") == tid)
                if e.height:
                    r.entries = np.zeros(r.dates.size)
                    pos = np.searchsorted(r.dates, e["date"].to_numpy().astype("datetime64[D]"))
                    np.add.at(r.entries, np.clip(pos, 0, r.dates.size - 1), e["n"].to_numpy())
        out.append(r)
    return out


def _trials_frame(recs: Sequence[_Rec], space: SearchSpace) -> pl.DataFrame:
    keys: list[str] = []
    for r in recs:
        for k in r.metrics:
            if k not in keys:
                keys.append(k)
    data: dict[str, list] = {"trial_id": [r.trial_id for r in recs], "status": [r.status for r in recs],
                             "source": [r.source or None for r in recs]}
    for n in space.names:
        data[f"param_{n}"] = [r.params.get(n) for r in recs]
    for k in keys:
        data[f"m_{k}"] = [float(r.metrics.get(k, float("nan"))) for r in recs]
    data["params"] = [_pjson(r.params) for r in recs]
    return pl.DataFrame(data, strict=False)


def run_study(evaluator: Any, space: SearchSpace, *, book: str, system: str, issue: int, attempt: int,
              method: Optional[str] = "auto", n_trials: Optional[int] = None, seed: int = 0, n_jobs: Any = "auto",
              cv: CPCVConfig = CPCVConfig(), wfo: Optional[WFOConfig] = WFOConfig(),
              objective: Objective = Objective(), min_trades: Optional[float] = None,
              selection: PlateauConfig = PlateauConfig(), cost: Any = None,
              parent_study: Optional[str] = None, notes: str = "", study_id: Optional[str] = None,
              storage: Optional[str] = None, resume: bool = False, tpe_batch: int = 8,
              checkpoint_every: int = 100, allow_large_grid: bool = False,
              abort_on_uniform_errors: bool = True,
              plateau_radius: Optional[float | Mapping[str, float]] = None,
              ledger_dir: Optional[Path] = None, studies_dir: Optional[Path] = None) -> StudyResult:
    """Run a fully logged study and assemble a :class:`contracts.StudyResult`.

    ``method="auto"`` (default): ``grid`` for ≤ 3 all-discrete params, else ``sobol``.
    ``method="grid"`` enumerates ``space.grid()`` (≤ 3 params unless ``allow_large_grid``);
    ``method="sobol"`` / ``"random"`` evaluate :func:`candidate_set` (``n_trials`` distinct
    configurations, seeded, fixed before any evaluation);
    ``method="tpe"`` runs a seeded Optuna TPE sampler for ``n_trials`` distinct
    configurations on the full-dev objective, asked/told in fixed batches of
    ``tpe_batch`` (independent of ``n_jobs`` → identical results for any ``n_jobs``).
    TPE's candidate set is data-dependent (``meta["candidate_set_data_dependent"]``): its
    CPCV / walk-forward "OOS" is contaminated (B3) and the gates skip the OOS gates.
    ``storage`` (Optuna RDB URL, e.g. ``sqlite:///…``) makes the sampler state durable;
    ``resume=True`` continues an existing study id from its recorded trials.
    ``min_trades`` overrides ``objective.min_trades``.  ``wfo=None`` skips the walk-forward.

    ``abort_on_uniform_errors`` (default True, L1): if the first ``max(n_jobs, 2)`` evaluated
    trials of this call all fail with the *same* exception type and message, the error is a
    data/setup error rather than a parameter effect, and the study stops with
    :class:`StudyError` (the failed trials are recorded and the ``trials`` event is logged
    with ``status="aborted"``).  Disable it for an evaluator whose parameter-specific errors
    legitimately share one message.

    ``plateau_radius`` (float, or ``{param: float}``; default 0.20, floor 0.10): the plateau
    neighbourhood pre-registered on the hypothesis card (DESIGN §4.2, red-team N3).  It is
    written to the ``study_created`` ledger row and the gates read it from there only.  It
    scales the ``plateau_scale="relative"`` params; ``plateau_step`` params are perturbed by
    their own step (a per-param radius naming one raises).

    The ``study_created`` row also records (R2-2 / R2-4) the evaluator's description and cost
    model version (compared by the gates), the traded ``symbols`` + ``conversion_legs`` of a real
    evaluator (holdout access scope and horizon), and ``studies_dir`` when it is not the default
    (the gates load the trial store from there).

    Resume (red-team N1 / N10): the study's ``study_created`` ledger row is the source of truth.
    ``book``, ``system``, ``issue``, ``attempt``, the resolved ``method``, ``seed``, the search
    space (``to_json``) and ``plateau_radius`` (None = take the ledger's) must equal the
    recorded values, else :class:`StudyError`.  Every trial records its ``source`` (grid /
    sobol / random / tpe) in the trial store and in ``study.trials``; the ``trials`` event logs
    ``n_by_source``.  ``candidate_set_data_dependent`` is the OR of the ledger's history and
    the sources of all trials — once True it is never reset.  A resume that changes the planned
    trial count logs an ``n_trials_changed`` event (old / new / already recorded / earlier gate
    runs), so optional stopping is auditable.
    """
    t0 = time.perf_counter()
    if getattr(evaluator, "requires_refit", False):
        raise NotImplementedError(
            "run_study: evaluators with requires_refit=True (ML, fit per window) are not supported in "
            "Phase 1 — the rule-based path evaluates each parameter set once on the dev window.")
    if isinstance(issue, bool) or not isinstance(issue, (int, np.integer)) or int(issue) < 0:
        raise ValueError(f"run_study: issue must be the hypothesis issue number (int >= 0), got {issue!r} "
                         f"(red-team R3-4)")
    issue = int(issue)
    method_requested = method
    method = _resolve_method(method, space)
    bname = config.get_book(book).name
    if config.get_book(getattr(evaluator, "book", bname)).name != bname:
        raise ValueError(f"evaluator book {evaluator.book!r} != study book {book!r}")
    if min_trades is not None:
        objective = Objective(objective.kind, objective.n_blocks, objective.lam, float(min_trades))
    grid: list[dict[str, Any]] = []
    if method == "grid":
        if len(space.params) > 3 and not allow_large_grid:
            raise ValueError("grid search is for <= 3 free parameters; use method='sobol' (or allow_large_grid=True)")
        grid = space.grid()
        if n_trials is not None and n_trials != len(grid):
            raise ValueError(f"grid has {len(grid)} configurations but n_trials={n_trials}")
    elif not n_trials:
        raise ValueError(f"method={method!r} needs n_trials")
    elif method in ("sobol", "random"):
        grid = candidate_set(space, n_trials, method=method, seed=seed)
    data_dependent = method not in DATA_INDEPENDENT_METHODS
    jobs = _auto_jobs(n_jobs)
    ppy = float(evaluator.periods_per_year)
    sid = study_id or ledger.new_study_id(bname, issue, attempt)
    wfo_desc = wfo.describe() if wfo is not None else "none"
    cv_plan = cv.describe(cv.embargo_days if cv.embargo_days is not None else "auto",
                          cv.purge_days if cv.purge_days is not None else
                          (cv.embargo_days if cv.embargo_days is not None else "auto"))
    cost_version = getattr(cost, "version", None) or getattr(evaluator, "cost_version", None) or "unknown"
    dev_window = list(getattr(evaluator, "dev_window", ("unknown", "unknown")))
    describe = getattr(evaluator, "describe", None)
    ev_desc = describe() if describe else {"evaluator": type(evaluator).__name__}

    radius_call = None if plateau_radius is None else normalise_plateau_radius(plateau_radius, space.names)
    if isinstance(radius_call, Mapping):
        absolute = [n for n in radius_call if next(p for p in space.params if p.name == n).plateau_step is not None]
        if absolute:
            raise ValueError(f"plateau_radius names {absolute}, which declare an absolute plateau_step: the radius "
                             f"does not apply to them (R2-1) — pre-register the step itself")
    n_planned = len(grid) if method != "tpe" else int(n_trials)
    row: dict[str, Any] = {}
    if resume:
        row = ledger.created_row(sid, ledger_dir=ledger_dir) or {}
        if not row:
            raise StudyError(f"resume=True but study {sid} is not in the ledger")
        radius = normalise_plateau_radius(row.get("plateau_radius"), space.names)
        recorded = {"book": row.get("book"), "system": row.get("system"), "issue": row.get("issue"),
                    "attempt": row.get("attempt"), "method": row.get("method"), "seed": row.get("seed"),
                    "search_space": row.get("search_space"), "plateau_radius": radius}
        called = {"book": bname, "system": system, "issue": issue, "attempt": attempt, "method": method,
                  "seed": seed, "search_space": space.to_json(),
                  "plateau_radius": radius if radius_call is None else radius_call}
        diff = {k: (recorded[k], called[k]) for k in recorded if _canon(recorded[k]) != _canon(called[k])}
        if diff:
            raise StudyError(
                f"resume of {sid} must use the study's pre-registered settings from its ledger study_created row; "
                f"these differ (ledger, call): {diff}. A different method / seed / space / radius is a new study "
                f"(new attempt id), never a continuation (red-team N1)")
        events = ledger.study_events(sid, ledger_dir=ledger_dir)
        prev_planned = next((e["n_trials_planned"] for e in reversed(events) if "n_trials_planned" in e), None)
        n_gate_runs = sum(e.get("event") == "gates" for e in events)
        if prev_planned is not None and int(prev_planned) != n_planned:
            ledger.log_event(sid, "n_trials_changed", ledger_dir=ledger_dir, n_trials_planned_old=int(prev_planned),
                             n_trials_planned_new=n_planned, n_trials_planned=n_planned,
                             n_trials_recorded=ledger.TrialRecorder(sid, studies_dir).count(),
                             gate_runs_before=n_gate_runs)
        ledger.log_event(sid, "resumed", ledger_dir=ledger_dir, method=method, seed=seed,
                         n_trials_planned_call=n_planned, gate_runs_before=n_gate_runs)
        data_dependent = data_dependent or ledger.candidate_set_data_dependent(sid, ledger_dir=ledger_dir)
    else:
        radius = PLATEAU_RADIUS_DEFAULT if radius_call is None else radius_call
        syms, legs = _evaluator_symbols(evaluator, bname)
        extra: dict[str, Any] = {}
        if syms:
            extra.update(symbols=syms, conversion_legs=legs)
        code = evaluator_code_fingerprint(evaluator)
        if code is not None:
            extra["evaluator_code"] = code
        if studies_dir is not None and Path(studies_dir).resolve() != Path(config.STUDIES_DIR).resolve():
            extra["studies_dir"] = str(Path(studies_dir).resolve())    # where the gates load the trial store
        ledger.create_study(
            ledger_dir=ledger_dir, study_id=sid, book=bname, system=system, issue=issue, attempt=attempt,
            parent_study=parent_study, dev_window=dev_window, cost_model_version=cost_version,
            cv_scheme=f"{cv_plan}+{wfo_desc}", search_space=space.to_json(), method=method,
            n_trials_planned=n_planned, seed=seed, plateau_radius=radius,
            candidate_set=method, candidate_set_data_dependent=data_dependent,
            objective=objective.describe(), selection=selection.describe(), evaluator=_py(ev_desc), notes=notes,
            **extra)

    bk = _Book(sid, studies_dir, space, objective, ppy, checkpoint_every,
               abort_after=max(jobs, 2) if abort_on_uniform_errors else 0)
    if resume:
        bk.restore(_load_existing(sid, studies_dir, default_source=str(row.get("method") or "")))
    runner = _Runner(evaluator, cost, jobs)
    worker_threads = None
    n_dupes = 0
    status = "aborted"
    try:
        if runner.pool is not None:
            worker_threads = runner.worker_threads()
        if method != "tpe":
            todo = [p for p in grid if _pjson(p) not in bk.seen]
            valid = [p for p in todo if space.is_valid(p)]
            if valid:
                log.info("study %s: %d %s configurations (%d valid) on %d worker(s)", sid, len(todo), method,
                         len(valid), jobs)
            for i in range(0, len(todo), max(checkpoint_every, 1)):
                chunk = todo[i:i + checkpoint_every]
                vchunk = [p for p in chunk if space.is_valid(p)]
                res = runner.map(vchunk)
                for p in chunk:
                    bk.add_result(p, next(res) if space.is_valid(p) else None, method)
                bk.flush()
        else:
            n_dupes = _run_tpe(bk, runner, space, sid, n_trials, seed, tpe_batch, storage)
        bk.flush()
        status = "complete"
    finally:
        runner.close()
        bk.flush()
        recs = bk.all()
        counts = {s: sum(r.status == s for r in recs) for s in STATUSES}
        by_source: dict[str, int] = {}
        for r in recs:
            by_source[r.source or "unknown"] = by_source.get(r.source or "unknown", 0) + 1
        # N1: a single TPE-proposed trial makes the candidate set data-dependent, forever
        data_dependent = data_dependent or by_source.get("tpe", 0) > 0
        ledger.log_event(sid, "trials", ledger_dir=ledger_dir, n_trials=len(recs), status=status,
                         n_ok=counts["ok"], n_low_trades=counts["low_trades"], n_invalid=counts["invalid"],
                         n_error=counts["error"], n_pruned=counts["pruned"], n_by_source=by_source,
                         candidate_set_data_dependent=data_dependent)

    recs = bk.all()
    trials = _trials_frame(recs, space)
    returns = _matrix(recs, "ret")
    tcounts = _matrix(recs, "counts")
    ecounts = _matrix(recs, "entries")
    evaluated = [r for r in recs if r.ret is not None]
    if not evaluated:
        raise StudyError(f"study {sid}: no configuration could be evaluated "
                         f"(errors: {[r.error for r in recs if r.error][:3]})")

    S = _build_surface(returns, trials, space, selection, tcounts, method, ecounts)
    cfg = PlateauConfig(**{**asdict(selection), "neighbourhood": S.mode})
    full = _select_rows(S, np.arange(S.R.shape[0]), objective, cfg, ppy)
    if full["index"] is None:
        selected, sel_info = {}, {"method": "plateau", "status": "no_eligible"}
    else:
        w, raw = full["index"], full["raw_index"]
        selected = _py_params(S.params[w])
        sel_info = {
            "method": "plateau", "status": "ok", "neighbourhood": cfg.describe(),
            "selected_trial": int(S.trial_ids[w]), "trial_id": int(S.trial_ids[w]),  # gates key
            "objective": full["objective"], "smoothed": full["smoothed"],
            "sharpe": full["gate_value"], "plateau_score": full["plateau_score"],
            "plateau_score_objective": full["plateau_score_objective"], "n_neighbours": full["n_neighbours"],
            "raw_argmax_trial": int(S.trial_ids[raw]), "raw_argmax_params": _py_params(S.params[raw]),
            "raw_argmax_objective": full["raw_objective"], "raw_argmax_smoothed": full["raw_smoothed"],
            "raw_argmax_plateau_score": full["raw_plateau_score"], "objective_spec": objective.describe(),
        }

    paths, splits_df, cmeta = cpcv_paths(returns, trials, space, cv, objective=objective, selection=cfg,
                                         trade_counts=tcounts, periods_per_year=ppy, _surface=S)
    if wfo is not None:
        wfo_oos, wfo_params, wmeta = walk_forward(returns, trials, space, wfo, objective=objective,
                                                  selection=cfg, trade_counts=tcounts,
                                                  periods_per_year=ppy, _surface=S)
    else:
        wfo_oos, wfo_params, wmeta = pl.DataFrame(), pl.DataFrame(), {"scheme": "none"}
    runtime = time.perf_counter() - t0
    cv_scheme = f"{cmeta['scheme']}+{wmeta['scheme']}"
    ok_trades = trials.filter(pl.col("status") == "ok")
    med_trades = float(ok_trades["m_n_trades"].median()) if ok_trades.height and "m_n_trades" in ok_trades.columns else float("nan")
    budget = med_trades / len(space.params) / cv.n_groups * (cv.n_groups - cv.k_test) if med_trades == med_trades else float("nan")
    # L1: the evaluation start actually used (after the input-availability clamp), if the
    # evaluator reports one; L4: data gaps in the study's date index + gap-straddling trades
    eval_start_eff = None
    fn = getattr(evaluator, "eval_start_effective", None)
    if callable(fn):
        try:
            eval_start_eff = fn()
        except Exception as e:  # diagnostic only; the trials already carry the real outcome
            eval_start_eff = f"unavailable ({type(e).__name__}: {e})"
    gaps = _data_gaps(S.dates, calendar_days=ppy > 300)
    gap_col = "m_trades_across_data_gap"
    ok_evaluated = trials.filter(pl.col("status").is_in(["ok", "low_trades"]))
    gap_trades = (int(np.nan_to_num(ok_evaluated[gap_col].max() or 0.0))
                  if gap_col in ok_evaluated.columns and ok_evaluated.height else None)
    meta = {
        "study_id": sid, "book": bname, "system": system, "issue": issue, "attempt": attempt,
        "space": space.to_json(), "candidate_set": method, "candidate_set_data_dependent": data_dependent,
        "embargo_capped": bool(cmeta["embargo_capped"]), "embargo_days_uncapped": int(cmeta["embargo_days_uncapped"]),
        "cv_scheme": cv_scheme, "cpcv": cmeta, "cpcv_splits": splits_df,
        "wfo": wmeta, "method": method, "method_requested": method_requested, "seed": seed,
        "tpe_batch": tpe_batch if method == "tpe" else None,
        "n_trials": len(recs), "n_ok": counts["ok"], "n_low_trades": counts["low_trades"],
        "n_invalid": counts["invalid"], "n_error": counts["error"], "n_tpe_duplicates": n_dupes,
        "errors": [{"trial_id": r.trial_id, "error": r.error} for r in recs if r.error][:50],
        "runtime_s": runtime, "cost_model_version": cost_version, "n_jobs": jobs,
        "plateau_radius": radius, "n_by_source": by_source, "search_space_obj": space,
        "objective": objective.describe(), "selection": cfg.describe(), "periods_per_year": ppy,
        "dev_window": [str(S.dates[0]), str(S.dates[-1])], "evaluator": ev_desc,
        "trade_counts": tcounts, "median_trades_ok": med_trades,
        "train_trades_per_param_per_fold": budget, "resumed": resume, "storage": storage,
        "mp_start_method": runner.start_method, "worker_threads": worker_threads,
        "eval_start_effective": eval_start_eff, "data_gaps": gaps,
        "m_trades_across_data_gap": gap_trades,
    }
    artifact_sha = ledger.write_study_artifacts(sid, {"cpcv_paths": paths, "wfo_oos": wfo_oos, "wfo_params": wfo_params,
                                                     "trade_counts": tcounts, "entry_counts": ecounts}, studies_dir)
    meta["entry_counts"] = ecounts
    ledger.log_event(sid, "selection", ledger_dir=ledger_dir, selected_params=selected, artifact_sha256=artifact_sha,
                     selection=_py({k: v for k, v in sel_info.items()}), cv_scheme=cv_scheme,
                     cpcv_paths=cmeta["n_paths"], cpcv_embargo_days=cmeta["embargo_days"],
                     cpcv_purge_days=cmeta["purge_days"], embargo_capped=meta["embargo_capped"],
                     embargo_days_uncapped=meta["embargo_days_uncapped"], candidate_set=method,
                     candidate_set_data_dependent=data_dependent, wfo_refits=wmeta.get("n_refits"),
                     wfo_drift_mean=wmeta.get("drift_mean"), wfo_drift_max=wmeta.get("drift_max"),
                     dev_window_actual=meta["dev_window"], runtime_s=round(runtime, 2), n_jobs=jobs,
                     eval_start_effective=eval_start_eff, data_gaps=gaps,
                     m_trades_across_data_gap=gap_trades)
    return StudyResult(study_id=sid, param_names=space.names, trials=trials, returns=returns,
                       selected_params=selected, selection=sel_info, cpcv_paths=paths,
                       wfo_oos=wfo_oos, wfo_params=wfo_params, meta=meta)


def _run_tpe(bk: _Book, runner: _Runner, space: SearchSpace, sid: str, n_trials: int, seed: int,
             batch: int, storage: Optional[str]) -> int:
    import optuna
    from optuna.trial import TrialState
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sampler = optuna.samplers.TPESampler(
            seed=seed, multivariate=True, constant_liar=True,
            n_startup_trials=min(20, max(5, n_trials // 5)),
            constraints_func=lambda ft: (ft.user_attrs.get("violation", 0.0),))
        study = optuna.create_study(direction="maximize", sampler=sampler, storage=storage,
                                    study_name=sid, load_if_exists=storage is not None)
    dists = space.distributions()
    known = {t.user_attrs.get("trial_id") for t in study.trials}
    by_id = {r.trial_id: r for r in bk.all()}
    # Crash recovery: close RUNNING optuna trials from their records (or FAIL if unrecorded) …
    for t in study.trials:
        if t.state == TrialState.RUNNING:
            r = by_id.get(t.user_attrs.get("trial_id"))
            if r is None:
                study.tell(t.number, state=TrialState.FAIL)
            else:
                _tell(study, optuna.trial.Trial(study, t._trial_id), r)
    # … and replay recorded trials the sampler has never seen (in-memory storage resume).
    for r in bk.all():
        if r.trial_id not in known and r.status != "error":
            val, viol = _tell_values(r)
            ft = optuna.trial.create_trial(params=_py_params(r.params), distributions=dists, value=val,
                                           user_attrs={"trial_id": r.trial_id, "violation": viol})
            study.add_trial(ft)
    n_dupes, max_asks = 0, n_trials * 20
    asks = 0
    while len(bk.all()) < n_trials and asks < max_asks:
        want = min(batch, n_trials - len(bk.all()))
        ots = [study.ask() for _ in range(want)]
        asks += want
        plist = [space.suggest(t) for t in ots]
        new_idx, seen_now = [], set()
        for i, p in enumerate(plist):
            k = _pjson(p)
            if k in bk.seen or k in seen_now:
                continue
            seen_now.add(k)
            if space.is_valid(p):
                new_idx.append(i)
        res = runner.map([plist[i] for i in new_idx])
        pending_tells = []
        for i, (t, p) in enumerate(zip(ots, plist)):
            k = _pjson(p)
            if k in bk.seen:                      # duplicate configuration: not a new trial
                n_dupes += 1
                pending_tells.append((t, bk.seen[k]))
                continue
            r = bk.add_result(p, next(res) if i in new_idx else None, "tpe")
            t.set_user_attr("trial_id", r.trial_id)
            pending_tells.append((t, r))
        bk.flush()                                 # checkpoint before telling the sampler
        for t, r in pending_tells:
            _tell(study, t, r)
    return n_dupes


def _tell_values(r: _Rec) -> tuple[float, float]:
    if r.status == "invalid":
        return -1e6, 1.0
    raw = float(r.metrics.get("objective_raw", -1e6))
    raw = raw if math.isfinite(raw) else -1e6
    return raw, (0.0 if r.status == "ok" else 1.0)


def _tell(study: Any, trial: Any, r: _Rec) -> None:
    from optuna.trial import TrialState
    if r.status == "error":
        study.tell(trial, state=TrialState.FAIL)
        return
    val, viol = _tell_values(r)
    trial.set_user_attr("violation", viol)
    study.tell(trial, val)
