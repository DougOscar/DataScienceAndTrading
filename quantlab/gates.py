"""Single-system validation gates (DESIGN §4.2 v1.2) and the validation report.

``evaluate_gates(study, evaluator, periods_per_year=...)`` turns a ``StudyResult`` into a
``GateReport``: one row per gate (value · threshold · PASS/FAIL/MANUAL/SKIPPED · one-line
interpretation in the DESIGN §5 style), an overall verdict, the trial-count / effective-trials
calculation and the pre-registered holdout band (§4.4).

Verdict: ``PASS`` only if every gate is PASS.  Any FAIL → ``FAIL``.  No FAIL but at least one
SKIPPED/MANUAL gate → ``INCOMPLETE`` (a skipped mandatory gate can never produce PASS; the
mechanism check is MANUAL unless a callable is supplied).

Thresholds live in ``GATE_THRESHOLDS`` and are exactly DESIGN §4.2 v1.2.  They are read-only;
do not move them after seeing results (recalibration is a DESIGN decision, §11).
``GATE_THRESHOLDS["wfo_oos_recent"]`` is the second condition of the ``wfo_oos`` row.

Gates and the series each one uses
----------------------------------
* ``dsr`` — Deflated Sharpe of the selected trial's full-dev daily returns.  Hurdle SR0 =
  √V0 · E[max of N], V0 = 1/(T−1), **N = raw trials of this study (never fewer than its trial
  store / the optimizer's logged count, R2-4) + raw trials of every other study in the ledger
  that shares its normalised system name OR its issue number** (any attempt, any book, created
  before or after it — red-team N2 / R2 N2b).  An explicit ``prior_trials`` may only raise
  that count.  Eigen / cluster / Li–Ji effective N and the cross-sectional Sharpe variance are
  diagnostics (R1 / B1).
* ``cscv_oos_loss`` — P(OOS Sharpe of the IS-best < 0) over the CSCV splits (16 blocks) of the
  full trial matrix; < 0.10.  PBO, slope and R² are diagnostics (R2 / M1).
* ``oos_sharpe`` — median annualised Sharpe across ``study.cpcv_paths``.
* ``wfo_oos`` — the walk-forward procedure's OOS series (``study.wfo_oos``): whole-span
  Sharpe ≥ 0.5 **and** Sharpe over its most recent third > 0 (B2).
* ``cost_stress_sharpe`` — the evaluator re-run on the selected params with
  ``base_cost.stressed()`` (1.5× spread + 1 pip slippage on all market and stop fills,
  bar-extreme stops) at swap multipliers {1.0} ∪ ``swap_band``; value = the minimum Sharpe.
* ``plateau`` — **always measured here** (never the optimizer's number, B4).  With an
  evaluator the judge runs it (N4, :func:`judge_plateau`): the selected configuration is
  re-evaluated perturbed along each numeric axis at ±½ and ±1 of the parameter's
  **pre-registered plateau scale** (R2-1: ``plateau_scale="relative"`` → r·|x|, r = the radius
  pre-registered in the study's ledger row, default 0.20, N3; ``plateau_step=S`` → S), ordered
  categoricals at the existing levels among ±1 / ±2, plus a ``joint`` axis moving all numeric params together
  (8 points).  A point passes if its full-dev Sharpe is ≥ 50 % of the re-evaluated peak and
  > 0; score = the MINIMUM over axes of the per-axis pass share (N8: irrelevant axes cannot
  lift a fragile one; 0.60 ⇒ ≥ 3 of 4 on every axis).
  Search bounds are not validity limits: points outside them are evaluated (and reported);
  invalid (constraint, or natural: positive params > 0, lookback ints ≥ 1) and erroring points
  fail.  These ≤ 4·d evaluations are judge diagnostics (logged with the
  gates event), not selection trials: they do not enter the DSR's N.  Without an evaluator the
  matrix-based ±radius box over the recorded trials is used and labelled as such
  (:func:`plateau_score`).
* ``positive_years`` / ``max_year_share`` — calendar years of the selected trial's returns.
* ``trade_count`` — (a) trades ≥ per-trade MinTRL (95 %) and (b) dev days ≥ daily MinTRL at the
  claimed Sharpe = min(selected full-dev Sharpe, CPCV-median OOS Sharpe).
* ``mechanism`` — component ablation (callable) or MANUAL.

SKIPs that block PASS: ``oos_sharpe``, ``wfo_oos`` and ``cscv_oos_loss`` when the candidate set
is data-dependent (B3 / N1 — read from the **ledger**: the created row, or any later event that
set ``candidate_set_data_dependent`` or logged TPE-sourced trials); ``oos_sharpe`` and
``cscv_oos_loss`` when the CPCV embargo cap binds (``meta["embargo_capped"]`` — m3).

The ledger is the source of truth (N2, R2-4)
--------------------------------------------
Book, system, issue, attempt, method, seed, search space, plateau radius and the
data-dependence flag are read from the study's ``study_created`` ledger row (and its later
events).  A study that is not in the ledger raises :class:`GateError`; so does a
``study.meta`` value that disagrees with the ledger (meta is only a cache), a ``study.trials`` /
``study.returns`` that differs from the study's trial store (:func:`verify_trial_store`), and
an evaluator / cost model other than the one recorded for the study
(:func:`check_evaluator_identity`).

Holdout exam (DESIGN §4.4, decided 2026-09-24)
----------------------------------------------
The band's horizon is the holdout start → the end of the data available now (locked year +
newer exports), from the data manifest (:func:`resolve_holdout_horizon`).  Before an unlock,
newer exports require :func:`rebuild_holdout_band` (rebuilds for the new horizon and
re-registers it in the ledger, logged).  :func:`run_holdout_exam` judges the realised holdout
against the band stored with the unlock and records PASS / FAIL / NOT_DECISIVE with its
horizon.  NOT_DECISIVE = every criterion passed but P(pass | zero edge) >
:data:`HOLDOUT_MAX_ZERO_EDGE_PASS`: the system waits for more data and is re-examined on the
full, longer span with a band rebuilt beforehand.  A FAIL is final (the ledger refuses anything
after it).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import polars as pl

from . import config, data, ledger, metrics
from . import stats as st
from .contracts import Evaluator, Outcome, StudyResult

# DESIGN §4.2 v1.2 — (comparator, threshold).  Read-only.
GATE_THRESHOLDS = MappingProxyType({
    "dsr": (">=", 0.95),
    "cscv_oos_loss": ("<", 0.10),           # P(OOS Sharpe of the IS-best < 0), CSCV 16 splits
    "oos_sharpe": (">=", 1.0),              # annualised, CPCV path median
    "wfo_oos": (">=", 0.5),                 # annualised, whole WFO-OOS span …
    "wfo_oos_recent": (">", 0.0),           # … and over its most recent third
    "cost_stress_sharpe": (">", 0.5),       # annualised, CostModel.stressed()
    "plateau": (">=", 0.60),                # share of neighbours with SR ≥ 50 % of peak and > 0
    "positive_years": (">=", 0.60),
    "max_year_share": ("<=", 0.40),         # "no single year > 40 % of total PnL"
    "trade_count": (">=", "MinTRL"),
    "mechanism": ("manual", None),
})
PLATEAU_PEAK_FRACTION = 0.5
PLATEAU_RADIUS = 0.20                       # ±20 % economic neighbourhood (card pre-registers; ledger row)
PLATEAU_RADIUS_MIN = 0.10                   # floor for a pre-registered radius (N3)
# R3-3: every numeric perturbation's outer tolerance is at least this share of the declared search
# range (high − low), the inner one half of it — whatever scale the author declared.  Read-only.
PLATEAU_MIN_RANGE_FRACTION = 0.05
WFO_RECENT_FRACTION = 1.0 / 3.0
MINTRL_PROB = 0.95
HOLDOUT_TARGET_COVERAGE = 0.90
HOLDOUT_MAX_ZERO_EDGE_PASS = st.DECISIVE_MAX_ZERO_EDGE_PASS   # read-only (DESIGN §4.4)
GATE_ORDER = ("dsr", "cscv_oos_loss", "oos_sharpe", "wfo_oos", "cost_stress_sharpe", "plateau",
              "positive_years", "max_year_share", "trade_count", "mechanism")
GATE_LABELS = {
    "dsr": "Deflated Sharpe probability", "cscv_oos_loss": "CSCV OOS loss probability",
    "oos_sharpe": "OOS Sharpe (CPCV median)", "wfo_oos": "Walk-forward procedure OOS",
    "cost_stress_sharpe": "Cost-stress Sharpe", "plateau": "Parameter plateau",
    "positive_years": "Positive years", "max_year_share": "Max single-year PnL share",
    "trade_count": "Trade count vs MinTRL", "mechanism": "Mechanism check",
}


class GateError(ValueError):
    """The gates cannot be run honestly: the study is not in the ledger, ``study.meta`` disagrees
    with the ledger, or a caller tried to lower the deflation count."""


_REMOVED = object()


def _cmp(op: str, v: float, thr: float) -> bool:
    if v is None or not np.isfinite(v):
        return False
    return {">=": v >= thr, ">": v > thr, "<": v < thr, "<=": v <= thr}[op]


@dataclass
class GateRow:
    gate: str
    value: float | None
    threshold: str
    status: str                 # PASS / FAIL / MANUAL / SKIPPED
    interpretation: str
    display: str | None = None  # formatted value (defaults to value)


@dataclass
class GateReport:
    study_id: str
    rows: list[GateRow]
    verdict: str
    holdout_band: dict[str, Any] | None
    effective_trials: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    periods_per_year: float = 260.0
    prior_gate_runs: list[dict[str, Any]] = field(default_factory=list)
    plateau_detail: dict[str, Any] = field(default_factory=dict)   # judge-run perturbations (N4)
    ledger_context: dict[str, Any] = field(default_factory=dict)   # identity read from the ledger (N2)
    holdout_band_run: dict[str, Any] | None = None   # this run's band; == holdout_band unless one is registered
    ledger_row: dict[str, Any] | None = None          # the gates event written by evaluate_gates(log=True)

    def row(self, gate: str) -> GateRow:
        for r in self.rows:
            if r.gate == gate:
                return r
        hint = " ('pbo' was replaced by 'cscv_oos_loss' in DESIGN v1.2; PBO is in diagnostics['pbo_detail'])" \
            if gate == "pbo" else ""
        raise KeyError(f"no gate {gate!r} in this report; gates: {[r.gate for r in self.rows]}{hint}")

    def values(self) -> dict[str, float | None]:
        return {r.gate: (None if r.value is None or not np.isfinite(r.value) else float(r.value))
                for r in self.rows}

    def statuses(self) -> dict[str, str]:
        return {r.gate: r.status for r in self.rows}

    def to_markdown(self) -> str:
        L = [f"# Validation — {self.study_id}", "",
             f"Gates: DESIGN §4.2 v1.2 (development data, after costs). Sharpe annualised from daily "
             f"%-equity returns (periods/year = {self.periods_per_year:g}).", ""]
        if self.prior_gate_runs:
            runs = "; ".join(f"run {r.get('gate_run', '?')}: {r.get('verdict')} ({r.get('at', '?')})"
                             for r in self.prior_gate_runs)
            L += [f"**Warning — this study has been gated {len(self.prior_gate_runs)} time(s) before** "
                  f"({runs}). Re-running gates until PASS is a search; read this verdict with that in mind.", ""]
        L += ["| Gate | Value | Threshold | Status | Interpretation |", "|---|---|---|---|---|"]
        for r in self.rows:
            val = r.display if r.display is not None else ("—" if r.value is None else f"{r.value:.3g}")
            L.append(f"| {GATE_LABELS.get(r.gate, r.gate)} | {val} | {r.threshold} | **{r.status}** | "
                     f"{r.interpretation} |")
        L += ["", f"## Verdict: **{self.verdict}**", ""]
        bad = [r for r in self.rows if r.status != "PASS"]
        if bad:
            L += ["Not passed: " + ", ".join(f"{GATE_LABELS.get(r.gate, r.gate)} ({r.status})" for r in bad) + ".", ""]
        et = self.effective_trials
        nan = float("nan")
        L += ["## Trials and deflation", "",
              f"- **N used = raw {et.get('n_trials', nan):g} trials** = {et.get('n_trials_study', nan):g} in this "
              f"study + {et.get('n_trials_prior', nan):g} from related earlier studies ({et.get('prior_source', 'n/a')})",
              f"- Hurdle SR0 = √(1/(T−1)) · E[max of N] = {et.get('sr0_annual', nan):.2f} annualised "
              f"(T = {et.get('n_obs')} days)",
              f"- Selected Sharpe {et.get('sr_annual', nan):.2f} annualised; skew {et.get('skew', nan):.2f}, "
              f"kurtosis {et.get('kurt', nan):.2f}",
              f"- DSR = {et.get('dsr', nan):.3f}; PSR(SR*=0) = {et.get('psr0', nan):.3f}",
              f"- Diagnostics only — effective N of this study: eigen {et.get('eigen', nan):.1f} · cluster "
              f"(ρ ≥ 0.5) {et.get('cluster', nan):.1f} · Li–Ji {et.get('liji', nan):.1f}; cross-trial Sharpe "
              f"variance {et.get('var_sr', nan):.3g} (per-period); v1.1-style DSR (N_eff + prior, V_cross) "
              f"{et.get('dsr_neff_cross', nan):.3f}; raw N with V_cross {et.get('dsr_raw_cross', nan):.3f}", ""]
        pdl = self.plateau_detail
        if pdl.get("points"):
            L += ["## Plateau perturbations (judge-run, not trials)", "",
                  f"Radius {pdl.get('radius')} ({pdl.get('radius_source')}); peak (re-evaluated) Sharpe "
                  f"{pdl.get('peak_sharpe', nan):.2f}; {pdl.get('n_evaluations', 0)} evaluations in "
                  f"{pdl.get('runtime_s', nan):.1f} s.", "",
                  f"Score = min over parameter axes of the per-axis pass share; weakest axis "
                  f"{pdl.get('weakest_axis')} ({pdl.get('pass_share_by_param')}). Selected at search-space edge: "
                  f"{', '.join(pdl.get('selected_at_edge') or []) or 'no'}.", "",
                  "| Param | Offset | Value | Outside search bounds | Sharpe | Status | Keeps ≥ 50% of peak |",
                  "|---|---|---|---|---|---|---|"]
            for q in pdl["points"]:
                sr = q.get("sharpe")
                L.append(f"| {q['param']} | {q['offset']} | {q['value']} | "
                         f"{'yes' if q.get('outside_search_bounds') else ''} | "
                         f"{'—' if sr is None or not np.isfinite(sr) else f'{sr:.2f}'} | {q['status']} | "
                         f"{'yes' if q.get('pass') else 'no'} |")
            L.append("")
        if self.holdout_band:
            hb = self.holdout_band
            a = hb.get("tail_level", nan)
            L += ["## Pre-registered holdout pass band (DESIGN §4.4 v1.2)", "",
                  f"Horizon: {hb.get('horizon_days')} trading days, {hb.get('holdout_start') or 'holdout start'} → "
                  f"{hb.get('horizon_end') or 'n/a'} (source: {hb.get('horizon_source', 'explicit')}; newer-than-locked "
                  f"data included: {_yn(hb.get('newer_data_included'))}). The exam uses the locked year plus all "
                  f"newer data available at the unlock; if more data is exported first, the band is rebuilt "
                  f"(gates.rebuild_holdout_band) and re-registered before unlocking.", "",
                  f"Source: {hb.get('source')} + stationary bootstrap (mean block {hb.get('mean_block', 0):.1f} d), "
                  f"{hb.get('n_boot')} samples, seed {hb.get('seed')}. "
                  f"Joint band: common tail level α = {a:.3f} (one-sided limits at α, trade range at α/2 each "
                  f"side); {hb.get('joint_coverage', nan):.0%} of the joint draws pass all four "
                  f"(target {hb.get('target_coverage', nan):.0%}).", "",
                  "| Criterion | Pass if |", "|---|---|",
                  f"| Holdout Sharpe (annualised) | ≥ {hb['sharpe_lo']:.2f} (median {hb['sharpe_median']:.2f}) |",
                  f"| Mean monthly return at leverage k = {hb['leverage']:.3g} | ≥ {hb['ret_at_budget_lo']:.2%} |",
                  f"| Max drawdown (unlevered) | ≤ {hb['max_dd_mag_hi']:.2%} |",
                  f"| Trade count | {_fmt_range(hb.get('trades_lo'), hb.get('trades_hi'))} "
                  f"(source: {hb.get('trades_source', 'n/a')}) |", "",
                  f"Power: a zero-edge holdout (the source series de-meaned, same volatility and trade rate) "
                  f"passes this band with probability {hb.get('p_pass_zero_edge', nan):.0%} "
                  f"({hb.get('n_power', 0)} draws)."]
            if hb.get("decisive") is False:
                L.append(f"**Not decisive at this horizon:** P(pass | zero edge) > "
                         f"{HOLDOUT_MAX_ZERO_EDGE_PASS:.0%}. If every criterion passes, the exam is recorded "
                         f"NOT_DECISIVE: the system stays `holdout_pending` and is re-examined on the full, "
                         f"longer holdout once more data is exported (DESIGN §4.4). A FAIL still kills it.")
            else:
                L.append(f"Decisive at this horizon (≤ {HOLDOUT_MAX_ZERO_EDGE_PASS:.0%}): the exam ends PASS or FAIL.")
            L.append("")
        if self.diagnostics:
            L += ["## Diagnostics (not gates)", ""]
            for k, v in self.diagnostics.items():
                L.append(f"- {k}: {_fmt(v)}")
            L.append("")
        return "\n".join(L)

    def write_markdown(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown())
        return p


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_fmt(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


def _yn(v) -> str:
    return "n/a" if v is None else ("yes" if v else "no")


def _fmt_range(lo, hi) -> str:
    if lo is None or hi is None or not (np.isfinite(lo) and np.isfinite(hi)):
        return "not available (no trade data) — criterion not applied"
    return f"within [{lo:.0f}, {hi:.0f}]"


# --------------------------------------------------------------------------- helpers
def selected_trial_id(study: StudyResult) -> int:
    """Selected trial: ``selection['trial_id']`` or the trial whose params equal selected_params."""
    tid = study.selection.get("trial_id") if study.selection else None
    if tid is not None:
        return int(tid)
    df = study.trials
    for name in study.param_names:
        df = df.filter(pl.col(f"param_{name}") == study.selected_params[name])
    if df.height == 0:
        raise ValueError("selected_params do not match any trial; set selection['trial_id']")
    return int(df["trial_id"][0])


def _trial_sharpes(study: StudyResult, periods_per_year: float) -> dict[int, float]:
    cols = [c for c in study.returns.columns if c != "date"]
    m, _ = st._matrix(study.returns.select(cols))
    srs = st.sharpe_per_period(m, axis=0) * math.sqrt(periods_per_year)
    return {int(c[1:]): float(s) for c, s in zip(cols, np.atleast_1d(srs))}


def _space_specs(study: StudyResult, space: Any = None) -> dict[str, dict[str, Any]]:
    """Declared parameter specs from ``space`` (the ledger's ``search_space``) or else
    ``meta['space']`` / ``meta['search_space']`` (SearchSpace json or object); {} when absent
    (observed ranges are used instead)."""
    meta = study.meta or {}
    sp = space if space is not None else meta.get("space", meta.get("search_space"))
    if sp is not None and hasattr(sp, "to_json"):
        sp = sp.to_json()
    if isinstance(sp, Mapping) and isinstance(sp.get("params"), (list, tuple)):
        return {p["name"]: dict(p) for p in sp["params"] if isinstance(p, Mapping) and "name" in p}
    return {}


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def plateau_score(study: StudyResult, periods_per_year: float = 260.0, *,
                  radius: float | Mapping[str, float] | None = None, space: Any = None) -> dict[str, Any]:
    """Matrix-based parameter plateau over the recorded trials (DESIGN §4.2 v1.2; red-team B4 + M5).

    The gate uses this only when no evaluator is available (fallback, labelled as such); with
    an evaluator it runs :func:`judge_plateau`, which does not depend on the grid step or on
    how densely a Sobol set happens to cover the ±radius box (N4).  ``space``: declared
    SearchSpace json (the gates pass the ledger's); default ``study.meta['space']``.

    Score = share of the selected configuration's neighbours whose full-dev Sharpe (computed
    here from ``study.returns``) is ≥ ``PLATEAU_PEAK_FRACTION`` × the selected Sharpe and > 0.
    The peak fraction is the DESIGN constant and cannot be changed by the caller.

    Neighbourhood (a box; a neighbour must satisfy every parameter):
    * numeric with a declared plateau scale (R2-1): |y − x| ≤ ``plateau_step``, or ≤ r·|x| for
      ``plateau_scale="relative"``; legacy spaces without one fall back to the two rules below;
    * numeric, strictly positive (declared ``low`` > 0, else all observed values > 0):
      |y − x| ≤ r·|x| (relative);
    * other numeric: |y − x| ≤ r·(high − low) of the declared range (``study.meta['space']``),
      falling back to the observed range;
    * unordered categorical: equal; ordered categorical (declared ``ordered``): adjacent levels;
    * if a numeric parameter has no other level within its radius (coarse grids), its adjacent
      levels (next lower / next higher observed value) count.
    ``r`` = ``radius`` (float, or dict name → float for per-parameter values pre-registered on
    the hypothesis card), default ``PLATEAU_RADIUS`` = 0.20.  ``invalid`` trials (never
    evaluated) are excluded; any other neighbour without a return column (errors) counts as a
    failing neighbour."""
    radii = dict(radius) if isinstance(radius, Mapping) else {}
    r_default = float(radius) if isinstance(radius, (int, float)) else PLATEAU_RADIUS
    trials = study.trials
    sel = selected_trial_id(study)
    sr = _trial_sharpes(study, periods_per_year)
    ids = np.array(trials["trial_id"].to_list())
    hit = np.nonzero(ids == sel)[0]
    if hit.size == 0:
        raise ValueError(f"selected trial {sel} not in study.trials")
    isel = int(hit[0])
    valid = np.ones(len(ids), bool)
    if "status" in trials.columns:
        valid = np.array([s != "invalid" for s in trials["status"].to_list()])
    specs = _space_specs(study, space)
    mask = valid.copy()
    rules: dict[str, str] = {}
    for name in study.param_names:
        vals = trials[f"param_{name}"].to_list()
        x = vals[isel]
        spec = specs.get(name, {})
        kind = spec.get("kind")
        numeric = kind in ("int", "float") or (kind is None and _is_num(x)
                                                and all(v is None or _is_num(v) for v in vals))
        if not numeric:
            if spec.get("ordered") and spec.get("choices"):
                ch = list(spec["choices"])
                pos = {c: i for i, c in enumerate(ch)}
                ix = pos.get(x)
                m = np.array([v in pos and ix is not None and abs(pos[v] - ix) <= 1 for v in vals])
                rules[name] = "ordered categorical: adjacent levels"
            else:
                m = np.array([v == x for v in vals])
                rules[name] = "categorical: equal"
            mask &= m
            continue
        r = float(radii.get(name, r_default))
        obs = np.array([np.nan if v is None else float(v) for v in vals])
        ok_obs = obs[valid & np.isfinite(obs)]
        lo = float(spec["low"]) if spec.get("low") is not None else float(np.min(ok_obs))
        hi = float(spec["high"]) if spec.get("high") is not None else float(np.max(ok_obs))
        positive = lo > 0 if spec.get("low") is not None else bool(ok_obs.size and ok_obs.min() > 0)
        if x is None:
            mask &= ~np.isfinite(obs)
            rules[name] = "missing value: equal"
            continue
        xf = float(x)
        if spec.get("plateau_step") is not None:             # R2-1: pre-registered absolute step
            tol, rule = float(spec["plateau_step"]), f"±{float(spec['plateau_step']):g} (plateau_step)"
            if spec.get("low") is not None and spec.get("high") is not None:
                tol = max(tol, PLATEAU_MIN_RANGE_FRACTION * (float(spec["high"]) - float(spec["low"])))
        elif spec.get("plateau_scale") == "relative" and xf > 0:
            tol, rule = r * abs(xf), f"±{r:.0%} relative (plateau_scale)"
            if spec.get("low") is not None and spec.get("high") is not None:
                tol = max(tol, PLATEAU_MIN_RANGE_FRACTION * (float(spec["high"]) - float(spec["low"])))
        elif positive and xf > 0:
            tol, rule = r * abs(xf), f"±{r:.0%} relative"
        else:
            tol, rule = r * (hi - lo), f"±{r:.0%} of range [{lo:g}, {hi:g}]"
        tol *= 1.0 + 1e-9
        m = np.isfinite(obs) & (np.abs(obs - xf) <= tol)
        levels = np.unique(ok_obs)
        if not np.any((levels != xf) & (np.abs(levels - xf) <= tol)):
            below, above = levels[levels < xf], levels[levels > xf]
            adj = ([below.max()] if below.size else []) + ([above.min()] if above.size else [])
            m |= np.isin(obs, adj)
            rule += " → adjacent levels (none inside radius)"
        rules[name] = rule
        mask &= m
    mask[isel] = False
    nb = np.nonzero(mask)[0]
    peak = float(sr.get(sel, float("nan")))
    nv = np.array([sr.get(int(ids[i]), float("-inf")) for i in nb], dtype=float)
    nv = np.where(np.isfinite(nv), nv, -np.inf)
    if nb.size and np.isfinite(peak) and peak > 0:
        score = float(np.mean((nv >= PLATEAU_PEAK_FRACTION * peak) & (nv > 0)))
    elif nb.size:
        score = 0.0
    else:
        score = float("nan")
    return {"plateau_score": score, "n_neighbours": int(nb.size), "peak_sharpe": peak,
            "neighbour_trials": [int(ids[i]) for i in nb], "rules": rules,
            "method": "economic ±radius box (" + "; ".join(f"{k}: {v}" for k, v in rules.items()) + ")"}


def _radius_for(radius: Any, name: str) -> float:
    if isinstance(radius, Mapping):
        return float(radius.get(name, PLATEAU_RADIUS))
    return float(radius if radius is not None else PLATEAU_RADIUS)


def _round_half_up(v: float) -> int:
    return int(math.floor(v + 0.5))


_AXIS_MULTS = (-1.0, -0.5, 0.5, 1.0)          # ±r and ±r/2 (relative) / ±step and ±step/2 (absolute)


def _axis_ladder(p: Any, x: float, radius: Any) -> tuple[list[tuple[str, float, Any]], str]:
    """The 4 perturbed values of numeric param ``p`` around ``x``: ``[(label, multiplier, value)]``
    in multiplier order (−1, −½, +½, +1) and the rule text.  ``plateau_scale="relative"`` moves
    by m·r·|x|; ``plateau_step=S`` by m·S (R2-1: the scale is pre-registered in the space).  Ints
    are rounded half up; a point that rounds back to x moves to the next int in its direction and
    the outer point is kept distinct from the inner one (so each side has two distinct ints)."""
    step = getattr(p, "plateau_step", None)
    scale = getattr(p, "plateau_scale", None)
    floor = PLATEAU_MIN_RANGE_FRACTION * (float(p.high) - float(p.low))
    if step is not None:
        unit, rule = float(step), f"±{float(step):g}/2, ±{float(step):g} (absolute plateau_step)"
        labels = [f"-{unit:g}", f"-{unit / 2:g}", f"+{unit / 2:g}", f"+{unit:g}"]
    elif scale == "relative":
        r = _radius_for(radius, p.name)
        unit, rule = r * abs(x), f"×(1 ± {r:g}/2), ×(1 ± {r:g}) (relative)"
        labels = [f"x{1 - r:g}", f"x{1 - r / 2:g}", f"x{1 + r / 2:g}", f"x{1 + r:g}"]
    else:
        raise GateError(f"parameter {p.name!r} declares no plateau scale (plateau_scale='relative' or "
                        f"plateau_step=); the space predates red-team R2-1 and cannot be judged — re-run the study")
    if unit < floor:                                # R3-3: 5 % of the declared range, read-only
        unit = floor
        rule += f" → floored at ±{floor:g}/2, ±{floor:g} ({PLATEAU_MIN_RANGE_FRACTION:.0%} of the range)"
        labels = [f"-{unit:g}", f"-{unit / 2:g}", f"+{unit / 2:g}", f"+{unit:g}"]
    raw = [x + m * unit for m in _AXIS_MULTS]
    if p.kind == "int":
        xi = _round_half_up(x)
        dn_in = _round_half_up(raw[1])
        dn_in = xi - 1 if dn_in >= xi else dn_in
        dn_out = _round_half_up(raw[0])
        dn_out = dn_in - 1 if dn_out >= dn_in else dn_out
        up_in = _round_half_up(raw[2])
        up_in = xi + 1 if up_in <= xi else up_in
        up_out = _round_half_up(raw[3])
        up_out = up_in + 1 if up_out <= up_in else up_out
        vals: list[Any] = [dn_out, dn_in, up_in, up_out]
    else:
        vals = [round(v, 12) for v in raw]
    return [(lab, m, v) for lab, m, v in zip(labels, _AXIS_MULTS, vals)], rule


def _numeric_flags(p: Any, v: Any) -> tuple[bool, bool]:
    """(outside the declared search bounds, naturally valid) for a numeric value."""
    lo, hi = float(p.low), float(p.high)
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    natural = not ((lo > 0 and v <= 0) or (p.kind == "int" and lo >= 1 and v < 1))
    return (not (lo - tol <= v <= hi + tol)), natural


def plateau_perturbations(space: Any, selected: Mapping[str, Any],
                          radius: float | Mapping[str, float] | None) -> list[dict[str, Any]]:
    """The pre-registered perturbation set around ``selected`` (N4, R2-1).

    Per axis:

    * numeric params move one at a time by their **declared** plateau scale (R2-1): a
      ``plateau_scale="relative"`` param to x·(1 − r), x·(1 − r/2), x·(1 + r/2), x·(1 + r); a
      ``plateau_step=S`` param to x − S, x − S/2, x + S/2, x + S (r = the study's radius from
      the ledger; per-param radii apply to relative params).  The outer move is floored at
      :data:`PLATEAU_MIN_RANGE_FRACTION` (5 %) of the declared range, the inner at half of it
      (R3-3).  Ints: see :func:`_axis_ladder`;
    * ordered categoricals: the levels at −2, −1, +1, +2 (up to 4 points); a level beyond either
      end does not exist: it is listed (``missing_level``) but not scored, so the axis share is
      over the levels that exist (a 3-level param at its centre or edge: 2 points, both must pass);
    * unordered categoricals are held fixed (no axis; numeric-valued ones are refused when the
      space is built).

    Joint axis ``"joint"`` (R2-1, when ≥ 2 numeric params): every numeric param moved together
    at ±r/2 and ±r of its own scale — all with the same sign (4 points) plus the two alternating
    sign patterns (+ − + …, − + − …) at r/2 and r (4 points): 8 points in all.

    The declared [low, high] are SEARCH limits, not validity limits (round 2b): points outside
    them are kept and evaluated (``outside_search_bounds = True``).  Only natural validity is
    enforced here (``natural_valid``): a strictly positive param (declared low > 0) must stay
    > 0, and an int with declared low ≥ 1 (lookback / period) must stay ≥ 1.
    Returns dicts ``param, offset, value, outside_search_bounds, natural_valid, params``."""
    pts: list[dict[str, Any]] = []
    ladders: dict[str, list[tuple[str, float, Any]]] = {}
    num_params = []
    for p in space.params:
        if p.name not in selected:
            continue
        x = selected[p.name]
        if p.kind == "categorical":
            if not p.ordered or x not in p.choices:
                continue
            ch = list(p.choices)
            i = ch.index(x)
            for dj in (-2, -1, 1, 2):
                j, lab = i + dj, f"level {dj:+d}"
                if 0 <= j < len(ch):
                    pts.append({"param": p.name, "offset": lab, "value": ch[j], "outside_search_bounds": False,
                                "natural_valid": True, "params": {**dict(selected), p.name: ch[j]}})
                else:
                    pts.append({"param": p.name, "offset": lab, "value": None, "outside_search_bounds": False,
                                "natural_valid": False, "missing_level": True,
                                "params": {**dict(selected), p.name: None}})
            continue
        if x is None:
            continue
        ladder, _ = _axis_ladder(p, float(x), radius)
        ladders[p.name] = ladder
        num_params.append(p)
        for lab, _m, v in ladder:
            outside, natural = _numeric_flags(p, v)
            pts.append({"param": p.name, "offset": lab, "value": v, "outside_search_bounds": outside,
                        "natural_valid": natural, "params": {**dict(selected), p.name: v}})
    if len(num_params) >= 2:
        alt = [1.0 if k % 2 == 0 else -1.0 for k in range(len(num_params))]
        patterns = ([("all -r", [-1.0] * len(num_params)), ("all -r/2", [-0.5] * len(num_params)),
                     ("all +r/2", [0.5] * len(num_params)), ("all +r", [1.0] * len(num_params))]
                    + [(f"alt{'+' if sg > 0 else '-'} {nm}", [sg * m * a for a in alt])
                       for sg in (1.0, -1.0) for nm, m in (("r/2", 0.5), ("r", 1.0))])
        for lab, mults in patterns:
            vals, outside_any, natural_all = {}, False, True
            for p, m in zip(num_params, mults):
                v = next(val for _l, mm, val in ladders[p.name] if mm == m)
                vals[p.name] = v
                o, n = _numeric_flags(p, v)
                outside_any |= o
                natural_all &= n
            pts.append({"param": "joint", "offset": lab, "value": dict(vals), "outside_search_bounds": outside_any,
                        "natural_valid": natural_all, "params": {**dict(selected), **vals}})
    return pts


def selected_at_edge(space: Any, selected: Mapping[str, Any]) -> list[str]:
    """Params whose selected value sits on a declared search bound (or first/last ordered level)."""
    out = []
    for p in space.params:
        x = selected.get(p.name)
        if x is None:
            continue
        if p.kind == "categorical":
            if p.ordered and x in p.choices and list(p.choices).index(x) in (0, len(p.choices) - 1):
                out.append(p.name)
        elif float(x) <= float(p.low) or float(x) >= float(p.high):
            out.append(p.name)
    return out


def judge_plateau(study: StudyResult, evaluator: Any, *, space: Any, radius: float | Mapping[str, float] | None,
                  periods_per_year: float = 260.0, n_jobs: Any = 1) -> dict[str, Any]:
    """Judge-run plateau (DESIGN §4.2 v1.2; red-team N4, N8; round 2b).

    Evaluates the selected configuration and every :func:`plateau_perturbations` point that is
    naturally valid and accepted by ``space.is_valid`` (the constraint; search bounds are NOT a
    validity check — points outside them are evaluated and reported) with
    ``evaluator(params, cost=None)`` (the evaluator's base cost, as in ``run_study``), on
    ``opt``'s runner (in-process for ``n_jobs=1``, else its capped process pool).

    A point passes if its full-dev Sharpe is ≥ ``PLATEAU_PEAK_FRACTION`` × peak and > 0, peak =
    the re-evaluated selected configuration (nothing passes when peak ≤ 0); invalid and erroring
    points fail; an ordered level beyond either end is listed but not scored.  **Score = the minimum over axes of
    the per-axis pass share**, the axes being every numeric / ordered parameter plus the
    ``"joint"`` axis (R2-1: all numeric params moved together, 8 points, when there are ≥ 2) —
    N8: an irrelevant axis scores 1.0 and cannot lift a failing relevant axis; with 4 points per
    axis the 0.60 gate means ≥ 3 of 4 on every axis and ≥ 5 of 8 jointly.  Unordered
    categoricals give no axis.  The pooled share and ``pass_share_by_param`` are diagnostics.
    NaN (gate SKIPPED) when there is no axis."""
    from . import opt
    t0 = time.perf_counter()
    ppy = float(periods_per_year)
    sel = dict(study.selected_params)
    pts = plateau_perturbations(space, sel, radius)
    for q in pts:
        if q.get("missing_level"):
            q["status"] = "missing_level"
        elif not q["natural_valid"]:
            q["status"] = "invalid_natural"
        elif not space.is_valid(q["params"]):
            q["status"] = "invalid"
        else:
            q["status"] = "pending"
    todo = [sel] + [q["params"] for q in pts if q["status"] == "pending"]
    jobs = min(opt._auto_jobs(n_jobs), max(1, len(todo)))
    runner = opt._Runner(evaluator, None, jobs)
    try:
        res = list(runner.map(todo))
    finally:
        runner.close()
    it = iter(res[1:])
    for q in pts:
        q["sharpe"] = None
        if q["status"] != "pending":
            continue
        o = next(it)
        if o["ok"]:
            q["sharpe"], q["status"] = metrics.sharpe(o["ret"], ppy), "ok"
        else:
            q["status"], q["error"] = "error", str(o.get("error"))[:300]
    sel_col = f"t{selected_trial_id(study)}"
    peak_study = (metrics.sharpe(study.returns[sel_col].fill_null(0.0), ppy)
                  if sel_col in study.returns.columns else float("nan"))
    if res[0]["ok"]:
        peak, peak_src = metrics.sharpe(res[0]["ret"], ppy), "re-evaluated"
    else:
        peak, peak_src = peak_study, f"study returns (re-evaluation failed: {str(res[0].get('error'))[:120]})"
    ok_peak = peak is not None and np.isfinite(peak) and peak > 0
    for q in pts:
        sr = q["sharpe"]
        q["pass"] = bool(ok_peak and q["status"] == "ok" and sr is not None and np.isfinite(sr)
                         and sr > 0 and sr >= PLATEAU_PEAK_FRACTION * peak)
    by_param: dict[str, list] = {}
    for q in pts:
        if q.get("missing_level"):          # a level beyond the end of an ordered categorical: reported only
            continue
        by_param.setdefault(q["param"], []).append(q["pass"])
    shares = {k: float(np.mean(v)) for k, v in by_param.items()}
    counts = {k: (int(sum(v)), len(v)) for k, v in by_param.items()}
    if shares:
        weakest = min(shares, key=lambda k: (shares[k], list(shares).index(k)))
        score = shares[weakest]
    else:
        weakest, score = None, float("nan")
    scored = [q["pass"] for q in pts if not q.get("missing_level")]
    pooled = float(np.mean(scored)) if scored else float("nan")
    return {"plateau_score": score, "score_rule": "min over parameter axes", "pooled_share": pooled,
            "weakest_axis": weakest, "weakest_axis_passes": counts.get(weakest), "n_axes": len(shares),
            "n_points": len(pts), "n_neighbours": len(pts), "n_evaluations": len(todo),
            "n_invalid": sum(q["status"] in ("invalid", "invalid_natural") for q in pts),
            "n_missing_levels": sum(q["status"] == "missing_level" for q in pts),
            "n_joint_points": sum(q["param"] == "joint" for q in pts),
            "n_outside_search_bounds": sum(bool(q["outside_search_bounds"]) for q in pts),
            "outside_search_bounds_points": [(q["param"], q["value"]) for q in pts if q["outside_search_bounds"]],
            "selected_at_edge": selected_at_edge(space, sel),
            "peak_sharpe": float(peak) if peak is not None else float("nan"), "peak_source": peak_src,
            "peak_sharpe_study": float(peak_study), "points": pts, "radius": radius,
            "pass_share_by_param": shares, "pass_count_by_param": counts,
            "runtime_s": time.perf_counter() - t0, "n_jobs": jobs,
            "method": (f"judge-run perturbations per axis at ±½ and ±1 of each param's pre-registered plateau "
                       f"scale (relative: r·|x|, r = {radius}; absolute: plateau_step), ordered categoricals ±1/±2 "
                       f"levels, plus a joint axis (all numeric params together); score = min over axes; "
                       f"{len(pts)} points, {len(todo)} evaluations")}


def _outcome_sharpe(o: Outcome, ppy: float) -> float:
    return metrics.sharpe(o.daily, ppy)


def _per_trade_returns(trades: pl.DataFrame) -> np.ndarray | None:
    if trades is None or trades.height == 0:
        return None
    if {"pnl_ccy", "equity_before"} <= set(trades.columns):
        t = trades
        if "skipped" in t.columns:
            t = t.filter(~pl.col("skipped").cast(pl.Boolean))
        return (t["pnl_ccy"] / t["equity_before"]).to_numpy().astype(float)
    if "ret" in trades.columns:
        return trades["ret"].to_numpy().astype(float)
    return None


def _trades_per_day(trades: pl.DataFrame | None, dates: pl.Series) -> np.ndarray | None:
    if trades is None or trades.height == 0 or "entry_ts" not in trades.columns:
        return None
    t = trades
    if "skipped" in t.columns:
        t = t.filter(~pl.col("skipped").cast(pl.Boolean))
    cnt = t.group_by(pl.col("entry_ts").dt.date().alias("date")).agg(pl.len().alias("n"))
    grid = pl.DataFrame({"date": dates}).join(cnt, on="date", how="left").fill_null(0)
    return grid["n"].to_numpy().astype(float)


def wfo_trades_per_day(study: StudyResult) -> tuple[np.ndarray | None, str]:
    """Daily trade counts of the walk-forward *procedure* on its OOS days (F4), aligned with
    ``study.wfo_oos`` sorted by date: ``wfo_oos['n_trades']`` if present with no nulls, else rebuilt from
    ``meta['trade_counts']`` (wide date + t<id>) following ``wfo_params.selected_trial`` per
    refit.  (None, reason) when unavailable."""
    w = study.wfo_oos
    if w is None or w.height == 0:
        return None, "no WFO series"
    w = w.sort("date")
    if "n_trades" in w.columns and w["n_trades"].null_count() == 0:
        return w["n_trades"].cast(pl.Float64).to_numpy(), "wfo_oos.n_trades"
    tc = (study.meta or {}).get("trade_counts")
    wp = study.wfo_params
    if (not isinstance(tc, pl.DataFrame) or tc.height == 0 or wp is None or wp.height == 0
            or "refit_id" not in w.columns or "selected_trial" not in wp.columns):
        return None, "no WFO trade counts"
    pick = dict(zip(wp["refit_id"].to_list(), wp["selected_trial"].to_list()))
    tcd = tc.with_columns(pl.col("date").cast(pl.Date))
    j = w.select(pl.col("date").cast(pl.Date), "refit_id").join(tcd, on="date", how="left")
    out = np.zeros(j.height)
    rid = j["refit_id"].to_numpy()
    for r, tid in pick.items():
        col = f"t{tid}" if tid is not None else None
        if col is None or col not in j.columns:
            continue
        sel = rid == r
        out[sel] = j[col].fill_null(0.0).to_numpy()[sel]
    return out, ("WFO procedure (meta.trade_counts along wfo_params; approximate — counted by exit date, "
                 "not entry)")


_MEDIAN_SPREAD: dict[tuple[str, str, str], float] = {}


def _median_spread(symbol: str, book: str) -> float:
    """Median dev-window D1 spread in points (display in the cost-stress row); NaN if unavailable.
    Cached per (symbol, book, manifest)."""
    key = (str(symbol), str(book), ledger.manifest_sha())
    if key not in _MEDIAN_SPREAD:
        try:
            from .costs import _median_dev_spread_points
            _MEDIAN_SPREAD[key] = float(_median_dev_spread_points(symbol, book=book))
        except Exception:  # noqa: BLE001 — display only
            _MEDIAN_SPREAD[key] = float("nan")
    return _MEDIAN_SPREAD[key]


def _resolve_base_cost(evaluator, base_cost):
    if base_cost is not None:
        return base_cost
    for attr in ("cost", "cost_model", "base_cost"):
        c = getattr(evaluator, attr, None)
        if c is not None and hasattr(c, "stressed"):
            return c
    return None


def _is_synthetic(evaluator: Any) -> bool:
    """SyntheticEvaluator (or a wrapper exposing it as ``.inner``, or ``is_synthetic = True``):
    no instrument, so the pip → point conversion of the stress model does not apply."""
    from .evaluators import SyntheticEvaluator
    return bool(isinstance(evaluator, SyntheticEvaluator) or getattr(evaluator, "is_synthetic", False)
                or isinstance(getattr(evaluator, "inner", None), SyntheticEvaluator))


def _resolve_spec(evaluator: Any, spec: Any) -> tuple[Any, str]:
    """Instrument spec for the 1-pip stress slippage (M2): explicit ``spec=`` → ``evaluator.spec``
    → ``costs.load_instrument(evaluator.symbol, book=evaluator.book)``.  (None, reason) if none."""
    sym = getattr(evaluator, "symbol", None)
    if spec is not None:
        ssym = getattr(spec, "symbol", None)
        if sym and ssym and str(ssym).upper() != str(sym).upper():
            raise GateError(f"spec= is for {ssym!r} but the evaluator trades {sym!r} (red-team N7): the stress "
                            f"slippage would be converted with the wrong instrument")
        return spec, "spec="
    s = getattr(evaluator, "spec", None)
    if s is not None:
        ssym = getattr(s, "symbol", None)
        if sym and ssym and str(ssym).upper() != str(sym).upper():
            raise GateError(f"evaluator.spec is for {ssym!r} but the evaluator trades {sym!r} (red-team N7)")
        return s, "evaluator.spec"
    if sym:
        from .costs import load_instrument
        try:
            return load_instrument(sym, book=getattr(evaluator, "book", "FBS")), f"load_instrument({sym})"
        except Exception as e:  # noqa: BLE001 — reported in the SKIPPED row
            return None, f"load_instrument({sym}) failed: {e}"
    return None, "evaluator has neither spec nor symbol"


def _study_trial_count(study: StudyResult) -> float:
    """This study's raw trial count: trials not ``invalid`` (never evaluated) — else
    ``meta['n_trials']`` — else the number of return columns."""
    t = study.trials
    if t is not None and t.height and "status" in t.columns:
        return float(t.filter(pl.col("status") != "invalid").height)
    n = (study.meta or {}).get("n_trials")
    if n is not None:
        return float(n)
    return float(len([c for c in study.returns.columns if c != "date"]))


def _canon(x: Any) -> Any:
    def _py(v):
        if isinstance(v, dict):
            return {str(k): _py(u) for k, u in v.items()}
        if isinstance(v, (list, tuple)):
            return [_py(u) for u in v]
        if isinstance(v, np.generic):
            return v.item()
        return v
    return json.loads(json.dumps(_py(x), sort_keys=True, default=str))


def _norm_radius(r: Any) -> Any:
    """Canonical radius (as ``opt.normalise_plateau_radius``) without importing opt eagerly."""
    from .opt import normalise_plateau_radius
    return normalise_plateau_radius(r)


def ledger_context(study: StudyResult, ledger_dir: Path | None = None) -> dict[str, Any]:
    """What the gates rely on, read from the study's ledger rows (red-team N1–N3).

    Returns book, system, issue, attempt, method, seed, space (``search_space`` json),
    ``plateau_radius`` (+ source), ``data_dependent`` (:func:`ledger.candidate_set_data_dependent`)
    and ``seq``.  Raises :class:`GateError` if the study is not in the ledger or if
    ``study.meta`` disagrees with the ledger on any of these keys (meta is a cache)."""
    row = ledger.created_row(study.study_id, ledger_dir=ledger_dir)
    where = ledger_dir if ledger_dir is not None else "the default research/ledger"
    if row is None:
        raise GateError(f"study {study.study_id} is not in the ledger ({where}); the gates read the study's "
                        f"identity, prior trials, plateau radius and candidate-set flag from its study_created "
                        f"row — run it through opt.run_study (or ledger.create_study) and pass ledger_dir=")
    radius_raw = row.get("plateau_radius")
    try:
        radius = _norm_radius(radius_raw)
    except (TypeError, ValueError) as e:
        raise GateError(f"study {study.study_id}: ledger plateau_radius {radius_raw!r} is not usable: {e}") from e
    ctx = {"book": row.get("book"), "system": row.get("system"), "issue": row.get("issue"),
           "attempt": row.get("attempt"), "method": row.get("method"), "seed": row.get("seed"),
           "space": row.get("search_space"), "plateau_radius": radius,
           "radius_source": ("ledger study_created row" if radius_raw is not None else
                             f"DESIGN default {PLATEAU_RADIUS:g} (not recorded in the ledger row)"),
           "data_dependent": ledger.candidate_set_data_dependent(study.study_id, ledger_dir=ledger_dir),
           "seq": row.get("seq"), "ledger_dir": str(where)}
    meta = study.meta or {}
    book_name = (lambda v: config.get_book(v).name if v else v)
    checks = [("book", "book", book_name), ("system", "system", None), ("issue", "issue", None),
              ("attempt", "attempt", None), ("seed", "seed", None), ("method", "method", None),
              ("candidate_set", "method", None), ("space", "space", None),
              ("plateau_radius", "plateau_radius", _norm_radius),
              ("candidate_set_data_dependent", "data_dependent", bool)]
    bad = {}
    for mk, ck, f in checks:
        if mk not in meta or meta[mk] is None or ctx.get(ck) is None:
            if mk == "candidate_set_data_dependent" and meta.get(mk) is not None and meta[mk] != ctx["data_dependent"]:
                bad[mk] = (meta[mk], ctx["data_dependent"])
            continue
        mv, lv = meta[mk], ctx[ck]
        if hasattr(mv, "to_json"):
            mv = mv.to_json()
        try:
            mv, lv = (f(mv), f(lv)) if f is not None else (mv, lv)
        except (TypeError, ValueError) as e:
            bad[mk] = (meta[mk], ctx[ck], repr(e))
            continue
        if _canon(mv) != _canon(lv):
            bad[mk] = (meta[mk] if not hasattr(meta[mk], "to_json") else "<SearchSpace>", ctx[ck])
    if bad:
        raise GateError(f"study {study.study_id}: study.meta disagrees with the ledger (meta, ledger): {bad}. "
                        f"The ledger is the source of truth; re-run or resume the study instead of editing meta")
    return ctx


def resolve_prior_trials(study: StudyResult, prior_trials: float | None = None,
                         ledger_dir: Path | None = None, *, ctx: dict[str, Any] | None = None
                         ) -> tuple[float, dict[str, Any]]:
    """Raw trials of related studies (M4 / N2) — :func:`ledger.related_prior_trials`: every other
    study in the ledger (created before or after this one) that shares its normalised system
    name or its issue number, whatever the attempt.  An explicit ``prior_trials`` below that count raises
    :class:`GateError` (it may only be higher — conservative).  Attempt > 1 with no prior found
    raises unless an explicit positive ``prior_trials`` is given."""
    ctx = ctx or ledger_context(study, ledger_dir)
    total, used, _ = ledger.related_prior_trials(study.study_id, ledger_dir=ledger_dir)
    attempt = ctx.get("attempt")
    info = {"prior_source": (f"ledger: {', '.join(used)}" if used else "ledger: no earlier related studies"),
            "prior_trials_ledger": float(total), "prior_studies": used, "book": ctx.get("book"),
            "system": ctx.get("system"), "issue": ctx.get("issue"), "attempt": attempt}
    if prior_trials is not None:
        pt = float(prior_trials)
        if not np.isfinite(pt) or pt < total:
            raise GateError(f"prior_trials={prior_trials!r} is below the ledger's count of related earlier trials "
                            f"({total:g} from {used}); an explicit prior may only raise N (red-team N2)")
        if attempt is not None and int(attempt) > 1 and pt <= 0:
            raise GateError(f"study {study.study_id} is attempt {attempt}: prior_trials must be > 0")
        info["prior_source"] = f"explicit prior_trials={pt:g} (≥ ledger {total:g})"
        return pt, info
    if attempt is not None and int(attempt) > 1 and (not used or total <= 0):
        raise GateError(f"study {study.study_id} is attempt {attempt} but the ledger holds no earlier study of "
                        f"system {ctx.get('system')!r} / issue {ctx.get('issue')!r}; the earlier attempts must be "
                        f"in the ledger (or pass a conservative prior_trials= > 0)")
    return float(total), info


def resolve_studies_dir(study_id: str, studies_dir: Path | None = None, ledger_dir: Path | None = None) -> Path:
    """Trial-store root of a study: ``studies_dir=`` → the ``studies_dir`` recorded in its
    ``study_created`` row → ``config.STUDIES_DIR``.  An explicit value that differs from the
    recorded one raises :class:`GateError` (the store is the one the optimizer wrote)."""
    row = ledger.created_row(study_id, ledger_dir=ledger_dir) or {}
    rec = row.get("studies_dir")
    if studies_dir is not None:
        if rec and Path(studies_dir).resolve() != Path(rec).resolve():
            raise GateError(f"study {study_id}: studies_dir={studies_dir} differs from the trial store recorded in "
                            f"the ledger ({rec})")
        return Path(studies_dir)
    return Path(rec) if rec else Path(config.STUDIES_DIR)


def verify_trial_store(study: StudyResult, *, studies_dir: Path | None = None,
                       ledger_dir: Path | None = None) -> dict[str, Any]:
    """Check the in-memory ``StudyResult`` against the study's trial store (red-team R2-4).

    Loads ``ledger.load_trials`` / ``ledger.load_trial_returns`` for ``study.study_id`` and
    requires: the same trial ids (every recorded trial present — no truncation, no foreign
    trials), the same status and ``source`` per trial, and the same return columns with equal
    values on every recorded date (0 / null elsewhere, as ``opt`` builds the matrix).  Any
    mismatch — e.g. a TPE study relabelled with a clean study's id — raises :class:`GateError`.

    R3-2: ``selection['trial_id']`` and ``selected_params`` must equal the ledger's ``selection``
    event, and ``cpcv_paths`` / ``wfo_oos`` / ``wfo_params`` (and ``meta['trade_counts']`` /
    ``['entry_counts']`` when present) must hash (:func:`ledger.frame_sha256`) to the values that
    event logged for the parquet artifacts ``opt.run_study`` wrote into the store.

    Returns ``n_store`` (store trials that are not ``invalid``), ``n_ledger`` (the optimizer's
    ``trials`` event count, :func:`ledger.logged_trial_count`), ``tpe_in_store``, ``studies_dir``
    and ``artifacts`` (the stored frames, which the gates use)."""
    sid = study.study_id
    sdir = resolve_studies_dir(sid, studies_dir, ledger_dir)
    store = ledger.load_trials(sid, sdir)
    if store.height == 0:
        raise GateError(f"study {sid}: no trial store under {sdir / sid}; the gates load the trials from the store "
                        f"written by opt.run_study (red-team R2-4) — pass studies_dir= if it lives elsewhere")
    mem = study.trials
    if mem is None or mem.height == 0 or "trial_id" not in mem.columns:
        raise GateError(f"study {sid}: study.trials is empty; it must be the study's full trial table")
    s_ids = sorted(int(x) for x in store["trial_id"].to_list())
    m_ids = sorted(int(x) for x in mem["trial_id"].to_list())
    if s_ids != m_ids:
        miss = sorted(set(s_ids) - set(m_ids))
        extra = sorted(set(m_ids) - set(s_ids))
        raise GateError(f"study {sid}: study.trials does not match the trial store ({len(m_ids)} in memory vs "
                        f"{len(s_ids)} recorded; missing {miss[:10]}{'…' if len(miss) > 10 else ''}, not recorded "
                        f"{extra[:10]}{'…' if len(extra) > 10 else ''}) — red-team R2-4: the gates never trust a "
                        f"hand-edited StudyResult")
    for colname in ("status", "source"):
        if colname in store.columns and colname in mem.columns:
            a = dict(zip(store["trial_id"].to_list(), store[colname].to_list()))
            b = dict(zip(mem["trial_id"].to_list(), mem[colname].to_list()))
            bad = [t for t in a if (a[t] or None) != (b.get(t) or None)]
            if bad:
                raise GateError(f"study {sid}: trial {colname} differs from the trial store for trials {bad[:10]} "
                                f"(R2-4)")
    rets = ledger.load_trial_returns(sid, sdir)
    s_cols = sorted(c for c in rets.columns if c != "date") if rets.height else []
    m_cols = sorted(c for c in study.returns.columns if c != "date")
    if s_cols != m_cols:
        raise GateError(f"study {sid}: study.returns columns differ from the trial store ({len(m_cols)} in memory vs "
                        f"{len(s_cols)} recorded) (R2-4)")
    if s_cols:
        w = rets.with_columns(pl.col("date").cast(pl.Date))
        m = study.returns.with_columns(pl.col("date").cast(pl.Date))
        j = w.join(m, on="date", how="full", coalesce=True, suffix="__mem")
        for c in s_cols:
            sv = j[c].cast(pl.Float64).to_numpy()
            mv = j[c + "__mem"].cast(pl.Float64).fill_null(0.0).fill_nan(0.0).to_numpy()
            has = np.isfinite(sv)
            if not (np.allclose(sv[has], mv[has], rtol=0.0, atol=1e-12) and np.all(mv[~has] == 0.0)):
                raise GateError(f"study {sid}: returns of {c} differ from the trial store (R2-4: content check)")
    status = store["status"].to_list() if "status" in store.columns else []
    n_store = float(sum(s != "invalid" for s in status)) if status else float(store.height)
    tpe = bool("source" in store.columns and (store["source"] == "tpe").any())
    # R3-2: selection and OOS artifacts must be the ones the optimizer logged
    sel_ev = ledger.study_events(sid, "selection", ledger_dir=ledger_dir)
    if not sel_ev:
        raise GateError(f"study {sid}: no selection event in the ledger; the gates only judge studies produced by "
                        f"opt.run_study (red-team R3-2)")
    sel_ev = sel_ev[-1]
    led_sel = sel_ev.get("selection") or {}
    led_tid = led_sel.get("trial_id", led_sel.get("selected_trial"))
    mem_tid = (study.selection or {}).get("trial_id")
    if led_tid is not None and (mem_tid is None or int(mem_tid) != int(led_tid)):
        raise GateError(f"study {sid}: selection trial {mem_tid} differs from the ledger's selection event "
                        f"({led_tid}) — red-team R3-2")
    if _canon(sel_ev.get("selected_params") or {}) != _canon(study.selected_params or {}):
        raise GateError(f"study {sid}: selected_params {study.selected_params} differ from the ledger's selection "
                        f"event {sel_ev.get('selected_params')} — red-team R3-2")
    hashes = sel_ev.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise GateError(f"study {sid}: the selection event records no artifact hashes (pre-R3-2 study); re-run it")
    meta = study.meta or {}
    mem_frames = {"cpcv_paths": study.cpcv_paths, "wfo_oos": study.wfo_oos, "wfo_params": study.wfo_params,
                  "trade_counts": meta.get("trade_counts"), "entry_counts": meta.get("entry_counts")}
    artifacts: dict[str, Any] = {}
    for name in ledger.STUDY_ARTIFACTS:
        want = hashes.get(name)
        if want is None:
            continue
        stored = ledger.load_study_artifact(sid, name, sdir)
        if ledger.frame_sha256(stored) != want:
            raise GateError(f"study {sid}: stored artifact {name} does not match its hash in the selection event "
                            f"(R3-2)")
        mf = mem_frames.get(name)
        optional = name in ("trade_counts", "entry_counts")        # meta caches; absent = read from the store
        if not (optional and mf is None) and ledger.frame_sha256(mf) != want:
            raise GateError(f"study {sid}: study.{name} differs from the series the optimizer stored (sha256) — "
                            f"red-team R3-2: OOS series cannot be recomputed or replaced after the study")
        artifacts[name] = stored
    return {"n_store": n_store, "n_ledger": ledger.logged_trial_count(sid, ledger_dir=ledger_dir),
            "tpe_in_store": tpe, "studies_dir": str(sdir), "artifacts": artifacts}


def check_evaluator_identity(study_id: str, evaluator: Any, base_cost: Any, *,
                             ledger_dir: Path | None = None) -> dict[str, Any]:
    """The evaluator passed to the gates must be the study's (red-team R2-4): its ``describe()``
    (or ``{"evaluator": <class name>}``, as ``opt.run_study`` records it) must equal the ledger
    row's ``evaluator``, and the base cost model version (``base_cost`` / ``evaluator.cost`` …,
    else ``evaluator.cost_version``) its ``cost_model_version``, and (R3-7) the SHA-256 of the
    strategy's source file recorded at creation (``opt.evaluator_code_fingerprint``) must equal the
    current one.  Raises :class:`GateError`."""
    from .opt import _py
    row = ledger.created_row(study_id, ledger_dir=ledger_dir) or {}
    led = row.get("evaluator")
    if led is None:
        raise GateError(f"study {study_id}: the ledger row records no evaluator description; the gates cannot "
                        f"check that the evaluator is the study's (R2-4) — run the study through opt.run_study")
    fn = getattr(evaluator, "describe", None)
    desc = fn() if callable(fn) else {"evaluator": type(evaluator).__name__}
    if _canon(_py(desc)) != _canon(led):
        raise GateError(f"study {study_id}: the evaluator passed to the gates is not the study's (ledger "
                        f"{json.dumps(_canon(led), sort_keys=True)[:300]} vs given "
                        f"{json.dumps(_canon(_py(desc)), sort_keys=True)[:300]}) — red-team R2-4")
    # same precedence as opt.run_study records it: the explicit cost model, else the evaluator's
    # cost_version, else its cost model (R3-7 harness note)
    bc = _resolve_base_cost(evaluator, None)
    ver = (getattr(base_cost, "version", None) if base_cost is not None else
           (getattr(evaluator, "cost_version", None) or getattr(bc, "version", None)))
    led_ver = row.get("cost_model_version")
    if ver is not None and led_ver not in (None, "unknown") and str(ver) != str(led_ver):
        raise GateError(f"study {study_id}: cost model version {ver!r} differs from the study's "
                        f"{led_ver!r} (ledger) — the gates must run on the study's cost model (R2-4)")
    from .opt import evaluator_code_fingerprint
    code_now = evaluator_code_fingerprint(evaluator)
    code_led = row.get("evaluator_code")
    if code_led is not None:
        if code_now is None or code_now.get("sha256") != code_led.get("sha256"):
            raise GateError(f"study {study_id}: the strategy / evaluator source {code_led.get('files')} changed since "
                            f"the study was run (sha256 {str(code_led.get('sha256'))[:12]} → "
                            f"{str((code_now or {}).get('sha256'))[:12]}) — red-team R3-7: re-run the study")
    elif code_now is not None:
        import warnings as _w
        _w.warn(f"study {study_id}: no evaluator source hash in the ledger row (pre-R3-7 study); the strategy code "
                f"cannot be checked against the study", stacklevel=3)
    return {"evaluator": _canon(led), "cost_model_version": led_ver, "evaluator_code": code_led}


def _judge_space(study: StudyResult, ctx: dict[str, Any], space: Any) -> tuple[Any, str]:
    """SearchSpace for the judge-run plateau: ``space=`` or ``meta['search_space_obj']`` (must
    match the ledger's ``search_space``), else rebuilt from the ledger json (constraint lost)."""
    js = ctx.get("space")
    if not isinstance(js, Mapping) or not isinstance(js.get("params"), (list, tuple)):
        return None, "no search space in the ledger row"
    obj = space if space is not None else (study.meta or {}).get("search_space_obj")
    if obj is not None:
        if _canon(obj.to_json()) != _canon(js):
            raise GateError(f"study {study.study_id}: the SearchSpace passed to the gates differs from the ledger's "
                            f"search_space")
        return obj, ""
    from .opt import space_from_json
    note = ("" if js.get("constraint") is None else
            f"constraint {js.get('constraint')!r} not available (pass space=): validity not checked, evaluator "
            f"errors count as failures")
    try:
        return space_from_json(js), note
    except ValueError as e:
        raise GateError(f"study {study.study_id}: the ledger's search space cannot be judged ({e}); spaces must "
                        f"declare a plateau scale per numeric param (red-team R2-1) — re-run the study") from e


# --------------------------------------------------------------------------- evaluation
def evaluate_gates(study: StudyResult, evaluator: Evaluator | None, *, periods_per_year: float,
                   selected_trades: pl.DataFrame | None = None,
                   mechanism_check: Callable[..., Any] | None = None, base_cost: Any = None,
                   swap_band: tuple[float, ...] = (0.5, 1.5), prior_trials: float | None = None,
                   ledger_dir: Path | None = None, holdout_days: int | None = None, pbo_splits: int = 16,
                   n_boot: int | None = None, seed: int | None = None, spec: Any = None, space: Any = None,
                   n_jobs: Any = 1, plateau_radius: Any = _REMOVED,
                   prior_effective_trials: Any = _REMOVED,
                   holdout_symbols: str | list[str] | None = None,
                   studies_dir: Path | None = None, log: bool = False) -> GateReport:
    """Run every DESIGN §4.2 v1.2 gate on a study.  See the module docstring for series choices.

    ``evaluator``: needed for the cost-stress gate (else SKIPPED) and, when
    ``selected_trades`` is None, to obtain the selected configuration's trades.
    ``base_cost``: the study's CostModel (else taken from ``evaluator.cost``/``cost_model``).
    ``spec``: InstrumentSpec for the stress's 1-pip slippage (else ``evaluator.spec``, else
    ``costs.load_instrument(evaluator.symbol)``); without one a non-synthetic evaluator's
    cost-stress row is SKIPPED (never the silent 1 pip = 1 point fallback).
    ``mechanism_check``: callable(study, evaluator) → bool or (bool, interpretation str).
    ``ledger_dir``: the ledger holding the study (it must be there — :class:`GateError`).
    ``prior_trials``: raw trials of related earlier studies; None → the ledger's count; an
    explicit value below the ledger's raises (N2).
    ``space``: the study's SearchSpace object (for ``is_valid`` in the judge-run plateau; else
    ``meta['search_space_obj']``, else rebuilt from the ledger).  ``n_jobs``: workers for the
    judge-run plateau evaluations (1 = in-process).
    ``holdout_days``: explicit horizon override for the pass band (tests).  Default (DESIGN §4.4,
    2026-09-24): holdout start → end of the data available now, read from the data manifest for
    the study's symbols registered in the ledger (R2-2: ``study_created`` row / registered band;
    ``holdout_symbols`` may only name them, or supply them when the ledger has none; else
    ``evaluator.symbol`` for a real evaluator, else ``meta['symbol(s)']``) and their conversion
    legs, by :func:`quantlab.data.holdout_horizon`; with no resolvable symbol, one year of periods
    (``horizon_source="default"`` — such a band must be rebuilt before any unlock).

    The in-memory study is checked against the ledger (red-team R2-4): its trials / returns must
    equal the trial store of ``study.study_id`` (:func:`verify_trial_store`; ``studies_dir``
    defaults to the one recorded in the ledger row, else ``config.STUDIES_DIR``), and the
    evaluator's description and cost model version must equal the ledger row's
    (:func:`check_evaluator_identity`); any mismatch raises :class:`GateError`.  The DSR's N for
    this study is the maximum of the in-memory count, the store's count and the optimizer's
    logged ``trials`` count.

    ``n_boot`` / ``seed`` are read-only (R2-3): the band and the budget diagnostic always use
    ``stats.HOLDOUT_BAND_N_BOOT`` and ``stats.holdout_band_seed(study_id)``; any other value
    raises :class:`GateError`.

    ``log=True`` (R3-1) appends the ``gates`` event for the report computed here — the only way a
    gate result reaches the ledger (the first run's band becomes the registered band); the row is
    returned in ``report.ledger_row``.
    Removed: ``plateau_radius`` (pre-registered via ``opt.run_study(plateau_radius=)`` and read
    from the ledger, N3) and ``prior_effective_trials`` (use ``prior_trials``) raise TypeError."""
    if plateau_radius is not _REMOVED:
        raise TypeError("evaluate_gates(plateau_radius=) was removed (red-team N3): the radius is pre-registered "
                        "from the hypothesis card via opt.run_study(plateau_radius=) and read from the study's "
                        "ledger row")
    if prior_effective_trials is not _REMOVED:
        raise TypeError("evaluate_gates(prior_effective_trials=) was removed (red-team N2): the DSR uses raw trial "
                        "counts read from the ledger; pass prior_trials= only to RAISE N above the ledger's count")
    ctx = ledger_context(study, ledger_dir)
    band_seed = st.holdout_band_seed(study.study_id)
    if n_boot is not None and int(n_boot) != st.HOLDOUT_BAND_N_BOOT:
        raise GateError(f"n_boot={n_boot!r}: the holdout band's bootstrap size is fixed at "
                        f"{st.HOLDOUT_BAND_N_BOOT} (red-team R2-3); omit n_boot")
    if seed is not None and int(seed) != band_seed:
        raise GateError(f"seed={seed!r}: the holdout band's seed is fixed per study (stats.holdout_band_seed = "
                        f"{band_seed}, red-team R2-3); omit seed")
    store = verify_trial_store(study, studies_dir=studies_dir, ledger_dir=ledger_dir)
    # R3-2: the trade / entry counts come from the store (the frames the optimizer logged)
    arts = store.get("artifacts") or {}
    if arts.get("trade_counts") is not None or arts.get("entry_counts") is not None:
        study = replace(study, meta={**(study.meta or {}),
                                     **{k: arts[k] for k in ("trade_counts", "entry_counts") if arts.get(k) is not None}})
    ev_ident = (check_evaluator_identity(study.study_id, evaluator, base_cost, ledger_dir=ledger_dir)
                if evaluator is not None else None)
    ppy = float(periods_per_year)
    ann = math.sqrt(ppy)
    meta = study.meta or {}
    rows: list[GateRow] = []
    diag: dict[str, Any] = {}
    sel = selected_trial_id(study)
    col = f"t{sel}"
    sel_daily = study.returns.select("date", pl.col(col).fill_null(0.0).alias("ret"))
    sel_sr_ann = metrics.sharpe(sel_daily, ppy)

    # B3 / N1: from the ledger (created row or any later event), never from meta alone
    data_dep = bool(ctx["data_dependent"]) or bool(store["tpe_in_store"])
    if study.trials is not None and "source" in study.trials.columns:
        data_dep = data_dep or bool((study.trials["source"] == "tpe").any())
    cpcv_meta = meta.get("cpcv") if isinstance(meta.get("cpcv"), Mapping) else {}
    emb_capped = bool(meta.get("embargo_capped") or cpcv_meta.get("embargo_capped"))
    skip_dd = ("candidate set is data-dependent (ledger: method=" + str(ctx.get("method")) + "): the configurations "
               "were chosen using the full dev sample, so this OOS series is not out of sample; re-run with a "
               "data-independent candidate set (grid / Sobol)")
    skip_emb = "embargo cap binds; use fewer CPCV groups"

    # ---- DSR (R1): V0 = 1/(T−1), N = raw trials (study + earlier attempts)
    # R2-4: never fewer than the trial store / the optimizer's logged count
    n_mem = _study_trial_count(study)
    n_study = max(n_mem, store["n_store"], store["n_ledger"])
    n_prior, pinfo = resolve_prior_trials(study, prior_trials, ledger_dir, ctx=ctx)
    dres = st.dsr_from_matrix(study.returns, col, periods_per_year=ppy, n_trials=n_study, extra_trials=n_prior)
    op, thr = GATE_THRESHOLDS["dsr"]
    ok = _cmp(op, dres.dsr, thr)
    rows.append(GateRow("dsr", dres.dsr, f"{op} {thr}", "PASS" if ok else "FAIL",
                        f"DSR {dres.dsr:.2f}: {dres.dsr:.0%} probability the true Sharpe exceeds the "
                        f"{dres.sr0_annual:.2f} expected from the best of N = {dres.n_trials:g} raw null trials "
                        f"({n_study:g} this study + {n_prior:g} from related earlier studies) over {dres.n_obs} days."))
    eff = {**dres.as_dict(), **dres.n_eff_by_method, **pinfo,
           "n_trials_in_memory": n_mem, "n_trials_store": store["n_store"], "n_trials_ledger": store["n_ledger"]}
    eff.pop("n_eff_by_method", None)

    # ---- CSCV OOS loss (R2); PBO etc. as diagnostics
    op, thr = GATE_THRESHOLDS["cscv_oos_loss"]
    pres = None
    try:
        pres = st.pbo_cscv(study.returns, n_splits=pbo_splits)
        diag["pbo_detail"] = {k: v for k, v in pres.summary().items()}
    except ValueError as e:
        diag["pbo_error"] = str(e)
    if data_dep or emb_capped:
        why = skip_dd if data_dep else skip_emb
        v = pres.prob_oos_loss if pres is not None else None
        rows.append(GateRow("cscv_oos_loss", v, f"{op} {thr}", "SKIPPED", f"Not gated: {why}."))
    elif pres is None:
        rows.append(GateRow("cscv_oos_loss", None, f"{op} {thr}", "SKIPPED",
                            f"CSCV not computable: {diag.get('pbo_error')}."))
    else:
        ok = _cmp(op, pres.prob_oos_loss, thr)
        rows.append(GateRow("cscv_oos_loss", pres.prob_oos_loss, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"The in-sample-best config lost money out of sample in {pres.prob_oos_loss:.0%} of "
                            f"{pres.n_combos} CSCV splits (median OOS Sharpe {np.median(pres.oos_perf) * ann:.2f}); "
                            f"PBO {pres.pbo:.2f} (diagnostic)."))

    # ---- OOS Sharpe (CPCV path median)
    op, thr = GATE_THRESHOLDS["oos_sharpe"]
    oos_med = float("nan")
    if study.cpcv_paths is not None and study.cpcv_paths.height:
        ps = (study.cpcv_paths.group_by("path_id").agg(
            (pl.col("ret").mean() / pl.col("ret").std(ddof=1) * ann).alias("sr")))["sr"].to_numpy()
        ps = ps[np.isfinite(ps)]
        oos_med = float(np.median(ps)) if ps.size else float("nan")
        q25, q75 = (np.quantile(ps, [0.25, 0.75]) if ps.size else (np.nan, np.nan))
        diag["cpcv_path_sharpe"] = {"median": oos_med, "p25": float(q25), "p75": float(q75),
                                    "min": float(ps.min()) if ps.size else float("nan"), "n_paths": int(ps.size)}
        if data_dep or emb_capped:
            rows.append(GateRow("oos_sharpe", oos_med, f"{op} {thr}", "SKIPPED",
                                f"Not gated: {skip_dd if data_dep else skip_emb} (median {oos_med:.2f} reported "
                                f"as a diagnostic only)."))
        else:
            ok = _cmp(op, oos_med, thr)
            rows.append(GateRow("oos_sharpe", oos_med, f"{op} {thr}", "PASS" if ok else "FAIL",
                                f"Median OOS Sharpe {oos_med:.2f} across {ps.size} CPCV paths (IQR {q25:.2f}–"
                                f"{q75:.2f}); {np.mean(ps > 0):.0%} of paths positive; in-sample {sel_sr_ann:.2f}."))
    else:
        rows.append(GateRow("oos_sharpe", None, f"{op} {thr}", "SKIPPED", "No CPCV paths in the study."))

    # ---- WFO procedure OOS (B2)
    op, thr = GATE_THRESHOLDS["wfo_oos"]
    op2, thr2 = GATE_THRESHOLDS["wfo_oos_recent"]
    thr_txt = f"{op} {thr} and recent ⅓ {op2} {thr2:g}"
    w = study.wfo_oos
    if w is None or w.height == 0 or "ret" not in w.columns:
        rows.append(GateRow("wfo_oos", None, thr_txt, "SKIPPED",
                            "No walk-forward OOS series in the study; the re-optimisation procedure is ungated."))
    else:
        w = w.sort("date")
        r = w["ret"].cast(pl.Float64).fill_null(0.0).to_numpy()
        n_rec = max(2, int(math.ceil(r.size * WFO_RECENT_FRACTION)))
        rec = r[-n_rec:]
        d_rec = w["date"][-n_rec:]
        sr_all, sr_rec = metrics.sharpe(r, ppy), metrics.sharpe(rec, ppy)
        diag["wfo_oos"] = {"sharpe_all": sr_all, "sharpe_recent": sr_rec, "n_days": int(r.size),
                           "recent_from": str(d_rec[0]), "recent_to": str(d_rec[-1])}
        txt = (f"WFO OOS Sharpe {sr_all:.2f} over {r.size} days ({w['date'][0]} → {w['date'][-1]}); "
               f"{sr_rec:.2f} over the most recent third ({d_rec[0]} → {d_rec[-1]})")
        disp = f"{sr_all:.2f} / recent {sr_rec:.2f}"
        if data_dep:
            rows.append(GateRow("wfo_oos", sr_all, thr_txt, "SKIPPED", f"Not gated: {skip_dd}. {txt}.", disp))
        else:
            ok = _cmp(op, sr_all, thr) and _cmp(op2, sr_rec, thr2)
            tail = ("." if ok else ("; the procedure has not worked recently." if _cmp(op, sr_all, thr)
                                    else "; the procedure does not hold up out of sample."))
            rows.append(GateRow("wfo_oos", sr_all, thr_txt, "PASS" if ok else "FAIL", txt + tail, disp))

    # ---- Cost stress
    op, thr = GATE_THRESHOLDS["cost_stress_sharpe"]
    bc = _resolve_base_cost(evaluator, base_cost)
    ispec, spec_src, synthetic = None, "", False
    if evaluator is not None and bc is not None:
        ispec, spec_src = _resolve_spec(evaluator, spec)
        synthetic = _is_synthetic(evaluator)
    if evaluator is None or bc is None:
        why = "no evaluator" if evaluator is None else "no base cost model (pass base_cost=)"
        rows.append(GateRow("cost_stress_sharpe", None, f"{op} {thr}", "SKIPPED",
                            f"Cost stress not run: {why}; the verdict cannot be PASS."))
    elif ispec is None and not synthetic:
        rows.append(GateRow("cost_stress_sharpe", None, f"{op} {thr}", "SKIPPED",
                            f"No instrument spec: cannot convert 1 pip to points ({spec_src}); pass spec=. "
                            f"The verdict cannot be PASS."))
    else:
        med_spread = float("nan")
        if ispec is not None:
            from .costs import pip_points
            pip_pts = float(pip_points(ispec))
            stressed = bc.stressed(spec=ispec)
            slip_pts = float(stressed.slippage_points - bc.slippage_points)
            sym = getattr(ispec, "symbol", "?")
            med_spread = _median_spread(sym, getattr(evaluator, "book", "FBS"))
            spread_txt = (f" = {slip_pts / med_spread:.2f}× the median dev spread of {med_spread:g} points"
                          if np.isfinite(med_spread) and med_spread > 0 else "; median spread n/a")
            pip_txt = f"slippage {slip_pts:g} points per fill for {sym}{spread_txt}"
        else:
            pip_pts, stressed = float("nan"), bc.stressed()
            slip_pts = float(stressed.slippage_points - bc.slippage_points)
            pip_txt = "synthetic evaluator: no instrument, pip conversion n/a"
        mults = sorted({1.0, *swap_band})
        srs = {}
        for mlt in mults:
            cm = replace(stressed, swap_multiplier=stressed.swap_multiplier * mlt)
            srs[mlt] = _outcome_sharpe(evaluator(dict(study.selected_params), cost=cm), ppy)
        worst = min(srs.values())
        ok = _cmp(op, worst, thr)
        rows.append(GateRow("cost_stress_sharpe", worst, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"Worst Sharpe {worst:.2f} at 1.5× spread + 1 stress unit of slippage (1 pip / tick / "
                            f"median spread by asset class) on all market and stop fills "
                            f"({pip_txt}), bar-extreme stops, swap ×{', '.join(f'{m:g}' for m in mults)} "
                            f"({', '.join(f'{v:.2f}' for v in srs.values())}); base {sel_sr_ann:.2f}."))
        diag["cost_stress_by_swap_mult"] = {f"{k:g}": v for k, v in srs.items()}
        diag["cost_stress_pip_points"] = {"pip_points": pip_pts, "spec_source": spec_src if ispec is not None
                                          else "synthetic", "slippage_points": stressed.slippage_points,
                                          "added_slippage_points": slip_pts, "median_spread_points": med_spread,
                                          "slippage_x_median_spread": (slip_pts / med_spread if np.isfinite(med_spread)
                                                                       and med_spread > 0 else float("nan"))}
        diag["cost_stress_version"] = getattr(stressed, "version", None)

    # ---- Plateau (B4, M5, N3, N4): measured here, radius from the ledger
    op, thr = GATE_THRESHOLDS["plateau"]
    popt = study.selection.get("plateau_score") if study.selection else None
    if popt is not None:
        diag["plateau_optimizer"] = {"score": popt, "neighbourhood": study.selection.get("neighbourhood")}
    radius = ctx["plateau_radius"]
    jspace, space_note = (None, "no evaluator") if evaluator is None else _judge_space(study, ctx, space)
    plateau_detail: dict[str, Any] = {}
    if evaluator is not None and jspace is not None:
        pj = judge_plateau(study, evaluator, space=jspace, radius=radius, periods_per_year=ppy, n_jobs=n_jobs)
        psc = pj["plateau_score"]
        plateau_detail = {**pj, "radius_source": ctx["radius_source"], "kind": "judge-run",
                          "points": [{k: q.get(k) for k in ("param", "offset", "value", "params", "sharpe", "status",
                                                            "outside_search_bounds", "pass")} for q in pj["points"]]}
        if space_note:
            plateau_detail["note"] = space_note
        diag["plateau"] = {"kind": "judge-run", "plateau_score": psc, "score_rule": pj["score_rule"],
                           "pooled_share": pj["pooled_share"], "weakest_axis": pj["weakest_axis"],
                           "n_points": pj["n_points"], "n_evaluations": pj["n_evaluations"],
                           "peak_sharpe": pj["peak_sharpe"], "peak_sharpe_study": pj["peak_sharpe_study"],
                           "radius": radius, "radius_source": ctx["radius_source"], "runtime_s": pj["runtime_s"],
                           "pass_share_by_param": pj["pass_share_by_param"],
                           "n_outside_search_bounds": pj["n_outside_search_bounds"],
                           "selected_at_search_space_edge": pj["selected_at_edge"]}
        unordered = [p.name for p in jspace.params if p.kind == "categorical" and not p.ordered]
        un_txt = (f"; unordered categorical(s) {', '.join(unordered)}: not judged — review at S2" if unordered else "")
        diag["plateau"]["unordered_not_judged"] = unordered
        if pj["n_axes"] == 0:
            rows.append(GateRow("plateau", None, f"{op} {thr}", "SKIPPED",
                                "No numeric or ordered parameter to perturb; plateau cannot be judged" + un_txt + "."))
        else:
            ok = _cmp(op, psc, thr)
            k_pass, k_n = pj["weakest_axis_passes"]
            ni, no = pj["n_invalid"], pj["n_outside_search_bounds"]
            edge = pj["selected_at_edge"]
            rows.append(GateRow("plateau", psc, f"{op} {thr}", "PASS" if ok else "FAIL",
                                f"Weakest parameter axis: {pj['weakest_axis']} keeps {k_pass}/{k_n} judge-run "
                                f"perturbations (±½, ±1 of each param's pre-registered plateau scale; r = {radius}, "
                                f"{ctx['radius_source']}; joint axis: {pj['n_joint_points']} points) at ≥ 50% of the peak "
                                f"Sharpe {pj['peak_sharpe']:.2f}; score = min over {pj['n_axes']} axes (pooled "
                                f"{pj['pooled_share']:.0%})"
                                + (f"; {no} point(s) outside the search bounds evaluated" if no else "")
                                + (f"; {ni} invalid counted as failed" if ni else "")
                                + (f"; selected at search-space edge ({', '.join(edge)})" if edge else "")
                                + (f"; {space_note}" if space_note else "") + un_txt
                                + f"; {pj['n_evaluations']} evaluations in {pj['runtime_s']:.1f} s; "
                                + ("broad optimum." if ok else "sharp, fragile optimum.")))
    else:
        why = "no evaluator" if evaluator is None else space_note
        pl_info = plateau_score(study, ppy, radius=radius, space=ctx.get("space") or {})
        psc = pl_info["plateau_score"]
        diag["plateau"] = {"kind": f"matrix-based fallback ({why})", "radius": radius,
                           "radius_source": ctx["radius_source"],
                           **{k: pl_info[k] for k in ("plateau_score", "n_neighbours", "peak_sharpe", "rules")}}
        plateau_detail = {"kind": f"matrix-based fallback ({why})", "radius": radius,
                          "radius_source": ctx["radius_source"], "plateau_score": psc,
                          "neighbour_trials": pl_info["neighbour_trials"], "n_evaluations": 0}
        if pl_info["n_neighbours"] == 0:
            rows.append(GateRow("plateau", None, f"{op} {thr}", "SKIPPED",
                                f"Matrix-based fallback ({why}): no trial inside the pre-registered neighbourhood "
                                f"({pl_info['method']}); plateau cannot be judged."))
        else:
            ok = _cmp(op, psc, thr)
            rows.append(GateRow("plateau", psc, f"{op} {thr}", "PASS" if ok else "FAIL",
                                f"Matrix-based fallback ({why}): {psc:.0%} of {pl_info['n_neighbours']} recorded "
                                f"neighbours keep ≥ 50% of the peak Sharpe {pl_info['peak_sharpe']:.2f} "
                                f"({pl_info['method']}); " + ("broad optimum." if ok else "sharp, fragile optimum.")))

    # ---- Time stability
    ts = st.time_stability(sel_daily)
    op, thr = GATE_THRESHOLDS["positive_years"]
    ok = _cmp(op, ts["pos_year_share"], thr)
    neg = ts["table"].filter(pl.col("pnl") <= 0)["year"].to_list()
    rows.append(GateRow("positive_years", ts["pos_year_share"], f"{op} {thr:.0%}", "PASS" if ok else "FAIL",
                        f"{ts['pos_year_share']:.0%} of {ts['n_years']} years positive"
                        + (f"; losing years {neg}." if neg else ".")))
    op, thr = GATE_THRESHOLDS["max_year_share"]
    ok = _cmp(op, ts["max_year_share"], thr)
    tbl = ts["table"]
    top = tbl.sort("pnl", descending=True)["year"][0] if tbl.height else None
    rows.append(GateRow("max_year_share", ts["max_year_share"], f"{op} {thr:.0%}", "PASS" if ok else "FAIL",
                        (f"Best year {top} carries {ts['max_year_share']:.0%} of total PnL"
                         if np.isfinite(ts["max_year_share"]) else "Total PnL ≤ 0: no share defined")
                        + ("." if ok else "; PnL is concentrated.")))

    # ---- Trade count vs MinTRL
    if selected_trades is None and evaluator is not None:
        try:
            selected_trades = evaluator(dict(study.selected_params), cost=bc).trades
        except Exception as e:  # noqa: BLE001 — report, do not crash the whole gate run
            diag["trades_error"] = repr(e)
    claimed = min(sel_sr_ann, oos_med) if np.isfinite(oos_med) else sel_sr_ann
    sk, ku = metrics.skew_kurt(sel_daily["ret"])
    mt_days = st.min_trl(claimed / ann, 0.0, np.nan_to_num(sk), 3.0 if not np.isfinite(ku) else ku,
                         MINTRL_PROB, ppy)
    n_days = sel_daily.height
    days_ok = n_days >= mt_days["periods"]
    tr = _per_trade_returns(selected_trades)
    if tr is not None and tr.size >= 3:
        tsk, tku = metrics.skew_kurt(tr)
        mt_tr = st.min_trl(st.sharpe_per_period(tr), 0.0, np.nan_to_num(tsk), 3.0 if not np.isfinite(tku) else tku,
                           MINTRL_PROB, 1.0)["periods"]
        tr_ok = tr.size >= mt_tr
        ok = days_ok and tr_ok
        disp = f"{tr.size} trades / need {mt_tr:.0f}"
        interp = (f"{tr.size} trades vs per-trade MinTRL {mt_tr:.0f}; {n_days} days vs daily MinTRL "
                  f"{mt_days['periods']:.0f} at claimed Sharpe {claimed:.2f} (95%).")
        val = float(tr.size)
    else:
        ok = False if not days_ok else None
        disp = f"{n_days} days / need {mt_days['periods']:.0f}"
        interp = (f"No per-trade data; {n_days} days vs daily MinTRL {mt_days['periods']:.0f} at claimed "
                  f"Sharpe {claimed:.2f}.")
        val = float(n_days)
    status = "PASS" if ok else ("FAIL" if ok is False else "SKIPPED")
    rows.append(GateRow("trade_count", val, "≥ MinTRL (95%)", status, interp, display=disp))
    diag["min_trl_days"] = mt_days["periods"]

    # ---- Mechanism
    if mechanism_check is None:
        rows.append(GateRow("mechanism", None, "ablation matches hypothesis", "MANUAL",
                            "Hypothesis-specific component ablation; to be judged against the card."))
    else:
        res = mechanism_check(study, evaluator)
        passed, txt = (res if isinstance(res, tuple) else (bool(res), "Component ablation result."))
        rows.append(GateRow("mechanism", float(bool(passed)), "ablation matches hypothesis",
                            "PASS" if passed else "FAIL", txt))

    # ---- Diagnostics + holdout band
    try:
        br = st.return_at_dd_budget(sel_daily, 0.10, n_boot=st.HOLDOUT_BAND_N_BOOT, periods_per_year=ppy,
                                    rng=np.random.default_rng(band_seed))
        diag["return_at_10pct_dd_budget"] = {"leverage": br.leverage, "mean_monthly": br.mean_monthly,
                                             "band_p5_p50_p95": br.mean_monthly_band, "label": br.label}
    except Exception as e:  # noqa: BLE001
        diag["return_at_budget_error"] = repr(e)
    band = None
    led_syms, led_legs = ledger.study_symbols(study.study_id, ledger_dir=ledger_dir)
    hsyms = holdout_symbols
    if led_syms:
        if hsyms is not None and sorted([hsyms] if isinstance(hsyms, str) else list(hsyms)) != sorted(led_syms):
            raise GateError(f"holdout_symbols={hsyms!r} differ from the study's symbols registered in the ledger "
                            f"{led_syms} (R2-2 / R2-3: symbols come from the ledger)")
        hsyms = led_syms
    horizon, hnote = resolve_holdout_horizon(ctx.get("book"), ppy, holdout_days=holdout_days,
                                             symbols=hsyms, evaluator=evaluator, meta=meta, legs=led_legs)
    if hnote:
        diag["holdout_horizon_note"] = hnote
    try:
        band = _build_holdout_band(study, horizon, ppy, n_boot=st.HOLDOUT_BAND_N_BOOT, seed=band_seed,
                                   selected_trades=selected_trades, sel_daily=sel_daily).as_dict()
    except ValueError as e:
        diag["holdout_band_error"] = str(e)
    band_run = band
    reg_band = ledger.registered_holdout_band(study.study_id, ledger_dir=ledger_dir)
    if reg_band is not None:
        band = reg_band
        diag["holdout_band_note"] = ("the band shown is the study's registered band (first gate run / rebuild); "
                                     "this run's band is a diagnostic only and never replaces it (R2-3)")
        if band_run:
            diag["holdout_band_this_run"] = {k: band_run.get(k) for k in ("sharpe_lo", "max_dd_mag_hi",
                                                                          "p_pass_zero_edge", "horizon_end")}

    prior_runs = []
    try:
        prior_runs = [{"gate_run": e.get("gate_run"), "verdict": e.get("verdict"), "at": e.get("at")}
                      for e in ledger.study_events(study.study_id, "gates", ledger_dir=ledger_dir)]
    except Exception as e:  # noqa: BLE001
        diag["prior_gate_runs_error"] = repr(e)

    statuses = [r.status for r in rows]
    verdict = "FAIL" if "FAIL" in statuses else ("PASS" if all(s == "PASS" for s in statuses) else "INCOMPLETE")
    lctx = {k: ctx.get(k) for k in ("book", "system", "issue", "attempt", "method", "seed", "plateau_radius",
                                    "radius_source", "data_dependent", "ledger_dir")}
    lctx["trial_store"] = store["studies_dir"]
    if ev_ident is not None:
        lctx["cost_model_version"] = ev_ident["cost_model_version"]
    report = GateReport(study.study_id, rows, verdict, band, eff, diag, ppy, prior_runs, plateau_detail, lctx,
                        band_run)
    if log:
        report.ledger_row = _log_gates(report, ledger_dir=ledger_dir)
    return report


def log_gates(*_: Any, **__: Any) -> None:
    """Removed (red-team R3-1): a report computed elsewhere (and possibly edited) must never reach
    the ledger.  Use ``evaluate_gates(..., log=True)``."""
    raise TypeError("gates.log_gates was removed (red-team R3-1): run evaluate_gates(..., log=True), which logs "
                    "the report it computed itself")


def _log_gates(report: GateReport, study_id: str | None = None, *, ledger_dir: Path | None = None) -> dict[str, Any]:
    """Append the gate results to the study's ledger row (event ``gates``, DESIGN §8).  Private:
    called only by :func:`evaluate_gates` (``log=True``) on the report it just computed (R3-1).

    Logs the study's **own** trial count (``n_trials_study`` — the N the DSR used for this study,
    i.e. never below the optimizer's logged count, R2-4) separately from the prior and the N used
    by DSR, so cumulative sums over attempts never double-count (m1), and a gate-run index (1, 2,
    …) so repeated gating of one study is visible (m9).

    Holdout band (R2-3): the band of the study's FIRST gates run that produces one is its
    registered band (``holdout_band``; checked with :func:`ledger.check_band_construction`).
    Every later run logs its band as ``holdout_band_diagnostic`` only — it never replaces the
    registered band (a later band must go through :func:`rebuild_holdout_band`)."""
    sid = study_id or report.study_id
    n_prev = len(ledger.study_events(sid, "gates", ledger_dir=ledger_dir))
    band_kw: dict[str, Any] = {}
    run_band = report.holdout_band_run if report.holdout_band_run is not None else report.holdout_band
    if run_band:
        if ledger.registered_holdout_band(sid, ledger_dir=ledger_dir) is None:
            try:
                ledger.check_band_construction(sid, run_band, ledger_dir=ledger_dir)
            except ledger.LedgerError as e:
                raise GateError(str(e)) from e
            band_kw["holdout_band"] = run_band
        else:
            band_kw["holdout_band_diagnostic"] = run_band
            band_kw["holdout_band_frozen"] = True
    et = report.effective_trials
    pdl = report.plateau_detail or {}
    plateau = {k: pdl.get(k) for k in ("kind", "radius", "radius_source", "plateau_score", "score_rule",
                                       "pooled_share", "weakest_axis", "pass_share_by_param", "selected_at_edge",
                                       "n_evaluations", "peak_sharpe", "peak_sharpe_study", "runtime_s", "note")
               if k in pdl}
    if pdl.get("points") is not None:
        plateau["points"] = [{k: q.get(k) for k in ("param", "offset", "params", "sharpe", "status",
                                                    "outside_search_bounds", "pass")} for q in pdl["points"]]
    if pdl.get("neighbour_trials") is not None:
        plateau["neighbour_trials"] = pdl["neighbour_trials"]
    return ledger.log_event(sid, "gates", ledger_dir=ledger_dir,
                            gates=report.values(), gate_status=report.statuses(), verdict=report.verdict,
                            gate_run=n_prev + 1, gate_version="4.2-v1.2",
                            n_trials_study=et.get("n_trials_study"), n_trials_prior=et.get("n_trials_prior"),
                            n_trials_prior_ledger=et.get("prior_trials_ledger"),
                            prior_studies=et.get("prior_studies"), n_trials_dsr=et.get("n_trials"),
                            effective_trials_study=float(et.get("n_eff", float("nan"))),
                            n_trials_ledger=et.get("n_trials_ledger"), n_trials_store=et.get("n_trials_store"),
                            plateau=plateau, ledger_context=report.ledger_context, **band_kw)


# --------------------------------------------------------------------------- holdout horizon / exam (§4.4)
def resolve_holdout_horizon(book: str | None, periods_per_year: float, *, holdout_days: int | None = None,
                            symbols: str | list[str] | None = None, evaluator: Any = None,
                            meta: Mapping[str, Any] | None = None,
                            legs: tuple[str, ...] | list[str] = ()) -> tuple[int | dict[str, Any], str | None]:
    """Horizon for the holdout band (DESIGN §4.4, 2026-09-24) and a note (None when clean).

    ``holdout_days`` given → that int (explicit override, tests).  Else symbols =
    ``symbols`` or ``evaluator.symbol`` (real evaluators only) or ``meta['symbols'|'symbol']``,
    and the horizon is :func:`quantlab.data.holdout_horizon` (manifest metadata only: holdout
    start → end of the available data over the symbols, their conversion legs and ``legs``).  Without a symbol or book: one year of periods, labelled
    ``horizon_source="default"``; the ledger refuses such a band at unlock."""
    if holdout_days is not None:
        return int(holdout_days), None
    meta = meta or {}
    syms = symbols
    if syms is None and evaluator is not None and not _is_synthetic(evaluator):
        syms = getattr(evaluator, "symbol", None)
    if syms is None:
        syms = meta.get("symbols") or meta.get("symbol")
    default = {"horizon_days": int(round(periods_per_year)), "horizon_source": "default"}
    if not syms or not book:
        return default, ("no symbol resolved for the holdout horizon: one year of periods used; rebuild the "
                         "band from the manifest (gates.rebuild_holdout_band) before any unlock")
    try:
        return data.holdout_horizon(book, syms, periods_per_year=periods_per_year, legs=legs), None
    except (ValueError, KeyError) as e:
        return default, f"holdout horizon from the manifest failed ({e}); one year of periods used"


def _build_holdout_band(study: StudyResult, horizon: int | Mapping[str, Any], ppy: float, *, n_boot: int,
                        seed: int, selected_trades: pl.DataFrame | None = None,
                        sel_daily: pl.DataFrame | None = None) -> st.HoldoutBand:
    """The band construction shared by :func:`evaluate_gates` and :func:`rebuild_holdout_band`."""
    tpd, tsrc = wfo_trades_per_day(study)
    has_wfo = study.wfo_oos is not None and study.wfo_oos.height > 0
    if tpd is None and has_wfo:
        # F4: the band is built on the WFO procedure; the dev-selected trial's trade rate is the
        # wrong series, so the trade criterion is left "not available".
        tsrc = f"not available: {tsrc} (WFO n_trades null/absent and no trade_counts to rebuild)"
    elif tpd is None:
        if sel_daily is None:
            sel_daily = study.returns.select("date", pl.col(f"t{selected_trial_id(study)}").fill_null(0.0).alias("ret"))
        tpd = _trades_per_day(selected_trades, sel_daily["date"])
        tsrc = "dev-selected trial's trades (no WFO series)" if tpd is not None else "none"
    return st.holdout_band(study, horizon, ppy, n_boot=n_boot, trades_per_day=tpd, seed=seed, trades_source=tsrc,
                           target_coverage=HOLDOUT_TARGET_COVERAGE, n_power=st.HOLDOUT_BAND_N_POWER)


def rebuild_holdout_band(study: StudyResult, *, periods_per_year: float, reason: str,
                         selected_trades: pl.DataFrame | None = None, ledger_dir: Path | None = None,
                         studies_dir: Path | None = None, symbols: Any = _REMOVED, n_boot: Any = _REMOVED,
                         seed: Any = _REMOVED) -> dict[str, Any]:
    """Rebuild the pre-registered holdout band for the horizon available NOW and re-register it
    (DESIGN §4.4).  Call it when data newer than the registered band's ``horizon_end`` has been
    exported, **before** ``/unlock-holdout`` — and, after a NOT_DECISIVE exam, before the re-exam
    unlock (never after reading the new data).

    Only the horizon changes (R2-3): the same construction, the fixed seed
    (``stats.holdout_band_seed(study_id)``) and ``n_boot`` (``stats.HOLDOUT_BAND_N_BOOT``), the
    study's symbols and conversion legs **from the ledger** (:func:`ledger.study_symbols`).
    ``symbols=`` / ``seed=`` / ``n_boot=`` overrides raise ``TypeError``.  The study is checked
    against its trial store first (:func:`verify_trial_store`).  The horizon comes from the
    manifest (:func:`quantlab.data.holdout_horizon`, metadata only) and must end strictly later
    than the registered band's.  The event is logged as ``holdout_band_registered`` with
    ``reason``, ``band_version`` and the horizon it replaces; the ledger refuses a rebuild after
    an unlock whose exam is pending, after a FAIL or PASS, or without newer data
    (``ledger._register_holdout_band``, reachable only from here — R3-1).  Returns the new band."""
    for name, v in (("symbols", symbols), ("n_boot", n_boot), ("seed", seed)):
        if v is not _REMOVED:
            raise TypeError(f"rebuild_holdout_band({name}=) is refused (red-team R2-3): the rebuild uses the same "
                            f"construction, the fixed seed / n_boot and the study's symbols from the ledger")
    ctx = ledger_context(study, ledger_dir)
    verify_trial_store(study, studies_dir=studies_dir, ledger_dir=ledger_dir)
    cur = ledger.registered_holdout_band(study.study_id, ledger_dir=ledger_dir) or {}
    syms, legs = ledger.study_symbols(study.study_id, ledger_dir=ledger_dir)
    if not syms:
        raise GateError(f"study {study.study_id}: no symbols registered in the ledger (study_created row or a "
                        f"registered band); the holdout horizon cannot be rebuilt")
    horizon = data.holdout_horizon(ctx["book"], syms, periods_per_year=periods_per_year, legs=legs)
    from datetime import datetime as _dt
    if cur.get("horizon_end") and _dt.fromisoformat(horizon["horizon_end"]) <= _dt.fromisoformat(str(cur["horizon_end"])):
        raise GateError(f"no newer data: the manifest ends {horizon['horizon_end']}, the registered band already "
                        f"covers up to {cur['horizon_end']}")
    band = _build_holdout_band(study, horizon, float(periods_per_year), n_boot=st.HOLDOUT_BAND_N_BOOT,
                               seed=st.holdout_band_seed(study.study_id),
                               selected_trades=selected_trades).as_dict()
    ledger._register_holdout_band(study_id=study.study_id, band=band, reason=reason, ledger_dir=ledger_dir)
    return band


def _same_band(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths where two band dicts differ (floats: rel 1e-9; NaN == NaN; ``manifest_sha`` ignored)."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k == "manifest_sha":
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k}")
            else:
                out += _same_band(a[k], b[k], f"{path}{k}.")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [path.rstrip(".")]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _same_band(x, y, f"{path}{i}.")]
    if _is_num(a) and _is_num(b):
        fa, fb = float(a), float(b)
        if (math.isnan(fa) and math.isnan(fb)) or math.isclose(fa, fb, rel_tol=1e-9, abs_tol=1e-12):
            return []
        return [path.rstrip(".")]
    return [] if a == b else [path.rstrip(".")]


def unlock_holdout(study: StudyResult, evaluator: Evaluator | None, *, periods_per_year: float,
                   user_confirmation: str, ledger_dir: Path | None = None, studies_dir: Path | None = None,
                   selected_trades: pl.DataFrame | None = None, mechanism_check: Callable[..., Any] | None = None,
                   base_cost: Any = None, spec: Any = None, space: Any = None, n_jobs: Any = 1) -> dict[str, Any]:
    """Unlock the holdout exam of a study (DESIGN §4.4; called by ``/unlock-holdout``) — R3-1.

    Nothing logged earlier is trusted.  The study is re-verified against its trial store and the
    ledger (:func:`verify_trial_store`, :func:`check_evaluator_identity`) and every gate is
    recomputed by :func:`evaluate_gates` (not logged) with the study's evaluator — so cost stress
    and plateau are re-run.  The unlock is refused unless:

    * every statistical gate is PASS and the mechanism gate is PASS — either from
      ``mechanism_check`` or, when it is MANUAL, from a ``mechanism_review`` event with
      ``passed=True`` (:func:`quantlab.ledger.record_mechanism_review`, the human S5 decision);
    * the band recomputed from the verified study for the current horizon equals the study's
      registered band (:func:`ledger.registered_holdout_band`) — a band edited after it was
      logged, or registered for another horizon, is refused (rebuild it first when newer data
      exists: :func:`rebuild_holdout_band`).

    Then ``ledger._record_holdout_unlock`` applies the family / exam-sequence / manifest / clean
    tree rules and writes the unlock row, with the recomputed verdict as ``verification``."""
    ctx = ledger_context(study, ledger_dir)
    reg = ledger.registered_holdout_band(study.study_id, ledger_dir=ledger_dir)
    if reg is None:
        raise GateError(f"study {study.study_id}: no registered holdout band (run evaluate_gates(..., log=True) at S5)")
    rep = evaluate_gates(study, evaluator, periods_per_year=periods_per_year, selected_trades=selected_trades,
                         mechanism_check=mechanism_check, base_cost=base_cost, spec=spec, space=space, n_jobs=n_jobs,
                         ledger_dir=ledger_dir, studies_dir=studies_dir, log=False)
    st = rep.statuses()
    stat_bad = {g: s for g, s in st.items() if g != "mechanism" and s != "PASS"}
    if stat_bad:
        raise GateError(f"study {study.study_id}: the recomputed gates are not all PASS ({stat_bad}); the holdout "
                        f"cannot be unlocked (R3-1)")
    mech = st.get("mechanism")
    mech_src = "mechanism_check"
    if mech != "PASS":
        reviews = ledger.study_events(study.study_id, "mechanism_review", ledger_dir=ledger_dir)
        if mech == "MANUAL" and reviews and reviews[-1].get("passed") is True:
            mech_src = f"mechanism_review ({reviews[-1].get('at')})"
        else:
            raise GateError(f"study {study.study_id}: mechanism gate is {mech} and no passing mechanism review is "
                            f"recorded (ledger.record_mechanism_review); the verdict is not PASS (R3-1)")
    run_band = rep.holdout_band_run
    if not run_band:
        raise GateError(f"study {study.study_id}: the holdout band could not be recomputed "
                        f"({rep.diagnostics.get('holdout_band_error')})")
    diff = _same_band(run_band, reg)
    if diff:
        hint = (" — the registered band is for another horizon; rebuild it (rebuild_holdout_band) if newer data was "
                "exported" if "horizon_end" in diff or "horizon_days" in diff else "")
        raise GateError(f"study {study.study_id}: the registered band differs from the band recomputed from the "
                        f"verified study in {diff[:8]}{hint} (red-team R3-1: a doctored band is never unlocked)")
    verification = {"verdict_recomputed": "PASS", "mechanism": mech_src, "gate_values": rep.values(),
                    "band_recomputed_equal": True, "gate_version": "4.2-v1.2"}
    return ledger._record_holdout_unlock(book=ctx["book"], system=ctx["system"], study_id=study.study_id,
                                         pass_band=reg, user_confirmation=user_confirmation, ledger_dir=ledger_dir,
                                         verification=verification)


def run_holdout_exam(*, book: str, system: str, study_id: str, holdout_daily, n_trades: float,
                     periods_per_year: float, ledger_dir: Path | None = None) -> dict[str, Any]:
    """Judge the pending holdout exam and record it (DESIGN §4.4).  Only after ``/unlock-holdout``.

    Uses the band stored with the unlock row (never a band passed in), applies
    :func:`quantlab.stats.holdout_check` to the realised holdout (the FULL span holdout start →
    the unlock's ``horizon_end``) and appends the ``holdout_exam`` event.  Returns the check
    result (``status`` ∈ PASS / FAIL / NOT_DECISIVE, criteria, values, horizon)."""
    stt = ledger.holdout_state(book, system, ledger_dir, study_id=study_id)
    if not stt["pending"]:
        raise GateError(f"{book}/{system}: no unlocked exam is pending")
    band = stt["last_unlock"]["pass_band"]
    res = st.holdout_check(band, holdout_daily, n_trades, periods_per_year)
    ledger.record_holdout_exam(book=book, system=system, study_id=study_id, result=res, ledger_dir=ledger_dir)
    return res
