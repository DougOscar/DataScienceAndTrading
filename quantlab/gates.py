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
  √V0 · E[max of N], V0 = 1/(T−1), **N = raw trials of this study + raw trials of every other
  study created before it in the ledger that shares its normalised system name OR its issue
  number** (any attempt, any book — red-team N2).  An explicit ``prior_trials`` may only raise
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
  re-evaluated perturbed along each numeric axis at ×(1 ± r/2) and ×(1 ± r) (± r/2, ± r of the
  declared range for params that are not strictly positive), r = the radius pre-registered in
  the study's ledger row (default 0.20, N3).  Score = share of perturbations whose full-dev
  Sharpe is ≥ 50 % of the re-evaluated peak and > 0; invalid / out-of-bounds / erroring
  points count as failures.  These ≤ 4·d evaluations are judge diagnostics (logged with the
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

The ledger is the source of truth (N2)
--------------------------------------
Book, system, issue, attempt, method, seed, search space, plateau radius and the
data-dependence flag are read from the study's ``study_created`` ledger row (and its later
events).  A study that is not in the ledger raises :class:`GateError`; so does a
``study.meta`` value that disagrees with the ledger (meta is only a cache).
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

from . import config, ledger, metrics
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
WFO_RECENT_FRACTION = 1.0 / 3.0
MINTRL_PROB = 0.95
HOLDOUT_TARGET_COVERAGE = 0.90
HOLDOUT_MAX_ZERO_EDGE_PASS = 0.30
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
                  "| Param | Offset | Value | Sharpe | Status | Keeps ≥ 50% of peak |", "|---|---|---|---|---|---|"]
            for q in pdl["points"]:
                sr = q.get("sharpe")
                L.append(f"| {q['param']} | {q['offset']} | {q['value']} | "
                         f"{'—' if sr is None or not np.isfinite(sr) else f'{sr:.2f}'} | {q['status']} | "
                         f"{'yes' if q.get('pass') else 'no'} |")
            L.append("")
        if self.holdout_band:
            hb = self.holdout_band
            a = hb.get("tail_level", nan)
            L += ["## Pre-registered holdout pass band (DESIGN §4.4 v1.2)", "",
                  f"Source: {hb.get('source')} + stationary bootstrap (mean block {hb.get('mean_block', 0):.1f} d), "
                  f"horizon {hb.get('horizon_days')} days, {hb.get('n_boot')} samples, seed {hb.get('seed')}. "
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
                L.append(f"**Not decisive on its own:** P(pass | zero edge) > "
                         f"{hb.get('max_zero_edge_pass', HOLDOUT_MAX_ZERO_EDGE_PASS):.0%}; a holdout pass is weak "
                         f"evidence (holdout length / renewing-holdout policy: DESIGN §11 #8).")
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
        if positive and xf > 0:
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


def plateau_perturbations(space: Any, selected: Mapping[str, Any],
                          radius: float | Mapping[str, float] | None) -> list[dict[str, Any]]:
    """The pre-registered perturbation set around ``selected`` (N4), one axis at a time.

    * numeric, strictly positive (declared ``low`` > 0): x·(1 − r), x·(1 − r/2), x·(1 + r/2),
      x·(1 + r);
    * other numeric: x ∓ r·(high − low), x ∓ r/2·(high − low);
    * ints are rounded (half up); a point that rounds back to x moves to the next int in its
      direction, and the outer point is kept distinct from the inner one (so each side has
      two distinct ints);
    * ordered categoricals: the adjacent levels that exist; unordered categoricals: held fixed;
    * a point outside the declared [low, high] is kept with ``in_bounds = False`` (the caller
      counts it as a failed neighbour — conservative).
    Returns dicts ``param, offset, value, in_bounds, params``."""
    pts: list[dict[str, Any]] = []
    for p in space.params:
        if p.name not in selected:
            continue
        x = selected[p.name]
        if p.kind == "categorical":
            if not p.ordered or x not in p.choices:
                continue
            i = list(p.choices).index(x)
            for j, lab in ((i - 1, "level -1"), (i + 1, "level +1")):
                if 0 <= j < len(p.choices):
                    pts.append({"param": p.name, "offset": lab, "value": p.choices[j], "in_bounds": True})
            continue
        if x is None:
            continue
        r = _radius_for(radius, p.name)
        xf, lo, hi = float(x), float(p.low), float(p.high)
        if lo > 0:
            raw = [(f"x{1 - r:g}", xf * (1 - r)), (f"x{1 - r / 2:g}", xf * (1 - r / 2)),
                   (f"x{1 + r / 2:g}", xf * (1 + r / 2)), (f"x{1 + r:g}", xf * (1 + r))]
        else:
            span = hi - lo
            raw = [(f"-{r:g}*range", xf - r * span), (f"-{r / 2:g}*range", xf - r / 2 * span),
                   (f"+{r / 2:g}*range", xf + r / 2 * span), (f"+{r:g}*range", xf + r * span)]
        if p.kind == "int":
            xi = _round_half_up(xf)
            dn_in = _round_half_up(raw[1][1])
            dn_in = xi - 1 if dn_in >= xi else dn_in
            dn_out = _round_half_up(raw[0][1])
            dn_out = dn_in - 1 if dn_out >= dn_in else dn_out
            up_in = _round_half_up(raw[2][1])
            up_in = xi + 1 if up_in <= xi else up_in
            up_out = _round_half_up(raw[3][1])
            up_out = up_in + 1 if up_out <= up_in else up_out
            vals = [(raw[0][0], dn_out), (raw[1][0], dn_in), (raw[2][0], up_in), (raw[3][0], up_out)]
        else:
            vals = [(lab, round(v, 12)) for lab, v in raw]
        tol = 1e-9 * max(1.0, abs(lo), abs(hi))
        for lab, v in vals:
            pts.append({"param": p.name, "offset": lab, "value": v, "in_bounds": lo - tol <= v <= hi + tol})
    for q in pts:
        q["params"] = {**dict(selected), q["param"]: q["value"]}
    return pts


def judge_plateau(study: StudyResult, evaluator: Any, *, space: Any, radius: float | Mapping[str, float] | None,
                  periods_per_year: float = 260.0, n_jobs: Any = 1) -> dict[str, Any]:
    """Judge-run plateau (DESIGN §4.2 v1.2; red-team N4).  Evaluates the selected configuration
    and each :func:`plateau_perturbations` point that is inside the declared bounds and valid
    (``space.is_valid``) with ``evaluator(params, cost=None)`` (the evaluator's base cost, as in
    ``run_study``), on ``opt``'s runner (in-process for ``n_jobs=1``, else its capped process
    pool).  Score = share of ALL perturbation points whose full-dev Sharpe is ≥
    ``PLATEAU_PEAK_FRACTION`` × peak and > 0, peak = the re-evaluated selected configuration
    (0 when peak ≤ 0); out-of-bounds, invalid and erroring points count as failures.  NaN (and
    the gate SKIPPED) only when there is no perturbable parameter."""
    from . import opt
    t0 = time.perf_counter()
    ppy = float(periods_per_year)
    sel = dict(study.selected_params)
    pts = plateau_perturbations(space, sel, radius)
    for q in pts:
        if not q["in_bounds"]:
            q["status"] = "out_of_bounds"
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
    score = float(np.mean([q["pass"] for q in pts])) if pts else float("nan")
    by_param: dict[str, list] = {}
    for q in pts:
        by_param.setdefault(q["param"], []).append(q["pass"])
    return {"plateau_score": score, "n_points": len(pts), "n_neighbours": len(pts),
            "n_evaluations": len(todo), "n_failed_unevaluated": sum(q["status"] in ("out_of_bounds", "invalid")
                                                                    for q in pts),
            "peak_sharpe": float(peak) if peak is not None else float("nan"), "peak_source": peak_src,
            "peak_sharpe_study": float(peak_study), "points": pts, "radius": radius,
            "pass_share_by_param": {k: float(np.mean(v)) for k, v in by_param.items()},
            "runtime_s": time.perf_counter() - t0, "n_jobs": jobs,
            "method": (f"judge-run perturbations ×(1 ± r/2), ×(1 ± r) per numeric axis (± r·range when not strictly "
                       f"positive), r = {radius}; {len(pts)} points, {len(todo)} evaluations")}


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
    """Raw trials of related earlier studies (M4 / N2) — :func:`ledger.related_prior_trials`:
    every other study created before this one that shares its normalised system name or its
    issue number, whatever the attempt.  An explicit ``prior_trials`` below that count raises
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
    return space_from_json(js), note


# --------------------------------------------------------------------------- evaluation
def evaluate_gates(study: StudyResult, evaluator: Evaluator | None, *, periods_per_year: float,
                   selected_trades: pl.DataFrame | None = None,
                   mechanism_check: Callable[..., Any] | None = None, base_cost: Any = None,
                   swap_band: tuple[float, ...] = (0.5, 1.5), prior_trials: float | None = None,
                   ledger_dir: Path | None = None, holdout_days: int | None = None, pbo_splits: int = 16,
                   n_boot: int = 2000, seed: int = 12345, spec: Any = None, space: Any = None,
                   n_jobs: Any = 1, plateau_radius: Any = _REMOVED,
                   prior_effective_trials: Any = _REMOVED) -> GateReport:
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
    ``holdout_days``: holdout horizon for the pass band (default: one year of periods).
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
    data_dep = bool(ctx["data_dependent"])
    if study.trials is not None and "source" in study.trials.columns:
        data_dep = data_dep or bool((study.trials["source"] == "tpe").any())
    cpcv_meta = meta.get("cpcv") if isinstance(meta.get("cpcv"), Mapping) else {}
    emb_capped = bool(meta.get("embargo_capped") or cpcv_meta.get("embargo_capped"))
    skip_dd = ("candidate set is data-dependent (ledger: method=" + str(ctx.get("method")) + "): the configurations "
               "were chosen using the full dev sample, so this OOS series is not out of sample; re-run with a "
               "data-independent candidate set (grid / Sobol)")
    skip_emb = "embargo cap binds; use fewer CPCV groups"

    # ---- DSR (R1): V0 = 1/(T−1), N = raw trials (study + earlier attempts)
    n_study = _study_trial_count(study)
    n_prior, pinfo = resolve_prior_trials(study, prior_trials, ledger_dir, ctx=ctx)
    dres = st.dsr_from_matrix(study.returns, col, periods_per_year=ppy, n_trials=n_study, extra_trials=n_prior)
    op, thr = GATE_THRESHOLDS["dsr"]
    ok = _cmp(op, dres.dsr, thr)
    rows.append(GateRow("dsr", dres.dsr, f"{op} {thr}", "PASS" if ok else "FAIL",
                        f"DSR {dres.dsr:.2f}: {dres.dsr:.0%} probability the true Sharpe exceeds the "
                        f"{dres.sr0_annual:.2f} expected from the best of N = {dres.n_trials:g} raw null trials "
                        f"({n_study:g} this study + {n_prior:g} from related earlier studies) over {dres.n_obs} days."))
    eff = {**dres.as_dict(), **dres.n_eff_by_method, **pinfo}
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
        if ispec is not None:
            from .costs import pip_points
            pip_pts = float(pip_points(ispec))
            stressed = bc.stressed(spec=ispec)
            pip_txt = f"1 pip = {pip_pts:g} points for {getattr(ispec, 'symbol', '?')}"
        else:
            pip_pts, stressed = float("nan"), bc.stressed()
            pip_txt = "synthetic evaluator: no instrument, pip conversion n/a"
        mults = sorted({1.0, *swap_band})
        srs = {}
        for mlt in mults:
            cm = replace(stressed, swap_multiplier=stressed.swap_multiplier * mlt)
            srs[mlt] = _outcome_sharpe(evaluator(dict(study.selected_params), cost=cm), ppy)
        worst = min(srs.values())
        ok = _cmp(op, worst, thr)
        rows.append(GateRow("cost_stress_sharpe", worst, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"Worst Sharpe {worst:.2f} at 1.5× spread + 1 pip slippage on all market and stop fills "
                            f"({pip_txt}), bar-extreme stops, swap ×{', '.join(f'{m:g}' for m in mults)} "
                            f"({', '.join(f'{v:.2f}' for v in srs.values())}); base {sel_sr_ann:.2f}."))
        diag["cost_stress_by_swap_mult"] = {f"{k:g}": v for k, v in srs.items()}
        diag["cost_stress_pip_points"] = {"pip_points": pip_pts, "spec_source": spec_src if ispec is not None
                                          else "synthetic", "slippage_points": stressed.slippage_points}
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
                          "points": [{k: q.get(k) for k in ("param", "offset", "value", "params", "sharpe",
                                                            "status", "pass")} for q in pj["points"]]}
        if space_note:
            plateau_detail["note"] = space_note
        diag["plateau"] = {"kind": "judge-run", "plateau_score": psc, "n_points": pj["n_points"],
                           "n_evaluations": pj["n_evaluations"], "peak_sharpe": pj["peak_sharpe"],
                           "peak_sharpe_study": pj["peak_sharpe_study"], "radius": radius,
                           "radius_source": ctx["radius_source"], "runtime_s": pj["runtime_s"],
                           "pass_share_by_param": pj["pass_share_by_param"]}
        if pj["n_points"] == 0:
            rows.append(GateRow("plateau", None, f"{op} {thr}", "SKIPPED",
                                "No numeric or ordered parameter to perturb; plateau cannot be judged."))
        else:
            ok = _cmp(op, psc, thr)
            nf = pj["n_failed_unevaluated"]
            rows.append(GateRow("plateau", psc, f"{op} {thr}", "PASS" if ok else "FAIL",
                                f"{psc:.0%} of {pj['n_points']} judge-run perturbations (±r/2, ±r per axis, r = "
                                f"{radius}, {ctx['radius_source']}) keep ≥ 50% of the peak Sharpe "
                                f"{pj['peak_sharpe']:.2f}"
                                + (f"; {nf} out of bounds / invalid counted as failed" if nf else "")
                                + (f"; {space_note}" if space_note else "")
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
        br = st.return_at_dd_budget(sel_daily, 0.10, n_boot=n_boot, periods_per_year=ppy,
                                    rng=np.random.default_rng(seed))
        diag["return_at_10pct_dd_budget"] = {"leverage": br.leverage, "mean_monthly": br.mean_monthly,
                                             "band_p5_p50_p95": br.mean_monthly_band, "label": br.label}
    except Exception as e:  # noqa: BLE001
        diag["return_at_budget_error"] = repr(e)
    band = None
    try:
        tpd, tsrc = wfo_trades_per_day(study)
        has_wfo = study.wfo_oos is not None and study.wfo_oos.height > 0
        if tpd is None and has_wfo:
            # F4: the band is built on the WFO procedure; the dev-selected trial's trade rate is the
            # wrong series, so the trade criterion is left "not available".
            tsrc = f"not available: {tsrc} (WFO n_trades null/absent and no trade_counts to rebuild)"
        elif tpd is None:
            tpd = _trades_per_day(selected_trades, sel_daily["date"])
            tsrc = "dev-selected trial's trades (no WFO series)" if tpd is not None else "none"
        hb = st.holdout_band(study, int(holdout_days or round(ppy)), ppy, n_boot=n_boot, trades_per_day=tpd,
                             seed=seed, trades_source=tsrc, target_coverage=HOLDOUT_TARGET_COVERAGE,
                             max_zero_edge_pass=HOLDOUT_MAX_ZERO_EDGE_PASS)
        band = hb.as_dict()
    except ValueError as e:
        diag["holdout_band_error"] = str(e)

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
    return GateReport(study.study_id, rows, verdict, band, eff, diag, ppy, prior_runs, plateau_detail, lctx)


def log_gates(report: GateReport, study_id: str | None = None, *, ledger_dir: Path | None = None) -> dict[str, Any]:
    """Append the gate results to the study's ledger row (event ``gates``, DESIGN §8).

    Logs the study's **own** trial count (``n_trials_study``) separately from the prior and the
    N used by DSR, so cumulative sums over attempts never double-count (m1), and a gate-run
    index (1, 2, …) so repeated gating of one study is visible (m9)."""
    sid = study_id or report.study_id
    n_prev = len(ledger.study_events(sid, "gates", ledger_dir=ledger_dir))
    et = report.effective_trials
    pdl = report.plateau_detail or {}
    plateau = {k: pdl.get(k) for k in ("kind", "radius", "radius_source", "plateau_score", "n_evaluations",
                                       "peak_sharpe", "peak_sharpe_study", "runtime_s", "note") if k in pdl}
    if pdl.get("points") is not None:
        plateau["points"] = [{k: q.get(k) for k in ("param", "offset", "params", "sharpe", "status", "pass")}
                             for q in pdl["points"]]
    if pdl.get("neighbour_trials") is not None:
        plateau["neighbour_trials"] = pdl["neighbour_trials"]
    return ledger.log_event(sid, "gates", ledger_dir=ledger_dir,
                            gates=report.values(), gate_status=report.statuses(), verdict=report.verdict,
                            gate_run=n_prev + 1, gate_version="4.2-v1.2",
                            n_trials_study=et.get("n_trials_study"), n_trials_prior=et.get("n_trials_prior"),
                            n_trials_prior_ledger=et.get("prior_trials_ledger"),
                            prior_studies=et.get("prior_studies"), n_trials_dsr=et.get("n_trials"),
                            effective_trials_study=float(et.get("n_eff", float("nan"))),
                            plateau=plateau, ledger_context=report.ledger_context,
                            holdout_band=report.holdout_band)
