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
  √V0 · E[max of N], V0 = 1/(T−1), **N = raw trials of this study + raw trials of earlier
  attempts** of the same system (read from the ledger, or ``prior_trials=``).  Eigen / cluster
  / Li–Ji effective N and the cross-sectional Sharpe variance are diagnostics (R1 / B1).
* ``cscv_oos_loss`` — P(OOS Sharpe of the IS-best < 0) over the CSCV splits (16 blocks) of the
  full trial matrix; < 0.10.  PBO, slope and R² are diagnostics (R2 / M1).
* ``oos_sharpe`` — median annualised Sharpe across ``study.cpcv_paths``.
* ``wfo_oos`` — the walk-forward procedure's OOS series (``study.wfo_oos``): whole-span
  Sharpe ≥ 0.5 **and** Sharpe over its most recent third > 0 (B2).
* ``cost_stress_sharpe`` — the evaluator re-run on the selected params with
  ``base_cost.stressed()`` (1.5× spread + 1 pip slippage on all market and stop fills,
  bar-extreme stops) at swap multipliers {1.0} ∪ ``swap_band``; value = the minimum Sharpe.
* ``plateau`` — **always recomputed here** (never the optimizer's number, B4) on an economic
  neighbourhood: numeric params within ±20 % (relative for strictly positive params, else of
  the declared range), categoricals equal, adjacent levels when no other level lies inside
  the radius (M5).  See :func:`plateau_score`.
* ``positive_years`` / ``max_year_share`` — calendar years of the selected trial's returns.
* ``trade_count`` — (a) trades ≥ per-trade MinTRL (95 %) and (b) dev days ≥ daily MinTRL at the
  claimed Sharpe = min(selected full-dev Sharpe, CPCV-median OOS Sharpe).
* ``mechanism`` — component ablation (callable) or MANUAL.

SKIPs that block PASS: ``oos_sharpe``, ``wfo_oos`` and ``cscv_oos_loss`` when the candidate set
is data-dependent (``meta["candidate_set_data_dependent"]``, or ``method == "tpe"`` when the key
is absent — B3); ``oos_sharpe`` and ``cscv_oos_loss`` when the CPCV embargo cap binds
(``meta["embargo_capped"]`` — m3).
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import polars as pl

from . import ledger, metrics
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
PLATEAU_RADIUS = 0.20                       # ±20 % economic neighbourhood (card may pre-register)
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
              f"study + {et.get('n_trials_prior', nan):g} from earlier attempts ({et.get('prior_source', 'n/a')})",
              f"- Hurdle SR0 = √(1/(T−1)) · E[max of N] = {et.get('sr0_annual', nan):.2f} annualised "
              f"(T = {et.get('n_obs')} days)",
              f"- Selected Sharpe {et.get('sr_annual', nan):.2f} annualised; skew {et.get('skew', nan):.2f}, "
              f"kurtosis {et.get('kurt', nan):.2f}",
              f"- DSR = {et.get('dsr', nan):.3f}; PSR(SR*=0) = {et.get('psr0', nan):.3f}",
              f"- Diagnostics only — effective N of this study: eigen {et.get('eigen', nan):.1f} · cluster "
              f"(ρ ≥ 0.5) {et.get('cluster', nan):.1f} · Li–Ji {et.get('liji', nan):.1f}; cross-trial Sharpe "
              f"variance {et.get('var_sr', nan):.3g} (per-period); v1.1-style DSR (N_eff + prior, V_cross) "
              f"{et.get('dsr_neff_cross', nan):.3f}; raw N with V_cross {et.get('dsr_raw_cross', nan):.3f}", ""]
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


def _space_specs(study: StudyResult) -> dict[str, dict[str, Any]]:
    """Declared parameter specs from ``meta['space']`` (SearchSpace json or object) or
    ``meta['search_space']``; {} when absent (observed ranges are used instead)."""
    meta = study.meta or {}
    sp = meta.get("space", meta.get("search_space"))
    if sp is not None and hasattr(sp, "to_json"):
        sp = sp.to_json()
    if isinstance(sp, Mapping) and isinstance(sp.get("params"), (list, tuple)):
        return {p["name"]: dict(p) for p in sp["params"] if isinstance(p, Mapping) and "name" in p}
    return {}


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def plateau_score(study: StudyResult, periods_per_year: float = 260.0, *,
                  radius: float | Mapping[str, float] | None = None) -> dict[str, Any]:
    """Judge-side parameter plateau (DESIGN §4.2 v1.2; red-team B4 + M5).

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
    specs = _space_specs(study)
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
    if spec is not None:
        return spec, "spec="
    s = getattr(evaluator, "spec", None)
    if s is not None:
        return s, "evaluator.spec"
    sym = getattr(evaluator, "symbol", None)
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


def _attempt_from_id(study_id: str) -> int | None:
    m = re.search(r"-a(\d+)$", study_id or "")
    return int(m.group(1)) if m else None


def resolve_prior_trials(study: StudyResult, prior_trials: float | None = None,
                         ledger_dir: Path | None = None) -> tuple[float, dict[str, Any]]:
    """Raw trials of earlier attempts of the study's system (M4).  Explicit ``prior_trials``
    wins; else book/system/attempt come from ``study.meta`` (then the study's own ledger row,
    then the ``-aN`` suffix of the id) and the ledger is summed (this study excluded, later
    attempts ignored).  Raises ``ValueError`` when attempt > 1 and no earlier trials can be
    found — cross-attempt deflation must never silently be 0."""
    if prior_trials is not None:
        return float(prior_trials), {"prior_source": "explicit prior_trials"}
    meta = study.meta or {}
    try:
        all_st = ledger.studies(ledger_dir)
    except Exception as e:  # noqa: BLE001 — unreadable ledger is handled below
        all_st, err = {}, repr(e)
    else:
        err = None
    own = all_st.get(study.study_id, {})
    book = meta.get("book") or own.get("book")
    system = meta.get("system") or own.get("system")
    attempt = meta.get("attempt") or own.get("attempt") or _attempt_from_id(study.study_id)
    total, used = 0.0, []
    if book and system:
        try:
            total, used = ledger.system_prior_trials(book, system, exclude_study=study.study_id,
                                                     max_attempt=attempt, ledger_dir=ledger_dir)
        except Exception as e:  # noqa: BLE001
            err = repr(e)
    info = {"prior_source": (f"ledger: {', '.join(used)}" if used else
                             ("ledger: no earlier studies" if book and system else
                              "none (book/system unknown)")),
            "book": book, "system": system, "attempt": attempt}
    if err:
        info["prior_error"] = err
    if attempt is not None and int(attempt) > 1 and (not used or total <= 0):
        raise ValueError(f"study {study.study_id} is attempt {attempt} but no earlier trials were found in the "
                         f"ledger for {book}/{system} ({err or 'no rows'}); pass prior_trials= explicitly")
    return float(total), info


# --------------------------------------------------------------------------- evaluation
def evaluate_gates(study: StudyResult, evaluator: Evaluator | None, *, periods_per_year: float,
                   selected_trades: pl.DataFrame | None = None,
                   mechanism_check: Callable[..., Any] | None = None, base_cost: Any = None,
                   swap_band: tuple[float, ...] = (0.5, 1.5), prior_trials: float | None = None,
                   ledger_dir: Path | None = None, plateau_radius: float | Mapping[str, float] | None = None,
                   holdout_days: int | None = None, pbo_splits: int = 16, n_boot: int = 2000,
                   seed: int = 12345, prior_effective_trials: float | None = None,
                   spec: Any = None) -> GateReport:
    """Run every DESIGN §4.2 v1.2 gate on a study.  See the module docstring for series choices.

    ``evaluator``: needed for the cost-stress gate (else SKIPPED) and, when
    ``selected_trades`` is None, to obtain the selected configuration's trades.
    ``base_cost``: the study's CostModel (else taken from ``evaluator.cost``/``cost_model``).
    ``spec``: InstrumentSpec for the stress's 1-pip slippage (else ``evaluator.spec``, else
    ``costs.load_instrument(evaluator.symbol)``); without one a non-synthetic evaluator's
    cost-stress row is SKIPPED (never the silent 1 pip = 1 point fallback).
    ``mechanism_check``: callable(study, evaluator) → bool or (bool, interpretation str).
    ``prior_trials``: raw trials of earlier attempts; None → read from the ledger
    (``ledger_dir``; raises if attempt > 1 and none found).  ``prior_effective_trials`` is a
    deprecated alias.  ``plateau_radius``: per-param radius pre-registered on the card.
    ``holdout_days``: holdout horizon for the pass band (default: one year of periods)."""
    if prior_effective_trials is not None:
        warnings.warn("prior_effective_trials is deprecated; the DSR gate uses raw trial counts — "
                      "pass prior_trials= (or nothing, to read the ledger)", DeprecationWarning, stacklevel=2)
        if prior_trials is None:
            prior_trials = prior_effective_trials
    ppy = float(periods_per_year)
    ann = math.sqrt(ppy)
    meta = study.meta or {}
    rows: list[GateRow] = []
    diag: dict[str, Any] = {}
    sel = selected_trial_id(study)
    col = f"t{sel}"
    sel_daily = study.returns.select("date", pl.col(col).fill_null(0.0).alias("ret"))
    sel_sr_ann = metrics.sharpe(sel_daily, ppy)

    data_dep = meta.get("candidate_set_data_dependent")
    if data_dep is None:
        data_dep = meta.get("method") == "tpe"
    cpcv_meta = meta.get("cpcv") if isinstance(meta.get("cpcv"), Mapping) else {}
    emb_capped = bool(meta.get("embargo_capped") or cpcv_meta.get("embargo_capped"))
    skip_dd = ("candidate set is data-dependent (method=" + str(meta.get("method")) + "): the configurations "
               "were chosen using the full dev sample, so this OOS series is not out of sample; re-run with a "
               "data-independent candidate set (grid / Sobol)")
    skip_emb = "embargo cap binds; use fewer CPCV groups"

    # ---- DSR (R1): V0 = 1/(T−1), N = raw trials (study + earlier attempts)
    n_study = _study_trial_count(study)
    n_prior, pinfo = resolve_prior_trials(study, prior_trials, ledger_dir)
    dres = st.dsr_from_matrix(study.returns, col, periods_per_year=ppy, n_trials=n_study, extra_trials=n_prior)
    op, thr = GATE_THRESHOLDS["dsr"]
    ok = _cmp(op, dres.dsr, thr)
    rows.append(GateRow("dsr", dres.dsr, f"{op} {thr}", "PASS" if ok else "FAIL",
                        f"DSR {dres.dsr:.2f}: {dres.dsr:.0%} probability the true Sharpe exceeds the "
                        f"{dres.sr0_annual:.2f} expected from the best of N = {dres.n_trials:g} raw null trials "
                        f"({n_study:g} this study + {n_prior:g} earlier) over {dres.n_obs} days."))
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

    # ---- Plateau (B4 + M5): always recomputed here
    op, thr = GATE_THRESHOLDS["plateau"]
    popt = study.selection.get("plateau_score") if study.selection else None
    if popt is not None:
        diag["plateau_optimizer"] = {"score": popt, "neighbourhood": study.selection.get("neighbourhood")}
    pl_info = plateau_score(study, ppy, radius=plateau_radius)
    psc = pl_info["plateau_score"]
    diag["plateau"] = {k: pl_info[k] for k in ("plateau_score", "n_neighbours", "peak_sharpe", "rules")}
    if pl_info["n_neighbours"] == 0:
        rows.append(GateRow("plateau", None, f"{op} {thr}", "SKIPPED",
                            f"No trial inside the pre-registered neighbourhood ({pl_info['method']}); plateau "
                            f"cannot be judged."))
    else:
        ok = _cmp(op, psc, thr)
        rows.append(GateRow("plateau", psc, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"{psc:.0%} of {pl_info['n_neighbours']} neighbours keep ≥ 50% of the peak Sharpe "
                            f"{pl_info['peak_sharpe']:.2f} ({pl_info['method']}); "
                            + ("broad optimum." if ok else "sharp, fragile optimum.")))

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
    return GateReport(study.study_id, rows, verdict, band, eff, diag, ppy, prior_runs)


def log_gates(report: GateReport, study_id: str | None = None, *, ledger_dir: Path | None = None) -> dict[str, Any]:
    """Append the gate results to the study's ledger row (event ``gates``, DESIGN §8).

    Logs the study's **own** trial count (``n_trials_study``) separately from the prior and the
    N used by DSR, so cumulative sums over attempts never double-count (m1), and a gate-run
    index (1, 2, …) so repeated gating of one study is visible (m9)."""
    sid = study_id or report.study_id
    n_prev = len(ledger.study_events(sid, "gates", ledger_dir=ledger_dir))
    et = report.effective_trials
    return ledger.log_event(sid, "gates", ledger_dir=ledger_dir,
                            gates=report.values(), gate_status=report.statuses(), verdict=report.verdict,
                            gate_run=n_prev + 1, gate_version="4.2-v1.2",
                            n_trials_study=et.get("n_trials_study"), n_trials_prior=et.get("n_trials_prior"),
                            n_trials_dsr=et.get("n_trials"),
                            effective_trials_study=float(et.get("n_eff", float("nan"))),
                            holdout_band=report.holdout_band)
