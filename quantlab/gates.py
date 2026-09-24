"""Single-system validation gates (DESIGN §4.2) and the validation report.

``evaluate_gates(study, evaluator, periods_per_year=...)`` turns a ``StudyResult`` into a
``GateReport``: one row per gate (value · threshold · PASS/FAIL/MANUAL/SKIPPED · one-line
interpretation in the DESIGN §5 style), an overall verdict, the effective-trials calculation
and the pre-registered holdout band (§4.4).

Verdict: ``PASS`` only if every gate is PASS.  Any FAIL → ``FAIL``.  No FAIL but at least one
SKIPPED/MANUAL gate → ``INCOMPLETE`` (a skipped mandatory gate can never produce PASS; the
mechanism check is MANUAL unless a callable is supplied).

Thresholds live in ``GATE_THRESHOLDS`` and are exactly DESIGN §4.2.  They are read-only; do
not move them after seeing results (recalibration is a DESIGN decision, §11 item 2).

Series used by each gate
------------------------
* DSR — the selected trial's full-dev daily returns, deflated by the effective number of
  trials of the study's trial return matrix (max of the eigen and cluster estimates, plus
  ``prior_effective_trials`` from earlier attempts).  The raw-N DSR is reported alongside.
* PBO — CSCV on the full trial return matrix (``study.returns``), 16 splits.
* OOS Sharpe — median annualised Sharpe across ``study.cpcv_paths``.
* Cost stress — the evaluator re-run on the selected params with ``base_cost.stressed()``
  (spread ×1.5, +1 pt slippage, bar_extreme stop fills) at swap multipliers
  {1.0} ∪ ``swap_band``; the gate value is the **minimum** Sharpe (DESIGN §4.3 swap band).
* Plateau — ``study.selection["plateau_score"]`` if the optimizer provided it, else
  recomputed (see :func:`plateau_score`).
* Time stability — calendar years of the selected trial's full-dev daily returns.
* Trade count — two conditions, both required: (a) per-trade MinTRL: number of trades ≥
  MinTRL computed on per-trade returns (pnl_ccy / equity_before) at 95 %; (b) the dev track
  record in days ≥ daily MinTRL at the *claimed* Sharpe = min(selected full-dev Sharpe,
  CPCV-median OOS Sharpe).  If trades are unavailable only (b) is checked and the row says so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import polars as pl

from . import ledger, metrics
from . import stats as st
from .contracts import Evaluator, Outcome, StudyResult

# DESIGN §4.2 — (comparator, threshold).  Read-only.
GATE_THRESHOLDS = MappingProxyType({
    "dsr": (">=", 0.95),
    "pbo": ("<", 0.30),
    "oos_sharpe": (">=", 1.0),              # annualised, CPCV path median
    "cost_stress_sharpe": (">", 0.5),       # annualised, CostModel.stressed()
    "plateau": (">=", 0.60),                # share of neighbours with SR ≥ 50 % of peak
    "positive_years": (">=", 0.60),
    "max_year_share": ("<=", 0.40),         # "no single year > 40 % of total PnL"
    "trade_count": (">=", "MinTRL"),
    "mechanism": ("manual", None),
})
PLATEAU_PEAK_FRACTION = 0.5
MINTRL_PROB = 0.95
GATE_ORDER = ("dsr", "pbo", "oos_sharpe", "cost_stress_sharpe", "plateau", "positive_years",
              "max_year_share", "trade_count", "mechanism")
GATE_LABELS = {
    "dsr": "Deflated Sharpe probability", "pbo": "PBO (CSCV)", "oos_sharpe": "OOS Sharpe (CPCV median)",
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

    def row(self, gate: str) -> GateRow:
        return next(r for r in self.rows if r.gate == gate)

    def values(self) -> dict[str, float | None]:
        return {r.gate: (None if r.value is None or not np.isfinite(r.value) else float(r.value))
                for r in self.rows}

    def statuses(self) -> dict[str, str]:
        return {r.gate: r.status for r in self.rows}

    def to_markdown(self) -> str:
        L = [f"# Validation — {self.study_id}", "",
             f"Gates: DESIGN §4.2 (development data, after costs). Sharpe annualised from daily "
             f"%-equity returns (periods/year = {self.periods_per_year:g}).", "",
             "| Gate | Value | Threshold | Status | Interpretation |", "|---|---|---|---|---|"]
        for r in self.rows:
            val = r.display if r.display is not None else ("—" if r.value is None else f"{r.value:.3g}")
            L.append(f"| {GATE_LABELS.get(r.gate, r.gate)} | {val} | {r.threshold} | **{r.status}** | "
                     f"{r.interpretation} |")
        L += ["", f"## Verdict: **{self.verdict}**", ""]
        bad = [r for r in self.rows if r.status != "PASS"]
        if bad:
            L += ["Not passed: " + ", ".join(f"{GATE_LABELS.get(r.gate, r.gate)} ({r.status})" for r in bad) + ".", ""]
        et = self.effective_trials
        L += ["## Effective number of trials", "",
              f"- Raw trials in this study: {et.get('n_trials_raw')}; prior attempts' effective trials added: "
              f"{et.get('prior_effective_trials', 0):g}",
              f"- Eigen (participation ratio): {et.get('eigen', float('nan')):.1f} · "
              f"Cluster (avg-linkage, ρ ≥ 0.5): {et.get('cluster', float('nan')):.1f} · "
              f"Li–Ji: {et.get('liji', float('nan')):.1f}",
              f"- **N_eff used = {et.get('n_eff', float('nan')):.1f}** (max of eigen and cluster + prior)",
              f"- Cross-trial Sharpe variance (per-period): {et.get('var_sr', float('nan')):.3g} → expected max "
              f"Sharpe under the null {et.get('sr0_annual', float('nan')):.2f} annualised",
              f"- Selected Sharpe {et.get('sr_annual', float('nan')):.2f} annualised over {et.get('n_obs')} days; "
              f"skew {et.get('skew', float('nan')):.2f}, kurtosis {et.get('kurt', float('nan')):.2f}",
              f"- DSR at N_eff = {et.get('dsr', float('nan')):.3f}; DSR at raw N = "
              f"{et.get('dsr_raw_n', float('nan')):.3f}; PSR(SR*=0) = {et.get('psr0', float('nan')):.3f}", ""]
        if self.holdout_band:
            hb = self.holdout_band
            L += ["## Pre-registered holdout pass band (DESIGN §4.4)", "",
                  f"Source: {hb.get('source')} + stationary bootstrap (mean block {hb.get('mean_block', 0):.1f} d), "
                  f"horizon {hb.get('horizon_days')} days, {hb.get('n_boot')} samples, seed {hb.get('seed')}.", "",
                  "| Criterion | Pass if |", "|---|---|",
                  f"| Holdout Sharpe (annualised) | ≥ {hb['sharpe_p10']:.2f} (p10; median {hb['sharpe_median']:.2f}) |",
                  f"| Mean monthly return at leverage k = {hb['leverage']:.3g} | ≥ {hb['ret_at_budget_p10']:.2%} (p10) |",
                  f"| Max drawdown (unlevered) | ≤ {hb['max_dd_mag_p95']:.2%} (p95) |",
                  f"| Trade count | {_fmt_range(hb.get('trades_lo'), hb.get('trades_hi'))} |", ""]
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


def plateau_score(study: StudyResult, periods_per_year: float = 260.0, *,
                  peak_fraction: float = PLATEAU_PEAK_FRACTION, grid_max_levels: int = 20,
                  radius: float = 0.1) -> dict[str, Any]:
    """Share of the selected configuration's neighbours whose Sharpe ≥ peak_fraction × the
    selected Sharpe (the "peak").

    Sharpe per trial: ``m_sharpe`` column if present, else computed from ``study.returns``.
    Neighbourhood: if every parameter has ≤ ``grid_max_levels`` distinct values (a grid), map
    each to its index among the sorted unique values and take the trials at Chebyshev
    distance exactly 1 in index space (all ±1-step moves incl. diagonals).  Otherwise
    (continuous/random search), min-max normalise numeric params to [0, 1] and take the trials
    within Chebyshev radius ``radius`` with identical non-numeric params; if fewer than 3 are
    found, use the 8 nearest (Euclidean, normalised).  Failed trials count as neighbours
    with Sharpe = −inf (a crashing neighbour is not a plateau)."""
    trials = study.trials
    sel = selected_trial_id(study)
    if "m_sharpe" in trials.columns:
        sr = dict(zip(trials["trial_id"].to_list(), trials["m_sharpe"].to_list()))
    else:
        sr = _trial_sharpes(study, periods_per_year)
    pcols = [f"param_{n}" for n in study.param_names]
    ids = np.array(trials["trial_id"].to_list())
    isel = int(np.nonzero(ids == sel)[0][0])
    levels = {c: trials[c].unique().drop_nulls().sort().to_list() for c in pcols}
    if all(len(v) <= grid_max_levels for v in levels.values()):
        idx = np.column_stack([np.array([levels[c].index(x) if x is not None else -99 for x in trials[c].to_list()])
                               for c in pcols])
        dist = np.abs(idx - idx[isel]).max(axis=1)
        nb = np.nonzero(dist == 1)[0]
        method = "grid ±1 step (Chebyshev)"
    else:
        num, cat = [], []
        for c in pcols:
            (num if trials[c].dtype.is_numeric() else cat).append(c)
        x = np.column_stack([trials[c].cast(pl.Float64).to_numpy() for c in num]) if num else np.zeros((len(ids), 0))
        rng_ = np.nanmax(x, 0) - np.nanmin(x, 0) if num else np.array([])
        xn = (x - np.nanmin(x, 0)) / np.where(rng_ > 0, rng_, 1) if num else x
        same = np.ones(len(ids), bool)
        for c in cat:
            same &= np.array([v == trials[c][isel] for v in trials[c].to_list()])
        d = np.abs(xn - xn[isel]).max(axis=1) if num else np.zeros(len(ids))
        nb = np.nonzero(same & (d <= radius) & (np.arange(len(ids)) != isel))[0]
        method = f"normalised radius {radius}"
        if nb.size < 3 and num:
            e = np.sqrt(((xn - xn[isel]) ** 2).sum(axis=1))
            e[isel] = np.inf
            nb = np.argsort(e)[:8]
            method = "8 nearest (normalised)"
    peak = float(sr.get(sel, float("nan")))
    vals = np.array([sr.get(int(ids[i]), None) for i in nb], dtype=float)
    vals = np.where(np.isfinite(vals), vals, -np.inf)
    score = float(np.mean(vals >= peak_fraction * peak)) if nb.size and np.isfinite(peak) and peak > 0 else 0.0
    return {"plateau_score": score, "n_neighbours": int(nb.size), "peak_sharpe": peak, "method": method}


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


def _resolve_base_cost(evaluator, base_cost):
    if base_cost is not None:
        return base_cost
    for attr in ("cost", "cost_model", "base_cost"):
        c = getattr(evaluator, attr, None)
        if c is not None and hasattr(c, "stressed"):
            return c
    return None


# --------------------------------------------------------------------------- evaluation
def evaluate_gates(study: StudyResult, evaluator: Evaluator | None, *, periods_per_year: float,
                   selected_trades: pl.DataFrame | None = None,
                   mechanism_check: Callable[..., Any] | None = None, base_cost: Any = None,
                   swap_band: tuple[float, ...] = (0.5, 1.5), prior_effective_trials: float = 0.0,
                   holdout_days: int | None = None, pbo_splits: int = 16, n_boot: int = 2000,
                   seed: int = 12345) -> GateReport:
    """Run every DESIGN §4.2 gate on a study.  See the module docstring for series choices.

    ``evaluator``: needed for the cost-stress gate (else SKIPPED) and, when
    ``selected_trades`` is None, to obtain the selected configuration's trades.
    ``base_cost``: the study's CostModel (else taken from ``evaluator.cost``/``cost_model``).
    ``mechanism_check``: callable(study, evaluator) → bool or (bool, interpretation str).
    ``prior_effective_trials``: effective trials of earlier attempts of the same system.
    ``holdout_days``: holdout horizon for the pass band (default: one year of periods)."""
    ppy = float(periods_per_year)
    ann = math.sqrt(ppy)
    rows: list[GateRow] = []
    diag: dict[str, Any] = {}
    sel = selected_trial_id(study)
    col = f"t{sel}"
    sel_daily = study.returns.select("date", pl.col(col).fill_null(0.0).alias("ret"))
    sel_sr_ann = metrics.sharpe(sel_daily, ppy)

    # ---- DSR
    dres = st.dsr_from_matrix(study.returns, col, periods_per_year=ppy, extra_trials=prior_effective_trials,
                              methods=("eigen", "cluster"))
    liji = st.effective_n_trials(study.returns, "liji")
    op, thr = GATE_THRESHOLDS["dsr"]
    ok = _cmp(op, dres.dsr, thr)
    rows.append(GateRow("dsr", dres.dsr, f"{op} {thr}", "PASS" if ok else "FAIL",
                        f"DSR {dres.dsr:.2f}: {dres.dsr:.0%} probability the true Sharpe exceeds the "
                        f"{dres.sr0_annual:.2f} expected from the best of {dres.n_eff:.0f} effective null trials "
                        f"(raw-N DSR {dres.dsr_raw_n:.2f})."))
    eff = {**dres.as_dict(), **{k: v for k, v in dres.n_eff_by_method.items()}, "liji": liji,
           "prior_effective_trials": prior_effective_trials}
    eff.pop("n_eff_by_method", None)

    # ---- PBO
    op, thr = GATE_THRESHOLDS["pbo"]
    try:
        pres = st.pbo_cscv(study.returns, n_splits=pbo_splits)
        ok = _cmp(op, pres.pbo, thr)
        rows.append(GateRow("pbo", pres.pbo, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"PBO {pres.pbo:.2f}: the in-sample-best config ranked below the OOS median in "
                            f"{pres.pbo:.0%} of {pres.n_combos} splits (beat it {1 - pres.pbo:.0%}); "
                            f"OOS loss probability {pres.prob_oos_loss:.0%}."))
        diag["pbo_detail"] = {k: v for k, v in pres.summary().items()}
    except ValueError as e:
        rows.append(GateRow("pbo", None, f"{op} {thr}", "SKIPPED", f"PBO not computable: {e}."))

    # ---- OOS Sharpe (CPCV path median)
    op, thr = GATE_THRESHOLDS["oos_sharpe"]
    oos_med = float("nan")
    if study.cpcv_paths is not None and study.cpcv_paths.height:
        ps = (study.cpcv_paths.group_by("path_id").agg(
            (pl.col("ret").mean() / pl.col("ret").std(ddof=1) * ann).alias("sr")))["sr"].to_numpy()
        ps = ps[np.isfinite(ps)]
        oos_med = float(np.median(ps)) if ps.size else float("nan")
        q25, q75 = (np.quantile(ps, [0.25, 0.75]) if ps.size else (np.nan, np.nan))
        ok = _cmp(op, oos_med, thr)
        rows.append(GateRow("oos_sharpe", oos_med, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"Median OOS Sharpe {oos_med:.2f} across {ps.size} CPCV paths (IQR {q25:.2f}–{q75:.2f}); "
                            f"{np.mean(ps > 0):.0%} of paths positive; in-sample {sel_sr_ann:.2f}."))
        diag["cpcv_path_sharpe"] = {"median": oos_med, "p25": float(q25), "p75": float(q75),
                                    "min": float(ps.min()) if ps.size else float("nan"), "n_paths": int(ps.size)}
    else:
        rows.append(GateRow("oos_sharpe", None, f"{op} {thr}", "SKIPPED", "No CPCV paths in the study."))

    # ---- Cost stress
    op, thr = GATE_THRESHOLDS["cost_stress_sharpe"]
    bc = _resolve_base_cost(evaluator, base_cost)
    if evaluator is None or bc is None:
        why = "no evaluator" if evaluator is None else "no base cost model (pass base_cost=)"
        rows.append(GateRow("cost_stress_sharpe", None, f"{op} {thr}", "SKIPPED",
                            f"Cost stress not run: {why}; the verdict cannot be PASS."))
    else:
        stressed = bc.stressed()
        mults = sorted({1.0, *swap_band})
        srs = {}
        for mlt in mults:
            cm = replace(stressed, swap_multiplier=stressed.swap_multiplier * mlt)
            srs[mlt] = _outcome_sharpe(evaluator(dict(study.selected_params), cost=cm), ppy)
        worst = min(srs.values())
        ok = _cmp(op, worst, thr)
        rows.append(GateRow("cost_stress_sharpe", worst, f"{op} {thr}", "PASS" if ok else "FAIL",
                            f"Worst Sharpe {worst:.2f} at 1.5× spread +1 pt slippage, bar-extreme stops, swap ×"
                            f"{', '.join(f'{m:g}' for m in mults)} ({', '.join(f'{v:.2f}' for v in srs.values())}); "
                            f"base {sel_sr_ann:.2f}."))
        diag["cost_stress_by_swap_mult"] = {f"{k:g}": v for k, v in srs.items()}
        diag["cost_stress_version"] = stressed.version

    # ---- Plateau
    op, thr = GATE_THRESHOLDS["plateau"]
    pscore = study.selection.get("plateau_score") if study.selection else None
    src = "optimizer"
    if pscore is None:
        pinfo = plateau_score(study, ppy)
        pscore, src = pinfo["plateau_score"], f"recomputed, {pinfo['method']}, {pinfo['n_neighbours']} neighbours"
    ok = _cmp(op, float(pscore), thr)
    rows.append(GateRow("plateau", float(pscore), f"{op} {thr}", "PASS" if ok else "FAIL",
                        f"{float(pscore):.0%} of neighbouring configs keep ≥ 50% of the peak Sharpe ({src}); "
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
        tpd = _trades_per_day(selected_trades, sel_daily["date"])
        hb = st.holdout_band(study, int(holdout_days or round(ppy)), ppy, n_boot=n_boot,
                             trades_per_day=tpd, seed=seed)
        band = hb.as_dict()
    except ValueError as e:
        diag["holdout_band_error"] = str(e)

    statuses = [r.status for r in rows]
    verdict = "FAIL" if "FAIL" in statuses else ("PASS" if all(s == "PASS" for s in statuses) else "INCOMPLETE")
    return GateReport(study.study_id, rows, verdict, band, eff, diag, ppy)


def log_gates(report: GateReport, study_id: str | None = None, *, ledger_dir: Path | None = None) -> dict[str, Any]:
    """Append the gate results to the study's ledger row (event ``gates``, DESIGN §8)."""
    return ledger.log_event(study_id or report.study_id, "gates", ledger_dir=ledger_dir,
                            gates=report.values(), gate_status=report.statuses(), verdict=report.verdict,
                            effective_trials=float(report.effective_trials.get("n_eff", float("nan"))),
                            holdout_band=report.holdout_band)
