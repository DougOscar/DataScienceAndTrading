"""Assemble the tear sheet's metrics dict, applicability rows and one-line interpretations
(DESIGN §5).  Gate *values* always come from the ledger's logged ``gates`` event, read
verbatim -- never recomputed.  A handful of display-only numbers not persisted by
``gates._log_gates`` (PSR against zero, the WFO OOS recent-third Sharpe, PBO) are recomputed
here from the same stored series with the same public functions the statistician uses, and are
always labelled as such so nobody mistakes them for a second, independent gate run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from .. import metrics as m
from .. import stats as st
from ..contracts import RiskType
from ..gates import GATE_LABELS, GATE_THRESHOLDS, WFO_RECENT_FRACTION
from . import _applicability as appl

Row = dict[str, Any]


def _row(metrics_name: str, shown: bool, why: str) -> Row:
    return {"metric": metrics_name, "shown": bool(shown), "why": why}


def _basic_win_stats(trades: pl.DataFrame) -> dict[str, Any]:
    """Win rate / profit factor from whichever P&L column is present (``pnl_ccy`` preferred,
    else ``pnl_points``) -- unlike ``evaluators.trade_stats`` this needs no ``entry_ts``/
    ``exit_ts`` (holding time is handled separately, only when those exist)."""
    t = trades.filter(~pl.col("skipped")) if "skipped" in trades.columns else trades
    col = "pnl_ccy" if "pnl_ccy" in t.columns else ("pnl_points" if "pnl_points" in t.columns else None)
    if col is None or t.height == 0:
        return {"available": False, "n_trades": float(t.height)}
    pnl = t[col].to_numpy().astype(float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    pf = float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else float("nan"))
    return {"available": True, "n_trades": float(pnl.size), "win_rate": float(np.mean(pnl > 0)), "profit_factor": pf}


# --------------------------------------------------------------------------- returns & risk
def returns_and_risk(daily: pl.DataFrame | None, ppy: float, budget: Any
                     ) -> tuple[dict[str, Any], dict[str, str], list[Row]]:
    metrics: dict[str, Any] = {}
    interp: dict[str, str] = {}
    rows: list[Row] = []
    if daily is None or daily.height < 2:
        why = "no daily-returns series for the selected configuration (trial store has no matching column)"
        for name in ("cagr", "mean_monthly", "ret_at_budget", "sharpe", "sortino", "calmar", "psr0",
                    "max_dd", "longest_dd_days", "ulcer", "cvar95", "worst_month", "pct_time_underwater"):
            rows.append(_row(name, False, why))
        return metrics, interp, rows

    r = daily["ret"]
    s = m.summary(daily, ppy)
    metrics.update({"cagr": s["cagr"], "mean_monthly": s["mean_monthly"], "sharpe": s["sharpe"],
                    "sortino": s["sortino"], "calmar": s["calmar"], "max_dd": s["max_dd"],
                    "longest_dd_days": s["longest_dd_days"], "ulcer": s["ulcer"], "cvar95": s["cvar95"],
                    "worst_month": s["worst_month"], "skew": s["skew"], "kurtosis": s["kurtosis"]})
    dd = m.drawdown(r)
    metrics["pct_time_underwater"] = float(np.mean(dd < 0)) if dd.size else float("nan")
    interp["cagr"] = f"CAGR {s['cagr']:.1%}: compound annual growth of the %-equity curve."
    interp["mean_monthly"] = f"Average calendar-month return {s['mean_monthly']:.2%} (unlevered, all months equal weight)."
    interp["sharpe"] = f"Annualised Sharpe {s['sharpe']:.2f} from daily returns (periods/year = {ppy:g})."
    interp["sortino"] = f"Sortino {s['sortino']:.2f}: like Sharpe but only penalises downside deviation."
    interp["calmar"] = f"Calmar {s['calmar']:.2f}: CAGR / max drawdown magnitude."
    interp["max_dd"] = f"Max drawdown {s['max_dd']:.1%} of the running peak (unlevered)."
    interp["longest_dd_days"] = f"Longest stretch below the running peak: {s['longest_dd_days']:.0f} trading days."
    interp["ulcer"] = f"Ulcer index {s['ulcer']:.3f}: RMS drawdown depth (penalises deep and long drawdowns)."
    interp["cvar95"] = f"Daily CVaR95 {s['cvar95']:.2%}: mean return on the worst 5% of days."
    interp["worst_month"] = f"Worst calendar month {s['worst_month']:.2%}."
    interp["pct_time_underwater"] = f"{metrics['pct_time_underwater']:.0%} of trading days were below the running peak."
    for name in ("cagr", "mean_monthly", "sharpe", "sortino", "calmar", "max_dd", "longest_dd_days",
                "ulcer", "cvar95", "worst_month", "pct_time_underwater"):
        rows.append(_row(name, True, "always reported (DESIGN §5)"))

    # PSR against zero: deterministic, not a §4.2 gate (DSR — the deflated, N-trial-adjusted
    # version — is read from the ledger below).  Same formula the statistician's stats.psr uses.
    sr_period = float(r.mean() / r.std(ddof=1)) if r.len() > 1 and r.std(ddof=1) else float("nan")
    psr0 = st.psr(sr_period, 0.0, float(r.len()), s["skew"], s["kurtosis"]) if np.isfinite(sr_period) else float("nan")
    metrics["psr0"] = psr0
    interp["psr0"] = (f"PSR(SR*=0) {psr0:.2f}: probability the true per-period Sharpe exceeds 0, given its "
                      f"skew/kurtosis over {r.len()} days -- NOT deflated for multiple testing (see DSR under "
                      f"Robustness for that).")
    rows.append(_row("psr0", True, "always reported (DESIGN §5); recomputed here (deterministic, not a gate)"))

    if budget is not None and np.isfinite(budget.leverage):
        lo, med, hi = budget.mean_monthly_band
        metrics["ret_at_budget"] = {"mean_monthly": budget.mean_monthly, "leverage": budget.leverage,
                                    "band_p5_p50_p95": [lo, med, hi], "label": budget.label}
        interp["ret_at_budget"] = (f"Return at the 10% DD budget: {budget.mean_monthly:.2%}/month at leverage "
                                   f"k={budget.leverage:.2f} (bootstrap band p5/p50/p95 = {lo:.2%}/{med:.2%}/"
                                   f"{hi:.2%}) -> classified **{budget.label}** (DESIGN §4.5).")
        rows.append(_row("ret_at_budget", True, "always reported (DESIGN §5, §4.5 classification)"))
    else:
        metrics["ret_at_budget"] = None
        rows.append(_row("ret_at_budget", False, "could not solve a leverage hitting the 10% DD budget "
                         "(bootstrapped drawdown never reaches it)"))
    return metrics, interp, rows


# --------------------------------------------------------------------------- costs
def costs(trades: pl.DataFrame | None) -> tuple[dict[str, Any], dict[str, str], list[Row]]:
    names = ("cost_pct_of_gross", "break_even_spread_multiple", "swap_share_of_costs")
    if trades is None or trades.height == 0:
        why = "no trades (no evaluator given, or the selected configuration produced none)"
        return {}, {}, [_row(n, False, why) for n in names]
    cb = m.cost_breakdown(trades)
    if not np.isfinite(cb["gross_pnl_ccy"]):
        why = "trades frame lacks the sizing cost columns (spread_cost_points / swap / commission / value_per_point)"
        return {}, {}, [_row(n, False, why) for n in names]
    metrics = {n: cb[n] for n in names}
    interp = {
        "cost_pct_of_gross": (f"Costs are {cb['cost_pct_of_gross']:.1%} of gross P&L "
                              f"(gross {cb['gross_pnl_ccy']:,.0f}, net {cb['net_pnl_ccy']:,.0f}, "
                              f"cost {cb['cost_total_ccy']:,.0f}, 100k nominal)."),
        "break_even_spread_multiple": ("Break-even spread multiple "
                                       f"{cb['break_even_spread_multiple']:.2f}x: spread would have to widen by "
                                       "this multiple (other costs held fixed) to erase the net P&L."
                                       if np.isfinite(cb["break_even_spread_multiple"]) else
                                       "Break-even spread multiple: not defined (net P&L <= 0 or no spread cost)."),
        "swap_share_of_costs": (f"Swap is {cb['swap_share_of_costs']:.0%} of total cost."
                                if np.isfinite(cb["swap_share_of_costs"]) else
                                "Swap share of costs: not defined (no net cost, or swap was a net credit)."),
    }
    rows = [_row(n, True, "always reported (DESIGN §5); approximated from the trades frame's cost columns")
           for n in names]
    return metrics, interp, rows


# --------------------------------------------------------------------------- robustness (ledger)
def robustness(gates_event: dict[str, Any] | None, pbo: dict[str, Any] | None, holdout: dict[str, Any],
               wfo_recent: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, str], list[Row]]:
    metrics: dict[str, Any] = {}
    interp: dict[str, str] = {}
    rows: list[Row] = []
    gate_names = ("dsr", "cscv_oos_loss", "oos_sharpe", "wfo_oos", "plateau", "cost_stress_sharpe",
                 "positive_years", "max_year_share")
    if gates_event is None:
        metrics["gate_verdict"] = "not validated"
        for n in gate_names:
            rows.append(_row(n, False, "no gates event logged for this study yet (S5 has not run)"))
    else:
        gv = gates_event.get("gates") or {}
        gs = gates_event.get("gate_status") or {}
        plateau = gates_event.get("plateau") or {}
        metrics["gate_verdict"] = gates_event.get("verdict")
        for n in gate_names:
            v = gv.get(n)
            metrics[n] = v
            status = gs.get(n)
            thr = GATE_THRESHOLDS.get(n)
            thr_txt = f"{thr[0]} {thr[1]}" if thr else ""
            label = GATE_LABELS.get(n, n)
            if n == "plateau":
                interp[n] = (f"{label} {('%.2f' % v) if v is not None else '—'} ({thr_txt}, status {status}): "
                            f"weakest parameter axis '{plateau.get('weakest_axis')}' keeps >= half the peak "
                            f"Sharpe on {('%.0f%%' % (v * 100)) if v is not None else 'n/a'} of its judge-run "
                            f"perturbations.")
            elif n == "wfo_oos":
                rec_txt = (f"; recent third (recomputed from the stored WFO OOS series, same "
                          f"{WFO_RECENT_FRACTION:.0%} window the gate uses) {wfo_recent['sharpe_recent']:.2f}"
                          if wfo_recent and wfo_recent.get("available") else "; recent third: not available")
                interp[n] = (f"{label} {('%.2f' % v) if v is not None else '—'} ({thr_txt}, status {status})"
                            f"{rec_txt}.")
            else:
                interp[n] = f"{label} {('%.3g' % v) if v is not None else '—'} ({thr_txt}, status {status})."
            rows.append(_row(n, True, f"logged gates event, gate_run {gates_event.get('gate_run')}"))
        metrics["wfo_oos_recent"] = (wfo_recent or {}).get("sharpe_recent")

    hs = holdout.get("status")
    metrics["holdout_status"] = hs if holdout.get("unlocked") else "not unlocked"
    interp["holdout_status"] = (f"Holdout: {hs}." if holdout.get("unlocked")
                                else "Holdout: not unlocked (checkpoint B has not run).")
    rows.append(_row("holdout_status", True, "from the ledger's holdout_unlock/holdout_exam rows (never from fresh holdout bars)"))

    if pbo and pbo.get("available"):
        metrics["pbo"] = pbo["pbo"]
        interp["pbo"] = (f"PBO {pbo['pbo']:.2f} (diagnostic, not a v1.2 gate — replaced by CSCV OOS loss): "
                         f"probability the in-sample-best configuration ranks at or below the out-of-sample "
                         f"median.")
        rows.append(_row("pbo", True, "diagnostic only; recomputed from the trial store with stats.pbo_cscv"))
    else:
        rows.append(_row("pbo", False, (pbo or {}).get("why", "trial store unavailable")))
    return metrics, interp, rows


# --------------------------------------------------------------------------- trade-level (risk type)
def trade_level(risk_type: RiskType, trades: pl.DataFrame | None, dates: pl.Series | None
                ) -> tuple[dict[str, Any], dict[str, str], list[Row]]:
    metrics: dict[str, Any] = {}
    interp: dict[str, str] = {}
    rows: list[Row] = []
    no_trades_why = "no trades (no evaluator given, or the selected configuration produced none)"

    def excl(name: str, why: str) -> None:
        rows.append(_row(name, False, why))

    def incl(name: str, why: str = "DESIGN §5, risk type " + risk_type.value) -> None:
        rows.append(_row(name, True, why))

    if trades is None or trades.height == 0:
        for name in ("raw_points_per_trade", "expectancy_r", "max_losing_streak_r",
                    "risk_realisation_error", "pnl_ccy_distribution", "risk_per_trade_distribution",
                    "mae_mfe", "worst_trade", "holding_time", "win_rate_profit_factor",
                    "turnover_exposure_cost_per_turnover"):
            excl(name, no_trades_why)
        return metrics, interp, rows

    # ``evaluators.trade_stats`` needs entry_ts/exit_ts (a full engine trades frame); win
    # rate / profit factor only need a P&L column, so those degrade separately from holding time.
    ts = _basic_win_stats(trades)
    metrics["win_rate"] = ts.get("win_rate")
    metrics["profit_factor"] = ts.get("profit_factor")
    metrics["n_trades"] = ts.get("n_trades")
    interp["win_rate"] = f"Win rate {ts.get('win_rate', float('nan')):.0%} over {ts.get('n_trades', 0):.0f} trades."
    interp["profit_factor"] = f"Profit factor {ts.get('profit_factor', float('nan')):.2f} (gross win / gross loss)."
    if risk_type is RiskType.D:
        r = appl.not_relevant_reason(risk_type, "win_rate_profit_factor")
        excl("win_rate_profit_factor", r)
    elif ts.get("available"):
        incl("win_rate_profit_factor")
    else:
        excl("win_rate_profit_factor", "no pnl_ccy/pnl_points column in the trades frame")

    if {"entry_ts", "exit_ts"} <= set(trades.columns):
        from ..evaluators import trade_stats
        hs = trade_stats(trades, dates=dates)
        metrics["expectancy_points"] = hs.get("expectancy_points")
        metrics["hold_days_p95"] = hs.get("hold_days_p95")
        interp["holding_time"] = (f"Holding time: median-scale {hs.get('avg_hold_bars', float('nan')):.1f} bars; "
                                  f"95th percentile {hs.get('hold_days_p95', float('nan')):.0f} trading days.")
        incl("holding_time")
    else:
        excl("holding_time", "no entry_ts/exit_ts in the trades frame")

    # type A: R-multiples, expectancy in R, risk-realisation error, max losing streak in R
    r_mult = m.r_multiples(trades)
    if risk_type is RiskType.A and r_mult.size:
        exp_r = float(r_mult.mean())
        streak = m.max_losing_streak_r(trades)
        rre = m.risk_realisation_error_stats(trades)
        metrics["expectancy_r"] = exp_r
        metrics["max_losing_streak_r"] = streak
        metrics["risk_realisation_error"] = rre
        interp["expectancy_r"] = f"Expectancy {exp_r:.2f} R over {r_mult.size} trades (mean P&L / planned risk)."
        interp["max_losing_streak_r"] = f"Longest losing streak: {streak} trades in a row with a negative R-multiple."
        interp["risk_realisation_error"] = (f"Risk-realisation error: mean {rre['mean']:+.1%} of target, 95th "
                                            f"percentile |error| {rre['p95_abs']:.1%} (lot-step rounding / gaps "
                                            f"vs the planned {risk_type.value}-type risk).")
        incl("expectancy_r")
        incl("max_losing_streak_r")
        incl("risk_realisation_error")
        excl("raw_points_per_trade", appl.not_relevant_reason(risk_type, "raw_points_per_trade"))
    else:
        for name in ("expectancy_r", "max_losing_streak_r", "risk_realisation_error"):
            why = (appl.not_relevant_reason(risk_type, name) or
                  "no risk_target_ccy in the trades frame (not fixed-fraction sizing)")
            excl(name, why)
        if risk_type is RiskType.A:
            excl("raw_points_per_trade", appl.not_relevant_reason(risk_type, "raw_points_per_trade"))
        elif "pnl_points" in trades.columns:
            tnp = trades.filter(~pl.col("skipped")) if "skipped" in trades.columns else trades
            exp_pts = float(tnp["pnl_points"].mean()) if tnp.height else float("nan")
            metrics["expectancy_points"] = exp_pts
            interp["raw_points_per_trade"] = f"Mean P&L per trade {exp_pts:+.1f} points."
            incl("raw_points_per_trade")
        else:
            excl("raw_points_per_trade", "no pnl_points column in the trades frame")

    # type B: P&L in currency / % equity, risk-per-trade distribution
    if risk_type is RiskType.B and "pnl_ccy" in trades.columns:
        t = trades.filter(~pl.col("skipped")) if "skipped" in trades.columns else trades
        pnl = t["pnl_ccy"].to_numpy().astype(float)
        eq0 = float(t["equity_before"][0]) if "equity_before" in t.columns and t.height else 100_000.0
        metrics["pnl_ccy_mean"] = float(pnl.mean()) if pnl.size else float("nan")
        metrics["pnl_pct_equity_mean"] = float((pnl / eq0).mean()) if pnl.size else float("nan")
        interp["pnl_ccy_distribution"] = (f"Mean P&L per trade {metrics['pnl_ccy_mean']:,.0f} (100k nominal), "
                                          f"{metrics['pnl_pct_equity_mean']:.2%} of the nominal account.")
        incl("pnl_ccy_distribution")
        rpt = m.risk_per_trade_ccy(trades)
        if rpt.size:
            metrics["risk_per_trade_ccy"] = {"mean": float(rpt.mean()), "p95": float(np.quantile(rpt, 0.95))}
            interp["risk_per_trade_distribution"] = (f"Risk per trade (stop distance x value): mean "
                                                      f"{rpt.mean():,.0f}, p95 {np.quantile(rpt, 0.95):,.0f} "
                                                      f"(100k nominal).")
            incl("risk_per_trade_distribution")
        else:
            excl("risk_per_trade_distribution", "no trade carries a stop_price")
    else:
        excl("pnl_ccy_distribution", appl.not_relevant_reason(risk_type, "pnl_ccy_distribution")
            or f"risk type {risk_type.value}, not B")
        excl("risk_per_trade_distribution", f"risk type {risk_type.value}, not B")

    # type C: MAE/MFE, worst trade, CVaR (daily CVaR already reported above), holding time
    if risk_type is RiskType.C:
        mm = m.mae_mfe_stats(trades)
        metrics["mae_mfe"] = mm
        interp["mae_mfe"] = (f"MAE median {mm['mae_p50']:.1f} pts (p95 {mm['mae_p95']:.1f}); "
                             f"MFE median {mm['mfe_p50']:.1f} pts (p95 {mm['mfe_p95']:.1f}): how far trades moved "
                             f"against/for the position before exit.")
        incl("mae_mfe")
        wt_pts, wt_ccy = m.worst_trade_points(trades), m.worst_trade_ccy(trades)
        metrics["worst_trade_points"] = wt_pts
        metrics["worst_trade_ccy"] = wt_ccy
        interp["worst_trade"] = f"Worst single trade: {wt_pts:+.1f} points ({wt_ccy:+,.0f} on the 100k nominal)."
        incl("worst_trade")
    else:
        excl("mae_mfe", appl.not_relevant_reason(risk_type, "mae_mfe") or f"risk type {risk_type.value}, not C")
        excl("worst_trade", appl.not_relevant_reason(risk_type, "worst_trade") or f"risk type {risk_type.value}, not C")

    # type D: turnover, exposure, cost per unit turnover -- the engine has no continuous-position
    # path (RuleEvaluator._mode raises NotImplementedError for risk_type D), so there is never
    # trade/position data to compute these from.
    excl("turnover_exposure_cost_per_turnover", appl.not_relevant_reason(risk_type,
        "turnover_exposure_cost_per_turnover") or
        "not produced by engine yet: the Phase 0 engine has no continuous / vol-targeted position path")
    return metrics, interp, rows
