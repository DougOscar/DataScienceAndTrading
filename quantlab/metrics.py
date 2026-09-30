"""Return-based performance metrics on the daily-returns convention (contracts, Phase 1).

All functions take a daily simple-return series (``pl.Series`` / numpy array, or a frame
with ``date``/``ret``) on the book's nominal account.  ``periods_per_year``: FX/metals
≈ 260 (Mon–Fri server days), crypto CFDs 365, B3 ≈ 252.  Sharpe is always annualised from
**daily** returns (DESIGN §5), never from per-trade statistics.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

PERIODS_PER_YEAR = {"forex": 260.0, "crypto": 365.0, "b3": 252.0}
GAP_TRADING_DAYS = 5   # L4: a data gap is "more than 5 trading days missing" between two rows


def data_gaps(dates, *, calendar_days: bool = False, min_missing: int = GAP_TRADING_DAYS
              ) -> list[tuple[str, str, int]]:
    """Gaps in a daily date index: ``(last_date_before, first_date_after, n_missing)`` for every
    pair of consecutive rows with more than ``min_missing`` trading days (weekdays; calendar days
    when ``calendar_days``, e.g. crypto) missing between them."""
    d = np.asarray(dates).astype("datetime64[D]")
    if d.size < 2:
        return []
    miss = ((d[1:] - d[:-1]).astype(np.int64) if calendar_days else np.busday_count(d[:-1], d[1:])) - 1
    idx = np.flatnonzero(miss > min_missing)
    return [(str(d[i]), str(d[i + 1]), int(miss[i])) for i in idx]


def as_array(r) -> np.ndarray:
    """Finite daily returns as a float array (accepts Series, frame with 'ret', array-like)."""
    return _arr(r)


def _arr(r) -> np.ndarray:
    if isinstance(r, pl.DataFrame):
        r = r["ret"]
    if isinstance(r, pl.Series):
        r = r.to_numpy()
    a = np.asarray(r, dtype=float)
    return a[np.isfinite(a)]


def sharpe(r, periods_per_year: float = 260.0) -> float:
    a = _arr(r)
    if a.size < 2:
        return float("nan")
    sd = a.std(ddof=1)
    return float("nan") if sd == 0 else float(a.mean() / sd * math.sqrt(periods_per_year))


def sortino(r, periods_per_year: float = 260.0) -> float:
    a = _arr(r)
    if a.size < 2:
        return float("nan")
    downside = np.sqrt(np.mean(np.minimum(a, 0.0) ** 2))
    return float("nan") if downside == 0 else float(a.mean() / downside * math.sqrt(periods_per_year))


def equity_curve(r, equity0: float = 1.0) -> np.ndarray:
    return equity0 * np.cumprod(1.0 + _arr(r))


def drawdown(r) -> np.ndarray:
    """Drawdown series (≤ 0) as a fraction of the running peak (peak includes the start)."""
    eq = equity_curve(r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return eq / peak - 1.0


def max_drawdown(r) -> float:
    dd = drawdown(r)
    return float(dd.min()) if dd.size else 0.0


def longest_drawdown_periods(r) -> int:
    """Longest run of consecutive periods spent below the running peak."""
    under = drawdown(r) < 0
    best = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        best = max(best, cur)
    return best


def cagr(r, periods_per_year: float = 260.0) -> float:
    a = _arr(r)
    if a.size == 0:
        return float("nan")
    growth = float(np.prod(1.0 + a))
    years = a.size / periods_per_year
    return float("nan") if growth <= 0 or years <= 0 else growth ** (1.0 / years) - 1.0


def calmar(r, periods_per_year: float = 260.0) -> float:
    mdd = max_drawdown(r)
    return float("nan") if mdd == 0 else cagr(r, periods_per_year) / abs(mdd)


def ulcer_index(r) -> float:
    dd = drawdown(r)
    return float(np.sqrt(np.mean(dd ** 2))) if dd.size else 0.0


def cvar(r, alpha: float = 0.95) -> float:
    """Expected shortfall of daily returns beyond the (1-alpha) quantile (a negative number)."""
    a = _arr(r)
    if a.size == 0:
        return float("nan")
    q = np.quantile(a, 1.0 - alpha)
    tail = a[a <= q]
    return float(tail.mean()) if tail.size else float(q)


def skew_kurt(r) -> tuple[float, float]:
    """Sample skewness and **raw** (non-excess, normal = 3) kurtosis — the PSR/DSR convention."""
    a = _arr(r)
    if a.size < 3:
        return float("nan"), float("nan")
    m, s = a.mean(), a.std(ddof=0)
    if s == 0:
        return float("nan"), float("nan")
    z = (a - m) / s
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


def period_returns(daily: pl.DataFrame, every: str = "1mo") -> pl.DataFrame:
    """Compound daily returns into calendar periods ('1mo', '1y'): columns period, ret, n_days."""
    return (daily.sort("date")
            .group_by_dynamic(pl.col("date").cast(pl.Datetime("ms")), every=every, label="left")
            .agg(((pl.col("ret") + 1.0).product() - 1.0).alias("ret"), pl.len().alias("n_days"))
            .rename({"date": "period"}))


def period_sharpe(daily: pl.DataFrame, every: str = "1y", periods_per_year: float = 260.0) -> pl.DataFrame:
    """Annualised Sharpe computed *within* each calendar period (yearly / monthly Sharpe plots)."""
    return (daily.sort("date")
            .group_by_dynamic(pl.col("date").cast(pl.Datetime("ms")), every=every, label="left")
            .agg((pl.col("ret").mean() / pl.col("ret").std(ddof=1) * math.sqrt(periods_per_year)).alias("sharpe"),
                 pl.len().alias("n_days"))
            .rename({"date": "period"}))


def summary(daily: pl.DataFrame, periods_per_year: float = 260.0) -> dict[str, float]:
    r = daily["ret"]
    sk, ku = skew_kurt(r)
    monthly = period_returns(daily, "1mo")["ret"]
    return {
        "sharpe": sharpe(r, periods_per_year), "sortino": sortino(r, periods_per_year),
        "cagr": cagr(r, periods_per_year), "max_dd": max_drawdown(r), "calmar": calmar(r, periods_per_year),
        "ulcer": ulcer_index(r), "cvar95": cvar(r, 0.95), "skew": sk, "kurtosis": ku,
        "longest_dd_days": float(longest_drawdown_periods(r)),
        "mean_monthly": float(monthly.mean()) if monthly.len() else float("nan"),
        "worst_month": float(monthly.min()) if monthly.len() else float("nan"),
        "n_days": float(r.len()),
    }


# =========================================================================== trade-level (report-builder, DESIGN §5)
# These read the sized-trades frame produced by ``sizing.apply_sizing`` (skipped trades are
# always excluded).  Everything here is a *display* statistic for the tear sheet — never a
# validation gate (gates live in ``quantlab.gates`` and are read from the ledger, not recomputed).

def _trades_not_skipped(trades: pl.DataFrame | None) -> pl.DataFrame | None:
    if trades is None or trades.height == 0:
        return None
    return trades.filter(~pl.col("skipped")) if "skipped" in trades.columns else trades


def r_multiples(trades: pl.DataFrame | None) -> np.ndarray:
    """Per-trade R-multiples (risk type A, DESIGN §5): ``pnl_ccy / risk_target_ccy``.  Empty
    array when ``trades`` has no rows, no ``risk_target_ccy`` column (not fixed-fraction
    sizing) or every risk target is null/non-positive (e.g. a stopless system)."""
    t = _trades_not_skipped(trades)
    if t is None or "risk_target_ccy" not in t.columns or "pnl_ccy" not in t.columns:
        return np.array([], dtype=float)
    rt = t["risk_target_ccy"].to_numpy().astype(float)
    pnl = t["pnl_ccy"].to_numpy().astype(float)
    mask = np.isfinite(rt) & (rt > 0) & np.isfinite(pnl)
    return (pnl[mask] / rt[mask]).astype(float)


def expectancy_r(trades: pl.DataFrame | None) -> float:
    """Mean R-multiple (type A expectancy in R, DESIGN §5); NaN when unavailable."""
    r = r_multiples(trades)
    return float(r.mean()) if r.size else float("nan")


def max_losing_streak_r(trades: pl.DataFrame | None) -> int:
    """Longest run of consecutive negative R-multiples, in trade (not calendar) order."""
    r = r_multiples(trades)
    if r.size == 0:
        return 0
    best = cur = 0
    for x in r < 0:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def risk_realisation_error_stats(trades: pl.DataFrame | None) -> dict[str, float]:
    """Type A "risk-realisation error" (actual risk vs target after lot rounding and gaps):
    mean signed error and the 95th percentile of its magnitude, both as a fraction of the
    target risk (``sizing.apply_sizing``'s ``risk_realisation_error``, skips -1 = trade
    skipped below the minimum lot)."""
    t = _trades_not_skipped(trades)
    if t is None or "risk_realisation_error" not in t.columns:
        return {"mean": float("nan"), "p95_abs": float("nan"), "n": 0.0}
    e = t["risk_realisation_error"].drop_nulls().to_numpy().astype(float)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"mean": float("nan"), "p95_abs": float("nan"), "n": 0.0}
    return {"mean": float(e.mean()), "p95_abs": float(np.quantile(np.abs(e), 0.95)), "n": float(e.size)}


def mae_mfe_stats(trades: pl.DataFrame | None) -> dict[str, float]:
    """Type C MAE/MFE summary in **points** (engine columns ``mae_points``/``mfe_points``):
    median and 95th percentile of the adverse and favourable excursion any open trade saw."""
    t = _trades_not_skipped(trades)
    if t is None or "mae_points" not in t.columns or "mfe_points" not in t.columns:
        return {"mae_p50": float("nan"), "mae_p95": float("nan"),
                "mfe_p50": float("nan"), "mfe_p95": float("nan"), "n": 0.0}
    mae = t["mae_points"].to_numpy().astype(float)
    mfe = t["mfe_points"].to_numpy().astype(float)
    return {"mae_p50": float(np.median(mae)), "mae_p95": float(np.quantile(mae, 0.95)),
            "mfe_p50": float(np.median(mfe)), "mfe_p95": float(np.quantile(mfe, 0.95)), "n": float(t.height)}


def worst_trade_points(trades: pl.DataFrame | None) -> float:
    """Worst single-trade P&L in points (``pnl_points``; symbol-comparable within one symbol)."""
    t = _trades_not_skipped(trades)
    if t is None or "pnl_points" not in t.columns or t.height == 0:
        return float("nan")
    return float(t["pnl_points"].min())


def worst_trade_ccy(trades: pl.DataFrame | None) -> float:
    """Worst single-trade P&L in account currency (``pnl_ccy``, DESIGN §5 type B/C)."""
    t = _trades_not_skipped(trades)
    if t is None or "pnl_ccy" not in t.columns or t.height == 0:
        return float("nan")
    return float(t["pnl_ccy"].min())


def risk_per_trade_ccy(trades: pl.DataFrame | None) -> np.ndarray:
    """Type B "distribution of risk per trade": realised risk in account currency
    (``stop_distance x value_per_point x lots``, ``risk_realised_ccy``) for trades that carry
    a stop; empty when the column is absent or every stop is null (no hard stop at all)."""
    t = _trades_not_skipped(trades)
    if t is None or "risk_realised_ccy" not in t.columns:
        return np.array([], dtype=float)
    r = t["risk_realised_ccy"].drop_nulls().to_numpy().astype(float)
    return r[np.isfinite(r)]


# Below this fraction of total cost, gross P&L is treated as "~= 0": cost_pct_of_gross is
# reported as not meaningful rather than as a percentage that can read past 1000% for a
# razor-thin, sign-noisy gross edge (dry run #23 finding #60: gross at ~5.5% of total cost
# printed as "1808%", which does not mean "costs ate 18x the edge" in any useful sense -- the
# edge itself is indistinguishable from zero at that scale). Documented, not tuned.
GROSS_NEAR_ZERO_FRACTION = 0.10


def cost_breakdown(trades: pl.DataFrame | None) -> dict[str, float]:
    """Approximate cost decomposition of a sized-trades frame, in account currency.

    Engine/sizing convention: ``pnl_ccy = pnl_points*value_per_point_acct_exit*lots + swap_ccy
    - commission_ccy`` (``sizing.apply_sizing``).  Fill prices already price in the spread paid
    at entry/exit (DESIGN §4.3: longs buy at Ask, sell at Bid); ``spread_cost_points`` is the
    engine's own diagnostic of how many points of spread were paid, so the spread-free
    ("gross") P&L is recovered by adding it back at the entry-time point value.  Swap and
    commission are isolated from the money-mode columns at the *exit*-time point value
    (an approximation when ``spec.swap_ccy`` differs from the quote currency — the exact
    conversion lives only inside ``apply_sizing``, red-team N1).  Returns NaN fields when the
    trades frame lacks the needed columns (no evaluator, or a non-engine trade source).

    ``cost_total_ccy`` = spread + commission + (swap only when it drags, i.e. ``-swap_ccy`` when
    positive; never a signed zero -- #61).  ``cost_pct_of_gross`` = cost_total / gross P&L,
    defined only when gross is positive *and* not "≈ 0" next to the cost total (below
    :data:`GROSS_NEAR_ZERO_FRACTION` of it, #60); ``cost_pct_of_gross_note`` is ``None`` when it
    is defined, else the plain-language reason it isn't (always a string, never left for the
    caller to reverse-engineer from a bare NaN).  ``break_even_spread_multiple`` = the multiple
    of the *current* spread cost at which total net P&L would hit zero, holding every other cost
    fixed: ``1 + net_pnl / spread_cost_ccy`` (only meaningful when both are positive)."""
    need = {"pnl_ccy", "spread_cost_points", "swap_points", "swap_money_per_lot", "commission_per_lot",
            "value_per_point_acct", "value_per_point_acct_exit", "lots"}
    t = _trades_not_skipped(trades)
    nan = float("nan")
    out = {"gross_pnl_ccy": nan, "net_pnl_ccy": nan, "cost_total_ccy": nan, "spread_cost_ccy": nan,
          "commission_ccy": nan, "swap_ccy": nan, "cost_pct_of_gross": nan,
          "cost_pct_of_gross_note": "no trades, or the trades frame lacks the sizing cost columns",
          "swap_share_of_costs": nan, "break_even_spread_multiple": nan}
    if t is None or not need <= set(t.columns) or t.height == 0:
        return out
    lots = t["lots"].to_numpy().astype(float)
    spread_ccy = t["spread_cost_points"].fill_null(0.0).to_numpy().astype(float) * \
        t["value_per_point_acct"].to_numpy().astype(float) * lots
    swap_ccy = (t["swap_points"].fill_null(0.0).to_numpy().astype(float) *
                t["value_per_point_acct_exit"].to_numpy().astype(float) * lots +
                t["swap_money_per_lot"].fill_null(0.0).to_numpy().astype(float) * lots)
    commission_ccy = t["commission_per_lot"].fill_null(0.0).to_numpy().astype(float) * lots
    net_pnl = float(t["pnl_ccy"].sum())
    spread_tot, swap_tot, commission_tot = float(spread_ccy.sum()), float(swap_ccy.sum()), float(commission_ccy.sum())
    gross = net_pnl + spread_tot
    swap_cost = -swap_tot if swap_tot < 0.0 else 0.0     # #61: max(-0.0, 0.0) is -0.0 in Python
    cost_total = spread_tot + commission_tot + swap_cost

    if gross <= 0:
        cost_pct_of_gross, cpg_note = nan, "undefined (gross P&L <= 0: costs can't be a share of a negative edge)"
    elif cost_total > 0 and gross < GROSS_NEAR_ZERO_FRACTION * cost_total:
        cost_pct_of_gross = nan
        cpg_note = (f"n/a (gross ≈ 0: gross is {gross / cost_total:.1%} of total cost, below the "
                    f"{GROSS_NEAR_ZERO_FRACTION:.0%} floor)")
    else:
        cost_pct_of_gross, cpg_note = cost_total / gross, None

    out.update(gross_pnl_ccy=gross, net_pnl_ccy=net_pnl, cost_total_ccy=cost_total, spread_cost_ccy=spread_tot,
              commission_ccy=commission_tot, swap_ccy=swap_tot,
              cost_pct_of_gross=cost_pct_of_gross, cost_pct_of_gross_note=cpg_note,
              swap_share_of_costs=(swap_cost / cost_total if cost_total > 0 else nan),
              break_even_spread_multiple=(1.0 + net_pnl / spread_tot if net_pnl > 0 and spread_tot > 0 else nan))
    for k, v in out.items():                             # #61: belt-and-suspenders, no -0.0 anywhere
        if isinstance(v, float) and v == 0.0:
            out[k] = 0.0
    return out
