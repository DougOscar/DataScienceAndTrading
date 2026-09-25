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
