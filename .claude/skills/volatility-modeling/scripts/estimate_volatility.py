#!/usr/bin/env python3
"""Multi-estimator volatility analysis with volatility cone construction.

Computes realized volatility using five estimators (close-to-close, Parkinson,
Garman-Klass, EWMA, GARCH), builds a volatility cone showing historical
percentile distribution, and classifies the current volatility regime.

Usage:
    python scripts/estimate_volatility.py                       # demo mode
    python scripts/estimate_volatility.py --periods-per-year 260  # FX/metals daily

Dependencies:
    uv pip install pandas numpy

To use with real data, load your own OHLCV DataFrame (e.g. from an MT5
export via quantlab's data loaders) in place of generate_demo_data().
"""

import argparse
import sys
from typing import Optional

import numpy as np
import pandas as pd


# ── Configuration ───────────────────────────────────────────────────
# Annualization (periods_per_year): FX/metals daily bars ~260 (5-day week;
# 252 also common), crypto CFDs 365 (7-day week), intraday = bars_per_day
# x trading_days_per_year. Always pass the value matching your instrument
# and bar interval rather than relying on this default.
ANNUALIZATION_FACTOR = 252
EWMA_LAMBDA = 0.94
WINDOWS = [7, 14, 30, 60, 90]
CONE_PERCENTILES = [5, 25, 50, 75, 95]


# ── Demo Data ───────────────────────────────────────────────────────
def generate_demo_data(
    n_days: int = 400, seed: int = 42, periods_per_year: float = ANNUALIZATION_FACTOR
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with illustrative FX/metals-scale volatility regimes.

    Creates three regimes (illustrative magnitudes only — measure actual
    levels on your own instrument/timeframe rather than assuming these):
    - Low vol  (days 0-150):   ~6% annualized (quiet FX major)
    - High vol (days 150-300): ~20% annualized (elevated / metals-like)
    - Normal   (days 300+):    ~12% annualized

    Args:
        n_days: Number of daily bars to generate.
        seed: Random seed for reproducibility.
        periods_per_year: Annualization factor used to de-annualize the
            target vol levels above into daily vol.

    Returns:
        DataFrame with columns: date, open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-06-01", periods=n_days, freq="D")

    # Daily vol by regime
    daily_vols = np.zeros(n_days)
    daily_vols[:150] = 0.06 / np.sqrt(periods_per_year)  # ~6% annualized
    daily_vols[150:300] = 0.20 / np.sqrt(periods_per_year)  # ~20% annualized
    daily_vols[300:] = 0.12 / np.sqrt(periods_per_year)  # ~12% annualized

    closes = np.zeros(n_days)
    closes[0] = 100.0

    for i in range(1, n_days):
        ret = rng.normal(0.0002, daily_vols[i])
        closes[i] = closes[i - 1] * np.exp(ret)

    # Generate OHLC from closes
    opens = np.zeros(n_days)
    highs = np.zeros(n_days)
    lows = np.zeros(n_days)
    opens[0] = closes[0] * (1 + rng.normal(0, 0.002))

    for i in range(1, n_days):
        opens[i] = closes[i - 1] * (1 + rng.normal(0, 0.003))
        intraday_range = abs(rng.normal(0, daily_vols[i])) * closes[i]
        highs[i] = max(opens[i], closes[i]) + intraday_range * rng.uniform(0.3, 1.0)
        lows[i] = min(opens[i], closes[i]) - intraday_range * rng.uniform(0.3, 1.0)
        # Ensure valid OHLC
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
        lows[i] = max(lows[i], 0.01)  # floor at near-zero

    highs[0] = closes[0] * 1.01
    lows[0] = closes[0] * 0.99

    volumes = rng.lognormal(mean=15, sigma=0.5, size=n_days)

    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ── Volatility Estimators ──────────────────────────────────────────
def vol_close_to_close(
    closes: pd.Series, window: int, annualize: int = ANNUALIZATION_FACTOR
) -> pd.Series:
    """Close-to-close realized volatility (rolling).

    Args:
        closes: Series of closing prices.
        window: Rolling window in days.
        annualize: Annualization factor.

    Returns:
        Series of annualized volatility estimates.
    """
    log_ret = np.log(closes / closes.shift(1))
    return log_ret.rolling(window).std(ddof=1) * np.sqrt(annualize)


def vol_parkinson(
    highs: pd.Series, lows: pd.Series, window: int, annualize: int = ANNUALIZATION_FACTOR
) -> pd.Series:
    """Parkinson high-low range volatility estimator (rolling).

    Args:
        highs: Series of high prices.
        lows: Series of low prices.
        window: Rolling window in days.
        annualize: Annualization factor.

    Returns:
        Series of annualized volatility estimates.
    """
    hl_sq = (np.log(highs / lows)) ** 2
    factor = 1.0 / (4.0 * np.log(2))
    return np.sqrt(hl_sq.rolling(window).mean() * factor) * np.sqrt(annualize)


def vol_garman_klass(
    opens: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    window: int,
    annualize: int = ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Garman-Klass OHLC volatility estimator (rolling).

    Args:
        opens: Series of open prices.
        highs: Series of high prices.
        lows: Series of low prices.
        closes: Series of closing prices.
        window: Rolling window in days.
        annualize: Annualization factor.

    Returns:
        Series of annualized volatility estimates.
    """
    hl = np.log(highs / lows)
    co = np.log(closes / opens)
    gk = 0.5 * hl**2 - (2.0 * np.log(2) - 1.0) * co**2
    return np.sqrt(gk.rolling(window).mean().clip(lower=0)) * np.sqrt(annualize)


def vol_ewma(
    closes: pd.Series, lam: float = EWMA_LAMBDA, annualize: int = ANNUALIZATION_FACTOR
) -> pd.Series:
    """EWMA (RiskMetrics) volatility estimator.

    Args:
        closes: Series of closing prices.
        lam: Decay factor (0.94 for daily).
        annualize: Annualization factor.

    Returns:
        Series of annualized EWMA volatility estimates.
    """
    log_ret = np.log(closes / closes.shift(1)).dropna()
    n = len(log_ret)
    ewma_var = np.zeros(n)

    # Initialize with first 20 observations or available data
    init_window = min(20, n)
    ewma_var[0] = np.var(log_ret.values[:init_window])

    for t in range(1, n):
        ewma_var[t] = lam * ewma_var[t - 1] + (1.0 - lam) * log_ret.values[t - 1] ** 2

    result = pd.Series(
        np.sqrt(ewma_var) * np.sqrt(annualize),
        index=log_ret.index,
        name="ewma_vol",
    )
    return result


def vol_garch_simple(
    closes: pd.Series,
    alpha: float = 0.08,
    beta: float = 0.88,
    annualize: int = ANNUALIZATION_FACTOR,
) -> pd.Series:
    """Simple GARCH(1,1) volatility with fixed parameters.

    Uses illustrative pre-set parameters rather than MLE fitting — fit
    alpha/beta on your own instrument for anything beyond a quick look.
    For full MLE fitting, see scripts/vol_forecast.py.

    Args:
        closes: Series of closing prices.
        alpha: Shock reaction coefficient.
        beta: Persistence coefficient.
        annualize: Annualization factor.

    Returns:
        Series of annualized GARCH volatility estimates.
    """
    log_ret = np.log(closes / closes.shift(1)).dropna()
    n = len(log_ret)

    omega = np.var(log_ret.values) * (1.0 - alpha - beta)
    omega = max(omega, 1e-10)

    garch_var = np.zeros(n)
    garch_var[0] = np.var(log_ret.values[:min(20, n)])

    for t in range(1, n):
        garch_var[t] = omega + alpha * log_ret.values[t - 1] ** 2 + beta * garch_var[t - 1]

    result = pd.Series(
        np.sqrt(garch_var) * np.sqrt(annualize),
        index=log_ret.index,
        name="garch_vol",
    )
    return result


# ── Volatility Cone ─────────────────────────────────────────────────
def build_volatility_cone(
    closes: pd.Series,
    windows: Optional[list[int]] = None,
    percentiles: Optional[list[int]] = None,
) -> tuple[dict[int, dict[int, float]], dict[int, float], dict[int, float]]:
    """Build a volatility cone from historical closing prices.

    Args:
        closes: Series of closing prices.
        windows: List of lookback windows in days.
        percentiles: List of percentiles to compute.

    Returns:
        Tuple of (cone_data, current_vol, current_percentile).
        cone_data: {window: {percentile: vol_value}}.
        current_vol: {window: current_vol_value}.
        current_percentile: {window: percentile_rank}.
    """
    if windows is None:
        windows = WINDOWS
    if percentiles is None:
        percentiles = CONE_PERCENTILES

    cone_data: dict[int, dict[int, float]] = {}
    current_vol: dict[int, float] = {}
    current_pctile: dict[int, float] = {}

    for w in windows:
        rv = vol_close_to_close(closes, w).dropna()
        if len(rv) < 10:
            continue
        cone_data[w] = {}
        for p in percentiles:
            cone_data[w][p] = float(np.percentile(rv.values, p))
        current_vol[w] = float(rv.iloc[-1])
        # Percentile rank of current vol
        current_pctile[w] = float(
            (rv.values < rv.iloc[-1]).sum() / len(rv) * 100
        )

    return cone_data, current_vol, current_pctile


# ── Regime Classification ───────────────────────────────────────────
def classify_regime(
    annualized_vol: float,
    low: float = 0.08,
    normal: float = 0.15,
    high: float = 0.25,
) -> str:
    """Classify volatility regime.

    The default thresholds (8% / 15% / 25% annualized) are illustrative,
    FX/metals-scale placeholders. Calibrate them per instrument from your
    own volatility cone (see build_volatility_cone) rather than assuming
    these levels — a quiet FX major and gold/silver will have different
    baselines, and crypto CFDs will run far higher.

    Args:
        annualized_vol: Annualized volatility as a decimal (e.g., 0.15 = 15%).
        low: Upper bound of the "low vol" regime.
        normal: Upper bound of the "normal vol" regime.
        high: Upper bound of the "high vol" regime.

    Returns:
        Regime label string.
    """
    if annualized_vol < low:
        return "LOW VOL (range-bound, mean-reversion favored)"
    elif annualized_vol < normal:
        return "NORMAL VOL (trending possible, balanced strategies)"
    elif annualized_vol < high:
        return "HIGH VOL (strong trends or sharp reversals)"
    else:
        return "CRISIS VOL (reduce size significantly)"


# ── Reporting ───────────────────────────────────────────────────────
def print_estimator_report(
    df: pd.DataFrame, window: int = 30, periods_per_year: float = ANNUALIZATION_FACTOR
) -> None:
    """Print volatility estimates from all estimators for a given window.

    Args:
        df: DataFrame with OHLCV columns.
        window: Lookback window in bars.
        periods_per_year: Annualization factor for the bar interval/market
            (see module docstring for FX/metals/crypto/intraday guidance).
    """
    closes = df["close"]
    opens = df["open"]
    highs = df["high"]
    lows = df["low"]

    cc = vol_close_to_close(closes, window, annualize=periods_per_year).iloc[-1]
    pk = vol_parkinson(highs, lows, window, annualize=periods_per_year).iloc[-1]
    gk = vol_garman_klass(opens, highs, lows, closes, window, annualize=periods_per_year).iloc[-1]
    ewma = vol_ewma(closes, annualize=periods_per_year).iloc[-1]
    garch = vol_garch_simple(closes, annualize=periods_per_year).iloc[-1]

    print(f"\n{'=' * 60}")
    print(f"  VOLATILITY ESTIMATES  ({window}-bar window)")
    print(f"{'=' * 60}")
    print(f"  {'Estimator':<20} {'Annualized Vol':>15} {'Per-Bar Vol':>12}")
    print(f"  {'-' * 47}")
    print(f"  {'Close-to-Close':<20} {cc * 100:>14.1f}% {cc / np.sqrt(periods_per_year) * 100:>11.2f}%")
    print(f"  {'Parkinson (H-L)':<20} {pk * 100:>14.1f}% {pk / np.sqrt(periods_per_year) * 100:>11.2f}%")
    print(f"  {'Garman-Klass':<20} {gk * 100:>14.1f}% {gk / np.sqrt(periods_per_year) * 100:>11.2f}%")
    print(f"  {'EWMA (λ=0.94)':<20} {ewma * 100:>14.1f}% {ewma / np.sqrt(periods_per_year) * 100:>11.2f}%")
    print(f"  {'GARCH(1,1)':<20} {garch * 100:>14.1f}% {garch / np.sqrt(periods_per_year) * 100:>11.2f}%")
    print()
    print(f"  Regime: {classify_regime(cc)}")


def print_cone_report(
    cone: dict[int, dict[int, float]],
    current: dict[int, float],
    pctile: dict[int, float],
) -> None:
    """Print volatility cone summary.

    Args:
        cone: Cone percentile data by window.
        current: Current vol by window.
        pctile: Current percentile rank by window.
    """
    print(f"\n{'=' * 70}")
    print("  VOLATILITY CONE")
    print(f"{'=' * 70}")
    header = f"  {'Window':>7}  {'5th':>7}  {'25th':>7}  {'50th':>7}  {'75th':>7}  {'95th':>7}  {'Curr':>7}  {'%ile':>5}"
    print(header)
    print(f"  {'-' * 64}")

    for w in sorted(cone.keys()):
        row = f"  {w:>5}d"
        for p in CONE_PERCENTILES:
            row += f"  {cone[w][p] * 100:>6.1f}%"
        row += f"  {current[w] * 100:>6.1f}%"
        row += f"  {pctile[w]:>4.0f}%"
        print(row)

    print()


def print_multi_window_report(
    df: pd.DataFrame, periods_per_year: float = ANNUALIZATION_FACTOR
) -> None:
    """Print volatility across multiple windows.

    Args:
        df: DataFrame with OHLCV columns.
        periods_per_year: Annualization factor for the bar interval/market.
    """
    closes = df["close"]

    print(f"\n{'=' * 55}")
    print("  VOLATILITY BY LOOKBACK WINDOW (Close-to-Close)")
    print(f"{'=' * 55}")
    print(f"  {'Window':>8}  {'Ann. Vol':>10}  {'Per-Bar Vol':>12}  {'Regime':>20}")
    print(f"  {'-' * 52}")

    for w in WINDOWS:
        cc = vol_close_to_close(closes, w, annualize=periods_per_year)
        if cc.dropna().empty:
            continue
        v = cc.iloc[-1]
        regime = classify_regime(v).split("(")[0].strip()
        print(f"  {w:>6}d  {v * 100:>9.1f}%  {v / np.sqrt(periods_per_year) * 100:>11.2f}%  {regime:>20}")


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    """Run the volatility estimation analysis."""
    parser = argparse.ArgumentParser(description="Multi-estimator volatility analysis")
    parser.add_argument("--days", type=int, default=400, help="Days of history")
    parser.add_argument("--window", type=int, default=30, help="Primary estimation window")
    parser.add_argument(
        "--periods-per-year",
        type=float,
        default=ANNUALIZATION_FACTOR,
        help="Annualization factor: ~260 FX/metals daily (252 also common), "
        "365 crypto CFDs, bars_per_day*trading_days for intraday",
    )
    args = parser.parse_args()

    print("Running in DEMO mode with synthetic data.")
    print("Swap generate_demo_data() for your own MT5/quantlab OHLCV loader for real data.\n")
    df = generate_demo_data(args.days, periods_per_year=args.periods_per_year)
    print(f"Generated {len(df)} daily bars with 3 volatility regimes:")
    print("  Days   1-150: Low vol  (~6% annualized)")
    print("  Days 151-300: High vol (~20% annualized)")
    print("  Days 301+:    Normal   (~12% annualized)")

    if len(df) < args.window + 5:
        print(f"Error: Need at least {args.window + 5} data points, got {len(df)}.")
        sys.exit(1)

    # Estimator comparison
    print_estimator_report(df, window=args.window, periods_per_year=args.periods_per_year)

    # Multi-window analysis
    print_multi_window_report(df, periods_per_year=args.periods_per_year)

    # Volatility cone
    cone, current, pctile = build_volatility_cone(df["close"])
    print_cone_report(cone, current, pctile)

    # Summary
    cc_30 = vol_close_to_close(df["close"], 30).iloc[-1]
    print(f"  Summary: 30-day realized vol is {cc_30 * 100:.1f}% annualized")
    print(f"  This sits at the {pctile.get(30, 0):.0f}th percentile of historical 30-day vol.")
    print(f"  Regime: {classify_regime(cc_30)}")
    print()


if __name__ == "__main__":
    main()
