#!/usr/bin/env python3
"""Detect the current market regime for an instrument using multiple indicators.

Computes volatility percentile (ATR), trend strength (ADX), Hurst exponent,
and Bollinger Band width to classify the market into a 4-quadrant regime model.

Usage:
    python scripts/detect_regime.py --demo

Dependencies:
    uv pip install pandas numpy

To use with real data, load your own OHLCV DataFrame (e.g. from an MT5
export via quantlab's data loaders) in place of generate_demo_data().
"""

import argparse

import numpy as np
import pandas as pd

# ── Configuration ───────────────────────────────────────────────────
# Regime detection parameters
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.0
VOL_LOOKBACK = 100
HURST_MAX_LAG = 50
EMA_FAST = 20
EMA_SLOW = 50


# ── Data Generation (Demo Mode) ────────────────────────────────────
def generate_demo_data(n_bars: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data with clear regime transitions.

    Creates four distinct regimes:
    - Bars 0-74: Low volatility uptrend
    - Bars 75-149: High volatility uptrend
    - Bars 150-224: Low volatility range
    - Bars 225-299: High volatility range (choppy)

    Args:
        n_bars: Total number of bars to generate.

    Returns:
        DataFrame with open, high, low, close, volume columns.
    """
    np.random.seed(42)
    prices = [100.0]
    volumes = []

    for i in range(1, n_bars):
        if i < 75:
            # Low vol uptrend
            drift = 0.002
            vol = 0.008
            base_vol = 1000
        elif i < 150:
            # High vol uptrend
            drift = 0.003
            vol = 0.025
            base_vol = 2000
        elif i < 225:
            # Low vol range
            mean_price = prices[-1]
            drift = (150 - prices[-1]) * 0.01  # Mean revert to 150
            vol = 0.006
            base_vol = 800
        else:
            # High vol range (choppy)
            drift = (150 - prices[-1]) * 0.005
            vol = 0.03
            base_vol = 1500

        ret = drift + vol * np.random.randn()
        prices.append(prices[-1] * (1 + ret))
        volumes.append(base_vol * (1 + 0.3 * abs(np.random.randn())))

    volumes.append(volumes[-1])  # Pad to match length
    prices_arr = np.array(prices)

    # Generate OHLC from close prices
    noise = np.abs(np.random.randn(n_bars)) * 0.005 + 0.002
    highs = prices_arr * (1 + noise)
    lows = prices_arr * (1 - noise)
    opens = np.roll(prices_arr, 1)
    opens[0] = prices_arr[0]

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices_arr,
        "volume": volumes,
    })


# ── Regime Indicators ──────────────────────────────────────────────
def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute Average True Range.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Smoothing period.

    Returns:
        ATR series.
    """
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_atr_percentile(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr_period: int = 14, lookback: int = 100
) -> pd.Series:
    """ATR percentile rank over a rolling window.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        atr_period: ATR smoothing period.
        lookback: Window for percentile ranking.

    Returns:
        Series of percentile values (0-1).
    """
    atr = compute_atr(high, low, close, atr_period)
    return atr.rolling(lookback).rank(pct=True)


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Compute Average Directional Index (Wilder smoothing).

    Uses Wilder smoothing (EMA with alpha=1/period), not the faster
    span-based EMA (alpha=2/(period+1)) — this matches the textbook ADX
    definition and its conventional 20/25 thresholds.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        period: Smoothing period.

    Returns:
        ADX series.
    """
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out the smaller DM
    mask_plus = plus_dm < minus_dm
    mask_minus = minus_dm < plus_dm
    plus_dm = plus_dm.copy()
    minus_dm = minus_dm.copy()
    plus_dm[mask_plus] = 0
    minus_dm[mask_minus] = 0

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    alpha = 1.0 / period  # Wilder smoothing
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx


def compute_trend_direction(close: pd.Series, period: int = 20) -> pd.Series:
    """Detect trend direction via EMA slope and price position.

    Args:
        close: Close prices.
        period: EMA period.

    Returns:
        Series: +1 (uptrend), -1 (downtrend), 0 (neutral).
    """
    ema = close.ewm(span=period, adjust=False).mean()
    slope = ema.diff(5)
    above = close > ema
    direction = pd.Series(0, index=close.index, dtype=int)
    direction[(slope > 0) & above] = 1
    direction[(slope < 0) & ~above] = -1
    return direction


def compute_bb_width_percentile(
    close: pd.Series, period: int = 20,
    std_dev: float = 2.0, lookback: int = 100
) -> pd.Series:
    """Bollinger Band width percentile.

    Args:
        close: Close prices.
        period: BB period.
        std_dev: Standard deviation multiplier.
        lookback: Window for percentile ranking.

    Returns:
        Series of BB width percentile values (0-1).
    """
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    width = (2 * std_dev * std) / sma
    return width.rolling(lookback).rank(pct=True)


def compute_hurst(series: pd.Series, max_lag: int = 50) -> float:
    """Estimate Hurst exponent via Rescaled Range (R/S) method.

    Args:
        series: Price or return series.
        max_lag: Maximum lag for R/S computation.

    Returns:
        Hurst exponent estimate.
    """
    values = series.dropna().values
    if len(values) < max_lag * 2:
        return 0.5  # Not enough data

    lags = range(2, min(max_lag, len(values) // 4))
    rs_means = []
    valid_lags = []

    for lag in lags:
        n_chunks = len(values) // lag
        if n_chunks < 1:
            continue
        rs_list = []
        for c in range(n_chunks):
            chunk = values[c * lag:(c + 1) * lag]
            if len(chunk) < lag:
                continue
            mean_c = np.mean(chunk)
            devs = np.cumsum(chunk - mean_c)
            r = np.max(devs) - np.min(devs)
            s = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_list.append(r / s)
        if rs_list:
            rs_means.append(np.mean(rs_list))
            valid_lags.append(lag)

    if len(valid_lags) < 5:
        return 0.5

    log_lags = np.log(np.array(valid_lags, dtype=float))
    log_rs = np.log(np.array(rs_means, dtype=float))
    coeffs = np.polyfit(log_lags, log_rs, 1)
    h = float(coeffs[0])
    # Clamp to reasonable range
    return max(0.0, min(1.0, h))


def compute_rolling_hurst(
    close: pd.Series, window: int = 100, max_lag: int = 50
) -> pd.Series:
    """Compute rolling Hurst exponent.

    Args:
        close: Close prices.
        window: Rolling window size.
        max_lag: Max lag for R/S computation.

    Returns:
        Series of Hurst exponent values.
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    hurst_values = pd.Series(np.nan, index=close.index)

    for i in range(window, len(log_returns)):
        segment = log_returns.iloc[i - window:i]
        hurst_values.iloc[i + 1] = compute_hurst(segment, max_lag)

    return hurst_values


# ── Regime Classification ──────────────────────────────────────────
def classify_regime(
    vol_pct: float, adx_val: float, hurst_val: float, trend_dir: int
) -> dict[str, str]:
    """Classify into the 4-quadrant regime model.

    Args:
        vol_pct: ATR percentile (0-1).
        adx_val: Current ADX value.
        hurst_val: Current Hurst exponent.
        trend_dir: +1 (up), -1 (down), 0 (neutral).

    Returns:
        Dict with volatility, trend, direction, hurst, and quadrant keys.
    """
    vol_regime = (
        "low" if vol_pct < 0.30
        else "high" if vol_pct > 0.70
        else "normal"
    )
    trend_regime = (
        "trending" if adx_val > 25
        else "ranging" if adx_val < 20
        else "transitional"
    )
    direction = (
        "up" if trend_dir > 0
        else "down" if trend_dir < 0
        else "neutral"
    )
    hurst_regime = (
        "mean_reverting" if hurst_val < 0.4
        else "trending" if hurst_val > 0.6
        else "random_walk"
    )

    # Quadrant
    if trend_regime == "trending" and vol_regime != "high":
        quadrant = "quiet_trend"
    elif trend_regime == "trending" and vol_regime == "high":
        quadrant = "volatile_trend"
    elif trend_regime != "trending" and vol_regime != "high":
        quadrant = "quiet_range"
    elif trend_regime != "trending" and vol_regime == "high":
        quadrant = "volatile_range"
    else:
        quadrant = "uncertain"

    return {
        "volatility": vol_regime,
        "trend": trend_regime,
        "direction": direction,
        "hurst": hurst_regime,
        "quadrant": quadrant,
    }


def get_strategy_recommendation(quadrant: str, direction: str) -> list[str]:
    """Return strategy recommendations for the current regime.

    Args:
        quadrant: Regime quadrant string.
        direction: Trend direction string.

    Returns:
        List of strategy recommendation strings.
    """
    recs: dict[str, list[str]] = {
        "quiet_trend": [
            "Trend following with full position size",
            f"Direction: {direction} — align entries with trend",
            "Use 2x ATR trailing stop",
            "Standard indicator periods (20/50 EMA, 14 RSI)",
        ],
        "volatile_trend": [
            "Momentum strategy with REDUCED size (50%)",
            f"Direction: {direction} — strong but noisy",
            "Use 3x ATR trailing stop to avoid shakeouts",
            "Shorter indicator periods (10/20 EMA, 7 RSI)",
        ],
        "quiet_range": [
            "Mean-reversion / grid strategy",
            "RSI oversold/overbought at range boundaries",
            "Use 1.5x ATR fixed stops at range edges",
            "Bollinger Band bounce entries",
        ],
        "volatile_range": [
            "DANGER ZONE — reduce to 25% size or sit out",
            "Both trend and mean-reversion strategies will whipsaw",
            "If trading: use 3x ATR stops, expect frequent stops",
            "Wait for regime to resolve before full-size entries",
        ],
    }
    return recs.get(quadrant, ["Regime unclear — reduce size and wait for clarity"])


# ── Display ─────────────────────────────────────────────────────────
def display_current_regime(
    df: pd.DataFrame,
    vol_pct: pd.Series,
    adx: pd.Series,
    trend_dir: pd.Series,
    bb_pct: pd.Series,
    hurst_series: pd.Series,
) -> None:
    """Print the current regime analysis.

    Args:
        df: OHLCV DataFrame.
        vol_pct: ATR percentile series.
        adx: ADX series.
        trend_dir: Trend direction series.
        bb_pct: BB width percentile series.
        hurst_series: Rolling Hurst series.
    """
    # Get latest valid values
    latest_vol = vol_pct.dropna().iloc[-1] if not vol_pct.dropna().empty else 0.5
    latest_adx = adx.dropna().iloc[-1] if not adx.dropna().empty else 20.0
    latest_trend = int(trend_dir.dropna().iloc[-1]) if not trend_dir.dropna().empty else 0
    latest_bb = bb_pct.dropna().iloc[-1] if not bb_pct.dropna().empty else 0.5
    latest_hurst = hurst_series.dropna().iloc[-1] if not hurst_series.dropna().empty else 0.5
    latest_close = df["close"].iloc[-1]

    regime = classify_regime(latest_vol, latest_adx, latest_hurst, latest_trend)

    print("\n" + "=" * 60)
    print("  MARKET REGIME ANALYSIS")
    print("=" * 60)
    print(f"\n  Price:              {latest_close:.4f}")
    print(f"  ATR Percentile:     {latest_vol:.2f} ({regime['volatility']} vol)")
    print(f"  BB Width Pct:       {latest_bb:.2f}")
    print(f"  ADX:                {latest_adx:.1f} ({regime['trend']})")
    print(f"  Trend Direction:    {regime['direction']}")
    print(f"  Hurst Exponent:     {latest_hurst:.3f} ({regime['hurst']})")
    print(f"\n  REGIME QUADRANT:    {regime['quadrant'].upper()}")
    print("-" * 60)

    recs = get_strategy_recommendation(regime["quadrant"], regime["direction"])
    print("\n  Strategy Recommendations:")
    for r in recs:
        print(f"    - {r}")

    # Regime history for last 20 bars
    print("\n" + "-" * 60)
    print("  REGIME HISTORY (last 20 bars)")
    print("-" * 60)
    print(f"  {'Bar':>5}  {'Close':>10}  {'Vol%':>6}  {'ADX':>6}  {'Dir':>5}  {'Regime':<18}")
    print(f"  {'---':>5}  {'-----':>10}  {'----':>6}  {'---':>6}  {'---':>5}  {'------':<18}")

    start_idx = max(0, len(df) - 20)
    for i in range(start_idx, len(df)):
        vp = vol_pct.iloc[i] if not np.isnan(vol_pct.iloc[i]) else 0.5
        ax = adx.iloc[i] if not np.isnan(adx.iloc[i]) else 20.0
        td = int(trend_dir.iloc[i]) if not np.isnan(trend_dir.iloc[i]) else 0
        hr = hurst_series.iloc[i] if not np.isnan(hurst_series.iloc[i]) else 0.5
        r = classify_regime(vp, ax, hr, td)
        dir_sym = "+" if td > 0 else "-" if td < 0 else "."
        print(
            f"  {i:>5}  {df['close'].iloc[i]:>10.4f}  "
            f"{vp:>6.2f}  {ax:>6.1f}  {dir_sym:>5}  {r['quadrant']:<18}"
        )

    print("\n" + "=" * 60)
    print("  This is analytical output, not a trading recommendation.")
    print("=" * 60 + "\n")


# ── Main ────────────────────────────────────────────────────────────
def main() -> None:
    """Run regime detection analysis."""
    parser = argparse.ArgumentParser(description="Market Regime Detection")
    parser.add_argument("--demo", action="store_true", help="Use synthetic demo data (currently the only supported mode)")
    parser.add_argument("--bars", type=int, default=300, help="Number of synthetic bars to generate")
    args = parser.parse_args()

    print("Running in DEMO mode with synthetic data...")
    print("Swap generate_demo_data() for your own MT5/quantlab OHLCV loader for real data.")
    df = generate_demo_data(args.bars)

    print(f"Loaded {len(df)} bars. Computing regime indicators...")

    # Compute all indicators
    vol_pct = compute_atr_percentile(
        df["high"], df["low"], df["close"], ATR_PERIOD, VOL_LOOKBACK
    )
    adx = compute_adx(df["high"], df["low"], df["close"], ADX_PERIOD)
    trend_dir = compute_trend_direction(df["close"], EMA_FAST)
    bb_pct = compute_bb_width_percentile(df["close"], BB_PERIOD, BB_STD, VOL_LOOKBACK)
    hurst_series = compute_rolling_hurst(df["close"], window=100, max_lag=HURST_MAX_LAG)

    display_current_regime(df, vol_pct, adx, trend_dir, bb_pct, hurst_series)


if __name__ == "__main__":
    main()
