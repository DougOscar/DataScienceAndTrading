"""Baseline strategy: SMA crossover with ATR-based stop loss / take profit."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd


@dataclass
class StrategyParams:
    fast: int = 20
    slow: int = 50
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    # Session filter — None means no filter (24/5 or whole dataset)
    session_start: int | None = None  # inclusive hour (0-23)
    session_end: int | None = None    # exclusive hour; open positions closed at boundary
    # Position sizing
    sizing_mode: str = "unit"         # "unit" | "vol_scaled" | "fixed_frac"
    risk_fraction: float = 0.01       # fraction of equity risked per trade (fixed_frac only)

    def as_dict(self) -> dict:
        return asdict(self)


class SMACrossoverStrategy:
    """Long/short trend-following strategy.

    Entry:
      * Long when the fast SMA crosses above the slow SMA.
      * Short when the fast SMA crosses below the slow SMA.
    Exit:
      * Opposite crossover (reverses the position).
      * Hard stop loss at entry - direction * sl_atr_mult * ATR.
      * Take profit at entry + direction * tp_atr_mult * ATR.
      * Forced close when the bar is the last inside the session window.
    Sizing:
      * "unit"       — one unit per trade (default, backward-compatible).
      * "vol_scaled" — size = 1 / ATR; PnL is in ATR-normalised units.
      * "fixed_frac" — size = equity * risk_fraction / (sl_atr_mult * ATR).
    """

    def __init__(self, params: StrategyParams | None = None):
        self.params = params or StrategyParams()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()
        out["sma_fast"] = out["close"].rolling(p.fast, min_periods=p.fast).mean()
        out["sma_slow"] = out["close"].rolling(p.slow, min_periods=p.slow).mean()

        prev_close = out["close"].shift()
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        prev_diff = (ind["sma_fast"] - ind["sma_slow"]).shift()
        diff = ind["sma_fast"] - ind["sma_slow"]
        long_cross = (diff > 0) & (prev_diff <= 0)
        short_cross = (diff < 0) & (prev_diff >= 0)

        signal = np.where(long_cross, 1, np.where(short_cross, -1, 0))

        p = self.params
        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        ind["signal"] = signal.astype(int)
        return ind


@dataclass
class RSIMeanReversionParams:
    """Parameters for the RSI mean-reversion strategy.

    Field names mirror those the Backtester reads (sl_atr_mult, tp_atr_mult,
    session_start, session_end, sizing_mode, risk_fraction) so the existing
    Backtester / walk_forward / robustness helpers work unchanged.
    """

    rsi_period: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.0
    adx_period: int = 14
    adx_max: float = 25.0
    use_adx_filter: bool = False
    # Midline exit (RSI crossing 50 against the trade) is described in the doc
    # but cannot be expressed cleanly through the existing Backtester's
    # signal/SL/TP/reversal contract — leave disabled until the Backtester
    # gains a custom-exit hook.
    use_midline_exit: bool = False
    # Backtester-required fields (session + sizing).
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def __post_init__(self):
        if not (self.rsi_lower < 50 < self.rsi_upper):
            raise ValueError(
                f"Constraint rsi_lower < 50 < rsi_upper violated "
                f"(got rsi_lower={self.rsi_lower}, rsi_upper={self.rsi_upper})"
            )

    def as_dict(self) -> dict:
        return asdict(self)


class RSIMeanReversionStrategy:
    """RSI mean-reversion with ATR-based stop loss / take profit.

    Entry:
      * Long when RSI crosses up through ``rsi_lower`` (oversold rejection).
      * Short when RSI crosses down through ``rsi_upper`` (overbought rejection).
    Exit:
      * Hard SL/TP at entry +/- atr_mult * ATR_entry.
      * Opposite RSI threshold crossover reverses the position.
      * Optional ADX regime filter: suppress entries while ADX > adx_max.
    """

    def __init__(self, params: RSIMeanReversionParams | None = None):
        self.params = params or RSIMeanReversionParams()

    def _wilder_ema(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        # ATR (simple rolling mean of TR — same convention as SMACrossoverStrategy).
        prev_close = out["close"].shift()
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()

        # RSI with Wilder's smoothing (alpha = 1 / rsi_period).
        delta = out["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = self._wilder_ema(gain, p.rsi_period)
        avg_loss = self._wilder_ema(loss, p.rsi_period)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            rsi = 100.0 - 100.0 / (1.0 + rs)
        out["rsi"] = rsi

        # ADX (Wilder) — only computed when filter is active to keep WFO fast.
        if p.use_adx_filter:
            up = out["high"].diff()
            dn = -out["low"].diff()
            plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
            minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
            plus_dm = pd.Series(plus_dm, index=out.index)
            minus_dm = pd.Series(minus_dm, index=out.index)
            atr_w = self._wilder_ema(tr, p.adx_period)
            with np.errstate(divide="ignore", invalid="ignore"):
                plus_di = 100.0 * self._wilder_ema(plus_dm, p.adx_period) / atr_w
                minus_di = 100.0 * self._wilder_ema(minus_dm, p.adx_period) / atr_w
                dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            out["adx"] = self._wilder_ema(dx, p.adx_period)
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        rsi = ind["rsi"]
        prev_rsi = rsi.shift()
        long_cross = (rsi > p.rsi_lower) & (prev_rsi <= p.rsi_lower)
        short_cross = (rsi < p.rsi_upper) & (prev_rsi >= p.rsi_upper)

        signal = np.where(long_cross, 1, np.where(short_cross, -1, 0))

        if p.use_adx_filter and "adx" in ind.columns:
            allow = (ind["adx"] < p.adx_max).fillna(False).to_numpy()
            signal = np.where(allow, signal, 0)

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        signal = np.where(ind["atr"].isna() | rsi.isna(), 0, signal)

        ind["signal"] = signal.astype(int)
        return ind


@dataclass
class LinearRegressionChannelParams:
    """Linear Regression Channel dual-mode strategy parameters."""

    lr_period: int = 50
    lr_mult: float = 2.0
    slope_threshold: float = 0.3
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    mode: str = "auto"   # "auto" | "trend" | "reversion"
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


def _rolling_ols_channel(close: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (reg_line_end, slope, std_err) arrays computed in O(N) total.

    For each bar t, ``reg_line_end[t]`` is the OLS regression line value at
    the right edge of the ``period``-bar window ending at t (i.e. the
    predicted price at bar t given the rolling fit).  ``std_err`` is the
    sample standard error of the regression.
    """
    n = len(close)
    out_line = np.full(n, np.nan)
    out_slope = np.full(n, np.nan)
    out_se = np.full(n, np.nan)
    if period < 2 or n < period:
        return out_line, out_slope, out_se

    # Precompute cumulative sums of y and y².
    y = close
    cum_y = np.concatenate(([0.0], np.cumsum(y)))
    cum_y2 = np.concatenate(([0.0], np.cumsum(y * y)))
    # cum_ky[k] = sum_{j=0..k-1} j * y[j]
    j_idx = np.arange(n, dtype=float)
    cum_jy = np.concatenate(([0.0], np.cumsum(j_idx * y)))

    N = period
    mean_x = (N - 1) / 2.0
    sum_xx_centered = N * (N * N - 1) / 12.0  # Σ(x - mean_x)² for x=0..N-1

    for i in range(N - 1, n):
        a = i - N + 1
        b = i + 1
        sum_y = cum_y[b] - cum_y[a]
        sum_y2 = cum_y2[b] - cum_y2[a]
        sum_jy = cum_jy[b] - cum_jy[a]
        # In the window, the x values are 0..N-1; the underlying j-indices
        # are a..i.  sum_xy (x = j - a) = Σ(j - a)y_j = sum_jy - a * sum_y.
        sum_xy = sum_jy - a * sum_y
        mean_y = sum_y / N
        sum_xy_centered = sum_xy - N * mean_x * mean_y
        slope = sum_xy_centered / sum_xx_centered
        intercept = mean_y - slope * mean_x
        # Residual sum of squares = Σ y² − slope·sum_xy_uncentered − intercept·sum_y
        ss_res = sum_y2 - slope * sum_xy - intercept * sum_y
        # Numerical floor.
        if ss_res < 0:
            ss_res = 0.0
        se = np.sqrt(ss_res / (N - 2)) if N > 2 else np.nan
        reg_end = intercept + slope * (N - 1)
        out_line[i] = reg_end
        out_slope[i] = slope
        out_se[i] = se
    return out_line, out_slope, out_se


class LinearRegressionChannelStrategy:
    """Linear regression channel — dual-mode (trend or reversion).

    Mode selection (when ``mode="auto"``):
      * ``|slope_norm| >= slope_threshold`` → trend mode (breakout in slope dir).
      * ``|slope_norm| < slope_threshold`` → reversion mode (touch outer band).

    Where ``slope_norm = slope × lr_period / ATR`` (number of ATR units the
    regression line traverses over one full window).

    Trend mode entries:
      * Long: slope_norm ≥ threshold AND close > upper_band AND prev close
        was inside the channel.
      * Short: slope_norm ≤ -threshold AND close < lower_band AND prev close
        was inside the channel.

    Reversion mode entries:
      * Long: |slope_norm| < threshold AND close ≤ lower_band.
      * Short: |slope_norm| < threshold AND close ≥ upper_band.

    Exit (via Backtester):
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite signal reverses the position.
    """

    def __init__(self, params: LinearRegressionChannelParams | None = None):
        self.params = params or LinearRegressionChannelParams()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        p = self.params
        out = df.copy()

        prev_close = out["close"].shift()
        tr = pd.concat(
            [
                out["high"] - out["low"],
                (out["high"] - prev_close).abs(),
                (out["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        out["atr"] = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()

        reg, slope, se = _rolling_ols_channel(out["close"].to_numpy(), p.lr_period)
        out["reg_line"] = reg
        out["reg_slope"] = slope
        out["reg_se"] = se
        out["upper_band"] = out["reg_line"] + p.lr_mult * out["reg_se"]
        out["lower_band"] = out["reg_line"] - p.lr_mult * out["reg_se"]
        # Normalized slope: ATR units traversed per window.
        with np.errstate(divide="ignore", invalid="ignore"):
            out["slope_norm"] = slope * p.lr_period / out["atr"].to_numpy()
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        close = ind["close"]
        prev_close = close.shift()
        upper = ind["upper_band"]
        lower = ind["lower_band"]
        sn = ind["slope_norm"]

        trending = sn.abs() >= p.slope_threshold
        flat = sn.abs() < p.slope_threshold

        mode = p.mode.lower()
        if mode == "trend":
            allow_trend = True
            allow_rev = False
        elif mode == "reversion":
            allow_trend = False
            allow_rev = True
        else:
            allow_trend = True
            allow_rev = True

        long_trend = (
            allow_trend
            & trending
            & (sn > 0)
            & (close > upper)
            & (prev_close <= upper.shift(1))
        )
        short_trend = (
            allow_trend
            & trending
            & (sn < 0)
            & (close < lower)
            & (prev_close >= lower.shift(1))
        )
        long_rev = allow_rev & flat & (close <= lower)
        short_rev = allow_rev & flat & (close >= upper)

        long_sig = long_trend | long_rev
        short_sig = short_trend | short_rev
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        nan_guard = ind["atr"].isna() | ind["reg_line"].isna() | ind["slope_norm"].isna()
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
