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
class HurstRegimeSwitcherParams:
    """Hurst-exponent regime switcher parameters.

    Combines an SMA-crossover sub-strategy (used in persistent / trending
    regimes where ``H > hurst_trend_threshold``) with an RSI mean-reversion
    sub-strategy (used in anti-persistent regimes where
    ``H < hurst_mean_rev_threshold``).  Random-walk regimes generate no
    signals.
    """

    hurst_window: int = 128
    hurst_n_min: int = 8
    hurst_step: int = 4
    hurst_trend_threshold: float = 0.60
    hurst_mean_rev_threshold: float = 0.40
    trend_fast: int = 20
    trend_slow: int = 50
    rsi_period: int = 14
    rsi_lower: float = 30.0
    rsi_upper: float = 70.0
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    # Regime-change exit cannot be expressed cleanly through the existing
    # Backtester's signal/SL/TP/reversal contract — leave disabled.
    use_regime_exit: bool = False
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


def _rs_hurst(window_returns: np.ndarray, n_min: int) -> float:
    """Compute Hurst exponent via R/S analysis on a 1-D return window.

    Returns ``np.nan`` if the window is too short or the fit is degenerate.
    """
    n = len(window_returns)
    if n < max(16, n_min * 2):
        return np.nan
    sizes: list[int] = []
    s = n_min
    while s <= n // 2:
        sizes.append(s)
        s *= 2
    if len(sizes) < 2:
        return np.nan

    log_n: list[float] = []
    log_rs: list[float] = []
    for size in sizes:
        n_sub = n // size
        if n_sub < 1:
            continue
        rs_values: list[float] = []
        for k in range(n_sub):
            chunk = window_returns[k * size : (k + 1) * size]
            mean_r = chunk.mean()
            cum_dev = np.cumsum(chunk - mean_r)
            R = float(cum_dev.max() - cum_dev.min())
            S = float(chunk.std(ddof=0))
            if S > 0:
                rs_values.append(R / S)
        if rs_values:
            log_n.append(np.log(size))
            log_rs.append(np.log(np.mean(rs_values)))
    if len(log_n) < 2:
        return np.nan
    slope, _intercept = np.polyfit(log_n, log_rs, 1)
    return float(slope)


def _rolling_hurst(close: pd.Series, window: int, n_min: int, step: int) -> pd.Series:
    """Rolling Hurst series, recomputed every ``step`` bars and forward-filled."""
    log_ret = np.log(close).diff().to_numpy()
    n = len(log_ret)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window, n, max(1, step)):
        w = log_ret[i - window + 1 : i + 1]
        if np.isnan(w).any():
            continue
        out[i] = _rs_hurst(w, n_min)
    s = pd.Series(out, index=close.index)
    return s.ffill()


class HurstRegimeSwitcherStrategy:
    """Adaptive strategy that switches between SMA trend and RSI mean-rev.

    Entry decision flow (at bar close ``t``):

    1. Estimate ``H(t)`` via R/S analysis on the last ``hurst_window`` log
       returns (recomputed every ``hurst_step`` bars, forward-filled in
       between).
    2. If ``H(t) > hurst_trend_threshold`` (persistent regime), emit a long
       on a fresh SMA golden cross / short on a fresh death cross.
    3. If ``H(t) < hurst_mean_rev_threshold`` (anti-persistent regime), emit
       a long when RSI crosses up through ``rsi_lower`` / short when RSI
       crosses down through ``rsi_upper``.
    4. Otherwise (random-walk regime): no trade.

    Exits via Backtester: SL/TP + signal reversal.
    """

    def __init__(self, params: HurstRegimeSwitcherParams | None = None):
        self.params = params or HurstRegimeSwitcherParams()

    def _wilder_ema(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

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

        out["sma_fast"] = out["close"].rolling(p.trend_fast, min_periods=p.trend_fast).mean()
        out["sma_slow"] = out["close"].rolling(p.trend_slow, min_periods=p.trend_slow).mean()

        delta = out["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = self._wilder_ema(gain, p.rsi_period)
        avg_loss = self._wilder_ema(loss, p.rsi_period)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            out["rsi"] = 100.0 - 100.0 / (1.0 + rs)

        out["hurst"] = _rolling_hurst(
            out["close"], p.hurst_window, p.hurst_n_min, max(1, p.hurst_step)
        )
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        # Trend sub-signal: fresh golden / death cross.
        diff = ind["sma_fast"] - ind["sma_slow"]
        prev_diff = diff.shift()
        trend_long = (diff > 0) & (prev_diff <= 0)
        trend_short = (diff < 0) & (prev_diff >= 0)

        # Mean-reversion sub-signal: RSI cross-back.
        rsi = ind["rsi"]
        prev_rsi = rsi.shift()
        mr_long = (rsi > p.rsi_lower) & (prev_rsi <= p.rsi_lower)
        mr_short = (rsi < p.rsi_upper) & (prev_rsi >= p.rsi_upper)

        trending = ind["hurst"] > p.hurst_trend_threshold
        mean_reverting = ind["hurst"] < p.hurst_mean_rev_threshold

        long_sig = (trending & trend_long) | (mean_reverting & mr_long)
        short_sig = (trending & trend_short) | (mean_reverting & mr_short)
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        nan_guard = (
            ind["atr"].isna()
            | ind["hurst"].isna()
            | ind["sma_slow"].isna()
            | ind["rsi"].isna()
        )
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
