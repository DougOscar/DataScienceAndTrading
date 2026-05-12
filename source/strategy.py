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
class FFTCycleFilterParams:
    """FFT cycle filter strategy parameters."""

    fft_window: int = 128
    min_cycle_bars: int = 8
    max_cycle_bars: int = 64
    n_harmonics: int = 0
    delta_min_atr: float = 0.1
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    min_power_fraction: float = 0.0
    # Projection-reversal / phase-cycle exits cannot be expressed cleanly
    # through the existing Backtester contract.
    use_projection_exit: bool = False
    use_cycle_exit: bool = False
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


def _fft_projection_delta(
    log_close: np.ndarray,
    hann: np.ndarray,
    f_min: int,
    f_max: int,
    n_harmonics: int,
    min_power_fraction: float,
) -> float:
    """One bar's FFT projection delta on the trailing window of log-prices.

    Returns ``Δ = x_projected − x_filtered[N-1]``.  Returns NaN if the
    `min_power_fraction` floor isn't met (insufficient signal in the band).
    """
    N = len(log_close)
    x_idx = np.arange(N, dtype=float)
    slope, intercept = np.polyfit(x_idx, log_close, 1)
    trend = intercept + slope * x_idx
    x_detrended = log_close - trend

    x_w = x_detrended * hann
    X = np.fft.rfft(x_w)
    n_freqs = len(X)  # N//2 + 1

    keep = np.zeros(n_freqs, dtype=bool)
    lo = max(1, f_min)
    hi = min(n_freqs, f_max + 1)
    if lo < hi:
        keep[lo:hi] = True

    power = (X.real ** 2 + X.imag ** 2)
    total_power = power[1:].sum()
    if total_power <= 0:
        return float("nan")
    in_band_power = power[keep].sum()
    if min_power_fraction > 0 and in_band_power / total_power < min_power_fraction:
        return float("nan")

    if n_harmonics > 0:
        in_band_idx = np.where(keep)[0]
        if len(in_band_idx) > n_harmonics:
            top = in_band_idx[np.argsort(-power[in_band_idx])[:n_harmonics]]
            new_keep = np.zeros_like(keep)
            new_keep[top] = True
            keep = new_keep

    X_filtered = np.where(keep, X, 0.0)

    # In-window reconstruction at the right edge (Hann-distorted; we accept
    # this rather than dividing by hann≈0 at the edge, which is unstable).
    x_recon = np.fft.irfft(X_filtered, n=N)
    last_filtered = x_recon[-1] + trend[-1]

    # Projection at i=N: cos(2π·f·N/N + φ) = cos(2π·f + φ) = cos(φ) for integer f.
    # Real-FFT scaling: 2/N for interior coefficients, 1/N for DC and Nyquist.
    coeffs = np.full(n_freqs, 2.0 / N)
    coeffs[0] = 1.0 / N
    if N % 2 == 0:
        coeffs[-1] = 1.0 / N
    proj_detrended = float(np.sum(coeffs * X_filtered.real))
    trend_N = intercept + slope * N
    x_projected = proj_detrended + trend_N
    return x_projected - last_filtered


def _rolling_fft_delta(
    close: pd.Series,
    fft_window: int,
    min_cycle_bars: int,
    max_cycle_bars: int,
    n_harmonics: int,
    min_power_fraction: float,
) -> pd.Series:
    """Per-bar FFT projection delta on a rolling ``fft_window`` of log-close."""
    N = int(fft_window)
    n = len(close)
    out = np.full(n, np.nan)
    if N < 4 or n < N:
        return pd.Series(out, index=close.index)

    log_close = np.log(close.to_numpy())
    if np.isnan(log_close).any():
        # Replace any NaNs (shouldn't occur in clean OHLC) with forward-fill.
        log_close = pd.Series(log_close).ffill().to_numpy()

    # Pre-compute Hann window of length N once.
    hann = 0.5 * (1.0 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))

    f_min = int(np.floor(N / max_cycle_bars))
    f_max = int(np.ceil(N / min_cycle_bars))

    for i in range(N - 1, n):
        window = log_close[i - N + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = _fft_projection_delta(
            window, hann, f_min, f_max, n_harmonics, min_power_fraction
        )
    return pd.Series(out, index=close.index)


class FFTCycleFilterStrategy:
    """Rolling FFT cycle filter with one-bar forward projection.

    Indicator pipeline (per bar ``t``):
      1. Take the last ``fft_window`` log-closes.
      2. Detrend (linear fit) and Hann-window.
      3. FFT → keep frequencies in the period band
         ``[min_cycle_bars, max_cycle_bars]`` (optionally only the top
         ``n_harmonics`` by power within the band).
      4. Compute one-bar-ahead projection via cosine evaluation of each
         retained harmonic at ``i=N``; add back the linear trend
         extrapolation.  ``Δ(t) = x_projected − x_filtered[N-1]``.

    Entry:
      * Long when ``Δ(t) > delta_min_atr × ATR`` and ``Δ(t-1) ≤ 0``.
      * Short when ``Δ(t) < -delta_min_atr × ATR`` and ``Δ(t-1) ≥ 0``.

    Exit (via Backtester):
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite fresh-shift signal reverses the position.
    """

    def __init__(self, params: FFTCycleFilterParams | None = None):
        self.params = params or FFTCycleFilterParams()

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

        out["fft_delta"] = _rolling_fft_delta(
            out["close"],
            p.fft_window,
            p.min_cycle_bars,
            p.max_cycle_bars,
            p.n_harmonics,
            p.min_power_fraction,
        )
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        # Invalid band → no signals (avoids confusing WFO scores).
        if p.min_cycle_bars >= p.max_cycle_bars or p.max_cycle_bars >= p.fft_window // 2:
            ind["signal"] = np.zeros(len(ind), dtype=int)
            return ind

        # Δ is a *log-price* difference; ATR is a *price* difference. To make
        # them comparable, convert ATR to log-price via dividing by close (an
        # ATR price-change of `atr` is roughly `atr / close` in log space for
        # small moves).
        atr_log = ind["atr"] / ind["close"]
        delta = ind["fft_delta"]
        prev_delta = delta.shift(1)

        thresh = p.delta_min_atr * atr_log
        long_sig = (delta > thresh) & (prev_delta <= 0)
        short_sig = (delta < -thresh) & (prev_delta >= 0)
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        nan_guard = ind["atr"].isna() | ind["fft_delta"].isna()
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
