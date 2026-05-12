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
class VWAPReversionParams:
    """VWAP intraday mean-reversion parameters.

    Sessions are defined by ``session_start`` / ``session_end``; VWAP and its
    bands reset at the start of each new session.
    """

    n_sigma: float = 2.0
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    # The dynamic VWAP-touch TP is approximated by a fixed ATR-based TP
    # (the Backtester only supports fixed-at-entry SL/TP).
    tp_atr_mult: float = 1.5
    vwap_warmup_bars: int = 10
    entry_cutoff_minutes: int = 30
    session_start: int = 9
    session_end: int = 18
    vwap_volume_col: str = "tick_vol"
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


class VWAPReversionStrategy:
    """Session-VWAP mean-reversion (intraday).

    Indicators (per in-session bar):
      * ``VWAP`` — volume-weighted cumulative typical price, resets each day.
      * ``VWAP_std`` — volume-weighted std of typical price around VWAP.
      * ``VWAP_upper = VWAP + n_sigma × VWAP_std``,
        ``VWAP_lower = VWAP − n_sigma × VWAP_std``.

    Entry:
      * Long when ``close <= VWAP_lower`` (past warmup, before entry cutoff).
      * Short when ``close >= VWAP_upper``.

    Exit (via Backtester):
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry`` (TP approximates the
        doc's dynamic VWAP-touch target).
      * Session-end forced close via Backtester's ``session_start`` /
        ``session_end`` handling.
      * Opposite band touch reverses the position.
    """

    def __init__(self, params: VWAPReversionParams | None = None):
        self.params = params or VWAPReversionParams()

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

        in_session_arr = (
            (out.index.hour >= p.session_start) & (out.index.hour < p.session_end)
        )
        if hasattr(in_session_arr, "to_numpy"):
            in_session_arr = in_session_arr.to_numpy()
        # Pick volume column, falling back if absent.
        if p.vwap_volume_col in out.columns:
            vol = out[p.vwap_volume_col].to_numpy(dtype=float)
        elif "tick_vol" in out.columns:
            vol = out["tick_vol"].to_numpy(dtype=float)
        elif "volume" in out.columns:
            vol = out["volume"].to_numpy(dtype=float)
        else:
            vol = np.ones(len(out), dtype=float)
        # Avoid zero/NaN volume dominating; replace with 1 for stable cumsum.
        vol = np.where((vol <= 0) | np.isnan(vol), 1.0, vol)

        tp = (out["high"] + out["low"] + out["close"]).to_numpy() / 3.0

        # Session keys → cumulative resets per day.
        session_id = out.index.normalize().astype("int64").to_numpy()

        # Zero out contributions outside the session.
        v_in = np.where(in_session_arr, vol, 0.0)
        tp_in = np.where(in_session_arr, tp, 0.0)

        df_helper = pd.DataFrame(
            {"v": v_in, "tpv": tp_in * v_in, "tp2v": (tp_in ** 2) * v_in},
            index=out.index,
        )
        cum_v = df_helper["v"].groupby(session_id).cumsum().to_numpy()
        cum_tpv = df_helper["tpv"].groupby(session_id).cumsum().to_numpy()
        cum_tp2v = df_helper["tp2v"].groupby(session_id).cumsum().to_numpy()

        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = cum_tpv / cum_v
            second_moment = cum_tp2v / cum_v
            variance = second_moment - vwap ** 2
            variance = np.where(variance < 0, 0.0, variance)
            vwap_std = np.sqrt(variance)

        # Bar count since session start.
        ones = pd.Series(in_session_arr.astype(int), index=out.index)
        bars_in_session = ones.groupby(session_id).cumsum().to_numpy()

        vwap_full = np.where(in_session_arr, vwap, np.nan)
        std_full = np.where(in_session_arr, vwap_std, np.nan)
        out["vwap"] = vwap_full
        out["vwap_std"] = std_full
        out["vwap_upper"] = vwap_full + p.n_sigma * std_full
        out["vwap_lower"] = vwap_full - p.n_sigma * std_full
        out["session_bar"] = bars_in_session
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        in_session = (ind.index.hour >= p.session_start) & (
            ind.index.hour < p.session_end
        )
        minutes_into_day = ind.index.hour * 60 + ind.index.minute
        cutoff_minute = p.session_end * 60 - p.entry_cutoff_minutes
        before_cutoff = minutes_into_day < cutoff_minute
        warmup_ok = ind["session_bar"] >= p.vwap_warmup_bars

        long_touch = (
            (ind["close"] <= ind["vwap_lower"])
            & in_session
            & before_cutoff
            & warmup_ok
            & (ind["vwap_std"] > 0)
        )
        short_touch = (
            (ind["close"] >= ind["vwap_upper"])
            & in_session
            & before_cutoff
            & warmup_ok
            & (ind["vwap_std"] > 0)
        )
        signal = np.where(long_touch, 1, np.where(short_touch, -1, 0))

        nan_guard = (
            ind["atr"].isna() | ind["vwap_upper"].isna() | ind["vwap_lower"].isna()
        )
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
