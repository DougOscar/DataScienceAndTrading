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
class KeltnerReversionParams:
    """Keltner Channel mean-reversion parameters.

    Two ATR periods are tracked:
      * ``atr_period`` — feeds the Backtester's SL/TP placement (the ``atr``
        column).  Same role as in other strategies.
      * ``kc_atr_period`` — Keltner ATR for the channel width (often shorter
        / more reactive than the stop ATR).
    """

    kc_period: int = 20
    kc_atr_period: int = 10
    kc_mult: float = 2.0
    atr_period: int = 14
    sl_atr_mult: float = 2.5
    # Approximates the EMA-touch take-profit: at entry the distance from the
    # outer band to the EMA is roughly ``kc_mult × ATR_kc``.  Using
    # ``tp_atr_mult ≈ kc_mult`` produces a similar effective RR.
    tp_atr_mult: float = 2.0
    adx_period: int = 14
    adx_max: float = 25.0
    use_adx_filter: bool = True
class EMARibbonParams:
    """4-EMA ribbon with RSI gate parameters."""

    ema_p1: int = 5
    ema_p2: int = 8
    ema_p3: int = 13
    ema_p4: int = 21
    rsi_period: int = 14
    rsi_overbought: float = 65.0
    rsi_oversold: float = 35.0
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    # Ribbon collapse exit (close-on-de-alignment) cannot be expressed cleanly
    # through the existing Backtester's signal/SL/TP/reversal contract — leave
    # disabled.
    use_ribbon_exit: bool = False
class DonchianBreakoutParams:
    """Donchian channel breakout (Turtle-style) parameters."""

    dc_entry: int = 20
    dc_exit: int = 10
    atr_period: int = 14
    sl_atr_mult: float = 3.0
    # Donchian's primary trailing stop is the dc_exit channel; ATR-based TP is
    # disabled by default per the doc.  An effectively-infinite tp_atr_mult
    # keeps the Backtester's TP logic inert.
    tp_atr_mult: float = 100.0
    use_tp: bool = False
class MACDHistogramParams:
    """MACD histogram momentum strategy parameters."""

    macd_fast: int = 12
    macd_slow: int = 26
    signal_period: int = 9
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.5
    vol_period: int = 20
    vol_ratio_min: float = 1.3
    use_vol_filter: bool = False
    # Deceleration exit and MACD zero-line gate cannot be expressed cleanly
    # through the existing Backtester's signal/SL/TP/reversal contract.
    use_decel_exit: bool = False
    use_zero_line_gate: bool = False
class BBSqueezeParams:
    """Bollinger Band squeeze breakout parameters.

    Field names follow the contract the Backtester reads (sl_atr_mult,
    tp_atr_mult, session_start, session_end, sizing_mode, risk_fraction).
    """

    bb_period: int = 20
    bb_mult: float = 2.0
    squeeze_lookback: int = 20
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    vol_period: int = 20
    vol_ratio_min: float = 1.5
    use_vol_filter: bool = False
    # Band-re-entry exit (close back inside the broken band) is described in the
    # doc but cannot be expressed cleanly through the existing Backtester's
    # signal/SL/TP/reversal contract — leave disabled until a custom-exit hook
    # is added.
    use_band_reentry_exit: bool = False
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    # Note: the constraint dc_exit < dc_entry is *not* enforced in
    # __post_init__ because the WFO driver iterates the full grid via
    # itertools.product and would crash on every invalid combo.  Invalid
    # configs simply produce zero signals (see DonchianBreakoutStrategy) so the
    # WFO score is -inf and the combo is effectively skipped.
    def __post_init__(self):
        if not self.macd_fast < self.macd_slow:
            raise ValueError(
                f"Constraint macd_fast < macd_slow violated "
                f"(got macd_fast={self.macd_fast}, macd_slow={self.macd_slow})"
            )

    def as_dict(self) -> dict:
        return asdict(self)


class KeltnerReversionStrategy:
    """Keltner Channel mean-reversion with ATR-based stop + EMA-target TP.

    Entry:
      * Long when ``close <= KC_lower`` (price touches/exceeds lower band).
      * Short when ``close >= KC_upper``.
    Exit (via Backtester):
      * Hard SL at ``entry ± sl_atr_mult × ATR_entry`` (stop ATR).
      * Hard TP at ``entry ± tp_atr_mult × ATR_entry`` — an ATR-based proxy
        for the doc's dynamic "EMA touch" target.
      * Opposite band touch reverses the position.
    Filters:
      * ADX regime filter (on by default per the doc): only enter when
        ``ADX < adx_max`` so we don't fight strong trends.
    """

    def __init__(self, params: KeltnerReversionParams | None = None):
        self.params = params or KeltnerReversionParams()

    def _wilder_ema(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
class EMARibbonStrategy:
    """EMA ribbon (4 EMAs) trend strategy with RSI trend filter.

    Entry:
      * Long on a fresh bullish ribbon alignment (``ema1 > ema2 > ema3 > ema4``
        on bar ``t`` but not on bar ``t-1``) provided ``RSI < rsi_overbought``.
      * Short on a fresh bearish alignment (``ema1 < ema2 < ema3 < ema4``)
        provided ``RSI > rsi_oversold``.
    Exit:
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite ribbon-alignment signal reverses the position.
    Notes:
      * Invalid `ema_p1 < ema_p2 < ema_p3 < ema_p4` combos emit zero signals
        (rather than raising at construction) so the WFO driver can iterate
        the full grid.
    """

    def __init__(self, params: EMARibbonParams | None = None):
        self.params = params or EMARibbonParams()

    def _wilder_ema(self, s: pd.Series, period: int) -> pd.Series:
        return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
class DonchianBreakoutStrategy:
    """Donchian channel breakout with ATR catastrophic stop.

    Entry:
      * Long when ``close > DC_high_entry`` (prior-bar N-period high).
      * Short when ``close < DC_low_entry`` (prior-bar N-period low).
    Exit:
      * Donchian trailing stop: when long and ``close < DC_low_exit`` (tighter
        N2-period prior low), flip to short.  Mirror for short → long.  The
        Backtester's signal-reversal logic implements this directly.
      * Catastrophic ATR stop at ``entry ± sl_atr_mult × ATR_entry``.
      * ATR-based TP is disabled by default (``tp_atr_mult=100`` — Backtester
        will never hit it); flip to a tighter value to re-enable.
    """

    def __init__(self, params: DonchianBreakoutParams | None = None):
        self.params = params or DonchianBreakoutParams()
class MACDHistogramStrategy:
    """MACD histogram zero-line crossover with ATR-based stop loss / take profit.

    Entry:
      * Long when ``Histogram > 0`` and was non-positive on the previous bar
        (fresh MACD line / signal line crossover from below).
      * Short when ``Histogram < 0`` and was non-negative on the previous bar.
    Exit:
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite zero-line crossover reverses the position.
    Filters:
      * Optional tick-volume confirmation (off by default).
      * Optional MACD zero-line gate (off — would need a separate exit hook).
    """

    def __init__(self, params: MACDHistogramParams | None = None):
        self.params = params or MACDHistogramParams()
class BBSqueezeStrategy:
    """Bollinger Band squeeze breakout with ATR-based stop loss / take profit.

    Entry:
      * Long when ``close > BB_upper`` and a squeeze (BB Width at its
        ``squeeze_lookback`` rolling minimum) was active on bar ``t-1`` or
        ``t-2`` — i.e. volatility was compressed and price is now breaking out.
      * Short when ``close < BB_lower`` with the same squeeze recency rule.
    Exit:
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite breakout signal reverses the position.
    Filters:
      * Optional volume confirmation: ``tick_vol / SMA(tick_vol) >= vol_ratio_min``.
    """

    def __init__(self, params: BBSqueezeParams | None = None):
        self.params = params or BBSqueezeParams()

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
        # Stop ATR — used by the Backtester for SL/TP placement.
        out["atr"] = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()
        # Keltner ATR — used only for the channel width.
        out["kc_atr"] = tr.rolling(p.kc_atr_period, min_periods=p.kc_atr_period).mean()

        out["kc_mid"] = out["close"].ewm(
            span=max(2, p.kc_period), adjust=False, min_periods=p.kc_period
        ).mean()
        out["kc_upper"] = out["kc_mid"] + p.kc_mult * out["kc_atr"]
        out["kc_lower"] = out["kc_mid"] - p.kc_mult * out["kc_atr"]

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
        out["atr"] = tr.rolling(p.atr_period, min_periods=p.atr_period).mean()

        for tag, period in (
            ("ema1", p.ema_p1),
            ("ema2", p.ema_p2),
            ("ema3", p.ema_p3),
            ("ema4", p.ema_p4),
        ):
            out[tag] = out["close"].ewm(
                span=max(2, period), adjust=False, min_periods=period
            ).mean()

        # RSI (Wilder).
        delta = out["close"].diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = self._wilder_ema(gain, p.rsi_period)
        avg_loss = self._wilder_ema(loss, p.rsi_period)
        with np.errstate(divide="ignore", invalid="ignore"):
            rs = avg_gain / avg_loss
            out["rsi"] = 100.0 - 100.0 / (1.0 + rs)
        # Donchian channels — prior bars only (shift by 1 to exclude bar t).
        prior_high = out["high"].shift(1)
        prior_low = out["low"].shift(1)
        out["dc_high_entry"] = prior_high.rolling(
            p.dc_entry, min_periods=p.dc_entry
        ).max()
        out["dc_low_entry"] = prior_low.rolling(
            p.dc_entry, min_periods=p.dc_entry
        ).min()
        out["dc_high_exit"] = prior_high.rolling(
            p.dc_exit, min_periods=p.dc_exit
        ).max()
        out["dc_low_exit"] = prior_low.rolling(
            p.dc_exit, min_periods=p.dc_exit
        ).min()
        ema_fast = out["close"].ewm(
            span=p.macd_fast, adjust=False, min_periods=p.macd_fast
        ).mean()
        ema_slow = out["close"].ewm(
            span=p.macd_slow, adjust=False, min_periods=p.macd_slow
        ).mean()
        out["macd"] = ema_fast - ema_slow
        out["macd_signal"] = out["macd"].ewm(
            span=p.signal_period, adjust=False, min_periods=p.signal_period
        ).mean()
        out["macd_hist"] = out["macd"] - out["macd_signal"]
        mid = out["close"].rolling(p.bb_period, min_periods=p.bb_period).mean()
        std = out["close"].rolling(p.bb_period, min_periods=p.bb_period).std(ddof=1)
        out["bb_mid"] = mid
        out["bb_upper"] = mid + p.bb_mult * std
        out["bb_lower"] = mid - p.bb_mult * std
        with np.errstate(divide="ignore", invalid="ignore"):
            out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / mid

        sq_min = out["bb_width"].rolling(
            p.squeeze_lookback, min_periods=p.squeeze_lookback
        ).min()
        out["squeeze"] = (out["bb_width"] <= sq_min).astype(int)

        if p.use_vol_filter and "tick_vol" in out.columns:
            vsma = out["tick_vol"].rolling(p.vol_period, min_periods=p.vol_period).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                out["vol_ratio"] = out["tick_vol"] / vsma
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        long_touch = ind["close"] <= ind["kc_lower"]
        short_touch = ind["close"] >= ind["kc_upper"]
        signal = np.where(long_touch, 1, np.where(short_touch, -1, 0))

        if p.use_adx_filter and "adx" in ind.columns:
            allow = (ind["adx"] < p.adx_max).fillna(False).to_numpy()
        # Invalid ordering → emit no signals (so WFO scores it -inf).
        if not (p.ema_p1 < p.ema_p2 < p.ema_p3 < p.ema_p4):
            ind["signal"] = np.zeros(len(ind), dtype=int)
            return ind

        bull = (ind["ema1"] > ind["ema2"]) & (ind["ema2"] > ind["ema3"]) & (ind["ema3"] > ind["ema4"])
        bear = (ind["ema1"] < ind["ema2"]) & (ind["ema2"] < ind["ema3"]) & (ind["ema3"] < ind["ema4"])

        fresh_bull = bull & ~bull.shift(1).fillna(False)
        fresh_bear = bear & ~bear.shift(1).fillna(False)

        long_sig = fresh_bull & (ind["rsi"] < p.rsi_overbought)
        short_sig = fresh_bear & (ind["rsi"] > p.rsi_oversold)
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))
        # Invalid combo (dc_exit >= dc_entry) → emit no signals so WFO scores
        # it -inf and moves on.
        if p.dc_exit >= p.dc_entry:
            ind["signal"] = np.zeros(len(ind), dtype=int)
            return ind

        close = ind["close"]
        long_entry = close > ind["dc_high_entry"]
        short_entry = close < ind["dc_low_entry"]

        # Trend state derived purely from price vs. the wider entry channel so
        # the dc_exit reversal only fires when we're "in trend" — flat-state
        # spurious flips are filtered out.
        raw_trend = np.where(long_entry, 1, np.where(short_entry, -1, np.nan))
        trend_state = pd.Series(raw_trend, index=ind.index).ffill().fillna(0)
        trend_prev = trend_state.shift(1).fillna(0)

        long_exit_to_short = (close < ind["dc_low_exit"]) & (trend_prev == 1)
        short_exit_to_long = (close > ind["dc_high_exit"]) & (trend_prev == -1)

        long_sig = long_entry | short_exit_to_long
        short_sig = short_entry | long_exit_to_short
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))
        hist = ind["macd_hist"]
        prev_hist = hist.shift()
        long_cross = (hist > 0) & (prev_hist <= 0)
        short_cross = (hist < 0) & (prev_hist >= 0)
        signal = np.where(long_cross, 1, np.where(short_cross, -1, 0))

        if p.use_zero_line_gate:
            allow_long = (ind["macd"] > 0).fillna(False).to_numpy()
            allow_short = (ind["macd"] < 0).fillna(False).to_numpy()
            signal = np.where(
                (signal == 1) & ~allow_long, 0,
                np.where((signal == -1) & ~allow_short, 0, signal),
            )
        sq_recent = (ind["squeeze"].shift(1).fillna(0) > 0) | (
            ind["squeeze"].shift(2).fillna(0) > 0
        )
        long_breakout = (ind["close"] > ind["bb_upper"]) & sq_recent
        short_breakout = (ind["close"] < ind["bb_lower"]) & sq_recent
        signal = np.where(long_breakout, 1, np.where(short_breakout, -1, 0))

        if p.use_vol_filter and "vol_ratio" in ind.columns:
            allow = (ind["vol_ratio"] >= p.vol_ratio_min).fillna(False).to_numpy()
            signal = np.where(allow, signal, 0)

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        nan_guard = (
            ind["atr"].isna()
            | ind["kc_upper"].isna()
            | ind["kc_lower"].isna()
        )
            | ind["ema4"].isna()
            | ind["rsi"].isna()
        )
            ind["atr"].isna() | ind["dc_high_entry"].isna() | ind["dc_low_entry"].isna()
        )
        nan_guard = ind["atr"].isna() | ind["macd_hist"].isna()
        nan_guard = ind["atr"].isna() | ind["bb_upper"].isna() | ind["bb_width"].isna()
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
