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
class StochasticDivergenceParams:
    """Stochastic oscillator strategy parameters (crossover mode).

    Note: only ``entry_mode="crossover"`` is implemented in v1.  The
    divergence mode requires programmatic swing-high / swing-low detection
    (rolling argmax/argmin pivots) which is left for a follow-up.
    """

    k_period: int = 14
    d_period: int = 3
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.0
    entry_mode: str = "crossover"
    use_d_filter: bool = False
    # Mid-line and divergence exits cannot be expressed cleanly through the
    # existing Backtester's contract.
    use_midline_exit: bool = False
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

class HMMRegimeFilterParams:
    """HMM regime-filter parameters.

    NOTE: the regime model is **sklearn's GaussianMixture**, not a full HMM —
    the doc calls for a Gaussian HMM with hmmlearn but that dependency isn't
    in requirements.txt and rolling-window retraining is not supported by the
    existing Backtester.  GMM gives ``P(state | obs)`` directly (no transition
    matrix) and is fit **once** on the first ``hmm_train_window`` bars per
    backtest call (so each WFO fold gets its own fit).  See the notebook for
    a full discussion of the simplification.
    """

    n_states: int = 3
    hmm_train_window: int = 504
    hmm_prob_threshold: float = 0.70
    hmm_exit_threshold: float = 0.40
    rv_window: int = 20
    vol_period: int = 20
    sma_fast: int = 20
    sma_slow: int = 50
    atr_period: int = 14
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.5
    # HMM-regime-driven exit cannot be expressed cleanly through the existing
    # Backtester's signal/SL/TP/reversal contract — leave disabled.
    use_hmm_exit: bool = False
class DualThrustParams:
    """Dual Thrust intraday breakout parameters."""

    dt_lookback: int = 4
    k1: float = 0.5
    k2: float = 0.5
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 2.5
    # Session window — required for daily aggregation and forced session-end
    # close.  Defaults match B3 (09:00–18:00 BRT).
    session_start: int = 9
    session_end: int = 18
    # Minutes before session end during which new entries are suppressed.
    entry_cutoff_minutes: int = 30
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

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
class StochasticDivergenceStrategy:
    """Stochastic %K oversold/overbought crossover with ATR stops.

    Entry (crossover mode):
      * Long when %K crosses **up** through ``stoch_oversold``.
      * Short when %K crosses **down** through ``stoch_overbought``.
      * Optional %D confirmation: %D < 50 for longs / %D > 50 for shorts.

    Exit (via Backtester):
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Opposite extreme crossover reverses the position.
    """

    def __init__(self, params: StochasticDivergenceParams | None = None):
        self.params = params or StochasticDivergenceParams()
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
class HMMRegimeFilterStrategy:
    """SMA-crossover trend strategy gated on a Gaussian-Mixture regime model.

    Observation features per bar:
      * log_return
      * realized_vol (rolling std of log_return over ``rv_window``)
      * normalized_volume (``tick_vol / SMA(tick_vol)``) — used only when
        ``tick_vol`` is present; otherwise two features.

    Regime model: ``sklearn.mixture.GaussianMixture`` with ``n_states``
    components, fit on the first ``hmm_train_window`` bars of the input.
    States are labeled by ascending mean of the log_return feature:

      * lowest mean → "bear"
      * highest mean → "bull"
      * any middle → "neutral"

    Entry:
      * Long on a fresh SMA golden cross **and** P(bull) ≥ ``hmm_prob_threshold``.
      * Short on a fresh SMA death cross **and** P(bear) ≥ ``hmm_prob_threshold``.

    Exit via Backtester: SL/TP + signal reversal.  Periodic GMM retraining is
    not implemented (the model is fit once per backtest call); each WFO fold
    therefore gets its own model fit, so on a 5-fold WFO the model is
    effectively re-trained 5 times.
    """

    def __init__(self, params: HMMRegimeFilterParams | None = None):
        self.params = params or HMMRegimeFilterParams()

    def _features(self, out: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, valid_mask) for the GMM."""
        p = self.params
        log_ret = np.log(out["close"]).diff()
        realized_vol = log_ret.rolling(p.rv_window, min_periods=p.rv_window).std(ddof=0)
        cols = [log_ret, realized_vol]
        if "tick_vol" in out.columns:
            vsma = out["tick_vol"].rolling(p.vol_period, min_periods=p.vol_period).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                norm_vol = out["tick_vol"] / vsma
            cols.append(norm_vol)
        X = np.column_stack([c.to_numpy() for c in cols])
        valid = ~np.any(np.isnan(X), axis=1) & ~np.any(np.isinf(X), axis=1)
        return X, valid
class DualThrustStrategy:
    """Intraday Dual Thrust breakout.

    Each session's upper/lower trigger is computed from the prior
    ``dt_lookback`` sessions' OHLC via the ``max(HH-LC, HC-LL)`` range
    definition, then shifted by today's session open ± ``k × Range``.

    Entry (intraday bar ``t``):
      * Long when ``close(t) > Upper_trigger(t)`` and we're within the session
        window minus the entry cutoff.
      * Short when ``close(t) < Lower_trigger(t)`` (same time window).

    Exit (via Backtester):
      * Hard SL/TP at ``entry ± atr_mult × ATR_entry``.
      * Session-end forced close (Backtester reads ``session_start`` /
        ``session_end`` from params).
      * Opposite trigger reverses the position via signal reversal.
    """

    def __init__(self, params: DualThrustParams | None = None):
        self.params = params or DualThrustParams()
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
        lowest_low = out["low"].rolling(p.k_period, min_periods=p.k_period).min()
        highest_high = out["high"].rolling(p.k_period, min_periods=p.k_period).max()
        denom = highest_high - lowest_low
        with np.errstate(divide="ignore", invalid="ignore"):
            k = 100.0 * (out["close"] - lowest_low) / denom
        out["stoch_k"] = k.where(denom > 0, 50.0)
        out["stoch_d"] = out["stoch_k"].rolling(
            p.d_period, min_periods=p.d_period
        ).mean()
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
        out["sma_fast"] = out["close"].rolling(p.sma_fast, min_periods=p.sma_fast).mean()
        out["sma_slow"] = out["close"].rolling(p.sma_slow, min_periods=p.sma_slow).mean()

        out["p_bull"] = np.nan
        out["p_bear"] = np.nan
        out["regime"] = ""

        X, valid = self._features(out)
        train_end = p.hmm_train_window
        valid_train = valid[:train_end].sum()
        if valid_train < max(50, p.n_states * 10):
            return out

        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            return out

        X_train = X[:train_end][valid[:train_end]]
        try:
            gm = GaussianMixture(
                n_components=p.n_states,
                covariance_type="full",
                random_state=42,
                max_iter=200,
                reg_covar=1e-5,
            ).fit(X_train)
        except Exception:
            return out

        # Predict probabilities on the *full* valid history. The model is fit
        # on the warm-up window; everything after is true out-of-sample.
        X_valid = X[valid]
        proba = gm.predict_proba(X_valid)

        # Label states by ascending log-return mean (column 0).
        order = np.argsort(gm.means_[:, 0])
        bear_idx, bull_idx = int(order[0]), int(order[-1])

        p_bull_full = np.full(len(out), np.nan)
        p_bear_full = np.full(len(out), np.nan)
        p_bull_full[valid] = proba[:, bull_idx]
        p_bear_full[valid] = proba[:, bear_idx]
        out["p_bull"] = p_bull_full
        out["p_bear"] = p_bear_full

        # In-session mask — daily aggregates are computed only on in-session bars.
        hour = out.index.hour
        in_session = (hour >= p.session_start) & (hour < p.session_end)
        session_bars = out[in_session].copy()

        if session_bars.empty:
            out["upper_trigger"] = np.nan
            out["lower_trigger"] = np.nan
            return out

        # Daily OHLC from in-session bars.
        date_key = session_bars.index.normalize()
        daily = pd.DataFrame(
            {
                "open": session_bars["open"].groupby(date_key).first(),
                "high": session_bars["high"].groupby(date_key).max(),
                "low": session_bars["low"].groupby(date_key).min(),
                "close": session_bars["close"].groupby(date_key).last(),
            }
        ).sort_index()

        # Range from the prior dt_lookback sessions (shift by 1 to exclude
        # today and the *current* row's close from look-ahead).
        prior_high = daily["high"].rolling(p.dt_lookback).max().shift(1)
        prior_low = daily["low"].rolling(p.dt_lookback).min().shift(1)
        prior_close_high = daily["close"].rolling(p.dt_lookback).max().shift(1)
        prior_close_low = daily["close"].rolling(p.dt_lookback).min().shift(1)
        range_today = np.maximum(
            prior_high - prior_close_low, prior_close_high - prior_low
        )
        daily["upper_trigger"] = daily["open"] + p.k1 * range_today
        daily["lower_trigger"] = daily["open"] - p.k2 * range_today

        # Broadcast daily triggers back to every intraday bar.
        intraday_dates = pd.Index(out.index.normalize())
        out["upper_trigger"] = daily["upper_trigger"].reindex(intraday_dates).to_numpy()
        out["lower_trigger"] = daily["lower_trigger"].reindex(intraday_dates).to_numpy()
        out["sma_fast"] = out["close"].rolling(p.trend_fast, min_periods=p.trend_fast).mean()
        out["sma_slow"] = out["close"].rolling(p.trend_slow, min_periods=p.trend_slow).mean()

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

        out["hurst"] = _rolling_hurst(
            out["close"], p.hurst_window, p.hurst_n_min, max(1, p.hurst_step)
        )
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
        if p.entry_mode != "crossover":
            # Divergence mode is documented but not implemented in v1.
            ind["signal"] = np.zeros(len(ind), dtype=int)
            return ind

        k = ind["stoch_k"]
        prev_k = k.shift()
        long_cross = (k > p.stoch_oversold) & (prev_k <= p.stoch_oversold)
        short_cross = (k < p.stoch_overbought) & (prev_k >= p.stoch_overbought)
        signal = np.where(long_cross, 1, np.where(short_cross, -1, 0))

        if p.use_d_filter:
            d = ind["stoch_d"]
            allow_long = (d < 50).fillna(False).to_numpy()
            allow_short = (d > 50).fillna(False).to_numpy()
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
        diff = ind["sma_fast"] - ind["sma_slow"]
        prev_diff = diff.shift()
        golden = (diff > 0) & (prev_diff <= 0)
        death = (diff < 0) & (prev_diff >= 0)

        long_sig = golden & (ind["p_bull"] >= p.hmm_prob_threshold)
        short_sig = death & (ind["p_bear"] >= p.hmm_prob_threshold)
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))
        in_session = (ind.index.hour >= p.session_start) & (
            ind.index.hour < p.session_end
        )
        # Entry cutoff: block new entries when we're within entry_cutoff_minutes
        # of the session end.
        minutes_into_day = ind.index.hour * 60 + ind.index.minute
        cutoff_minute = p.session_end * 60 - p.entry_cutoff_minutes
        before_cutoff = minutes_into_day < cutoff_minute

        long_breakout = (ind["close"] > ind["upper_trigger"]) & in_session & before_cutoff
        short_breakout = (
            (ind["close"] < ind["lower_trigger"]) & in_session & before_cutoff
        )
        signal = np.where(long_breakout, 1, np.where(short_breakout, -1, 0))

        nan_guard = (
            ind["atr"].isna() | ind["upper_trigger"].isna() | ind["lower_trigger"].isna()
        )
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

        nan_guard = ind["atr"].isna() | ind["reg_line"].isna() | ind["slope_norm"].isna()
        nan_guard = ind["atr"].isna() | ind["stoch_k"].isna()
        nan_guard = (
            ind["atr"].isna() | ind["sma_slow"].isna() | ind["p_bull"].isna()
        )
            ind["atr"].isna()
            | ind["hurst"].isna()
            | ind["sma_slow"].isna()
            | ind["rsi"].isna()
        )
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
