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
    session_start: int | None = None
    session_end: int | None = None
    sizing_mode: str = "unit"
    risk_fraction: float = 0.01

    def as_dict(self) -> dict:
        return asdict(self)


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
        return out

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        ind = self.compute_indicators(df)
        p = self.params

        diff = ind["sma_fast"] - ind["sma_slow"]
        prev_diff = diff.shift()
        golden = (diff > 0) & (prev_diff <= 0)
        death = (diff < 0) & (prev_diff >= 0)

        long_sig = golden & (ind["p_bull"] >= p.hmm_prob_threshold)
        short_sig = death & (ind["p_bear"] >= p.hmm_prob_threshold)
        signal = np.where(long_sig, 1, np.where(short_sig, -1, 0))

        if p.session_start is not None and p.session_end is not None:
            in_session = (
                (ind.index.hour >= p.session_start)
                & (ind.index.hour < p.session_end)
            )
            signal = np.where(in_session, signal, 0)

        nan_guard = (
            ind["atr"].isna() | ind["sma_slow"].isna() | ind["p_bull"].isna()
        )
        signal = np.where(nan_guard, 0, signal)

        ind["signal"] = signal.astype(int)
        return ind
