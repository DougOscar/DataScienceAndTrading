"""``TradingEnv`` — a Gymnasium environment for discrete long/short trading.

The environment is the *only* place trading semantics live; the agent merely
maps observations to one of three actions. Those semantics are deliberately
identical to :class:`source.backtest.Backtester` so that what the agent
optimises during training matches what the backtest replay reports:

* **Action space** ``Discrete(3)`` → ``{0: go-short, 1: hold, 2: go-long}``.
  ``hold`` keeps the current position (which is why position is *sticky* and
  flat is only reachable at the start of an episode — exactly how the
  Backtester treats a ``signal == 0``). True flat-on-zero is a v2 feature
  (``use_flat_close_signal``) that needs a Backtester exit-hook extension.
* **Reward** at the transition from bar ``t`` to ``t+1`` is the log-return of
  the held position minus a transaction-cost charge on any position change::

      r = position_t · (log close_{t+1} − log close_t)
          − tx_cost · |position_t − position_{t-1}|

  Log-return units are dimensionless and comparable across assets/timeframes.

* **Observation** is a flattened window of ``obs_window`` bars of z-scored
  features plus two position-state scalars. All normalisation is *online*
  (rolling lookback up to and including the current bar) — no global scaler is
  fit, so no future statistics leak into the training window.

The env holds arrays aligned 1:1 with the input DataFrame's rows, so a bar
pointer ``t`` indexes both the env state and ``df.iloc[t]`` — this is what lets
:func:`source.rl.train.evaluate_policy_to_signals` map actions straight back to
a Backtester ``signal`` column.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import gymnasium as gym
from gymnasium import spaces

if TYPE_CHECKING:  # avoid a hard import cycle / heavy import at module load
    from ..strategy import DeepRLTradingParams


# Order matters: the observation flattens these columns per bar.
FEATURE_COLUMNS = [
    "log_return",
    "realized_vol",
    "rsi",
    "atr_rel",
    "volume_ratio",
    "bb_pos",
]

# Discrete action index -> target position / Backtester signal.
ACTION_TO_SIGNAL = {0: -1, 1: 0, 2: 1}


def compute_feature_panel(df: pd.DataFrame, params: "DeepRLTradingParams") -> pd.DataFrame:
    """Compute the raw (un-normalised) feature panel for one OHLCV frame.

    Returns a frame indexed like ``df`` with ``close`` (kept raw, for reward)
    plus the six :data:`FEATURE_COLUMNS`. Volume falls back to a constant when
    the source lacks a ``tick_vol`` column so the env still runs.
    """
    out = pd.DataFrame(index=df.index)
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    out["close"] = close

    # 1-bar log return
    out["log_return"] = np.log(close / close.shift(1))

    # realised volatility (rolling std of log returns)
    out["realized_vol"] = out["log_return"].rolling(
        params.rv_window, min_periods=params.rv_window
    ).std()

    # Wilder RSI, scaled to [0, 1]
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out["rsi"] = (100.0 - 100.0 / (1.0 + rs)) / 100.0

    # ATR (simple mean of True Range), expressed relative to price (unit-free)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(params.atr_period, min_periods=params.atr_period).mean()
    out["atr_rel"] = atr / close

    # relative volume
    if "tick_vol" in df.columns:
        tick = df["tick_vol"].astype(float)
    elif "volume" in df.columns:
        tick = df["volume"].astype(float)
    else:
        tick = pd.Series(1.0, index=df.index)
    vol_sma = tick.rolling(params.vol_period, min_periods=params.vol_period).mean()
    out["volume_ratio"] = tick / vol_sma.replace(0.0, np.nan)

    # Bollinger band position in [-1, +1]-ish band units
    bb_mid = close.rolling(params.bb_period, min_periods=params.bb_period).mean()
    bb_std = close.rolling(params.bb_period, min_periods=params.bb_period).std()
    denom = (params.bb_mult * bb_std).replace(0.0, np.nan)
    out["bb_pos"] = (close - bb_mid) / denom

    return out


def _rolling_zscore(panel: pd.DataFrame, window: int) -> pd.DataFrame:
    """Causal rolling z-score of each feature column (no look-ahead)."""
    z = pd.DataFrame(index=panel.index)
    for col in FEATURE_COLUMNS:
        s = panel[col]
        mean = s.rolling(window, min_periods=window).mean()
        std = s.rolling(window, min_periods=window).std()
        z[col] = (s - mean) / std.where(std > 0, 1.0)
    return z


class TradingEnv(gym.Env):
    """Single-asset discrete long/short trading environment.

    Parameters
    ----------
    df:
        OHLCV frame (datetime index; columns ``open/high/low/close`` and
        ideally ``tick_vol``).
    params:
        :class:`source.strategy.DeepRLTradingParams`.
    mode:
        ``"train"`` picks a random ``episode_len`` slice each ``reset`` and
        applies the drawdown circuit-breaker; ``"eval"`` runs one deterministic
        pass over the whole valid range with the breaker disabled.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, params: "DeepRLTradingParams", mode: str = "train"):
        super().__init__()
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")
        self.params = params
        self.mode = mode

        panel = compute_feature_panel(df, params)
        z = _rolling_zscore(panel, params.rolling_z_window)

        # Raw arrays aligned 1:1 with df rows.
        self._close = panel["close"].to_numpy(dtype=np.float64)
        self._logret = panel["log_return"].to_numpy(dtype=np.float64)
        self._feat = z[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self._hour = df.index.hour.to_numpy() if hasattr(df.index, "hour") else np.zeros(len(df), dtype=int)

        # First bar at which a full obs window of valid (non-NaN) features exists.
        finite_rows = np.isfinite(self._feat).all(axis=1)
        first_valid = int(np.argmax(finite_rows)) if finite_rows.any() else len(df)
        self._valid_start = max(first_valid + params.obs_window, params.obs_window)
        self._n = len(df)
        if self._n - self._valid_start < 2:
            raise ValueError(
                f"Not enough valid bars ({self._n}) after warm-up "
                f"(valid_start={self._valid_start}) to run an episode."
            )

        self.obs_window = params.obs_window
        self.n_features = len(FEATURE_COLUMNS)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_window * self.n_features + 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # episode state (set in reset)
        self.t = self._valid_start
        self._end = self._n - 1
        self.position = 0
        self._prev_position = 0
        self.bars_in_position = 0
        self._equity_mult = 1.0
        self._peak_mult = 1.0

    # ------------------------------------------------------------------ obs
    def _get_obs(self) -> np.ndarray:
        win = self._feat[self.t - self.obs_window + 1 : self.t + 1]  # (obs_window, n_features)
        flat = win.reshape(-1)
        pos_state = np.array(
            [self.position, self.bars_in_position / max(self.params.episode_len, 1)],
            dtype=np.float32,
        )
        return np.concatenate([flat, pos_state]).astype(np.float32)

    def _in_session(self, t: int) -> bool:
        if self.params.session_start is None or self.params.session_end is None:
            return True
        return self.params.session_start <= self._hour[t] < self.params.session_end

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if self.mode == "train":
            hi = self._n - self.params.episode_len - 1
            lo = self._valid_start
            start = int(self.np_random.integers(lo, hi + 1)) if hi > lo else lo
            self.t = start
            self._end = min(start + self.params.episode_len, self._n - 1)
        else:  # eval — full deterministic pass
            self.t = self._valid_start
            self._end = self._n - 1

        self.position = 0
        self._prev_position = 0
        self.bars_in_position = 0
        self._equity_mult = 1.0
        self._peak_mult = 1.0
        return self._get_obs(), {}

    # ----------------------------------------------------------------- step
    def step(self, action):
        action = int(action)
        signal = ACTION_TO_SIGNAL[action]  # -1, 0 (hold), +1

        prev_position = self.position
        if signal == 0:
            new_position = prev_position  # hold
        else:
            new_position = signal

        # B3 session mask: force flat outside the trading window.
        if not self._in_session(self.t):
            new_position = 0

        ret = self._logret[self.t + 1] if self.t + 1 < self._n else 0.0
        if not np.isfinite(ret):
            ret = 0.0
        tx = self.params.tx_cost * abs(new_position - prev_position)
        reward = float(new_position * ret - tx)

        # equity / drawdown bookkeeping (fractional, via exp of cumulative logret)
        self._equity_mult *= float(np.exp(new_position * ret))
        self._peak_mult = max(self._peak_mult, self._equity_mult)
        drawdown = 1.0 - self._equity_mult / self._peak_mult

        if new_position == prev_position and new_position != 0:
            self.bars_in_position += 1
        elif new_position == 0:
            self.bars_in_position = 0
        else:
            self.bars_in_position = 1

        self._prev_position = prev_position
        self.position = new_position
        self.t += 1

        terminated = False
        truncated = self.t >= self._end
        if self.mode == "train" and drawdown > self.params.max_drawdown_fraction:
            terminated = True

        obs = self._get_obs() if self.t <= self._end else self._last_obs_safe()
        info = {
            "position": new_position,
            "signal": signal,
            "drawdown": drawdown,
            "equity_mult": self._equity_mult,
            "bar": self.t,
        }
        return obs, reward, terminated, truncated, info

    def _last_obs_safe(self) -> np.ndarray:
        # Clamp pointer for the terminal obs so slicing never runs off the end.
        self.t = min(self.t, self._n - 1)
        return self._get_obs()

    @property
    def valid_start(self) -> int:
        return self._valid_start
