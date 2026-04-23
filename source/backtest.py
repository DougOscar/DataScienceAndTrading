"""Event-driven backtester (bar by bar) for the baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry: float
    exit: float
    reason: str
    pnl_points: float
    bars_held: int


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.Series
    signals: pd.DataFrame
    params: dict = field(default_factory=dict)


class Backtester:
    """Bar-close execution with intrabar SL/TP handling.

    Execution rules:
      * Signals are evaluated on bar close of bar `t`.
      * New positions open at that close.
      * Existing positions check SL then TP against the current bar's
        high/low; if both would trigger on the same bar we assume the
        stop triggers first (pessimistic assumption).
      * An opposite signal closes the position at that bar's close and
        opens the reverse position.
    """

    def __init__(self, strategy, slippage_points: float = 0.0):
        self.strategy = strategy
        self.slippage_points = slippage_points

    def run(self, df: pd.DataFrame) -> BacktestResult:
        sig = self.strategy.generate_signals(df)
        p = self.strategy.params

        highs = sig["high"].to_numpy()
        lows = sig["low"].to_numpy()
        closes = sig["close"].to_numpy()
        atrs = sig["atr"].to_numpy()
        signals = sig["signal"].to_numpy()
        index = sig.index

        trades: List[Trade] = []
        equity = np.zeros(len(sig))
        running_pnl = 0.0

        position = 0
        entry_price = 0.0
        entry_time = None
        entry_i = 0
        sl = tp = np.nan
        slip = self.slippage_points

        for i in range(len(sig)):
            close = closes[i]
            high = highs[i]
            low = lows[i]
            sgn = signals[i]
            atr = atrs[i]

            if position != 0:
                exit_price = None
                reason = None
                if position == 1:
                    if low <= sl:
                        exit_price, reason = sl, "SL"
                    elif high >= tp:
                        exit_price, reason = tp, "TP"
                else:
                    if high >= sl:
                        exit_price, reason = sl, "SL"
                    elif low <= tp:
                        exit_price, reason = tp, "TP"

                if exit_price is None and sgn == -position:
                    exit_price, reason = close, "SIGNAL"

                if exit_price is not None:
                    pnl = (exit_price - entry_price) * position - 2 * slip
                    running_pnl += pnl
                    trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=index[i],
                            direction=position,
                            entry=entry_price,
                            exit=exit_price,
                            reason=reason,
                            pnl_points=pnl,
                            bars_held=i - entry_i,
                        )
                    )
                    position = 0

            if position == 0 and sgn != 0 and not np.isnan(atr):
                position = int(sgn)
                entry_price = close
                entry_time = index[i]
                entry_i = i
                sl = entry_price - position * p.sl_atr_mult * atr
                tp = entry_price + position * p.tp_atr_mult * atr

            equity[i] = running_pnl

        if position != 0:
            pnl = (closes[-1] - entry_price) * position - 2 * slip
            running_pnl += pnl
            trades.append(
                Trade(
                    entry_time=entry_time,
                    exit_time=index[-1],
                    direction=position,
                    entry=entry_price,
                    exit=closes[-1],
                    reason="EOD",
                    pnl_points=pnl,
                    bars_held=(len(sig) - 1) - entry_i,
                )
            )
            equity[-1] = running_pnl

        trades_df = pd.DataFrame([asdict(t) for t in trades])
        equity_s = pd.Series(equity, index=index, name="equity_points")
        return BacktestResult(
            trades=trades_df, equity=equity_s, signals=sig, params=p.as_dict()
        )
