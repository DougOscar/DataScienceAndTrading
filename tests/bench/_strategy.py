"""Benchmark-only SMA-cross + ATR stop/target strategy (vectorised Polars).

This is **not** a quantlab strategy (Phase 0 ships ``data``/``engine``/``costs``/
``sizing``/``ledger`` only -- strategies are strategy-engineer's job in later
phases). It exists purely so the performance-engineer benchmarks in this
directory have a realistic, non-trivial signal frame -- ATR-based stop/target,
a real cross condition -- to drive ``run_backtest``/``apply_sizing`` with,
instead of timing an engine that is only ever fed trivial all-flat signals.

Vectorised end-to-end (rolling means/ATR + shift), so it is causal by
construction: every output at row ``i`` depends only on rows ``<= i``.
"""
from __future__ import annotations

import polars as pl


def sma_cross_atr_signals(
    bars: pl.DataFrame,
    *,
    fast: int = 20,
    slow: int = 50,
    atr_period: int = 14,
    atr_mult_stop: float = 2.0,
    atr_mult_target: float = 3.0,
) -> pl.DataFrame:
    """Long when the fast SMA crosses above the slow SMA, short on cross below.

    Stop/target distances are ``ATR(atr_period) * multiplier`` in price units
    (``contracts.SIGNAL_COLUMNS``). Rolling windows use ``min_periods`` equal
    to the window (no partial windows), so warm-up rows carry a null ATR ->
    null signal, which is causal and matches ``contracts.py``'s "signal on row
    i is known at its close" rule. ``fast``/``slow`` need not both be warm at
    the same row: before ``slow`` has enough history the cross state is held
    at a constant "undetermined" (0) value on both sides of the comparison, so
    no spurious entry fires until a real cross is observed.
    """
    high = pl.col("high")
    low = pl.col("low")
    close = pl.col("close")
    prev_close = close.shift(1)
    tr = pl.max_horizontal(
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    )
    atr = tr.rolling_mean(atr_period, min_periods=atr_period)

    fast_sma = close.rolling_mean(fast, min_periods=fast)
    slow_sma = close.rolling_mean(slow, min_periods=slow)
    raw_state = (
        pl.when(fast_sma > slow_sma).then(1)
        .when(fast_sma < slow_sma).then(-1)
        .otherwise(0)
    )

    out = bars.select(
        raw_state.alias("_state"),
        atr.alias("_atr"),
    ).with_columns(
        pl.col("_state").shift(1).alias("_prev_state"),
    ).select(
        pl.when(pl.col("_atr").is_null())
        .then(None)
        .when(pl.col("_state") != pl.col("_prev_state"))
        .then(pl.col("_state"))
        .otherwise(None)
        .cast(pl.Int8)
        .alias("signal"),
        (pl.col("_atr") * atr_mult_stop).alias("stop_dist"),
        (pl.col("_atr") * atr_mult_target).alias("target_dist"),
    )
    return out
