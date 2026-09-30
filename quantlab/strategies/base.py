"""Shared, minimal indicator helpers for system strategy modules (``quantlab/strategies/<slug>.py``).

Kept intentionally small (the strategy-engineer's own charter: "keep the public API small").
The True Range / ATR pattern is extracted here because it was already duplicated, computed the
same way, in two places: ``tests/bench/_strategy.py``'s ``sma_cross_atr_signals`` and
``tests/test_evaluators.py``'s ``SmaCross`` -- exactly the "don't hand-roll one-offs inside a
notebook [or a strategy module]" case the strategy-engineer's Implementation rules call out.
Nothing else lives here; add to this module only when a helper is genuinely shared, not on
spec.
"""

from __future__ import annotations

import polars as pl

__all__ = ["true_range", "atr", "wilder_atr"]


def true_range() -> pl.Expr:
    """True range: ``max(high-low, |high-prev_close|, |low-prev_close|)``.

    A Polars expression (not a materialised column) -- causal by construction (``prev_close``
    is ``close.shift(1)``, so row *i* only ever reads rows ``<= i``), safe to use directly
    inside a strategy's ``signals()``, e.g. ``bars.select(base.true_range().alias("tr"))``.
    """
    high, low, close = pl.col("high"), pl.col("low"), pl.col("close")
    prev_close = close.shift(1)
    return pl.max_horizontal(high - low, (high - prev_close).abs(), (low - prev_close).abs())


def atr(period: int, *, min_periods: int | None = None) -> pl.Expr:
    """Rolling-mean ATR expression over :func:`true_range`.

    ``min_periods`` defaults to ``period`` (no partial windows): warm-up rows are null rather
    than a noisy short-window estimate, matching ``contracts.py``'s "signal known at its close"
    rule -- a strategy that gates its own signal on a null ATR (as every example strategy in
    this repo does) stays causal for free.
    """
    if period < 1:
        raise ValueError(f"atr: period must be >= 1, got {period}")
    return true_range().rolling_mean(period, min_samples=period if min_periods is None else min_periods)


def wilder_atr(period: int) -> pl.Expr:
    """Wilder's Average True Range: MT5's ``iATR`` definition, distinct from :func:`atr`.

    Seeded at row ``period - 1`` (same "no partial windows" convention as :func:`atr`: a
    simple mean of the first ``period`` true-range values, via ``true_range().rolling_mean``),
    then smoothed forward with Wilder's own recursion, ``ATR[i] = (ATR[i-1] * (period - 1) +
    TR[i]) / period`` -- **not** a plain rolling mean of ``true_range()`` throughout, which is
    what :func:`atr` computes and a materially different number after the first window. Added
    for dry run #23 (``toy_tsmom``), whose card pins down "Wilder ATR on Bid bars" explicitly;
    kept here rather than in the strategy module because any future system stopping off an
    MT5-style ATR needs the same recursion, not a fresh one-off.

    Causal by construction: built from a Polars expression chain (:func:`true_range`,
    ``rolling_mean``, ``ewm_mean``), each of which only ever reads rows ``<= i`` to produce row
    ``i`` -- the recursive smoothing step is exactly what ``ewm_mean(adjust=False)`` already
    computes, seeded via a one-off substitution at the window's first valid row rather than at
    row 0 (``ewm_mean``'s own default seed), so it matches Wilder's published recursion exactly,
    not merely converges to it.
    """
    if period < 1:
        raise ValueError(f"wilder_atr: period must be >= 1, got {period}")
    tr = true_range()
    seed = tr.rolling_mean(period, min_samples=period)
    row = pl.int_range(pl.len())
    seeded = (
        pl.when(row < period - 1).then(None)
        .when(row == period - 1).then(seed)
        .otherwise(tr)
    )
    return seeded.ewm_mean(alpha=1.0 / period, adjust=False, ignore_nulls=True)
