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

__all__ = ["true_range", "atr"]


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
