"""DRY RUN toy -- ``toy_tsmom``: H4 EURUSD ROC-sign time-series momentum.

Dry run #23 (DESIGN §10 Phase 2 exit test); card at
``research/dryrun/systems/FBS/0023_toy_tsmom/hypothesis.md``. Not a real research candidate -- a toy
system used to exercise S2/S3 of the pipeline end to end.

Rules (card, verbatim in spirit)
---------------------------------
* ``ROC_t = close_t / close_{t-lookback} - 1``, at every H4 bar close ``t``.
* **Long cross** at bar ``t``: ``ROC_{t-1} <= 0`` and ``ROC_t > 0``.
* **Short cross** at bar ``t``: ``ROC_{t-1} >= 0`` and ``ROC_t < 0``.
* One position at a time; an opposite cross while in a position reverses (close + open at the
  same next-bar open); a cross while flat opens.
* **Stop**: fixed at entry, never trailed. ``stop_mult`` x Wilder ATR(14), value at the signal
  bar ``t`` (not the fill bar) -- this module attaches that distance to the signal row, and
  ``quantlab.engine`` reads it at fill time (``stop_dist[j-1]`` for a fill at row ``j``), which
  already *is* "the signal bar's own value" by the ``contracts.py`` convention -- no extra work
  needed here for that edge case.
* No target.
* **Time exit**: close at the open of bar ``entry_bar + hold`` (bars, not calendar time). A
  reversal or a new entry starts a new clock.
* **Exit priority within a bar** (stop before a same-bar time/reversal exit) and **lot
  rounding** are the engine's/sizing's job (``quantlab.engine._chk_long``/``_chk_short``,
  ``quantlab.sizing.floor_to_step``), not duplicated here.

Edge cases the card pins down, and how this module resolves them
------------------------------------------------------------------
* **ROC exactly 0 at t is not a cross.** The cross conditions require ``ROC_t > 0`` / ``< 0``
  strictly, so ``ROC_t == 0`` never fires either branch. ``ROC_{t-1} == 0`` satisfies *both*
  ``<= 0`` and ``>= 0`` at once (by construction of the two independent ``<=``/``>=``
  comparisons below), so it is available as the prior side for either direction's cross, per
  the card's "a zero only counts as the <=0/>=0 side at t-1".
* **Time exit is scheduled without ever needing to look ahead.** For every bar ``i``, this
  module tracks ``episode_start[i]`` -- the row index of the most recent cross at or before
  ``i`` (a plain forward-fill of the cross rows, so only past/current rows feed it) -- and
  marks the time exit at the unique row where ``i - episode_start[i] == hold``. Because
  ``episode_start`` jumps to the new cross's row the instant a reversal (or a fresh post-stop
  entry) occurs, an old episode's schedule is automatically superseded the moment a newer cross
  changes ``episode_start`` -- it can never fire after a newer position has already opened, and
  never needs to "know" the next cross in advance. The stop firing early (engine-only, path-
  dependent on M1 data this module never sees) makes any later time-exit row a no-op by
  construction (``contracts.py``: the engine only acts when ``signal[i-1] != position``).
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..contracts import Params, RiskType
from . import base

# Fixed by the card ("ATR(14)"); not a free parameter.
_ATR_PERIOD = 14

__all__ = ["ToyTsmomParams", "ToyTsmom", "STRATEGY"]


@dataclass(frozen=True)
class ToyTsmomParams(Params):
    """Free parameters (hypothesis.md "Free parameters (pre-registered...)" table).

    Defaults are the card's own mid-grid prior (used, unmodified, by the S3 baseline and by
    every evaluator call that doesn't override a parameter) -- picking the lower of the two
    middle grid levels where the level count is even, matching the card's own worked example
    ("Expected trades/year ... at mid-grid (lookback 60)"):

    | name      | range [lo, hi], grid step (levels)      | plateau scale         | mid-grid default |
    |-----------|------------------------------------------|------------------------|-------------------|
    | lookback  | [10, 120], step 10 (12 levels)            | relative (r = 0.20)    | 60                |
    | stop_mult | [1.0, 4.0], step 0.5 (7 levels)           | relative (r = 0.20)    | 2.5               |
    | hold      | [6, 60], step 6 (10 levels)                | relative (r = 0.20)    | 30                |

    The plateau scale column is copied verbatim from the card for S4 (optimization-architect)
    to use when it builds the real ``opt.SearchSpace``; nothing in this module applies it.
    """

    lookback: int = 60
    stop_mult: float = 2.5
    hold: int = 30


class ToyTsmom:
    """H4 ROC-sign time-series momentum with an ATR stop and a bar-count time exit.

    Risk type A (DESIGN §5): fixed-fraction risk against a price stop, lot size varies
    (``quantlab.sizing.apply_sizing(mode="fixed_fraction")``, chosen automatically by
    ``quantlab.evaluators.RuleEvaluator`` from ``risk_type``).
    """

    name = "toy_tsmom"
    risk_type = RiskType.A
    params_cls = ToyTsmomParams

    def __init__(self, params: ToyTsmomParams | None = None) -> None:
        self.params = params or ToyTsmomParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        close = pl.col("close")

        roc = close / close.shift(p.lookback) - 1.0
        roc_prev = roc.shift(1)
        atr14 = base.wilder_atr(_ATR_PERIOD)

        # No entry without a computable stop (ATR) or a computable cross (both ROC values) --
        # warm-up rows stay null ("keep current state", i.e. flat until the first real cross).
        warm = roc.is_null() | roc_prev.is_null() | atr14.is_null()
        raw_cross = (
            pl.when(warm).then(None)
            .when((roc_prev <= 0.0) & (roc > 0.0)).then(1)
            .when((roc_prev >= 0.0) & (roc < 0.0)).then(-1)
            .otherwise(None)
            .cast(pl.Int8)
        )

        # episode_start[i]: row index of the most recent cross at or before row i (forward-fill
        # of the cross rows themselves) -- see the module docstring's time-exit note.
        row = pl.int_range(pl.len())
        cross_row = pl.when(raw_cross.is_not_null()).then(row).otherwise(None)
        episode_start = cross_row.forward_fill()
        time_exit = episode_start.is_not_null() & ((row - episode_start) == p.hold) & raw_cross.is_null()

        signal = (
            pl.when(raw_cross.is_not_null()).then(raw_cross)
            .when(time_exit).then(0)
            .otherwise(None)
            .cast(pl.Int8)
        )
        stop_dist = atr14 * p.stop_mult

        return bars.select(
            signal.alias("signal"),
            stop_dist.alias("stop_dist"),
            pl.lit(None, dtype=pl.Float64).alias("target_dist"),
        )


STRATEGY = ToyTsmom
