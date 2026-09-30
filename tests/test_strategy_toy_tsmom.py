"""Tests for ``quantlab.strategies.toy_tsmom`` (DRY RUN toy, dry run #23, DESIGN §10 Phase 2).

Every price path below is hand-built and hand-verified (see the derivation notes in each
test's docstring) -- no reliance on real market data, so these tests are fast and fully
deterministic. A shared helper (:func:`_bars_from_close`) only assembles the
``contracts.BAR_COLUMNS`` frame from an explicit close-price path; the actual *logic* (which
prices, which row) is inlined in each test so every expected value is auditable on its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab.contracts import BAR_COLUMNS
from quantlab.costs import CostModel, InstrumentSpec
from quantlab.engine import run_backtest
from quantlab.strategies import toy_tsmom
from quantlab.strategies.toy_tsmom import ToyTsmom, ToyTsmomParams
from quantlab.testing import (
    assert_engine_causal,
    assert_no_lookahead,
    assert_strategy_source_clean,
    synthetic_bars,
)

# A generic FX-like spec (point=1e-5), independent of any real broker export -- same pattern as
# quantlab.systems' own template notebook audit cell and tests/test_evaluators.py's SmaCross spec.
SPEC = InstrumentSpec(
    symbol="EURUSD_TEST", digits=5, point=1e-5, contract_size=100_000.0, tick_size=1e-5,
    volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
    swap_mode="points", swap_long=0.0, swap_short=0.0, swap_3day=2,
    commission_per_lot_rt=0.0, stops_level=0.0, calibrated=True,
)


def _bars_from_close(
    close: list[float],
    *,
    wicks: dict[int, tuple[float, float]] | None = None,
    spread: float = 0.0,
    ts: list[datetime] | None = None,
) -> pl.DataFrame:
    """Build a ``contracts.BAR_COLUMNS`` frame from an explicit close-price path.

    ``open[i] = close[i-1]`` (``open[0] = close[0]``, the same convention ``testing.
    synthetic_bars`` uses); ``high``/``low`` default to ``close +/- 0.002`` unless ``wicks``
    overrides a row with an explicit ``(high, low)`` pair -- used to plant an intrabar stop
    touch without disturbing the close-price path the ROC/cross logic reads (only ``close``
    feeds ``ROC``). ``ts`` defaults to one bar every 4 hours from a fixed Monday anchor; pass
    an explicit list (e.g. with one oversized gap) to prove the bar-count logic below is blind
    to calendar time.
    """
    n = len(close)
    close_arr = np.asarray(close, dtype=np.float64)
    open_arr = np.empty(n)
    open_arr[0] = close_arr[0]
    open_arr[1:] = close_arr[:-1]
    high_arr = close_arr + 0.002
    low_arr = close_arr - 0.002
    for i, (h, l) in (wicks or {}).items():
        high_arr[i] = h
        low_arr[i] = l
    if ts is None:
        start = datetime(2020, 1, 6)  # a Monday
        ts = [start + timedelta(hours=4 * i) for i in range(n)]
    df = pl.DataFrame({
        "ts": ts, "open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr,
        "spread": np.full(n, float(spread)), "spread_max": np.full(n, float(spread)),
        "tick_vol": np.full(n, 100, dtype=np.int64),
    }).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"))
    return df.select(list(BAR_COLUMNS))


# =============================================================================================
# mandatory audits
# =============================================================================================
def test_no_lookahead_audit():
    """Exhaustive (no ``max_forced_cuts``) truncation + future-perturbation audit, default
    (mid-grid) parameters -- the exact class the S3 baseline and every evaluator call use."""
    bars = synthetic_bars(700, seed=11, timeframe="H4")
    assert_no_lookahead(ToyTsmom, bars)


def test_engine_causal_h4():
    bars = synthetic_bars(500, seed=12, timeframe="H4")
    assert_engine_causal(ToyTsmom(), bars, SPEC, timeframe="H4", n_checks=8, seed=1, min_history=150)


def test_strategy_source_is_clean():
    assert_strategy_source_clean(toy_tsmom.__file__)


# =============================================================================================
# known-answer test (signal logic + hand-computed trade)
# =============================================================================================
def test_known_answer_single_long_cross_and_time_exit():
    """20 flat bars @ 1.0, then 20 flat bars @ 1.01 (``lookback=5``, ``stop_mult=2.0``, ``hold=3``).

    ``ROC_t = close_t/close_{t-5} - 1`` is 0 for every ``t`` except ``t in [10, 14]``, where the
    lookback window straddles the jump (``ROC_t = 1.01/1.0 - 1 = 0.01``) -- except ``t=20``, the
    row the jump itself lands on (``close_20/close_15 - 1 = 1.01/1.0 - 1 = 0.01``, still positive:
    the jump is the *only* bar whose own close differs from the bar 5 rows back's, well past
    t=14). Hand values for ``t=15..24``:

    | t  | close | close_{t-5} | ROC_t  |
    |----|-------|-------------|--------|
    | 15 | 1.00  | 1.00        | 0      |
    | 19 | 1.00  | 1.00        | 0      |
    | 20 | 1.01  | 1.00        | 0.01   |  <- ROC_prev(=ROC_19=0) <= 0 and ROC_20 > 0: LONG CROSS
    | 21 | 1.01  | 1.00        | 0.01   |
    | 24 | 1.01  | 1.00        | 0.01   |
    | 25 | 1.01  | 1.01        | 0      |

    True range is 0.004 at every row except row 20 (the jump: ``TR_20 = max(0.004,
    |1.012-1.0|=0.012, |1.008-1.0|=0.008) = 0.012``). Wilder ATR(14) seeds at row 13 (mean of
    TR[0:14] = 0.004) and stays 0.004 through row 19 (every TR since the seed is 0.004 too), then
    at row 20: ``ATR_20 = (0.004*13 + 0.012)/14 = 0.0045714285714285714``. ``stop_dist_20 =
    2.0 * ATR_20 = 0.009142857142857143``.

    Entry fills at bar 21 (``open_21 = close_20 = 1.01``, spread 0). Time exit is scheduled at
    row ``20 + hold(3) = 23`` (``episode_start=20`` from row 20 onward, ``23 - 20 == 3``), firing
    at the open of bar 24 = ``entry_bar(21) + hold(3)``, exactly the card's rule.
    """
    close = [1.0] * 20 + [1.01] * 20
    bars = _bars_from_close(close)
    params = ToyTsmomParams(lookback=5, stop_mult=2.0, hold=3)
    sig = ToyTsmom(params).signals(bars)

    assert sig.height == bars.height
    non_null = sig.with_row_index().filter(pl.col("signal").is_not_null())
    assert non_null["index"].to_list() == [20, 23]
    assert non_null["signal"].to_list() == [1, 0]
    assert non_null["stop_dist"][0] == pytest.approx(0.009142857142857143, abs=1e-12)
    assert sig["target_dist"].is_null().all()

    res = run_backtest(bars, sig, SPEC, CostModel())
    t = res.trades
    assert t.height == 1
    row = t.row(0, named=True)
    assert row["entry_idx"] == 21 and row["exit_idx"] == 24          # entry_bar + hold, in bars
    assert row["direction"] == 1
    assert row["entry_price"] == pytest.approx(1.01, abs=1e-12)       # open(21) = close(20), no spread
    assert row["stop_price"] == pytest.approx(1.01 - 0.009142857142857143, abs=1e-9)
    assert row["exit_reason"] == "signal"                             # the scheduled time exit, not a stop


# =============================================================================================
# rule-conformance tests
# =============================================================================================
def test_roc_exactly_zero_is_never_a_cross():
    """``lookback=2``; rows 0-13 flat @ 1.0, row 14 jumps to 1.02 (a genuine long cross:
    ``ROC_13=0, ROC_14=0.02``), rows 15-16 stay @ 1.02.

    ``ROC_15 = close_15/close_13 - 1 = 1.02/1.00 - 1 = 0.02`` (still elevated: the lookback
    window still spans the jump). ``ROC_16 = close_16/close_14 - 1 = 1.02/1.02 - 1 = 0`` (the
    window has now fully slid past the jump on both ends) -- with ``ROC_prev(=ROC_15)=0.02 >= 0``,
    the short condition is one comparison away from firing (``roc_prev >= 0``) but is blocked by
    ``roc_16 < 0`` failing (``roc_16 == 0`` exactly): the card's "ROC exactly 0 at t is not a
    cross" rule, in the one case (a clearly-signed ``roc_prev``) where it actually matters.
    """
    close = [1.0] * 14 + [1.02, 1.02, 1.02]
    bars = _bars_from_close(close)
    sig = ToyTsmom(ToyTsmomParams(lookback=2, stop_mult=1.0, hold=50)).signals(bars)
    s = sig["signal"]
    assert s[14] == 1     # the baseline cross (sanity: ROC_13=0 <= 0, ROC_14=0.02 > 0)
    assert s[15] is None
    assert s[16] is None  # ROC_16 == 0 exactly: never a cross, despite ROC_prev clearly > 0


@pytest.mark.parametrize("bump,expected", [(1.02, 1), (0.98, -1)])
def test_roc_prev_zero_counts_as_either_side_for_the_next_cross(bump, expected):
    """``lookback=2``; rows 0-13 flat @ 1.0 (``ROC_13 = 0`` exactly), row 14 jumps to ``bump``.

    ``ROC_13 == 0`` satisfies *both* ``<= 0`` and ``>= 0`` at once, so it is available as the
    prior side for either a long cross (``bump > 1.0``) or a short cross (``bump < 1.0``) --
    the card's "a zero only counts as the <=0/>=0 side at t-1".
    """
    close = [1.0] * 14 + [bump]
    bars = _bars_from_close(close)
    sig = ToyTsmom(ToyTsmomParams(lookback=2, stop_mult=1.0, hold=50)).signals(bars)
    assert sig["signal"][14] == expected


def test_no_target_ever():
    """"Target: none" (card) on any bar series, any parameters, including the mid-grid prior."""
    bars = synthetic_bars(300, seed=5, timeframe="H4")
    sig = ToyTsmom().signals(bars)
    assert sig["target_dist"].is_null().all()


def test_reversal_closes_and_reopens_in_one_bar():
    """An opposite cross while in a position reverses at the *same* next-bar open (card).

    ``lookback=2``, ``stop_mult=50`` (wide enough that the price path below never touches it --
    isolates the reversal mechanics from the stop), ``hold=20`` (long enough not to fire in this
    30-bar window). Rows 0-15 flat @ 1.0; rows 16-17 @ 1.01 (long cross at row 16: ``ROC_15=0 <=
    0, ROC_16=0.01 > 0``); rows 18-29 @ 0.99 (short cross at row 18: ``ROC_17=0.01 >= 0``,
    ``ROC_18 = 0.99/1.01 - 1 < 0``).
    """
    close = [1.0] * 16 + [1.01] * 2 + [0.99] * 12
    bars = _bars_from_close(close)
    params = ToyTsmomParams(lookback=2, stop_mult=50.0, hold=20)
    sig = ToyTsmom(params).signals(bars)
    non_null = sig.with_row_index().filter(pl.col("signal").is_not_null())
    assert non_null["index"].to_list() == [16, 18]
    assert non_null["signal"].to_list() == [1, -1]

    trades = run_backtest(bars, sig, SPEC, CostModel()).trades.sort("entry_idx")
    assert trades.height == 2
    first, second = trades.row(0, named=True), trades.row(1, named=True)
    # The long from the row-16 cross closes at the row-18 cross's own fill bar (19 = 18+1)...
    assert first["entry_idx"] == 17 and first["exit_idx"] == 19
    assert first["direction"] == 1 and first["exit_reason"] == "signal"
    # ...and the new short opens at that very same bar (one position at a time, no gap bar).
    assert second["entry_idx"] == 19 and second["direction"] == -1


def test_stop_hit_before_scheduled_time_exit_is_not_double_exited():
    """"The stop, if touched intrabar, comes before a time/reversal exit at the next open" (card).

    ``lookback=2``, ``stop_mult=1.0``, ``hold=5``. Rows 0-15 flat @ 1.0, row 16 jumps to 1.01
    (long cross, entry fills at bar 17 -- ``ATR_16 = 0.0045714285714285714`` by the same Wilder
    recursion as the known-answer test above, so ``stop_price = 1.01 - 0.0045714285714285714 =
    1.0054285714285714``). Rows 17-18 stay @ 1.01 (close); row 19's *close* also stays @ 1.01 but
    its wick dips to ``low=1.0`` -- well under the stop -- so the engine's own intrabar SL/TP
    check (unrelated to this module, which only ever attaches a fixed ``stop_dist``) closes the
    trade at row 19, reason ``"stop"``. The module's own schedule would otherwise have placed a
    time-exit ``0`` at row ``16 + hold(5) = 21``; it must be a no-op (the position is already
    flat by then), not a second, spurious trade.
    """
    close = [1.0] * 16 + [1.01] * 19
    bars = _bars_from_close(close, wicks={19: (1.012, 1.0)})
    params = ToyTsmomParams(lookback=2, stop_mult=1.0, hold=5)
    sig = ToyTsmom(params).signals(bars)
    assert sig["signal"][21] == 0   # the schedule still fires as a signal row...

    trades = run_backtest(bars, sig, SPEC, CostModel()).trades
    assert trades.height == 1       # ...but produces no second trade: engine no-ops on signal==position
    row = trades.row(0, named=True)
    assert row["entry_idx"] == 17 and row["exit_idx"] == 19
    assert row["exit_reason"] == "stop"
    assert row["exit_price"] == pytest.approx(1.01 - 0.0045714285714285714, abs=1e-9)
    assert row["stop_price"] == pytest.approx(row["exit_price"], abs=1e-12)


def test_stop_measured_from_the_fill_price_not_the_bar_open():
    """"Level = entry fill price (mp) stop_mult x ATR(14)_t" (card) -- with a nonzero spread, the
    fill price differs from the bar's raw open, and the stop must track the *fill*, not the open.

    Same long-cross setup as the stop-priority test above (``lookback=2``, ``stop_mult=1.0``,
    entry cross at row 16), but with a 20-point spread: the long entry fills at ``open_17 +
    20*point = 1.01 + 0.0002 = 1.0102`` (``quantlab.engine``'s own spread convention), so the
    stop must be ``1.0102 - ATR_16 = 1.0102 - 0.0045714285714285714``, not ``1.01 - ATR_16``.
    """
    close = [1.0] * 16 + [1.01] * 24
    bars = _bars_from_close(close, spread=20.0)
    params = ToyTsmomParams(lookback=2, stop_mult=1.0, hold=50)
    sig = ToyTsmom(params).signals(bars)

    trades = run_backtest(bars, sig, SPEC, CostModel()).trades
    assert trades.height == 1
    row = trades.row(0, named=True)
    fill = 1.01 + 20 * SPEC.point
    assert row["entry_price"] == pytest.approx(fill, abs=1e-12)
    assert row["stop_price"] == pytest.approx(fill - 0.0045714285714285714, abs=1e-9)


def test_hold_counts_bars_not_calendar_time():
    """Time exit is measured in bars, "bars are counted in bars, not calendar time, so a
    weekend counts as 0 bars" (card). ``lookback=2``, ``hold=4``; the long cross at row 14
    (rows 0-13 flat @ 1.0, row 14 -> 1.02, flat after) is followed by an oversized (3-day)
    calendar gap inserted before row 17's timestamp -- the schedule must still fire at row
    ``14 + hold(4) = 18``, unaffected by that gap, because this module never reads ``ts``.
    """
    close = [1.0] * 14 + [1.02] * 11
    start = datetime(2020, 1, 6)
    ts = [start + timedelta(hours=4 * i) for i in range(15)]
    for i in range(15, len(close)):
        gap = timedelta(days=3) if i == 17 else timedelta(hours=4)
        ts.append(ts[-1] + gap)
    bars = _bars_from_close(close, ts=ts)

    params = ToyTsmomParams(lookback=2, stop_mult=1.0, hold=4)
    sig = ToyTsmom(params).signals(bars)
    non_null = sig.with_row_index().filter(pl.col("signal").is_not_null())
    assert non_null["index"].to_list() == [14, 18]
    assert non_null["signal"].to_list() == [1, 0]
