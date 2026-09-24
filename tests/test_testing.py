"""quantlab.testing: synthetic bars and the look-ahead auditor.

DESIGN §10 exit test: "signals computed on data up to t = signals on full data,
at t". We prove :func:`testing.assert_no_lookahead` actually enforces this by
running it against one clean strategy and two differently-leaky ones.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

import polars as pl
import pytest

from quantlab import contracts, testing
from quantlab.contracts import Params, RiskType
from quantlab.costs import InstrumentSpec

POINT = 0.0001


def _spec(**over):
    base = dict(
        symbol="EURUSD", digits=4, point=POINT, contract_size=100_000.0, tick_size=POINT,
        volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
        swap_mode="points", swap_long=0.0, swap_short=0.0, swap_3day=2,
        commission_per_lot_rt=0.0, stops_level=0.0, calibrated=True,
    )
    base.update(over)
    return InstrumentSpec(**base)


@dataclass(frozen=True)
class SmaCrossParams(Params):
    fast: int = 10
    slow: int = 30


@dataclass(frozen=True)
class NoParams(Params):
    pass


def _null_extra_columns(n: int) -> dict[str, pl.Series]:
    return {
        "stop_dist": pl.Series([None] * n, dtype=pl.Float64),
        "target_dist": pl.Series([None] * n, dtype=pl.Float64),
    }


class CleanSmaCross:
    """+1/-1 by fast-vs-slow rolling mean of ``close`` — a plain causal window, no look-ahead."""

    name: ClassVar[str] = "clean_sma_cross"
    risk_type: ClassVar[RiskType] = RiskType.C

    def __init__(self, params: SmaCrossParams = SmaCrossParams()) -> None:
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        fast = bars["close"].rolling_mean(self.params.fast)
        slow = bars["close"].rolling_mean(self.params.slow)
        raw = (fast > slow).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(fast.is_null() | slow.is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class LeakyShiftStrategy:
    """Bug: decides today's signal from *tomorrow's* close via ``shift(-1)``."""

    name: ClassVar[str] = "leaky_shift"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        future_close = bars["close"].shift(-1)
        raw = (future_close > bars["close"]).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(future_close.is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class LeakyFullSampleNormStrategy:
    """Bug: z-scores ``close`` against the *whole input frame's* mean/std.

    This never reads a future *value* directly (no ``shift(-1)``), so scrambling
    the order of future bars doesn't change it — a permutation leaves the sample
    mean/std unchanged. But it still fails the **truncation** check: cutting the
    frame at ``t`` changes which rows are in the sample, hence the mean/std,
    hence the historical z-scores and signals. This is why the auditor runs
    both checks rather than just one.
    """

    name: ClassVar[str] = "leaky_full_sample_norm"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        close = bars["close"]
        z = (close - close.mean()) / close.std()
        raw = (z > 0).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(z.is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class SparsePeekStrategy:
    """Bug: an SMA cross is only confirmed if the *next* bar's close would favour it -- a
    1-bar look-ahead that only shows up on the small number of crossover rows (M2: plain
    random cut-point sampling misses this most of the time)."""

    name: ClassVar[str] = "sparse_peek"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        fast = bars["close"].rolling_mean(20)
        slow = bars["close"].rolling_mean(50)
        above = fast > slow
        cross_up = above & ~above.shift(1).fill_null(True)
        cross_dn = (~above) & above.shift(1).fill_null(False) & slow.is_not_null()
        nxt = bars["close"].shift(-1) - bars["close"]              # <-- future
        sig = pl.select(
            pl.when(cross_up & (nxt > 0)).then(pl.lit(1, pl.Int8))
            .when(cross_dn & (nxt < 0)).then(pl.lit(-1, pl.Int8))
            .when(cross_up | cross_dn).then(pl.lit(0, pl.Int8))
            .otherwise(pl.lit(None, pl.Int8))
        ).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class LoadsItsOwnDataStrategy:
    """Bug: fetches extra data itself from inside signals() instead of using only the
    `bars` it was given -- a look-ahead through the data layer, invisible to any amount
    of truncating/perturbing `bars` alone (M2)."""

    name: ClassVar[str] = "loads_its_own_data"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        from quantlab import data                     # local import: resolved at call time

        _extra = data.load_bars("EURUSD", "D1")        # forbidden -- see quantlab.testing docstring
        raw = (bars["close"] > bars["close"].shift(1)).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(bars["close"].shift(1).is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class WrongShapeStrategy:
    name: ClassVar[str] = "wrong_shape"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        n = max(bars.height - 1, 0)
        return pl.DataFrame({"signal": pl.Series([1] * n, dtype=pl.Int8), **_null_extra_columns(n)})


# --------------------------------------------------------------------------- synthetic_bars
def test_synthetic_bars_shape_and_columns():
    bars = testing.synthetic_bars(500, seed=0, timeframe="H1")
    assert bars.height == 500
    assert bars.columns == list(contracts.BAR_COLUMNS)
    assert bars["ts"].is_sorted() and bars["ts"].n_unique() == 500


def test_synthetic_bars_are_ohlc_valid_and_weekday_only():
    bars = testing.synthetic_bars(400, seed=3, timeframe="H1")
    assert (bars["low"] <= bars["open"]).all() and (bars["open"] <= bars["high"]).all()
    assert (bars["low"] <= bars["close"]).all() and (bars["close"] <= bars["high"]).all()
    assert (bars["low"] <= bars["high"]).all()
    assert set(bars["ts"].dt.weekday().unique().to_list()) <= {1, 2, 3, 4, 5}
    assert (bars["spread"] == bars["spread_max"]).all()


def test_synthetic_bars_reproducible_with_same_seed():
    a = testing.synthetic_bars(200, seed=42)
    b = testing.synthetic_bars(200, seed=42)
    assert a.equals(b)


def test_synthetic_bars_rejects_unknown_timeframe():
    with pytest.raises(ValueError):
        testing.synthetic_bars(10, seed=0, timeframe="M3")


# --------------------------------------------------------------------------- assert_no_lookahead
def test_clean_strategy_passes_the_lookahead_audit():
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    testing.assert_no_lookahead(CleanSmaCross(), bars, n_checks=15, seed=2, min_history=60)


def test_leaky_shift_strategy_fails_the_audit():
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(LeakyShiftStrategy(), bars, n_checks=15, seed=2, min_history=60)


def test_leaky_full_sample_normalisation_fails_the_audit():
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="truncation"):
        testing.assert_no_lookahead(LeakyFullSampleNormStrategy(), bars, n_checks=15, seed=2, min_history=60)


def test_wrong_shape_strategy_fails_fast():
    bars = testing.synthetic_bars(300, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="rows"):
        testing.assert_no_lookahead(WrongShapeStrategy(), bars, n_checks=5, seed=0, min_history=60)


def test_bars_too_short_raises_value_error():
    bars = testing.synthetic_bars(50, seed=0, timeframe="H1")
    with pytest.raises(ValueError):
        testing.assert_no_lookahead(CleanSmaCross(), bars, min_history=200)


# --------------------------------------------------------------------------- M2: stronger checks
def test_sparse_one_bar_peek_now_always_fails_the_audit():
    """Red-team M2: with plain random cut points this only failed 12/20 seeds (the leak
    only shows up on the handful of crossover rows). Forcing every active cut point makes
    it deterministic -- every seed must now fail."""
    for seed in range(10):
        bars = testing.synthetic_bars(600, seed=seed, timeframe="H1")
        with pytest.raises(AssertionError, match="look-ahead detected"):
            testing.assert_no_lookahead(SparsePeekStrategy(), bars, n_checks=5, seed=seed, min_history=60)


def test_strategy_that_loads_its_own_data_now_fails_the_audit():
    """Red-team M2: this strategy called quantlab.data.load_bars() itself and passed the
    audit 20/20 seeds, because only the `bars` argument was ever truncated/perturbed."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="must not load data themselves"):
        testing.assert_no_lookahead(LoadsItsOwnDataStrategy(), bars, n_checks=5, seed=0, min_history=60)


# --------------------------------------------------------------------------- align_higher_timeframe
def test_align_higher_timeframe_never_exposes_an_unclosed_htf_bar():
    htf = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 4), datetime(2024, 1, 1, 8)],
        "close": [1.0, 2.0, 3.0],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    ltf = pl.DataFrame({
        "ts": [datetime(2024, 1, 1, h) for h in range(12)],
        "close": [float(h) for h in range(12)],
    }).with_columns(pl.col("ts").cast(pl.Datetime("ms")))

    joined = testing.align_higher_timeframe(ltf, htf)
    assert joined.height == ltf.height

    # Before the first HTF bar (00:00-04:00) has closed, nothing is attached at all.
    before_close = joined.filter(pl.col("ts") < datetime(2024, 1, 1, 4))
    assert before_close["htf_ts"].is_null().all()

    # One minute-equivalent (here: one hour) before a close, the bar is still forming.
    just_before = joined.filter(pl.col("ts") == datetime(2024, 1, 1, 3)).row(0, named=True)
    assert just_before["htf_ts"] is None

    # At/after the close, that bar (and only that one) is attached -- inclusive boundary.
    at_close = joined.filter(pl.col("ts") == datetime(2024, 1, 1, 4)).row(0, named=True)
    assert at_close["htf_ts"] == datetime(2024, 1, 1, 0) and at_close["htf_close"] == 1.0

    still_same_bar = joined.filter(pl.col("ts") == datetime(2024, 1, 1, 7)).row(0, named=True)
    assert still_same_bar["htf_ts"] == datetime(2024, 1, 1, 0)   # bar[04:00] hasn't closed (closes 08:00)
    assert still_same_bar["htf_close"] == 1.0

    next_bar = joined.filter(pl.col("ts") == datetime(2024, 1, 1, 8)).row(0, named=True)
    assert next_bar["htf_ts"] == datetime(2024, 1, 1, 4) and next_bar["htf_close"] == 2.0


def test_align_higher_timeframe_rejects_empty_htf():
    ltf = testing.synthetic_bars(10, seed=0, timeframe="H1")
    with pytest.raises(ValueError):
        testing.align_higher_timeframe(ltf, ltf.head(0))


# --------------------------------------------------------------------------- assert_engine_causal
def test_assert_engine_causal_passes_for_a_clean_strategy_without_m1():
    bars = testing.synthetic_bars(300, seed=7, timeframe="H1")
    testing.assert_engine_causal(CleanSmaCross(), bars, _spec(), n_checks=8, seed=1, min_history=60)


def test_assert_engine_causal_detects_a_manufactured_violation(monkeypatch):
    """Decoupled from engine.py's own current state (it may be mid-fix concurrently):
    monkeypatches run_backtest itself so this exercises the *checker's* comparison logic,
    not any particular bug in the real engine."""
    import quantlab.engine as engine_module

    bars = testing.synthetic_bars(300, seed=7, timeframe="H1")
    real_run_backtest = engine_module.run_backtest

    def _corrupting_run_backtest(bars_arg, signals_arg, spec_arg, *args, **kwargs):
        result = real_run_backtest(bars_arg, signals_arg, spec_arg, *args, **kwargs)
        if bars_arg.height < bars.height and result.trades.height > 0:
            # Pretend the first already-closed trade's exit price came out different --
            # standing in for "the truncated run saw data the full run didn't".
            bad_exit_price = result.trades["exit_price"].to_list()
            bad_exit_price[0] = bad_exit_price[0] + 1.0
            result = dataclasses.replace(
                result, trades=result.trades.with_columns(pl.Series("exit_price", bad_exit_price))
            )
        return result

    monkeypatch.setattr(engine_module, "run_backtest", _corrupting_run_backtest)
    with pytest.raises(AssertionError, match="engine causality violated"):
        testing.assert_engine_causal(CleanSmaCross(), bars, _spec(), n_checks=8, seed=1, min_history=60)
