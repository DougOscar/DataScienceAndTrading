"""quantlab.testing: synthetic bars and the look-ahead auditor.

DESIGN §10 exit test: "signals computed on data up to t = signals on full data,
at t". We prove :func:`testing.assert_no_lookahead` actually enforces this by
running it against one clean strategy and two differently-leaky ones.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest

from quantlab import contracts, data, testing
from quantlab.contracts import Params, RiskType
from quantlab.costs import InstrumentSpec

# Bound at import time, before any audit runs -- the M2 residual escape (red team,
# 2026-09-24 re-verification): a reference to `load_bars` taken before the sandbox even
# exists is untouched by rebinding `quantlab.data.load_bars`, but not by a guard checked
# inside the function body itself, which this same object still runs (see ImportBoundLoadBars).
from quantlab.data import load_bars as _import_bound_load_bars

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


class ImportBoundLoadBarsStrategy:
    """M2 residual escape: calls the `load_bars` object bound at *import time* (before the
    sandbox existed), not via `quantlab.data.load_bars` attribute access. This is exactly
    what let it pass the pre-fix auditor 20/20 seeds (rebinding the module attribute never
    touches an already-imported reference); the fix checks a flag inside the function body
    itself, which this is still the same object as."""

    name: ClassVar[str] = "import_bound_load_bars"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        _extra = _import_bound_load_bars("EURUSD", "D1")     # forbidden, imported at module load
        raw = (bars["close"] > bars["close"].shift(1)).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(bars["close"].shift(1).is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class PrivateHelperStrategy:
    """M2 residual escape: calls the private, unguarded-by-the-old-sandbox
    `data._load_resampled` directly instead of the public `load_bars`."""

    name: ClassVar[str] = "private_helper"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        row = data._resolve_source(data.catalog(), "EURUSD", None)
        _extra = data._load_resampled(Path(row["file"]), "M1", "D1").collect()
        raw = (bars["close"] > bars["close"].shift(1)).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(bars["close"].shift(1).is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class RawParquetStrategy:
    """M2 residual escape: reads the catalog's raw Parquet file directly with polars,
    bypassing `quantlab.data` entirely."""

    name: ClassVar[str] = "raw_parquet"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        row = data._resolve_source(data.catalog(), "EURUSD", None)
        _extra = pl.scan_parquet(row["file"]).limit(10).collect()      # forbidden raw read
        raw = (bars["close"] > bars["close"].shift(1)).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(bars["close"].shift(1).is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class ClosureCacheStrategy:
    """M2 residual escape: fetches data once in `__init__` and closes over it -- invisible
    to any amount of sandboxing `signals()` alone, since the data is already sitting on the
    instance by the time `assert_no_lookahead` ever sees it. Only constructing the strategy
    *inside* the sandbox (the `factory=` argument) can catch this."""

    name: ClassVar[str] = "closure_cache"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def __init__(self) -> None:
        self.extra = data.load_bars("EURUSD", "D1")     # forbidden only when built via factory=

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        raw = (bars["close"] > bars["close"].shift(1)).cast(pl.Int8) * 2 - 1
        sig = pl.select(pl.when(bars["close"].shift(1).is_null()).then(None).otherwise(raw).cast(pl.Int8)).to_series()
        return pl.DataFrame({"signal": sig, **_null_extra_columns(bars.height)})


class HoldEveryRowStrategy:
    """Always in the market (±1, never null/flat) -- the N6 adversarial case: the pre-fix
    `_forced_cut_points` forced a cut at *every* row for a strategy like this (every row is
    "active": non-null and non-zero), making the audit's cost quadratic in the bar count."""

    name: ClassVar[str] = "hold_every_row"
    risk_type: ClassVar[RiskType] = RiskType.C
    params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        mean = bars["close"].rolling_mean(50, min_samples=1)      # causal: no backward-fill
        sig = (bars["close"] > mean).cast(pl.Int8) * 2 - 1
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
    testing.assert_no_lookahead(CleanSmaCross, bars, n_checks=15, seed=2, min_history=60)


def test_leaky_shift_strategy_fails_the_audit():
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(LeakyShiftStrategy, bars, n_checks=15, seed=2, min_history=60)


def test_leaky_full_sample_normalisation_fails_the_audit():
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="truncation"):
        testing.assert_no_lookahead(LeakyFullSampleNormStrategy, bars, n_checks=15, seed=2, min_history=60)


def test_wrong_shape_strategy_fails_fast():
    bars = testing.synthetic_bars(300, seed=1, timeframe="H1")
    with pytest.raises(AssertionError, match="rows"):
        testing.assert_no_lookahead(WrongShapeStrategy, bars, n_checks=5, seed=0, min_history=60)


def test_bars_too_short_raises_value_error():
    bars = testing.synthetic_bars(50, seed=0, timeframe="H1")
    with pytest.raises(ValueError):
        testing.assert_no_lookahead(CleanSmaCross, bars, min_history=200)


# --------------------------------------------------------------------------- M2: stronger checks
def test_sparse_one_bar_peek_now_always_fails_the_audit():
    """Red-team M2: with plain random cut points this only failed 12/20 seeds (the leak
    only shows up on the handful of crossover rows). Forcing every active cut point makes
    it deterministic -- every seed must now fail."""
    for seed in range(10):
        bars = testing.synthetic_bars(600, seed=seed, timeframe="H1")
        with pytest.raises(AssertionError, match="look-ahead detected"):
            testing.assert_no_lookahead(SparsePeekStrategy, bars, n_checks=5, seed=seed, min_history=60)


def test_strategy_that_loads_its_own_data_now_fails_the_audit():
    """Red-team M2: this strategy called quantlab.data.load_bars() itself and passed the
    audit 20/20 seeds, because only the `bars` argument was ever truncated/perturbed."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="must not fetch their own data"):
        testing.assert_no_lookahead(LoadsItsOwnDataStrategy, bars, n_checks=5, seed=0, min_history=60)


# --------------------------------------------------------------------------- M2 residual: sandbox escapes
def test_import_bound_load_bars_now_fails_the_audit():
    """Red-team M2 residual (2026-09-24 re-verification): `from quantlab.data import
    load_bars` bound before the sandbox exists passed 1/1 -- rebinding the module attribute
    `data.load_bars` never touches an already-imported reference. The fix checks a flag
    inside `load_bars` itself, which this is still the same function object as."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(ImportBoundLoadBarsStrategy, bars, n_checks=5, seed=0, min_history=60)


def test_private_helper_now_fails_the_audit():
    """Red-team M2 residual: `quantlab.data._load_resampled` (private, called directly
    instead of through `load_bars`) passed the pre-fix auditor."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(PrivateHelperStrategy, bars, n_checks=5, seed=0, min_history=60)


def test_raw_parquet_scan_now_fails_the_audit():
    """Red-team M2 residual: `pl.scan_parquet(<catalog file>)`, bypassing quantlab.data
    entirely, passed the pre-fix auditor."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(RawParquetStrategy, bars, n_checks=5, seed=0, min_history=60)


def test_closure_cache_in_init_is_caught_because_factory_is_mandatory():
    """Red-team M2/R2-1: data fetched once in `__init__` and closed over is caught because
    the audit always constructs the strategy inside the sandbox."""
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(AssertionError, match="look-ahead detected"):
        testing.assert_no_lookahead(ClosureCacheStrategy, bars, n_checks=5, seed=0, min_history=60)


def test_assert_no_lookahead_rejects_missing_factory_and_prebuilt_instances():
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    with pytest.raises(ValueError, match="factory"):
        testing.assert_no_lookahead(bars=bars)
    with pytest.raises(TypeError, match="pre-built instance"):
        testing.assert_no_lookahead(CleanSmaCross(), bars)


def test_assert_no_lookahead_factory_builds_a_working_clean_strategy():
    """A clean strategy audited via `factory=` (the form /research-cycle should use) must
    still pass, and behave identically to passing an already-built instance."""
    bars = testing.synthetic_bars(600, seed=1, timeframe="H1")
    testing.assert_no_lookahead(factory=CleanSmaCross, bars=bars, n_checks=15, seed=2, min_history=60)


# --------------------------------------------------------------------------- N6: auditor performance
def test_dense_always_in_market_strategy_audits_768_bars_quickly():
    """N6: `_forced_cut_points` used to force a cut at every "active" (non-null, non-zero)
    row, so a strategy that is always in the market (like this one) forced a cut at nearly
    every row -- O(n) `signals()` calls, each O(n). Forcing cuts only where the signal
    *changes* (plus a 200-cut cap) must keep this fast even on a strategy that is never
    flat and never null."""
    bars = testing.synthetic_bars(768, seed=1, timeframe="H1")
    t0 = time.time()
    testing.assert_no_lookahead(HoldEveryRowStrategy, bars, n_checks=25, seed=0, min_history=60)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"audit took {elapsed:.1f}s (budget: 3s, N6)"


def test_forced_cut_points_uses_changes_not_every_active_row():
    bars = testing.synthetic_bars(768, seed=1, timeframe="H1")
    full = HoldEveryRowStrategy().signals(bars)
    forced = testing._forced_cut_points(full, bars.height)
    active = int(((full["signal"].is_not_null()) & (full["signal"] != 0)).sum())
    assert active > 700                                    # this strategy is "active" almost everywhere
    assert 0 < len(forced) < active                         # but far fewer rows are actual *changes*


def test_forced_cut_points_exhaustive_by_default_capped_only_on_request():
    """R2-2: gate runs must test every change point; a cap is an explicit quick-mode opt-in."""
    n = 5000
    sig = pl.Series([1 if i % 2 == 0 else -1 for i in range(n)], dtype=pl.Int8)
    full = pl.DataFrame({"signal": sig})
    assert len(testing._forced_cut_points(full, n)) > 4000          # exhaustive by default (R2-2)
    assert len(testing._forced_cut_points(full, n, cap=testing.QUICK_MAX_FORCED_CUTS)) <= 200


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


def test_assert_engine_causal_with_m1_on_real_data_requires_and_forwards_timeframe():
    """Walk-forward shape: HTF slices + one full M1 frame (red-team M1).  Real EURUSD, small window."""
    h1 = data.load_bars("EURUSD", "H1", start=datetime(2019, 3, 4), end=datetime(2019, 4, 6))
    m1 = data.load_bars("EURUSD", "M1", start=datetime(2019, 3, 4), end=datetime(2019, 4, 6))
    spec = _spec(point=1e-5)
    with pytest.raises(ValueError, match="timeframe"):
        testing.assert_engine_causal(CleanSmaCross(), h1, spec, m1=m1, min_history=60)
    testing.assert_engine_causal(CleanSmaCross(), h1, spec, m1=m1, timeframe="H1",
                                 n_checks=6, seed=3, min_history=60)


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


# --------------------------------------------------------------------------- R2-2: exhaustive cuts
@dataclass(frozen=True)
class ChoppyPeekStrategy:
    """Signal flips often (many change points) and peeks one bar ahead on a handful of rows."""

    name: ClassVar[str] = "choppy_peek"
    risk_type: ClassVar[RiskType] = RiskType.C
    params: Params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        n = bars.height
        sig = [1 if (i // 3) % 2 == 0 else -1 for i in range(n)]
        closes = bars["close"].to_list()
        for i in range(700, n - 1, 400):                     # ~a few leaky rows
            sig[i] = 1 if closes[i + 1] > closes[i] else -1   # peeks at bar i+1
        return pl.DataFrame({"signal": pl.Series(sig, dtype=pl.Int8), **_null_extra_columns(n)})


def test_sparse_peek_in_a_choppy_strategy_is_caught_by_the_exhaustive_audit():
    bars = testing.synthetic_bars(2500, seed=11, timeframe="H1")
    with pytest.raises(AssertionError):
        testing.assert_no_lookahead(ChoppyPeekStrategy, bars, n_checks=0, seed=0, min_history=60)


# --------------------------------------------------------------------------- R2-1: sandbox robustness
@dataclass(frozen=True)
class ReloadsDataStrategy:
    name: ClassVar[str] = "reloads_data"
    risk_type: ClassVar[RiskType] = RiskType.C
    params: Params = NoParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        import importlib
        importlib.reload(data)
        return pl.DataFrame({"signal": pl.Series([0] * bars.height, dtype=pl.Int8),
                             **_null_extra_columns(bars.height)})


def test_readers_are_restored_even_if_the_strategy_reloads_quantlab_data():
    import pyarrow.parquet as pq

    originals = (pl.read_parquet, pl.scan_parquet, pl.read_csv, pl.scan_csv, pq.read_table)
    bars = testing.synthetic_bars(300, seed=4, timeframe="H1")
    try:
        testing.assert_no_lookahead(ReloadsDataStrategy, bars, n_checks=1, seed=0, min_history=60)
    except Exception:
        pass
    assert (pl.read_parquet, pl.scan_parquet, pl.read_csv, pl.scan_csv, pq.read_table) == originals
    assert data.AUDIT_IN_PROGRESS.get() is False
    data.load_bars("EURUSD", "H1", start=datetime(2019, 3, 4), end=datetime(2019, 3, 6))


# --------------------------------------------------------------------------- static lint
def test_lint_accepts_a_clean_strategy_module():
    src = Path(__file__).with_name("_lint_fixture_clean.py")
    src.write_text("""\"\"\"clean\"\"\"
from dataclasses import dataclass
import polars as pl
from quantlab.contracts import Params
N = 20


@dataclass(frozen=True)
class P(Params):
    n: int = 20


class S:
    def signals(self, bars):
        return bars.select(pl.col("close").rolling_mean(N))
""")
    try:
        assert testing.lint_strategy_source(src) == []
        testing.assert_strategy_source_clean(src)
    finally:
        src.unlink()


@pytest.mark.parametrize("snippet", [
    "from quantlab.data import load_bars",
    "from quantlab import data",
    "from . import data",
    "import pyarrow.parquet as pq",
    "import pyarrow.dataset as ds",
    "from polars import scan_parquet",
    "import subprocess",
    "import importlib",
    "import pandas as pd",
    "import polars as pl\nCACHE = pl.read_parquet('x.parquet')",
    "import numpy as np\nclass S:\n    def signals(self, b):\n        return np.load('a.npy')",
    "class S:\n    def signals(self, b):\n        return open('x').read()",
    "import polars as pl\nclass S:\n    def signals(self, b):\n        return getattr(pl, 'scan_' + 'parquet')",
    "print('import-time side effect')",
])
def test_lint_flags_each_bypass_pattern(snippet):
    assert testing.lint_strategy_source(snippet + "\n"), snippet
