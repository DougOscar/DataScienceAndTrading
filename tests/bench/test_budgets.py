"""Performance-engineer benchmark guards for DESIGN §7's budgets.

All of these are marked ``slow`` (real, dev-period data; ``pytest -m slow`` to
run) and each asserts *with headroom* below the DESIGN §7 number, so a
regression is caught well before it would actually blow the budget:

* Budget 1/2 -- "one 9-year H1 backtest with M1 intrabar resolution < 1s
  warm": ``test_pipeline_under_budget`` runs the full
  load H1/H4/D1 -> load M1 -> signals -> run_backtest(m1=) -> apply_sizing ->
  daily_equity pipeline on real EURUSD H1, XAUUSD H4 and USDJPY D1 dev-period
  data and asserts each stays under 0.7s warm (vs. the 1s budget).
* Budget 3 -- the 3.7M M1-bar engine kernel: this used to take ~1.2s because
  ``run_backtest`` built a Python list from *every* bar's timestamp
  (``bars["ts"].to_list()``) just to look up the O(n_trades) entry/exit
  timestamps. Fixed to ``Series.gather(idx).to_list()`` (§ engine.py); this
  test pins the ~15-20x faster result so a re-introduction of the O(n_bars)
  pattern anywhere in the wrapper is caught immediately, well before the
  existing ``tests/test_engine.py`` 1.5s ceiling would even notice.
* Budget 4 -- optimisation-study throughput: how many H1+M1 backtests/sec one
  process gets, and what a shared-memory ``ProcessPoolExecutor`` adds on top
  (DESIGN §7: "no DataFrame pickled per task -- shared-memory / memory-mapped
  arrays"). Both are floor assertions (>=), not exact numbers -- see the perf
  engineer's report for the fuller sweep across worker counts.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, os.path.dirname(__file__))  # for the `_strategy`/`_shared_mem`/`_pool_worker` siblings

from _shared_mem import bars_to_shm, close_and_unlink  # noqa: E402
from _strategy import sma_cross_atr_signals  # noqa: E402

from quantlab import costs, data  # noqa: E402
from quantlab.engine import run_backtest  # noqa: E402
from quantlab.sizing import apply_sizing, daily_equity  # noqa: E402

DEV_START = "2016-05-02"
DEV_END = "2025-05-14"  # DESIGN §4.1: FBS holdout starts 2025-05-15 -- never request past this in dev work

pytestmark = pytest.mark.slow


# --------------------------------------------------------------------------- budgets 1 & 2: full pipeline

def _run_pipeline(symbol: str, htf: str):
    bars = data.load_bars(symbol, htf, start=DEV_START, end=DEV_END)
    m1 = data.load_bars(symbol, "M1", start=DEV_START, end=DEV_END)
    sig = sma_cross_atr_signals(bars)
    spec = costs.load_instrument(symbol, book="FBS")
    cost = costs.CostModel(version="fbs-v0-uncalibrated")
    result = run_backtest(bars, sig, spec, cost, m1=m1)
    # account_ccy=spec.quote_ccy sidesteps conversion_rate() (a separate,
    # not-in-scope-here data.py cost) for symbols quoted in something other
    # than USD (e.g. USDJPY) -- this budget is about the engine/sizing
    # pipeline itself, not FX-conversion latency.
    sized = apply_sizing(result.trades, spec, mode="fixed_fraction", risk_fraction=0.01,
                          account_ccy=spec.quote_ccy)
    daily = daily_equity(bars, result, sized)
    return bars, m1, result, sized, daily


@pytest.mark.parametrize("symbol,htf", [("EURUSD", "H1"), ("XAUUSD", "H4"), ("USDJPY", "D1")])
def test_pipeline_under_budget(symbol, htf):
    warnings.filterwarnings("ignore")
    _run_pipeline(symbol, htf)  # warm-up: resample cache, OS page cache, numba JIT

    t0 = time.perf_counter()
    bars, m1, result, sized, daily = _run_pipeline(symbol, htf)
    elapsed = time.perf_counter() - t0

    assert result.trades.height > 0, f"{symbol} {htf}: benchmark strategy produced no trades over the dev window"
    assert daily.height > 0
    # DESIGN §7 budget is 1.0s; 0.7s leaves ~30% headroom while still catching
    # a real regression (measured warm: ~0.21-0.28s on the dev laptop spec).
    assert elapsed < 0.7, (
        f"{symbol} {htf}: full pipeline took {elapsed:.3f}s warm "
        f"(budget: <1.0s, guard threshold 0.7s) -- n_htf={bars.height} n_m1={m1.height}"
    )


# --------------------------------------------------------------------------- budget 3: 3.7M M1-bar kernel

def _synthetic_m1_bars_and_signals(n: int, seed: int = 0):
    """Same generator as tests/test_engine.py's 3.7M-bar benchmark (kept
    independent/self-contained here rather than imported, so this guard does
    not depend on that file's internals)."""
    rng = np.random.default_rng(seed)
    step = rng.normal(0, 0.00005, n).cumsum()
    open_ = 1.1000 + step
    high = open_ + np.abs(rng.normal(0, 0.00003, n))
    low = open_ - np.abs(rng.normal(0, 0.00003, n))
    close = open_ + rng.normal(0, 0.00001, n)
    spread = rng.integers(1, 4, n).astype(np.float64)

    ts_start = datetime(2020, 1, 1)
    ts = pl.datetime_range(ts_start, ts_start + timedelta(minutes=n - 1), interval="1m",
                            eager=True, time_unit="ms")
    bars = pl.DataFrame({
        "ts": ts, "open": open_, "high": high, "low": low, "close": close, "spread": spread,
    }).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )

    prob = 0.0005
    r = rng.random(n)
    has_signal = r < prob
    sig_val = np.where(r < prob / 2, 1, -1).astype(np.int8)
    signals = pl.DataFrame({"_sig": sig_val, "_has": has_signal}).select(
        pl.when(pl.col("_has")).then(pl.col("_sig")).otherwise(None).cast(pl.Int8).alias("signal"),
        pl.when(pl.col("_has")).then(0.0050).otherwise(None).alias("stop_dist"),
        pl.when(pl.col("_has")).then(0.0100).otherwise(None).alias("target_dist"),
    )
    return bars, signals


def test_3_7m_m1_kernel_regression_guard():
    from quantlab.costs import CostModel, InstrumentSpec

    n = 3_700_000
    bars, signals = _synthetic_m1_bars_and_signals(n)
    spec = InstrumentSpec(
        symbol="EURUSD", digits=4, point=0.0001, contract_size=100_000.0, tick_size=0.0001,
        volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
        swap_mode="points", swap_long=-2.0, swap_short=0.5, swap_3day=2,
        commission_per_lot_rt=7.0, stops_level=0.0, calibrated=True,
    )
    cost = CostModel()

    run_backtest(bars, signals, spec, cost)  # numba warm-up

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        run_backtest(bars, signals, spec, cost)
        times.append(time.perf_counter() - t0)
    best = min(times)

    # Post-fix this runs in ~0.04-0.09s on the dev laptop; 0.3s leaves a
    # >3x margin while still catching a return of the O(n_bars)
    # `bars["ts"].to_list()` pattern this test exists to guard against (that
    # pattern alone cost ~1.1s of the pre-fix ~1.2s total).
    assert best < 0.3, f"3.7M M1-bar kernel took {best:.3f}s warm (guard: <0.3s; DESIGN §7 kernel budget ~1.2s->this)"


# --------------------------------------------------------------------------- budget 4: optimisation-study throughput

def _param_grid(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    grid = []
    for _ in range(n):
        fast = int(rng.integers(5, 30))
        slow = fast + int(rng.integers(10, 60))
        grid.append(dict(
            fast=fast, slow=slow,
            atr_period=int(rng.integers(7, 21)),
            atr_mult_stop=float(rng.uniform(1.0, 3.0)),
            atr_mult_target=float(rng.uniform(2.0, 5.0)),
        ))
    return grid


def test_single_process_throughput_floor():
    """Regression floor for single-process trials/sec (data loaded once, many
    parameter sets) -- sizes how big a Phase 1 study can be without any
    parallelism at all (DESIGN §7: "optimisation study (~500 trials x CPCV)
    < 1h" needs only a few trials/sec to clear easily)."""
    warnings.filterwarnings("ignore")
    symbol, htf = "EURUSD", "H1"
    bars = data.load_bars(symbol, htf, start=DEV_START, end=DEV_END)
    m1 = data.load_bars(symbol, "M1", start=DEV_START, end=DEV_END)
    spec = costs.load_instrument(symbol, book="FBS")
    cost = costs.CostModel(version="fbs-v0-uncalibrated")

    grid = _param_grid(40)
    run_backtest(bars, sma_cross_atr_signals(bars), spec, cost, m1=m1)  # warm-up

    t0 = time.perf_counter()
    for params in grid:
        sig = sma_cross_atr_signals(bars, **params)
        run_backtest(bars, sig, spec, cost, m1=m1)
    elapsed = time.perf_counter() - t0
    per_sec = len(grid) / elapsed

    # Measured ~20-25 trials/sec on the dev laptop; floor of 8 gives a wide
    # margin for slower CI hardware while still catching a large regression.
    assert per_sec >= 8.0, f"single-process throughput {per_sec:.1f} trials/sec (floor: 8.0)"


def test_process_pool_throughput_beats_single_process():
    """Shared-memory ProcessPoolExecutor must beat single-process throughput
    by a healthy margin, proving the "no per-task DataFrame pickle" pattern
    (DESIGN §7) is actually wired up and not silently falling back to
    per-task serialisation of the whole bars/M1 frame."""
    warnings.filterwarnings("ignore")
    import pickle
    from concurrent.futures import ProcessPoolExecutor

    import _pool_worker

    symbol, htf = "EURUSD", "H1"
    bars = data.load_bars(symbol, htf, start=DEV_START, end=DEV_END)
    m1 = data.load_bars(symbol, "M1", start=DEV_START, end=DEV_END)
    spec = costs.load_instrument(symbol, book="FBS")
    cost = costs.CostModel(version="fbs-v0-uncalibrated")
    # Needs enough trials that per-worker fixed start-up (shared-memory
    # attach + one numba warm-up call per worker, ~tens-to-hundreds of ms
    # each) amortises away -- at grid sizes as small as ~40 that fixed cost
    # can dominate and make the pool *slower* than single-process, which is
    # real but not what this test is checking.
    grid = _param_grid(150)

    # single-process reference (also proves the pool's answers are correct)
    run_backtest(bars, sma_cross_atr_signals(bars), spec, cost, m1=m1)
    t0 = time.perf_counter()
    reference = []
    for params in grid:
        sig = sma_cross_atr_signals(bars, **params)
        reference.append(run_backtest(bars, sig, spec, cost, m1=m1).trades.height)
    single_elapsed = time.perf_counter() - t0
    single_per_sec = len(grid) / single_elapsed

    n_workers = max(2, min(8, (os.cpu_count() or 4) - 2))  # DESIGN §7: leave 2 threads for the system
    h1_handles, h1_spec = bars_to_shm(bars)
    m1_handles, m1_spec = bars_to_shm(m1)
    try:
        t0 = time.perf_counter()
        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_pool_worker.init,
            initargs=(h1_spec, m1_spec, spec, cost),
        ) as ex:
            pool_results = list(ex.map(_pool_worker.run_one, grid, chunksize=4))
        pool_elapsed = time.perf_counter() - t0
    finally:
        close_and_unlink(h1_handles)
        close_and_unlink(m1_handles)

    pool_per_sec = len(grid) / pool_elapsed
    assert [r["n_trades"] for r in pool_results] == reference, (
        "process-pool trade counts diverged from the single-process reference "
        "-- shared-memory reconstruction is not bit-identical to the real bars/M1 frames"
    )
    # Lenient (>= 1.2x, not the ~3x this repo's laptop actually sees) because
    # this is a shared/noisy box and the fixed per-worker startup cost (numba
    # warm-up + shared-memory attach) matters more at small grid sizes like
    # this test's -- see the perf engineer's report for the fuller sweep.
    assert pool_per_sec >= single_per_sec * 1.2, (
        f"pool ({n_workers} workers) throughput {pool_per_sec:.1f}/s did not beat "
        f"single-process {single_per_sec:.1f}/s by the expected margin"
    )
