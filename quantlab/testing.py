"""Test helpers shared across quantlab: synthetic bars and the look-ahead auditor.

DESIGN §10 (Phase 0 exit test): "signals computed on data up to t = signals on
full data, at t". :func:`assert_no_lookahead` proves this for any
:class:`contracts.Strategy` with four independent checks:

1. **truncation** — ``strategy.signals(bars[:t+1])`` matches the full-run
   signals on rows ``<= t``, at every cut point ``t`` where the full-run signal
   is active (non-null and non-zero — this is where a k-bar look-ahead like
   ``.shift(-1)`` is actually visible) plus a random sample of the rest. Random
   sampling alone is not enough: a leak that only shows up on ~1-2% of rows
   (a sparse crossover strategy, say) is easily missed by chance (see
   ``research/audits/2026-09-23_phase0_redteam.md``, finding M2).
2. **future-perturbation** — replacing everything strictly after ``t`` with a
   *fresh, independently-drawn* random walk (several draws per cut point,
   never a permutation of the real future rows — a permutation preserves the
   future's mean/std/max, which hides a full-sample-normalisation leak from
   this particular check, though truncation alone already catches that one)
   must leave rows ``<= t`` unchanged too. This catches look-aheads that read
   future rows by *position* (``.shift(-1)``) rather than by a value that
   happens to look plausible under truncation.
3. **sandboxed data access** — every call to ``strategy.signals(...)`` made
   during the audit has ``quantlab.data.load_bars``/``conversion_rate``
   patched to raise: a :class:`contracts.Strategy` must be a pure function of
   the ``bars`` frame it is handed, never a caller of the data layer itself
   (a strategy that fetches its own D1 bars for "context" can read straight
   through any cut point — M2 evidence). A strategy that genuinely needs a
   higher timeframe must receive it already causally joined onto ``bars`` by
   its caller, via :func:`align_higher_timeframe` — the one sanctioned way to
   mix timeframes.
4. :func:`assert_engine_causal` extends the same idea to
   ``quantlab.engine.run_backtest`` itself (imported lazily, so this module
   stays decoupled from the engine): the *fills*, not just the signals, must
   not change for a trade that has already closed once you cut the data.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Iterator

import numpy as np
import polars as pl

from . import contracts

_STEP: dict[str, timedelta] = {
    "M1": timedelta(minutes=1), "M5": timedelta(minutes=5), "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30), "H1": timedelta(hours=1), "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
}

_PRICE_COLUMNS = ("open", "high", "low", "close", "spread", "spread_max", "tick_vol")
_N_FUTURE_DRAWS = 3   # independent random-walk redraws checked per cut point (M2b: "several draws")


def synthetic_bars(n: int, *, seed: int, start: str = "2020-01-06", timeframe: str = "H1",
                    point: float = 1e-5, spread_points: float = 10.0) -> pl.DataFrame:
    """``n`` GBM-ish bars on a weekday-only (FX-like) calendar; valid per :mod:`contracts`.

    ``ts_utc`` treats the naive clock as UTC directly — there is no "book" to
    localise against here, and library-level tests don't need one.
    """
    if timeframe not in _STEP:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(_STEP)}")
    step = _STEP[timeframe]
    rng = np.random.default_rng(seed)

    timestamps: list[datetime] = []
    t = datetime.fromisoformat(start)
    while len(timestamps) < n:
        if t.weekday() < 5:                          # Mon-Fri only, like an FX weekly session
            timestamps.append(t)
        t = t + step

    decimals = max(0, round(-np.log10(point)))
    log_returns = rng.normal(loc=0.0, scale=0.004, size=n)
    close = np.round(np.exp(np.cumsum(log_returns)), decimals)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(loc=0.0, scale=point * 20, size=n)) + point
    high = np.round(np.maximum(open_, close) + wick, decimals)
    low = np.round(np.minimum(open_, close) - wick, decimals)
    high = np.maximum(high, np.maximum(open_, close))     # rounding must never break OHLC validity
    low = np.minimum(low, np.minimum(open_, close))
    tick_vol = rng.integers(1, 500, size=n)

    df = pl.DataFrame({
        "ts": timestamps,
        "open": open_, "high": high, "low": low, "close": close,
        "spread": np.full(n, float(spread_points)),
        "spread_max": np.full(n, float(spread_points)),
        "tick_vol": tick_vol.astype(np.int64),
    }).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    df = df.with_columns(pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"))
    return df.select(list(contracts.BAR_COLUMNS))


# --------------------------------------------------------------------------- multi-timeframe helper
def _infer_span(ts: pl.Series) -> timedelta:
    """Median positive gap between consecutive timestamps — robust to weekend/holiday gaps."""
    diffs = sorted(d for d in ts.diff().drop_nulls().to_list() if d.total_seconds() > 0)
    if not diffs:
        raise ValueError("need at least 2 distinct timestamps to infer a timeframe span")
    return diffs[len(diffs) // 2]


def align_higher_timeframe(bars_ltf: pl.DataFrame, bars_htf: pl.DataFrame) -> pl.DataFrame:
    """Join ``bars_htf`` onto ``bars_ltf``, using only HTF bars that have already **closed**.

    This is the sanctioned way for a strategy to use more than one timeframe: the caller
    builds this joined frame *before* calling ``strategy.signals()``, and the strategy reads
    the attached ``htf_*`` columns like any other column of its input — it never calls
    ``quantlab.data.load_bars`` itself (which :func:`assert_no_lookahead` forbids from
    inside ``signals()``, see the module docstring).

    For each LTF row, attaches the last HTF bar whose own window has fully closed at or
    before that LTF row's ``ts`` (its open time) — an as-of (backward) join on the HTF
    bar's **close** time (``ts`` + the HTF span, inferred from ``bars_htf`` itself via
    :func:`_infer_span`), never on its open time. Joining on the open time would expose
    an HTF bar's still-forming range to any LTF row inside that same window — exactly the
    kind of look-ahead this module exists to catch.

    Returns ``bars_ltf`` with every ``bars_htf`` column renamed ``htf_<col>`` (so
    ``htf_ts`` is the attached bar's own open time, ``htf_close`` its close price, etc.);
    these are null on LTF rows that come before the first HTF bar has closed.
    """
    if bars_htf.height == 0:
        raise ValueError("bars_htf is empty")
    span = _infer_span(bars_htf["ts"])
    htf = (
        bars_htf.sort("ts")
        .rename({c: f"htf_{c}" for c in bars_htf.columns})
        .with_columns((pl.col("htf_ts") + span).alias("__htf_close_ts"))
    )
    return (
        bars_ltf.sort("ts")
        .join_asof(htf, left_on="ts", right_on="__htf_close_ts", strategy="backward")
        .drop("__htf_close_ts")
    )


# --------------------------------------------------------------------------- sandboxed data access
class _ForbiddenDataAccess(RuntimeError):
    """Raised by the sandbox stub; always converted to an AssertionError before it escapes
    :func:`assert_no_lookahead`/:func:`assert_engine_causal` (see :func:`_run_signals`)."""


_STRATEGY_DATA_ACCESS_MESSAGE = (
    "strategies must not load data themselves: quantlab.data.load_bars/conversion_rate was "
    "called from inside strategy.signals(). A Strategy.signals() must be a pure function of "
    "the `bars` frame it is given (DESIGN §4.3/§10) -- a strategy that reads its own extra "
    "window of data can trivially see past any truncation point. Multi-timeframe strategies "
    "must receive the higher-timeframe bars already causally joined onto `bars` by their "
    "caller -- see quantlab.testing.align_higher_timeframe -- and take that as an ordinary "
    "extra input column, not fetch it themselves."
)


@contextmanager
def _sandbox_strategy_data_access() -> Iterator[None]:
    """While active, any call to ``quantlab.data.load_bars``/``conversion_rate`` raises."""
    from . import data as data_module

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise _ForbiddenDataAccess(_STRATEGY_DATA_ACCESS_MESSAGE)

    original_load_bars = data_module.load_bars
    original_conversion_rate = data_module.conversion_rate
    data_module.load_bars = _raise          # type: ignore[assignment]
    data_module.conversion_rate = _raise    # type: ignore[assignment]
    try:
        yield
    finally:
        data_module.load_bars = original_load_bars
        data_module.conversion_rate = original_conversion_rate


def _run_signals(strategy: Any, bars: pl.DataFrame, *, context: str) -> pl.DataFrame:
    """``strategy.signals(bars)`` with ``quantlab.data`` sandboxed; converts a forbidden
    data-access attempt into the same :class:`AssertionError` shape as any other leak."""
    with _sandbox_strategy_data_access():
        try:
            return strategy.signals(bars)
        except _ForbiddenDataAccess as exc:
            raise AssertionError(f"look-ahead detected ({context}): {exc}") from exc


# --------------------------------------------------------------------------- look-ahead audit
def _check_shape(signals: pl.DataFrame, expected_len: int, context: str) -> None:
    missing = [c for c in contracts.SIGNAL_COLUMNS if c not in signals.columns]
    if missing:
        raise AssertionError(f"{context}: signals frame is missing columns {missing}")
    if signals.height != expected_len:
        raise AssertionError(f"{context}: signals frame has {signals.height} rows, expected {expected_len}")


def _mismatch_mask(a: pl.Series, b: pl.Series, tol: float) -> pl.Series:
    both_null = a.is_null() & b.is_null()
    either_null = a.is_null() | b.is_null()
    diff = (a.fill_null(0.0).cast(pl.Float64) - b.fill_null(0.0).cast(pl.Float64)).abs()
    return (~both_null) & (either_null | (diff > tol))


def _assert_prefix_matches(full: pl.DataFrame, other: pl.DataFrame, *, upto: int, bars: pl.DataFrame,
                            context: str) -> None:
    a = full.slice(0, upto)
    b = other.slice(0, upto)
    for col in contracts.SIGNAL_COLUMNS:
        tol = 0.0 if col == "signal" else 1e-12
        mask = _mismatch_mask(a[col], b[col], tol)
        if mask.any():
            i = int(mask.arg_true()[0])
            raise AssertionError(
                f"look-ahead detected ({context}): column {col!r} differs at row {i} "
                f"(ts={bars['ts'][i]}): full-run={a[col][i]!r} vs this-run={b[col][i]!r}"
            )


def _infer_decimals(closes: np.ndarray) -> int:
    """Max decimal places seen in a (small, recent) sample of closes."""
    max_decimals = 0
    for x in closes[-100:]:
        text = f"{round(float(x), 8):.8f}".rstrip("0").rstrip(".")
        decimals = len(text.split(".")[1]) if "." in text else 0
        max_decimals = max(max_decimals, decimals)
    return max_decimals


def _independent_future_walk(bars: pl.DataFrame, t: int, rng: np.random.Generator) -> pl.DataFrame:
    """Replace everything strictly after ``t`` with a fresh, independently-drawn random walk.

    Unlike a permutation of the real future rows (which leaves the future's mean/std/max
    exactly as they were), this generates brand-new prices from scratch — anchored near bar
    ``t``'s close but with their own level shift and volatility rescale — so it also catches
    leaks that read a *statistic* of the future rather than a specific future *value*
    (though truncation alone already catches full-sample normalisation). ``ts``/``ts_utc``
    are left untouched: only price/spread/volume columns after ``t`` are redrawn.
    """
    n = bars.height
    future_idx = np.arange(t + 1, n)
    m = future_idx.size
    if m == 0:
        return bars

    close_hist = bars["close"].to_numpy()[: t + 1].astype(np.float64)
    anchor = float(close_hist[-1])
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ret = np.diff(np.log(np.clip(close_hist, 1e-12, None)))
    hist_scale = float(np.nanstd(log_ret)) if log_ret.size >= 2 else 0.0
    if not np.isfinite(hist_scale) or hist_scale <= 0:
        hist_scale = 0.004
    scale = hist_scale * rng.uniform(0.5, 2.0)              # "rescale": needn't match the real future's vol
    level_shift = rng.normal(loc=0.0, scale=hist_scale * 10.0)

    decimals = _infer_decimals(close_hist)
    point = 10.0 ** (-decimals) if decimals > 0 else 1.0

    log_returns = rng.normal(loc=0.0, scale=scale, size=m)
    close = np.round(anchor * np.exp(level_shift + np.cumsum(log_returns)), decimals)
    open_ = np.empty(m)
    open_[0] = anchor
    open_[1:] = close[:-1]
    wick = np.abs(rng.normal(loc=0.0, scale=point * 20, size=m)) + point
    high = np.maximum(np.round(np.maximum(open_, close) + wick, decimals), np.maximum(open_, close))
    low = np.minimum(np.round(np.minimum(open_, close) - wick, decimals), np.minimum(open_, close))

    spread_hist = bars["spread"].to_numpy()[: t + 1].astype(np.float64)
    spread_level = float(np.nanmedian(spread_hist)) if spread_hist.size else 10.0
    spread = np.full(m, max(spread_level, 0.0))

    tick_hist = bars["tick_vol"].to_numpy()[: t + 1]
    tick_level = max(1, int(np.nanmedian(tick_hist))) if tick_hist.size else 100
    tick_vol = rng.integers(max(1, tick_level // 2), tick_level * 2 + 2, size=m)

    future_values = {
        "open": open_, "high": high, "low": low, "close": close,
        "spread": spread, "spread_max": spread, "tick_vol": tick_vol,
    }
    out = bars
    for col in _PRICE_COLUMNS:
        values = out[col].to_numpy().copy()
        values[future_idx] = future_values[col]
        out = out.with_columns(pl.Series(col, values, dtype=out[col].dtype))
    return out


def _forced_cut_points(full: pl.DataFrame, n: int) -> set[int]:
    """Every row (and its predecessor) where the full-run signal is active (non-null,
    non-zero) — a k-bar look-ahead is only visible when a cut point falls in the window
    it affects, and a sparse strategy (a crossover system, say) can be active on well
    under 2% of rows, which random sampling alone reliably misses (M2)."""
    active = (full["signal"].is_not_null() & (full["signal"] != 0)).to_numpy()
    active_idx = np.nonzero(active)[0]
    forced: set[int] = set()
    for r in active_idx.tolist():
        for cand in (r - 1, r):
            if 0 <= cand <= n - 2:
                forced.add(int(cand))
    return forced


def assert_no_lookahead(strategy: Any, bars: pl.DataFrame, *, n_checks: int = 25, seed: int = 0,
                        min_history: int = 200) -> None:
    """Prove a :class:`contracts.Strategy` only uses bars up to and including the current one.

    Cut points ``t`` are every row flagged by :func:`_forced_cut_points` plus up to
    ``n_checks`` more sampled uniformly at random from ``[min_history, len(bars) - 1)``.
    At each cut point:

    1. **truncation** — ``strategy.signals(bars[:t+1])`` matches the full-run signals on
       rows ``<= t`` (null-aware, float tolerance ``1e-12``).
    2. **future-perturbation** — several independent :func:`_independent_future_walk`
       redraws each leave rows ``<= t`` unchanged.

    Every ``strategy.signals(...)`` call is sandboxed (module docstring, point 3): a call
    to ``quantlab.data.load_bars``/``conversion_rate`` from inside ``signals()`` raises.

    Raises :class:`AssertionError` naming the first offending timestamp and column.
    """
    n = bars.height
    if n <= min_history + 1:
        raise ValueError(f"bars has {n} rows; need more than min_history + 1 = {min_history + 1}")
    rng = np.random.default_rng(seed)

    full = _run_signals(strategy, bars, context="full run")
    _check_shape(full, n, "full run")

    forced = _forced_cut_points(full, n)
    pool = [t for t in range(min_history, n - 1) if t not in forced]
    rng.shuffle(pool)
    cut_points = sorted(forced) + pool[: min(n_checks, len(pool))]

    for t in cut_points:
        context = f"truncation@t={t}"
        truncated = bars.slice(0, t + 1)
        sig_trunc = _run_signals(strategy, truncated, context=context)
        _check_shape(sig_trunc, t + 1, context)
        _assert_prefix_matches(full, sig_trunc, upto=t + 1, bars=bars, context=context)

        for draw in range(_N_FUTURE_DRAWS):
            pctx = f"future-perturbation@t={t},draw={draw}"
            perturbed = _independent_future_walk(bars, t, rng)
            sig_pert = _run_signals(strategy, perturbed, context=pctx)
            _check_shape(sig_pert, n, pctx)
            _assert_prefix_matches(full, sig_pert, upto=t + 1, bars=bars, context=pctx)


# --------------------------------------------------------------------------- engine-level causality
def assert_engine_causal(strategy: Any, bars: pl.DataFrame, spec: Any, m1: pl.DataFrame | None = None,
                          *, n_checks: int = 10, seed: int = 0, min_history: int = 200) -> None:
    """Prove ``quantlab.engine.run_backtest`` itself is causal, not just the strategy's signals.

    Runs the backtest once on the full ``bars`` (and optional ``m1``, passed through
    untouched), and again on ``bars`` truncated at several random cut points ``t`` — this
    deliberately mirrors walk-forward/CPCV code that loads M1 once and backtests fold
    slices of the HTF bars (the real-world scenario in which a look-ahead in the engine's
    HTF->M1 bar-mapping went undetected, DESIGN §10 red team finding M1). Every trade that
    closed strictly *before* the cut bar (``exit_idx < t``) must be identical between the
    two runs; a mismatch means data from beyond the cut changed an already-resolved trade.
    (Deliberately strict, not ``<=``: the trade open exactly at bar ``t`` is expected to
    differ — the truncated run legitimately force-closes it there as `eod`, since as far
    as it knows that is the end of the data, while the full run keeps holding it. That is
    ordinary truncation behaviour, not a causality bug.)

    ``spec`` is a ``costs.InstrumentSpec``; ``quantlab.engine.run_backtest`` is imported
    lazily so this module stays decoupled from the engine.
    """
    from .engine import run_backtest

    n = bars.height
    if n <= min_history + 1:
        raise ValueError(f"bars has {n} rows; need more than min_history + 1 = {min_history + 1}")

    full_sig = _run_signals(strategy, bars, context="engine-causal full run")
    full_trades = run_backtest(bars, full_sig, spec, m1=m1).trades

    rng = np.random.default_rng(seed)
    candidates = np.arange(min_history, n - 1)
    rng.shuffle(candidates)
    compare_cols = ("entry_idx", "exit_idx", "direction", "entry_price", "exit_price", "pnl_points")

    for raw_t in candidates[: min(n_checks, candidates.size)]:
        t = int(raw_t)
        context = f"engine-causal@t={t}"
        bars_trunc = bars.slice(0, t + 1)
        sig_trunc = _run_signals(strategy, bars_trunc, context=context)
        trunc_trades = run_backtest(bars_trunc, sig_trunc, spec, m1=m1).trades

        full_closed = full_trades.filter(pl.col("exit_idx") < t).sort("entry_idx")
        trunc_closed = trunc_trades.filter(pl.col("exit_idx") < t).sort("entry_idx")
        if full_closed.height != trunc_closed.height:
            raise AssertionError(
                f"engine causality violated ({context}): {full_closed.height} trades closed before "
                f"the cut on the full run vs {trunc_closed.height} on the truncated run"
            )
        for col in compare_cols:
            tol = 0.0 if full_closed[col].dtype in (pl.Int8, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64) else 1e-9
            mask = _mismatch_mask(full_closed[col], trunc_closed[col], tol)
            if mask.any():
                i = int(mask.arg_true()[0])
                raise AssertionError(
                    f"engine causality violated ({context}): column {col!r} differs on the trade closed "
                    f"at exit_idx={trunc_closed['exit_idx'][i]}: full-run={full_closed[col][i]!r} vs "
                    f"truncated-run={trunc_closed[col][i]!r}"
                )
