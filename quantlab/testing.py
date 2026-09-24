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
   during the audit runs with :data:`quantlab.data.AUDIT_IN_PROGRESS` set: a
   :class:`contracts.Strategy` must be a pure function of the ``bars`` frame
   it is handed, never a caller of the data layer itself (a strategy that
   fetches its own D1 bars for "context" can read straight through any cut
   point — M2 evidence). A strategy that genuinely needs a higher timeframe
   must receive it already causally joined onto ``bars`` by its caller, via
   :func:`align_higher_timeframe` — the one sanctioned way to mix timeframes.

   The guard is a flag checked *inside* every :mod:`quantlab.data` entry
   point itself (``load_bars``, ``_load_resampled``, ``conversion_rate``, ...),
   not a rebind of the ``quantlab.data`` module's attributes — a red-team
   escape (M2 residual) showed that rebinding only stops the exact spelling
   ``data.load_bars(...)``: a reference imported earlier
   (``from quantlab.data import load_bars``), a private helper called
   directly, or data fetched once in ``__init__`` and closed over all sailed
   straight through it. Checking the flag inside the function itself catches
   every calling convention, because it is the same code object no matter how
   it was reached. Two gaps remain, both handled separately: reading the raw
   Parquet catalog file directly (this module also patches ``polars``' and
   ``pyarrow.parquet``'s readers for the duration of the sandboxed call), and
   data fetched in a strategy's own ``__init__`` *before* the audit starts —
   see the ``factory=`` argument to :func:`assert_no_lookahead`.
4. :func:`assert_engine_causal` extends the same idea to
   ``quantlab.engine.run_backtest`` itself (imported lazily, so this module
   stays decoupled from the engine): the *fills*, not just the signals, must
   not change for a trade that has already closed once you cut the data.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

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
    """Raised by the raw-reader sandbox stubs (see :func:`_sandbox_strategy_data_access`);
    always converted to an AssertionError before it escapes :func:`assert_no_lookahead`/
    :func:`assert_engine_causal` (see :func:`_run_sandboxed`). ``quantlab.data``'s own entry
    points raise :class:`quantlab.data.AuditedDataAccess` instead, caught the same way."""


_RAW_READ_MESSAGE = (
    "a raw Parquet/CSV reader (polars or pyarrow) was called from inside an audited "
    "strategy.signals()/__init__. Strategies must not read the data catalog directly -- go "
    "through the `bars` frame they are given (DESIGN §4.3/§10). Multi-timeframe strategies "
    "must receive the higher-timeframe bars already causally joined onto `bars` by their "
    "caller -- see quantlab.testing.align_higher_timeframe -- and take that as an ordinary "
    "extra input column, not fetch it themselves."
)

# Readers a strategy could use to bypass quantlab.data entirely and reach the raw Parquet
# catalog files (M2 residual: `pl.scan_parquet(<catalog file>)` passed the pre-fix audit
# because only quantlab.data's own functions were sandboxed). Patched for the duration of
# every sandboxed call; quantlab.data's own (legitimate, unsandboxed) reads never run while
# this patch is active, because quantlab.data.AUDIT_IN_PROGRESS makes its own entry points
# raise first -- see the module docstring, point 3.
_RAW_READERS: tuple[tuple[Any, str], ...] = (
    (pl, "read_parquet"), (pl, "scan_parquet"), (pl, "read_csv"), (pl, "scan_csv"),
)


def _raise_raw_read(*_args: Any, **_kwargs: Any) -> Any:
    raise _ForbiddenDataAccess(_RAW_READ_MESSAGE)


@contextmanager
def _sandbox_strategy_data_access() -> Iterator[None]:
    """While active: :data:`quantlab.data.AUDIT_IN_PROGRESS` is set (checked inside every
    quantlab.data entry point itself, regardless of how it's called -- module docstring,
    point 3), and every raw Parquet/CSV reader in :data:`_RAW_READERS` (plus
    ``pyarrow.parquet.read_table``, if pyarrow is importable) is patched to raise.
    """
    from . import data as data_module

    targets: list[tuple[Any, str, Any]] = [(obj, name, getattr(obj, name)) for obj, name in _RAW_READERS]
    try:
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - pyarrow ships with polars in this project
        pass
    else:
        targets.append((pq, "read_table", pq.read_table))

    token = data_module.AUDIT_IN_PROGRESS.set(True)
    for obj, name, _orig in targets:
        setattr(obj, name, _raise_raw_read)
    try:
        yield
    finally:
        # Restore every reader first, each on its own, so nothing a strategy does (e.g.
        # importlib.reload(quantlab.data), which invalidates `token`) can leave the process
        # with patched readers (red-team R2-1).
        for obj, name, orig in targets:
            try:
                setattr(obj, name, orig)
            except Exception:  # pragma: no cover - setattr on a module attribute
                pass
        try:
            data_module.AUDIT_IN_PROGRESS.reset(token)
        except ValueError:
            # token belongs to a reloaded module's ContextVar; clear the current one instead.
            from . import data as reloaded
            reloaded.AUDIT_IN_PROGRESS.set(False)


def _run_sandboxed(fn: Callable[[], Any], *, context: str) -> Any:
    """Call ``fn()`` with the data-access sandbox active (module docstring, point 3);
    converts a forbidden data-access attempt -- from ``quantlab.data``'s own guard or a
    patched raw reader -- into the same :class:`AssertionError` shape as any other leak."""
    from . import data as data_module

    with _sandbox_strategy_data_access():
        try:
            return fn()
        except (_ForbiddenDataAccess, data_module.AuditedDataAccess) as exc:
            raise AssertionError(f"look-ahead detected ({context}): {exc}") from exc


def _run_signals(strategy: Any, bars: pl.DataFrame, *, context: str) -> pl.DataFrame:
    """``strategy.signals(bars)`` sandboxed -- see :func:`_run_sandboxed`."""
    return _run_sandboxed(lambda: strategy.signals(bars), context=context)


def _construct_via_factory(factory: Callable[[], Any], *, context: str) -> Any:
    """``factory()`` sandboxed -- catches data fetched in a strategy's own ``__init__``
    (M2 residual: a strategy that loads data once outside ``signals()`` and closes over it
    was invisible to any amount of sandboxing ``signals()`` alone). See
    :func:`assert_no_lookahead`'s ``factory`` argument."""
    return _run_sandboxed(factory, context=context)


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


QUICK_MAX_FORCED_CUTS = 200   # opt-in quick mode only (N6); gate runs are exhaustive (R2-2)


def _forced_cut_points(full: pl.DataFrame, n: int, cap: int | None = None) -> set[int]:
    """Rows (and their predecessor) where the full-run signal *changes* value — not every
    row where it merely happens to be active (N6). A k-bar look-ahead is only visible when
    a cut point falls in the window it affects, and a sparse strategy (a crossover system,
    say) can change on well under 2% of rows, which random sampling alone reliably misses
    (M2). Using "changed" rather than "active, non-null, non-zero" (the pre-N6 rule) matters
    for the opposite case too: a dense, always-in-the-market strategy (±1 on every row,
    never null) used to force a cut at *every* row, making the audit's cost O(n) calls to
    ``signals()`` each O(n) — quadratic overall, and impractical on a multi-year frame.
    A transition into/out of null (i.e. into/out of "keep current state") also counts as a
    change; two consecutive nulls do not (there is nothing new to test there).

    ``cap=None`` (the default, and mandatory for gate runs) keeps **every** change point:
    capping lets a sparse one-bar peek hide between sampled cuts (red-team R2-2). A cap
    (e.g. :data:`QUICK_MAX_FORCED_CUTS`) samples evenly and deterministically across the
    sorted change points -- only for quick local iteration.
    """
    changed = full["signal"].ne_missing(full["signal"].shift(1)).to_numpy()
    changed_idx = np.nonzero(changed)[0]
    forced: set[int] = set()
    for r in changed_idx.tolist():
        for cand in (r - 1, r):
            if 0 <= cand <= n - 2:
                forced.add(int(cand))
    if cap is not None and len(forced) > cap:
        ordered = sorted(forced)
        keep = np.linspace(0, len(ordered) - 1, cap)
        forced = {ordered[i] for i in np.unique(np.round(keep).astype(int))}
    return forced


def assert_no_lookahead(factory: Callable[[], Any] | None = None, bars: pl.DataFrame | None = None, *,
                        n_checks: int = 25, seed: int = 0, min_history: int = 200,
                        max_forced_cuts: int | None = None) -> None:
    """Prove a :class:`contracts.Strategy` only uses bars up to and including the current one.

    ``factory`` is a zero-arg callable that builds the strategy -- the strategy **class**
    itself works (``assert_no_lookahead(MyStrategy, bars)``), or a lambda binding params.
    It is required (red-team R2-1): the strategy is constructed *inside* the sandbox, so
    data fetched in ``__init__`` and closed over is caught too. Pre-built instances are
    rejected because they can carry data captured before the audit started.

    Module-level I/O (data cached at import time) and deliberate bypasses of the sandbox
    (raw file reads, subprocesses) cannot be caught at runtime; :func:`lint_strategy_source`
    covers those statically and is required alongside this audit.

    Cut points ``t`` are **every** signal-change row (:func:`_forced_cut_points`; pass
    ``max_forced_cuts`` only for quick local iteration, never for gate runs) plus up to
    ``n_checks`` more sampled uniformly at random from ``[min_history, len(bars) - 1)``.
    At each cut point:

    1. **truncation** — ``strategy.signals(bars[:t+1])`` matches the full-run signals on
       rows ``<= t`` (null-aware, float tolerance ``1e-12``).
    2. **future-perturbation** — several independent :func:`_independent_future_walk`
       redraws each leave rows ``<= t`` unchanged.

    Every ``strategy.signals(...)`` call (and, with ``factory=``, the construction call
    itself) is sandboxed (module docstring, point 3).

    Raises :class:`AssertionError` naming the first offending timestamp and column.
    """
    if bars is None:
        raise ValueError("assert_no_lookahead: `bars` is required")
    if factory is None:
        raise ValueError("assert_no_lookahead: `factory` is required (e.g. the strategy class)")
    if not callable(factory) or (hasattr(factory, "signals") and not isinstance(factory, type)):
        raise TypeError("assert_no_lookahead: pass a factory (the strategy class or a zero-arg "
                        "callable), not a pre-built instance -- instances can hold data captured "
                        "before the audit started (red-team R2-1)")
    strategy = _construct_via_factory(factory, context="strategy construction (factory)")

    n = bars.height
    if n <= min_history + 1:
        raise ValueError(f"bars has {n} rows; need more than min_history + 1 = {min_history + 1}")
    rng = np.random.default_rng(seed)

    full = _run_signals(strategy, bars, context="full run")
    _check_shape(full, n, "full run")

    forced = _forced_cut_points(full, n, cap=max_forced_cuts)
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
                          *, timeframe: str | None = None, n_checks: int = 10, seed: int = 0,
                          min_history: int = 200) -> None:
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

    ``spec`` is a ``costs.InstrumentSpec``; ``timeframe`` is forwarded to ``run_backtest``
    (required there whenever ``m1`` is given). ``quantlab.engine.run_backtest`` is imported
    lazily so this module stays decoupled from the engine.
    """
    from .engine import run_backtest

    if m1 is not None and timeframe is None:
        raise ValueError("assert_engine_causal: pass timeframe= when m1 is given")

    n = bars.height
    if n <= min_history + 1:
        raise ValueError(f"bars has {n} rows; need more than min_history + 1 = {min_history + 1}")

    full_sig = _run_signals(strategy, bars, context="engine-causal full run")
    full_trades = run_backtest(bars, full_sig, spec, m1=m1, timeframe=timeframe).trades

    rng = np.random.default_rng(seed)
    candidates = np.arange(min_history, n - 1)
    rng.shuffle(candidates)
    compare_cols = ("entry_idx", "exit_idx", "direction", "entry_price", "exit_price", "pnl_points")

    for raw_t in candidates[: min(n_checks, candidates.size)]:
        t = int(raw_t)
        context = f"engine-causal@t={t}"
        bars_trunc = bars.slice(0, t + 1)
        sig_trunc = _run_signals(strategy, bars_trunc, context=context)
        trunc_trades = run_backtest(bars_trunc, sig_trunc, spec, m1=m1, timeframe=timeframe).trades

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


# --------------------------------------------------------------------------- static strategy lint
# Runtime sandboxing can't see I/O done at import time or deliberate bypasses (raw file reads,
# subprocesses, reloading quantlab.data); red-team R2-1.  Strategy modules are therefore also
# checked statically: they may compute on the `bars` they receive, nothing else.
_FORBIDDEN_MODULES = (
    "quantlab.data", "subprocess", "importlib", "socket", "urllib", "http", "requests",
    "pickle", "shelve", "sqlite3", "ctypes", "multiprocessing", "pyarrow", "pandas", "duckdb",
)
_FORBIDDEN_CALL_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "vars"}
_FORBIDDEN_ATTR_PREFIXES = ("read_", "scan_", "sink_", "write_")
_FORBIDDEN_ATTRS = {"load", "loadtxt", "fromfile", "genfromtxt", "memmap", "system", "popen",
                    "read_text", "read_bytes", "open"}
_ALLOWED_MODULE_LEVEL_CALLS = {"dataclass", "field", "frozenset", "tuple", "set", "dict", "list",
                               "TypeVar", "compile_pattern", "Enum", "namedtuple", "ClassVar"}


def _module_matches(name: str) -> bool:
    return any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_MODULES)


def _call_name(node: Any) -> str:
    import ast

    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return ""


def lint_strategy_source(source_or_path: str | Path) -> list[str]:
    """Return static violations for a strategy module (empty list = clean).

    Forbidden: importing data-access / I/O / process modules (``quantlab.data``, pyarrow,
    pandas, subprocess, importlib, …); importing or calling polars ``read_*``/``scan_*``/
    ``sink_*``/``write_*``; builtins ``open``/``exec``/``eval``/``__import__``; numpy
    ``load``/``fromfile``/…; ``Path.read_text``/``read_bytes``; ``os.system``/``popen``; and
    any module-level call other than simple constructors/decorators (import-time work is
    where cached data hides).  ``/research-cycle`` requires this to be clean alongside
    :func:`assert_no_lookahead` (DESIGN §3 S2).
    """
    import ast

    is_path = isinstance(source_or_path, Path) or (
        "\n" not in str(source_or_path) and str(source_or_path).endswith(".py"))
    src = Path(source_or_path).read_text() if is_path else str(source_or_path)
    tree = ast.parse(src)
    out: list[str] = []

    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for a in node.names:
                if _module_matches(a.name):
                    out.append(f"line {line}: import of forbidden module {a.name!r}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            absolute = ("quantlab." + mod if mod else "quantlab") if node.level else mod
            names = {a.name for a in node.names}
            if _module_matches(absolute) or (absolute == "quantlab" and "data" in names):
                out.append(f"line {line}: import from forbidden module {absolute!r} {sorted(names)}")
            for a in node.names:
                if a.name.startswith(_FORBIDDEN_ATTR_PREFIXES) or a.name in _FORBIDDEN_ATTRS:
                    out.append(f"line {line}: import of I/O function {mod}.{a.name}")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if isinstance(node.func, ast.Name) and name in _FORBIDDEN_CALL_NAMES:
                out.append(f"line {line}: call to forbidden builtin {name}()")
            elif isinstance(node.func, ast.Name) and name == "getattr" and len(node.args) >= 2:
                attr = node.args[1]
                if not isinstance(attr, ast.Constant) or (isinstance(attr.value, str) and (
                        attr.value.startswith(_FORBIDDEN_ATTR_PREFIXES) or attr.value in _FORBIDDEN_ATTRS)):
                    out.append(f"line {line}: dynamic getattr() (possible I/O bypass)")
            elif isinstance(node.func, ast.Attribute) and (
                    name.startswith(_FORBIDDEN_ATTR_PREFIXES) or name in _FORBIDDEN_ATTRS):
                out.append(f"line {line}: call to I/O method .{name}()")

    # Module-level statements: only imports, defs, classes, docstrings and constant-ish assignments.
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            calls = [n for n in ast.walk(value) if isinstance(n, ast.Call)] if value is not None else []
            bad = [c for c in calls if _call_name(c) not in _ALLOWED_MODULE_LEVEL_CALLS]
            if bad:
                out.append(f"line {stmt.lineno}: module-level call {_call_name(bad[0])}() "
                           "(import-time work is forbidden in strategy modules)")
            continue
        out.append(f"line {stmt.lineno}: module-level {type(stmt).__name__} statement "
                   "(only imports, defs, classes and constants are allowed)")
    return out


def assert_strategy_source_clean(source_or_path: str | Path) -> None:
    violations = lint_strategy_source(source_or_path)
    if violations:
        raise AssertionError("strategy source lint failed:\n  " + "\n  ".join(violations))
