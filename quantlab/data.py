"""Lazy Parquet data loader (DESIGN §1, §4.1, §4.3).

Everything in ``quantlab`` reaches raw data through this module.  It:

* exposes the manifest as a typed catalog (:func:`catalog`);
* resamples the M1 archives to any of the MT5-compatible timeframes, entirely
  in **naive broker server time** (:func:`load_bars`);
* enforces the per-book holdout lock (DESIGN §4.1) — a dev-period read never
  needs anything special, a holdout-period read requires an explicit
  ``system`` with a ledger unlock, or it raises :class:`contracts.HoldoutLocked`;
* infers the price increment and currency pair of a symbol, and converts an
  amount between currencies using an as-of (backward-looking) M1 close.

Design choices worth knowing when reading this file
----------------------------------------------------
**Resampling is purely a naive-clock operation.**  MT5 timestamps in the
Parquet store are the broker's own clock reading with no timezone attached.
Grouping M1 bars into M5/M15/M30/H1/H4/D1 bars only cares about that clock
reading — never about the timezone — because that is what "aligned to server
midnight" means in the MT5 world.  Polars' ``group_by_dynamic`` anchors dynamic
windows at the Unix epoch (1970-01-01 00:00:00, itself a midnight), so a
window size that divides a day evenly (1/5/15/30/60/240 minutes) always lands
on server-midnight-aligned boundaries and a "1d" window always lands on
server calendar days.  Timezone only enters the picture afterwards, to build
``ts_utc`` from the resampled ``ts``.

**The holdout guard filters on the resampled bar's own window, not just its
open time.**  A D1 or H4 bar "belongs" to the dev period only if its *entire*
``[ts, ts + timeframe)`` window is before the cutoff.  Filtering only on
``ts`` would let a bar whose open is (say) 20:00 on the last dev day but whose
close reaches into the first holdout day leak holdout information into a
bar the caller believes is dev-only.  So the cutoff test is
``ts + timeframe <= cutoff``, which both drops such partial/straddling bars
and — as a side effect — makes the plain M1 case degenerate to the intuitive
``ts < cutoff``.

**Caching stores the full resampled series, independent of any holdout
cutoff.**  The (symbol, native timeframe, target timeframe) resample is the
expensive step; the ``start``/``end``/holdout window is a cheap filter applied
on every call, from cache or not.  This means the *cache itself* is built from
the entire M1 file (including whatever is currently past the holdout boundary)
— that's fine, because nothing keyed only by the cache is ever handed to a
caller unfiltered; :func:`load_bars` always re-applies the window (and the
straddling-bar rule above) after reading the cache.  The cache key is
``(source file name, size, mtime, target timeframe)`` so a re-exported MT5
file (bigger, newer mtime) invalidates it automatically.

**DST.**  Only the FBS book's timezone (Europe/Helsinki) observes DST; B3's
(America/Sao_Paulo) has not since 2019, before our data starts.  Two regimes,
depending on whether the *market* actually closes over the transition:

* **FX/metals** (closed on the Sundays EU DST changes happen — the market
  never holds a bar inside the transition hour itself, only needs correct
  absolute times either side of it):

  * *Ambiguous* local time (the repeated hour at the autumn fall-back):
    resolved to the **earliest** UTC offset (i.e. as if DST were still in
    effect).  This keeps ``ts_utc`` from ever stepping backwards — the instant
    right after the ambiguous hour then requires an extra +1h jump forward in
    UTC terms, which is a discontinuity but never a decrease or a duplicate.
  * *Non-existent* local time (the skipped hour at the spring-forward):
    resolved by shifting the naive wall clock forward by the DST gap (1h — the
    only gap size used by any book here) and localising the shifted value,
    i.e. the conventional "shift forward" rule.  Polars' own
    ``non_existent="null"`` covers the raising case; we then heal the nulls
    with the shifted result.

  Both are applied uniformly, without inspecting individual dates, so any
  Europe/Helsinki DST transition across the full data history (2016-2026+) is
  handled the same way.

* **Crypto** (trades 24/7, so it is the only market that can have *real* ticks
  land inside an EU DST transition hour): earlier analysis (see the old
  ``research/audits/2026-09-23_phase0_redteam.md`` Minor 3) concluded from the
  spring-forward evidence alone that crypto's server clock never implements
  the EU DST transition, and localised it with a single fixed, non-DST offset
  all year round. **That conclusion was wrong (finding N2).**
  ``research/audits/probes/p16_crypto_clock_xcorr.py`` cross-correlates
  |1-minute returns| between BTCUSD and XAUUSD/EURUSD, keyed by *naive server
  time*: the peak sits at **lag 0 in both summer and winter, every year
  checked (2019, 2022, 2024)**. Two clocks that read the same wall-clock
  number at the same instant, all year, in both DST regimes, are — by
  construction — running the *same* DST rule. Keying the same series by the
  old fixed-offset ``ts_utc`` instead moves the *summer* peak to **lag +60
  min**: the fixed-offset rule silently mislabels every summer crypto bar
  1 hour into the future relative to the true instant. So crypto is localised
  with the **same zoneinfo rule as FX/metals** (:func:`_localize_expr`), not a
  fixed offset.

  That leaves a real, narrower anomaly to explain: unlike FX/metals, crypto
  *does* have raw M1 rows whose naive server time falls inside the local hour
  that a DST-observing clock skips at the spring-forward (``ts.dt.hour() ==
  3`` on the transition Sunday, Europe/Helsinki). Two checks settle what these
  rows are:

  * **Same-day FX/metals comparison.** FX is flat closed all Sunday until its
    late-evening weekly open (``load_bars("EURUSD", "M1", start=<that Sunday>,
    end=<+1 day>)`` returns 0 rows for every spring-forward Sunday in the
    dev period) — there is no FX data to collide with, and no way for FX
    itself to ever exercise this path, which is exactly why the FX/metals
    branch above only *theoretically* needs a non-existent-hour rule.
  * **Adjacent-hour comparison, within BTCUSD itself.** The 03:xx rows are not
    flat, repeated or synthetic filler: OHLC keeps moving tick-by-tick and
    ``tick_vol`` stays in the same 1-14 range as the surrounding 02:xx/04:xx
    minutes (2019-03-31, hand-checked). So the *feed* really did keep ticking,
    once a minute, straight through what the platform's own DST rule says
    should not exist — a genuine broker/platform bug in the older
    scheduler that generated crypto minute labels without the FX/metals
    session-close logic that (incidentally) makes the DST skip a non-issue
    for them. The bug is date-scoped, not permanent: per-year counts of these
    rows (``research/audits/probes/reverify_2026-09-24/p16_crypto_clock_xcorr.out``
    investigation; counts reproduced for all three crypto symbols) run
    48-60 rows/year for 2017-2020, then drop to a stray 4 rows/year from 2021
    onward — consistent with a scheduler fix around 2021, leaving only
    boundary jitter.

  These rows are **broker artefacts**: keeping them (via the FX/metals
  shift-forward heal, which was the pre-N2 bug — see Minor 3) collides them
  with the real, already-present next-hour data and produces duplicate
  ``ts_utc``; inventing a distinct offset for them would report an hour of
  trading that, under the platform's own stated clock convention, never
  happened. :func:`load_bars` therefore **drops** them outright for crypto
  symbols (never shifts/heals them), and :func:`calendar_anomalies` reports
  the dropped minutes per date so they stay visible to anyone who needs them.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from . import config, contracts, ledger

# --------------------------------------------------------------------------- look-ahead audit guard
# Set (and reset) by quantlab.testing around an audited strategy.signals() call (or, via the
# `factory` argument, around the strategy's own construction) -- see quantlab.testing's module
# docstring and its M2 fix. Checked at the top of every function in this module that can hand
# a caller real price data, so the guard holds regardless of *how* that function was reached:
# `import quantlab.data as data; data.load_bars(...)`, `from quantlab.data import load_bars`
# bound before the audit even started, a private helper called directly, data fetched once in
# a strategy's own `__init__` and closed over, etc. Rebinding the module attribute (the pre-M2-
# fix approach) only ever caught the first of those.
AUDIT_IN_PROGRESS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "quantlab_data_audit_in_progress", default=False
)


class AuditedDataAccess(RuntimeError):
    """Raised by any :mod:`quantlab.data` entry point called while
    :data:`AUDIT_IN_PROGRESS` is set -- i.e. from inside a look-ahead-audited
    ``strategy.signals()`` call or, via ``factory=``, the strategy's own construction.
    Strategies must be a pure function of the ``bars`` frame they are given (DESIGN
    §4.3/§10); see :mod:`quantlab.testing`."""


def _forbid_during_audit(entry_point: str) -> None:
    if AUDIT_IN_PROGRESS.get():
        raise AuditedDataAccess(
            f"quantlab.data.{entry_point} was called while a look-ahead audit is in progress. "
            "Strategies must not fetch their own data -- a Strategy.signals() (and, if audited "
            "via `factory=`, its __init__) must be a pure function of the `bars` frame it is "
            "given. Multi-timeframe strategies must receive the higher-timeframe bars already "
            "causally joined onto `bars` by their caller -- see quantlab.testing.align_higher_timeframe."
        )


# --------------------------------------------------------------------------- timeframes
_TF_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_TF_EVERY: dict[str, str] = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
_TF_TIMEDELTA: dict[str, timedelta] = {k: timedelta(minutes=v) for k, v in _TF_MINUTES.items()}

_DST_GAP = timedelta(hours=1)          # the only DST delta used by any book's server_tz
# m2 (Phase 1 red team): the as-of conversion lookup must see the cross's last close *before*
# the first requested timestamp even when that timestamp follows a weekend / holiday gap
# (Monday 00:00 window starts, the data start of a cross).  7 days covers any market closure
# in the catalog; it only ever extends the window backwards (never towards the holdout).
_ASOF_LOOKBACK = timedelta(days=7)
_RAW_COLUMNS = ("ts", "open", "high", "low", "close", "spread", "tick_vol")

# Bump whenever `_resample_full`'s semantics change: this is folded into the cache key
# (Minor 8) so a stale on-disk cache from before the change can never be served silently.
# v3: crypto's spring-forward-gap M1 rows (N2) are now dropped before aggregation, not after.
_RESAMPLE_SCHEMA_VERSION = "v3-2026-09-24"


# --------------------------------------------------------------------------- catalog
def catalog(book: str | None = None, *, include_ticks: bool = False) -> pl.DataFrame:
    """Typed view of ``data/manifest.json``.

    Columns: ``symbol, market, book, timeframe, kind, file (abs path), start, end, rows``.
    Bars only unless ``include_ticks=True``.  ``book`` filters to one book's markets.
    """
    manifest: dict[str, dict[str, Any]] = json.loads(config.MANIFEST_PATH.read_text())
    rows: list[dict[str, Any]] = []
    for rel_path, meta in manifest.items():
        if not include_ticks and meta["kind"] != "bars":
            continue
        rows.append({
            "symbol": meta["symbol"],
            "market": meta["market"],
            "book": config.MARKET_TO_BOOK.get(meta["market"]),
            "timeframe": meta["timeframe"],
            "kind": meta["kind"],
            "file": str((config.ROOT / rel_path).resolve()),
            "start": meta["start"],
            "end": meta["end"],
            "rows": meta["rows"],
        })
    schema = {"symbol": pl.Utf8, "market": pl.Utf8, "book": pl.Utf8, "timeframe": pl.Utf8,
              "kind": pl.Utf8, "file": pl.Utf8, "start": pl.Utf8, "end": pl.Utf8, "rows": pl.Int64}
    df = pl.DataFrame(rows, schema=schema)
    df = df.with_columns(
        pl.col("start").str.to_datetime(strict=False),
        pl.col("end").str.to_datetime(strict=False),
    )
    if book is not None:
        want = config.get_book(book).name
        df = df.filter(pl.col("book") == want)
    return df.sort(["market", "symbol", "timeframe"])


def _resolve_source(cat: pl.DataFrame, symbol: str, source_file: str | Path | None) -> dict[str, Any]:
    """Pick the manifest row backing ``load_bars`` (DESIGN: M1-first, latest-end tiebreak)."""
    if source_file is not None:
        path = Path(source_file)
        path = path.resolve() if path.is_absolute() else (config.ROOT / path).resolve()
        match = cat.filter(pl.col("file") == str(path))
        if match.height == 0:
            raise ValueError(f"source_file {source_file!r} is not a bars entry in the catalog")
        found = match.row(0, named=True)
        if found["symbol"] != symbol:
            raise ValueError(
                f"source_file {source_file!r} holds {found['symbol']!r} data (market "
                f"{found['market']!r}), not {symbol!r}; pass the matching symbol or a source_file "
                f"that actually belongs to it"
            )
        return found
    candidates = cat.filter((pl.col("symbol") == symbol) & (pl.col("timeframe") == "M1"))
    if candidates.height == 0:
        raise ValueError(f"no M1 bars for symbol {symbol!r} in the catalog")
    if candidates.height > 1:
        candidates = candidates.sort("end", descending=True)     # WDO: two M1 files -> latest end
    return candidates.row(0, named=True)


# --------------------------------------------------------------------------- resampling + cache
def _cache_file(src_path: Path, timeframe: str, *, extra: str = "") -> Path:
    """``extra`` folds any additional resampling parameter (currently: the crypto DST-gap
    tz, see :func:`_load_resampled`) into the cache key so it can never collide with a
    cache entry built under a different value."""
    stat = src_path.stat()
    key = f"{src_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{timeframe}|{_RESAMPLE_SCHEMA_VERSION}|{extra}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:20]
    return config.CACHE_DIR / f"{src_path.stem}_{timeframe}_{digest}.parquet"


def _resample_full(src_path: Path, timeframe: str, *, drop_non_existent_local_tz: str | None = None) -> pl.DataFrame:
    """Resample the *entire* native file to ``timeframe`` bars (no start/end filter).

    ``drop_non_existent_local_tz`` (crypto only, N2 — module docstring's "DST" section):
    drop raw M1 rows whose naive server time falls inside a spring-forward gap under this
    tz *before* aggregating, so a bucket coarser than the gap itself (H4/D1) isn't quietly
    built from those broker-artefact minutes.
    """
    _forbid_during_audit("_resample_full")
    lf = pl.scan_parquet(src_path).select(list(_RAW_COLUMNS)).sort("ts")
    if drop_non_existent_local_tz is not None:
        lf = lf.filter(~_non_existent_local_mask("ts", drop_non_existent_local_tz))
    out = (
        lf.group_by_dynamic("ts", every=_TF_EVERY[timeframe], closed="left", label="left")
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("spread").first().cast(pl.Float64).alias("spread"),
            pl.col("spread").max().cast(pl.Float64).alias("spread_max"),
            pl.col("tick_vol").sum().alias("tick_vol"),
        )
    )
    return out.collect()


def _load_resampled(src_path: Path, native_tf: str, timeframe: str, *,
                     drop_non_existent_local_tz: str | None = None) -> pl.LazyFrame:
    """Full series at ``timeframe`` (ts, OHLC, spread, spread_max, tick_vol), cached (DESIGN §7).

    ``drop_non_existent_local_tz``: see :func:`_resample_full`. Threaded through here too
    (rather than filtered by the caller after the fact) so it also applies to the native
    M1 pass-through path and is folded into the cache key -- a call with vs. without it
    must never share a cached file.
    """
    _forbid_during_audit("_load_resampled")
    if timeframe == native_tf:
        # Pass-through: no groupby needed, so no cache either (predicate/projection
        # pushdown on the raw file is already fast; caching would just duplicate it).
        lf = (
            pl.scan_parquet(src_path)
            .select(list(_RAW_COLUMNS))
            .with_columns(
                pl.col("spread").cast(pl.Float64),
                pl.col("spread").cast(pl.Float64).alias("spread_max"),
            )
            .sort("ts")
        )
        if drop_non_existent_local_tz is not None:
            lf = lf.filter(~_non_existent_local_mask("ts", drop_non_existent_local_tz))
        return lf
    cache_path = _cache_file(src_path, timeframe, extra=drop_non_existent_local_tz or "")
    if cache_path.exists():
        return pl.scan_parquet(cache_path)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full = _resample_full(src_path, timeframe, drop_non_existent_local_tz=drop_non_existent_local_tz)
    # Atomic write (Minor 8): write to a private temp file in the same directory (so
    # os.replace is a same-filesystem rename, not a copy) then swap it into place, so a
    # concurrent reader either sees no file or a fully-written one -- never a partial read.
    fd, tmp_name = tempfile.mkstemp(dir=config.CACHE_DIR, prefix=f".{cache_path.stem}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        full.write_parquet(tmp_path)
        os.replace(tmp_path, cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return full.lazy()


# --------------------------------------------------------------------------- timezone
def _localize_expr(col: str, tz_name: str) -> pl.Expr:
    """Naive server time -> tz-aware UTC, for markets that actually close over the DST
    transition weekend (FX/metals/B3).  See module docstring for the DST convention."""
    naive = pl.col(col)
    primary = naive.dt.replace_time_zone(tz_name, ambiguous="earliest", non_existent="null")
    shifted = (naive + _DST_GAP).dt.replace_time_zone(tz_name, ambiguous="earliest", non_existent="null")
    return pl.coalesce([primary, shifted]).dt.convert_time_zone("UTC")


def _non_existent_local_mask(col: str, tz_name: str) -> pl.Expr:
    """True where a naive local time in ``col`` does not exist under ``tz_name`` — the
    EU spring-forward gap (module docstring, "DST" section / N2). FX/metals/B3 never have
    real rows here in practice (they are flat closed over the whole transition weekend);
    crypto does, and those rows are broker artefacts that :func:`load_bars` drops and
    :func:`calendar_anomalies` reports, rather than shifting/healing them like
    :func:`_localize_expr` does for the (theoretical, for FX/metals/B3) non-existent case.
    """
    return pl.col(col).dt.replace_time_zone(tz_name, ambiguous="earliest", non_existent="null").is_null()


def _to_naive_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise TypeError(f"expected a datetime, ISO string or None, got {type(value)!r}")


# --------------------------------------------------------------------------- load_bars
def load_bars(symbol: str, timeframe: str = "M1", *, book: str | None = None,
              start: datetime | str | None = None, end: datetime | str | None = None,
              system: str | None = None, include_holdout: bool = False,
              source_file: str | Path | None = None) -> pl.DataFrame:
    """Load bars for ``symbol`` at ``timeframe``, dev-period only unless unlocked.

    Returns exactly :data:`contracts.BAR_COLUMNS`, sorted by ``ts``.  See the
    module docstring for the resampling and holdout-guard rules.
    """
    _forbid_during_audit("load_bars")
    if timeframe not in _TF_MINUTES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(_TF_MINUTES)}")

    cat = catalog()
    row = _resolve_source(cat, symbol, source_file)
    native_tf = row["timeframe"]
    if _TF_MINUTES[timeframe] < _TF_MINUTES[native_tf]:
        raise ValueError(f"{symbol}: cannot resample native {native_tf} bars up to finer {timeframe}")

    inferred_book = row["book"]
    if inferred_book is None:
        raise ValueError(f"{symbol}: market {row['market']!r} has no entry in config.MARKET_TO_BOOK")
    book_obj = config.get_book(book) if book is not None else config.get_book(inferred_book)
    if book_obj.name != inferred_book:
        raise ValueError(
            f"book mismatch for {symbol!r}: its market {row['market']!r} belongs to book "
            f"{inferred_book!r}, not {book_obj.name!r}"
        )

    start_dt = _to_naive_datetime(start)
    end_dt = _to_naive_datetime(end)
    holdout_start = book_obj.holdout_start
    resolved_end = end_dt if end_dt is not None else holdout_start
    if resolved_end > holdout_start:
        unlocked = system is not None and ledger.is_holdout_unlocked(book_obj.name, system)
        if not (include_holdout and unlocked):
            raise contracts.HoldoutLocked(
                f"{book_obj.name}/{symbol}: requested window ends {resolved_end}, which reaches the "
                f"holdout (starts {holdout_start}); pass include_holdout=True with a `system` that has "
                f"an unlock recorded in the ledger to access it"
            )

    span = _TF_TIMEDELTA[timeframe]
    src_path = Path(row["file"])
    # Crypto (N2, module docstring "DST" section): drop spring-forward-gap M1 rows before
    # resampling, so they can never contaminate a coarser bucket either. Every other market
    # is unaffected (they are flat closed over the whole transition weekend in practice).
    dst_gap_tz = book_obj.tz_name if row["market"] == "crypto" else None
    lf = _load_resampled(src_path, native_tf, timeframe, drop_non_existent_local_tz=dst_gap_tz)
    if start_dt is not None:
        lf = lf.filter(pl.col("ts") >= start_dt)
    # Whole-bucket containment: drops any bar (partial or not) whose window
    # would reach at/past `resolved_end`, which is what keeps a straddling
    # D1/H4/etc. bar from leaking holdout minutes into the last dev bar.
    lf = lf.filter((pl.col("ts") + span) <= resolved_end)

    bars = lf.collect()
    # Crypto now uses the same zoneinfo DST rule as FX/metals/B3 (N2 -- see module docstring);
    # its gap rows are already gone, so `_localize_expr`'s shift-heal branch is dead for it too.
    tz_expr = _localize_expr("ts", book_obj.tz_name)
    bars = bars.with_columns(
        tz_expr.alias("ts_utc"),
        pl.col("spread").cast(pl.Float64),
        pl.col("spread_max").cast(pl.Float64),
    )
    return bars.select(list(contracts.BAR_COLUMNS)).sort("ts")


# --------------------------------------------------------------------------- point size
def infer_point(symbol: str, *, sample: int = 2500) -> float:
    """Price increment inferred from the data (max decimals seen in a sample of closes).

    Dev-period only (Minor 5): like every other read in this module, this never looks
    at raw rows at/after the book's holdout cutoff, even though it only reduces them to
    a decimal count. The file is still scanned lazily (not through :func:`load_bars`,
    which would pay for a resample this doesn't need) so the head/tail sampling stays cheap.
    """
    _forbid_during_audit("infer_point")
    cat = catalog()
    row = _resolve_source(cat, symbol, None)
    if row["book"] is None:
        raise ValueError(f"{symbol}: market {row['market']!r} has no entry in config.MARKET_TO_BOOK")
    holdout_start = config.get_book(row["book"]).holdout_start
    path = row["file"]
    lf = pl.scan_parquet(path).select("ts", "close").filter(pl.col("ts") < holdout_start).sort("ts")
    head = lf.head(sample).select("close").collect()["close"]
    tail = lf.tail(sample).select("close").collect()["close"]
    values = pl.concat([head, tail]).drop_nulls()
    if values.len() == 0:
        raise ValueError(f"{symbol}: no dev-period close prices to infer the point size from")
    max_decimals = 0
    for x in values.to_list():
        text = f"{round(float(x), 8):.8f}".rstrip("0").rstrip(".")
        decimals = len(text.split(".")[1]) if "." in text else 0
        max_decimals = max(max_decimals, decimals)
    return 10.0 ** (-max_decimals)


# --------------------------------------------------------------------------- calendar anomalies
def calendar_anomalies(symbol: str) -> pl.DataFrame:
    """Server-time calendar anomalies in ``symbol``'s raw M1 feed, for auditors.

    The definition of "anomalous" is market-specific:

    * **FX/metals/B3** (sessions that close over the weekend): M1 rows labelled as falling
      on a **Sunday** in naive server time. FX/metals sessions should not normally have
      Sunday bars, but a handful exist (2016 onboarding for several FX pairs; isolated
      minutes around the 2024-10-27 EU DST fall-back) and DESIGN keeps the data faithful to
      the raw MT5 feed rather than silently dropping them. One consequence worth knowing:
      resampling to D1 (or any coarser timeframe) will happily produce a tiny, otherwise-
      invisible Sunday-dated bar from just a handful of Sunday minutes — this helper is how
      to find the underlying rows without re-deriving them from the raw M1 file every time.
    * **Crypto** (trades 24/7 — a Sunday bar is completely normal for it, so the check above
      would be meaningless noise): M1 rows whose naive server time falls inside an EU
      spring-forward gap hour (N2 — module docstring's "DST" section). These are the exact
      rows :func:`load_bars` now drops for crypto symbols; they are reported here (rather
      than silently vanishing) so the excluded minutes stay visible and countable.

    Returns one row per anomalous **date** (dev-period only): ``date``, ``n_minutes``,
    ``first_ts``, ``last_ts`` (naive server time). Empty (but correctly typed) if none.
    """
    _forbid_during_audit("calendar_anomalies")
    cat = catalog()
    row = _resolve_source(cat, symbol, None)
    if row["book"] is None:
        raise ValueError(f"{symbol}: market {row['market']!r} has no entry in config.MARKET_TO_BOOK")
    book_obj = config.get_book(row["book"])
    base = pl.scan_parquet(row["file"]).select("ts").filter(pl.col("ts") < book_obj.holdout_start)
    if row["market"] == "crypto":
        flagged = base.filter(_non_existent_local_mask("ts", book_obj.tz_name))
    else:
        flagged = base.filter(pl.col("ts").dt.weekday() == 7)
    m1 = flagged.collect()
    if m1.height == 0:
        return pl.DataFrame(schema={
            "date": pl.Date, "n_minutes": pl.UInt32, "first_ts": pl.Datetime("ms"), "last_ts": pl.Datetime("ms"),
        })
    return (
        m1.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by("date")
        .agg(
            pl.len().cast(pl.UInt32).alias("n_minutes"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("ts").max().alias("last_ts"),
        )
        .sort("date")
    )


# --------------------------------------------------------------------------- currencies
_METAL_CCY: dict[str, tuple[str, str]] = {
    "XAUUSD": ("XAU", "USD"), "XAGUSD": ("XAG", "USD"),
    "Palladium": ("XPD", "USD"), "Platinum": ("XPT", "USD"),
}


def symbol_currencies(symbol: str) -> tuple[str, str]:
    """(base, quote) currency codes for a manifest symbol."""
    if symbol in _METAL_CCY:
        return _METAL_CCY[symbol]
    match = catalog().filter(pl.col("symbol") == symbol)
    if match.height == 0:
        raise ValueError(f"{symbol!r} not found in the bars catalog")
    market = match.row(0, named=True)["market"]
    if market == "b3":
        return (symbol, "BRL")
    if len(symbol) == 6 and symbol.isalpha():          # forex pairs and BTCUSD/ETHUSD/LTCUSD alike
        return (symbol[:3].upper(), symbol[3:].upper())
    raise ValueError(f"cannot infer base/quote currencies for {symbol!r} (market {market!r})")


def _pair_index(book: str) -> dict[tuple[str, str], str]:
    """(base, quote) -> symbol, for every non-B3 instrument in a book."""
    cat = catalog(book=book).filter(pl.col("market") != "b3")
    index: dict[tuple[str, str], str] = {}
    for sym in cat["symbol"].unique().to_list():
        try:
            index[symbol_currencies(sym)] = sym
        except ValueError:
            continue
    return index


def _asof_closes(symbol: str, ts_utc: pl.Series, *, book: str, system: str | None,
                  include_holdout: bool) -> pl.Series:
    """Close of the last M1 bar that had **already closed** at/before each ``ts_utc``.

    A bar with open time ``ts`` closes at ``ts + 1 minute``; only its close is actually
    known at that instant, so the as-of match is on that close time, not on ``ts`` itself
    (M3: matching on ``ts`` would return the close of the bar that is still *forming* at
    ``ts`` — a price only known 1 minute later). ``end=hi`` (not ``hi + 1min``) is exactly
    enough to cover the bar that closes right at ``hi`` and, as a side effect, fixes
    Minor 7: a ``ts_utc`` in the last minute before the holdout cutoff no longer pushes
    the internal ``load_bars`` window past the cutoff and spuriously raises
    :class:`contracts.HoldoutLocked`.

    The cross is loaded from ``min(ts) - 7 days`` (m2) so a window that starts right after a
    weekend still finds Friday's close.  A timestamp before the cross's first close (series
    start) falls back to the open of the M1 bar that opened at/before it; a timestamp before
    the series' first bar still raises.
    """
    _forbid_during_audit("_asof_closes")
    book_obj = config.get_book(book)
    naive_local = (
        ts_utc.dt.convert_time_zone(book_obj.tz_name).dt.replace_time_zone(None).cast(pl.Datetime("ms"))
    )
    lo, hi = naive_local.min(), naive_local.max()
    m1 = load_bars(symbol, "M1", book=book, start=lo - _ASOF_LOOKBACK, end=hi,
                   system=system, include_holdout=include_holdout)
    m1 = m1.select("ts", "close").with_columns((pl.col("ts") + timedelta(minutes=1)).alias("__close_ts"))
    target = pl.DataFrame({"__ts_local": naive_local}).with_row_index("__order")
    joined = (
        target.sort("__ts_local")
        .join_asof(m1.select("__close_ts", "close"), left_on="__ts_local", right_on="__close_ts",
                   strategy="backward")
        .sort("__order")
    )
    if joined["close"].null_count() > 0:
        # Timestamps before the cross's first *close* (only possible at the very start of its
        # series: the lookback covers every market closure).  The latest price known at such a
        # ``ts`` is the *open* of an M1 bar that opened at/before it (its first tick), which is
        # causal; used only where no close exists, so every other value is unchanged.
        miss = joined.filter(pl.col("close").is_null())
        opens = load_bars(symbol, "M1", book=book, start=miss["__ts_local"].min(),
                          end=miss["__ts_local"].max() + timedelta(minutes=1),
                          system=system, include_holdout=include_holdout).select("ts", "open")
        fill = (
            miss.select("__order", "__ts_local").sort("__ts_local")
            .join_asof(opens, left_on="__ts_local", right_on="ts", strategy="backward")
        )
        if fill["open"].null_count() > 0:
            raise ValueError(
                f"{symbol}: no M1 bar had opened yet at/before some requested timestamps "
                "(before the series start?)"
            )
        joined = (
            joined.join(fill.select("__order", "open"), on="__order", how="left")
            .with_columns(pl.coalesce("close", "open").alias("close"))
            .sort("__order")
        )
    return joined["close"].rename("rate")


def conversion_rate(from_ccy: str, to_ccy: str, ts_utc: pl.Series, *, book: str = "FBS",
                     system: str | None = None, include_holdout: bool = False) -> pl.Series:
    """Rate to convert an amount in ``from_ccy`` to ``to_ccy`` at each ``ts_utc`` (as-of, backward).

    Uses the direct or inverse pair in ``book`` if one exists, else triangulates
    through USD.  ``system``/``include_holdout`` are forwarded to :func:`load_bars`
    so this respects the holdout guard exactly like any other read.

    ``ts_utc`` must be a timezone-aware ``Datetime`` series (M3): a naive series is
    *not* assumed to already be UTC, because in practice callers pass naive **server**
    time (e.g. a trade's ``entry_ts``), and silently reinterpreting that as UTC shifts
    every lookup 2-3 hours into the future — a look-ahead through this very interface.
    Localise explicitly first, e.g. ``ts.dt.replace_time_zone(book_tz, ambiguous=...,
    non_existent=...).dt.convert_time_zone("UTC")``.
    """
    _forbid_during_audit("conversion_rate")
    from_ccy, to_ccy = from_ccy.upper(), to_ccy.upper()
    if not (isinstance(ts_utc.dtype, pl.Datetime) and ts_utc.dtype.time_zone is not None):
        raise ValueError(
            f"conversion_rate: ts_utc must be a timezone-aware Datetime series, got dtype "
            f"{ts_utc.dtype!r} ({'naive' if isinstance(ts_utc.dtype, pl.Datetime) else 'not a Datetime'}); "
            "a naive series is never silently assumed to be UTC (M3) -- localise it explicitly first"
        )
    if len(ts_utc) == 0:
        return pl.Series("rate", [], dtype=pl.Float64)
    if from_ccy == to_ccy:
        return pl.Series("rate", [1.0] * len(ts_utc))

    index = _pair_index(book)
    direct = index.get((from_ccy, to_ccy))
    if direct is not None:
        return _asof_closes(direct, ts_utc, book=book, system=system, include_holdout=include_holdout)
    inverse = index.get((to_ccy, from_ccy))
    if inverse is not None:
        closes = _asof_closes(inverse, ts_utc, book=book, system=system, include_holdout=include_holdout)
        return (1.0 / closes).rename("rate")
    if "USD" not in (from_ccy, to_ccy):
        via_from = conversion_rate(from_ccy, "USD", ts_utc, book=book, system=system, include_holdout=include_holdout)
        via_to = conversion_rate("USD", to_ccy, ts_utc, book=book, system=system, include_holdout=include_holdout)
        return (via_from * via_to).rename("rate")
    raise ValueError(f"no direct, inverse or USD-triangulated path from {from_ccy} to {to_ccy} in book {book!r}")
