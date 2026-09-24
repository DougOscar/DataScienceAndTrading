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

* **Crypto** (trades 24/7, so it is the only market with *real* bars sitting
  inside an EU DST transition hour — evidence from the raw feed itself settles
  what its clock does there): at the autumn fall-back the feed has exactly one
  pass through the repeated local hour (60 M1 rows, not 120 — see
  ``research/audits/probes/p15_crypto_dst_ts_utc.py``), and at the spring
  gap it has a full 48-60 rows sitting inside the hour that should not exist.
  Both facts together mean crypto's server clock **never implements the EU
  DST transition at all** — it just keeps incrementing through both the gap
  and the repeat, unlike FX/metals which genuinely stop trading over the
  transition weekend. Applying the FX/metals rule above to that kind of clock
  makes the "shifted" non-existent hour collide exactly with the real,
  unshifted next hour (both resolve to the same UTC instant — a duplicate —
  because shifting by the DST gap and then applying the *post*-transition
  offset is algebraically identical to applying the *pre*-transition offset
  directly, and a real, unambiguous bar already sits there under the
  post-transition offset). The fix is therefore not another per-row special
  case: crypto is localised with a single **fixed, non-DST ("standard time")
  offset for the whole book, all year round** (:func:`_standard_utc_offset`),
  which is trivially monotonic (a constant shift) and matches the fall-back
  evidence (no repeat) as well as the spring-forward evidence (no gap).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl

from . import config, contracts, ledger

# --------------------------------------------------------------------------- timeframes
_TF_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}
_TF_EVERY: dict[str, str] = {"M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m", "H1": "1h", "H4": "4h", "D1": "1d"}
_TF_TIMEDELTA: dict[str, timedelta] = {k: timedelta(minutes=v) for k, v in _TF_MINUTES.items()}

_DST_GAP = timedelta(hours=1)          # the only DST delta used by any book's server_tz
_RAW_COLUMNS = ("ts", "open", "high", "low", "close", "spread", "tick_vol")

# Bump whenever `_resample_full`'s semantics change: this is folded into the cache key
# (Minor 8) so a stale on-disk cache from before the change can never be served silently.
_RESAMPLE_SCHEMA_VERSION = "v2-2026-09-23"

# A winter reference date for reading each book's non-DST ("standard time") UTC offset.
# Only used for 24/7 (crypto) instruments -- see the module docstring's "DST" section.
_STANDARD_TIME_REFERENCE = datetime(2024, 1, 15)


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
def _cache_file(src_path: Path, timeframe: str) -> Path:
    stat = src_path.stat()
    key = f"{src_path.name}|{stat.st_size}|{int(stat.st_mtime)}|{timeframe}|{_RESAMPLE_SCHEMA_VERSION}"
    digest = hashlib.sha1(key.encode()).hexdigest()[:20]
    return config.CACHE_DIR / f"{src_path.stem}_{timeframe}_{digest}.parquet"


def _resample_full(src_path: Path, timeframe: str) -> pl.DataFrame:
    """Resample the *entire* native file to ``timeframe`` bars (no start/end filter)."""
    lf = pl.scan_parquet(src_path).select(list(_RAW_COLUMNS)).sort("ts")
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


def _load_resampled(src_path: Path, native_tf: str, timeframe: str) -> pl.LazyFrame:
    """Full series at ``timeframe`` (ts, OHLC, spread, spread_max, tick_vol), cached (DESIGN §7)."""
    if timeframe == native_tf:
        # Pass-through: no groupby needed, so no cache either (predicate/projection
        # pushdown on the raw file is already fast; caching would just duplicate it).
        return (
            pl.scan_parquet(src_path)
            .select(list(_RAW_COLUMNS))
            .with_columns(
                pl.col("spread").cast(pl.Float64),
                pl.col("spread").cast(pl.Float64).alias("spread_max"),
            )
            .sort("ts")
        )
    cache_path = _cache_file(src_path, timeframe)
    if cache_path.exists():
        return pl.scan_parquet(cache_path)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    full = _resample_full(src_path, timeframe)
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


def _standard_utc_offset(tz_name: str) -> timedelta:
    """``tz_name``'s non-DST ("standard time") UTC offset, read off a fixed winter date.

    Used only for 24/7 (crypto) instruments (module docstring, "DST" section) — their
    server clock never observes the EU DST transition, so it is localised with this one
    fixed offset all year round instead of :func:`_localize_expr`'s per-row zoneinfo rule.
    """
    offset = ZoneInfo(tz_name).utcoffset(_STANDARD_TIME_REFERENCE)
    if offset is None:  # pragma: no cover - zoneinfo always returns an offset for aware dt
        raise ValueError(f"could not determine a UTC offset for {tz_name!r}")
    return offset


def _localize_expr_fixed_offset(col: str, tz_name: str) -> pl.Expr:
    """Naive server time -> tz-aware UTC using a single, constant (non-DST) offset."""
    offset = _standard_utc_offset(tz_name)
    return (pl.col(col) - offset).dt.replace_time_zone("UTC")


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
    lf = _load_resampled(src_path, native_tf, timeframe)
    if start_dt is not None:
        lf = lf.filter(pl.col("ts") >= start_dt)
    # Whole-bucket containment: drops any bar (partial or not) whose window
    # would reach at/past `resolved_end`, which is what keeps a straddling
    # D1/H4/etc. bar from leaking holdout minutes into the last dev bar.
    lf = lf.filter((pl.col("ts") + span) <= resolved_end)

    bars = lf.collect()
    # Crypto never observes the EU DST transition (module docstring, "DST" section) -> a
    # fixed, non-DST offset all year round; every other market uses the real zoneinfo rule.
    tz_expr = (_localize_expr_fixed_offset if row["market"] == "crypto" else _localize_expr)(
        "ts", book_obj.tz_name
    )
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

    Currently flags M1 rows labelled as falling on a **Sunday** in naive server time —
    FX/metals/crypto sessions should not normally have Sunday bars, but a handful exist
    (2016 onboarding for several FX pairs; isolated minutes around the 2024-10-27 EU
    DST fall-back) and DESIGN keeps the data faithful to the raw MT5 feed rather than
    silently dropping them. One consequence worth knowing: resampling to D1 (or any
    coarser timeframe) will happily produce a tiny, otherwise-invisible Sunday-dated bar
    from just a handful of Sunday minutes — this helper is how to find the underlying
    rows without re-deriving them from the raw M1 file every time.

    Returns one row per anomalous **date** (dev-period only): ``date``, ``n_minutes``,
    ``first_ts``, ``last_ts`` (naive server time). Empty (but correctly typed) if none.
    """
    cat = catalog()
    row = _resolve_source(cat, symbol, None)
    if row["book"] is None:
        raise ValueError(f"{symbol}: market {row['market']!r} has no entry in config.MARKET_TO_BOOK")
    holdout_start = config.get_book(row["book"]).holdout_start
    m1 = (
        pl.scan_parquet(row["file"]).select("ts")
        .filter((pl.col("ts") < holdout_start) & (pl.col("ts").dt.weekday() == 7))
        .collect()
    )
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
    """
    book_obj = config.get_book(book)
    naive_local = (
        ts_utc.dt.convert_time_zone(book_obj.tz_name).dt.replace_time_zone(None).cast(pl.Datetime("ms"))
    )
    lo, hi = naive_local.min(), naive_local.max()
    m1 = load_bars(symbol, "M1", book=book, start=lo - timedelta(minutes=5), end=hi,
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
        raise ValueError(
            f"{symbol}: no M1 bar had closed yet at/before some requested timestamps "
            "(before the series start?)"
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
