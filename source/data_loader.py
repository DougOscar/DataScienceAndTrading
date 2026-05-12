"""Load OHLC CSV files that follow the `<ASSET>_<TF>_<START>_<END>.csv` scheme.

Two loading patterns are supported:

* **Eager** (``load_all``) — reads every CSV up front and returns the full
  DataFrame in memory.  Convenient for small datasets and notebooks that
  exercise everything at once.
* **Lazy** (``load_all_lazy`` + :class:`LazyDataset`) — returns metadata only;
  the DataFrame is materialised on the first ``.load()`` call and can be
  released with ``.unload()``.  Use this when iterating over many
  (asset, timeframe) combinations or when running parallel workers that
  should each hold only their own slice.

The bar-close backtest still consumes a full DataFrame, but the orchestration
layer can keep one resampled frame in RAM at a time instead of all of them.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<asset>[A-Za-z0-9]+)"
    r"_(?P<timeframe>[A-Za-z0-9]+)"
    r"_(?P<start>\d{12})"
    r"_(?P<end>\d{12})\.csv$"
)

_OHLC_COLS = ("open", "high", "low", "close")
_RESAMPLE_SOURCE_TFS = {"M1", "M5", "M15", "M30"}

COL_NAMES = {
    '<DATE>': 'date',
    '<TIME>': 'time',
    '<OPEN>': 'open',
    '<HIGH>': 'high',
    '<LOW>': 'low',
    '<CLOSE>': 'close',
    '<TICKVOL>': 'tick_vol',
    '<VOL>': 'volume',
    '<SPREAD>': 'spread'
}


@dataclass
class DatasetMeta:
    asset: str
    timeframe: str
    start: pd.Timestamp
    end: pd.Timestamp
    path: Path
    group: str = ""   # subdirectory name, e.g. "forex", "b3"


def _parse_filename(path: Path, group: str = "") -> DatasetMeta | None:
    m = FILENAME_RE.match(path.name)
    if m is None:
        return None
    return DatasetMeta(
        asset=m["asset"],
        timeframe=m["timeframe"],
        start=pd.to_datetime(m["start"], format="%Y%m%d%H%M"),
        end=pd.to_datetime(m["end"], format="%Y%m%d%H%M"),
        path=path,
        group=group,
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    lowered = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for target in ("datetime", "date", "time", "timestamp",
                   "open", "high", "low", "close", "volume"):
        if target in lowered:
            rename[lowered[target]] = target
    df = df.rename(columns=COL_NAMES)
    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str).str.strip() + " "
                + df["time"].astype(str).str.strip(),
                errors="coerce",
            )
            df.drop(["date", "time"], axis=1, inplace=True)
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError("No datetime column found (expected datetime/date+time/timestamp).")

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for col in _OHLC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    df = df.dropna(subset=list(_OHLC_COLS))
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and return a clean OHLC(V) DataFrame indexed by datetime."""
    path = Path(path)
    last_err: Exception | None = None
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 4:
                break
            else:
                continue
        except Exception as e:
            last_err = e
    if df.shape[1] >= 4:
        return _normalize(df)
    else:
        raise ValueError(f"Could not parse {path}: {last_err}")


def load_all(data_dir: str | Path = "data") -> Dict[Tuple[str, str], Tuple[DatasetMeta, pd.DataFrame]]:
    """Eagerly load every CSV in ``data_dir`` (and its subdirectories).

    Returns ``{(asset, timeframe): (meta, df)}``.  Backwards-compatible — for
    new pipelines prefer :func:`load_all_lazy` so each DataFrame is only
    materialised when needed and can be released between (asset, tf) iterations.
    """
    data_dir = Path(data_dir)
    out: Dict[Tuple[str, str], Tuple[DatasetMeta, pd.DataFrame]] = {}
    if not data_dir.exists():
        return out

    def _scan(directory: Path, group: str) -> None:
        for path in sorted(directory.glob("*.csv")):
            meta = _parse_filename(path, group)
            if meta is None:
                continue
            df = load_csv(path)
            out[(meta.asset, meta.timeframe)] = (meta, df)

    _scan(data_dir, "")
    for subdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        _scan(subdir, subdir.name)

    return out


def discover_datasets(data_dir: str | Path = "data") -> Dict[Tuple[str, str], DatasetMeta]:
    """Scan ``data_dir`` and return ``{(asset, tf): DatasetMeta}`` *without* reading any rows.

    Use this when you only need to know what's available — e.g. to iterate
    (asset, tf) combinations and load each one inside a worker process.
    """
    data_dir = Path(data_dir)
    out: Dict[Tuple[str, str], DatasetMeta] = {}
    if not data_dir.exists():
        return out

    def _scan(directory: Path, group: str) -> None:
        for path in sorted(directory.glob("*.csv")):
            meta = _parse_filename(path, group)
            if meta is None:
                continue
            out[(meta.asset, meta.timeframe)] = meta

    _scan(data_dir, "")
    for subdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        _scan(subdir, subdir.name)
    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLC(V) frame to a coarser frequency (e.g. '1H', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "sum"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "tick_vol" in df.columns:
        agg["tick_vol"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=list(_OHLC_COLS))


@dataclass
class LazyDataset:
    """Metadata + on-demand materialisation of a single (asset, tf) DataFrame.

    The DataFrame is **not** read until :meth:`load` is called the first time.
    The cached frame can be released with :meth:`unload` so a long-running
    notebook doesn't accumulate every (asset, tf) it has ever touched.

    ``resample_to`` (optional) is a pandas frequency string (``"1h"``, ``"4h"``,
    ``"1D"``…).  If provided **and** the source is one of M1/M5/M15/M30, the
    loader resamples after reading; otherwise the raw frame is returned.
    """

    meta: DatasetMeta
    resample_to: str | None = None
    _cached: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def asset(self) -> str:
        return self.meta.asset

    @property
    def source_timeframe(self) -> str:
        return self.meta.timeframe

    @property
    def timeframe(self) -> str:
        return self.resample_to or self.meta.timeframe

    @property
    def group(self) -> str:
        return self.meta.group

    @property
    def is_loaded(self) -> bool:
        return self._cached is not None

    def load(self) -> pd.DataFrame:
        """Read (and resample, if requested) the underlying CSV, caching the result."""
        if self._cached is None:
            df = load_csv(self.meta.path)
            if (
                self.resample_to is not None
                and self.meta.timeframe.upper() in _RESAMPLE_SOURCE_TFS
            ):
                df = resample_ohlc(df, self.resample_to)
            self._cached = df
        return self._cached

    def unload(self) -> None:
        """Drop the cached DataFrame so it can be garbage-collected."""
        self._cached = None

    @contextmanager
    def using(self) -> Iterator[pd.DataFrame]:
        """Context manager: ``with ds.using() as df:`` loads on entry, unloads on exit.

        Pattern for streaming through many (asset, tf) combos with bounded RAM::

            for ds in datasets.values():
                with ds.using() as df:
                    result = Backtester(strategy).run(df)
                    save(result)
                # df is dropped here — only one frame in memory at a time
        """
        try:
            yield self.load()
        finally:
            self.unload()


def load_all_lazy(
    data_dir: str | Path = "data",
    *,
    resample_to: str | None = None,
) -> Dict[Tuple[str, str], LazyDataset]:
    """Discover every CSV under ``data_dir`` and return a dict of :class:`LazyDataset`.

    Nothing is read from disk until each dataset's ``.load()`` is called.
    Pass ``resample_to`` to apply a uniform resample rule to every dataset
    (skipped automatically for sources that aren't M1/M5/M15/M30).
    """
    metas = discover_datasets(data_dir)
    return {key: LazyDataset(meta=meta, resample_to=resample_to) for key, meta in metas.items()}


def build_lazy_grid(
    data_dir: str | Path = "data",
    group_timeframes: Dict[str, list[str]] | None = None,
) -> Dict[str, Dict[str, Dict[str, LazyDataset]]]:
    """Return ``{group: {timeframe: {asset: LazyDataset}}}`` without loading any CSV.

    Mirrors the ``group_tfs`` shape used in the notebooks but holds
    :class:`LazyDataset` instances instead of materialised DataFrames.

    ``group_timeframes`` maps group → list of pandas resample rules (e.g.
    ``{"forex": ["1h", "4h", "1D"]}``).  Pass ``None`` to skip resampling and
    keep each dataset at its source TF.
    """
    metas = discover_datasets(data_dir)
    grid: Dict[str, Dict[str, Dict[str, LazyDataset]]] = {}
    for (asset, _src_tf), meta in metas.items():
        group = meta.group
        if group == "":
            continue
        target_tfs = (
            group_timeframes.get(group, [meta.timeframe])
            if group_timeframes is not None
            else [meta.timeframe]
        )
        grid.setdefault(group, {})
        for tf in target_tfs:
            grid[group].setdefault(tf, {})
            # Resample only when the source is intraday-bar data; otherwise pass through.
            resample = tf if meta.timeframe.upper() in _RESAMPLE_SOURCE_TFS else None
            grid[group][tf][asset] = LazyDataset(meta=meta, resample_to=resample)
    return grid


def iter_windowed_bars(
    df: pd.DataFrame,
    lookback: int,
    *,
    chunk_size: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Yield rolling windows of ``df`` that always include the last ``lookback`` bars.

    Useful for streaming-style indicator computation: each window contains
    ``lookback`` warm-up bars followed by ``chunk_size`` "live" bars, so any
    rolling indicator with a lookback ≤ ``lookback`` can be computed on the
    window without seeing earlier history.

    Parameters
    ----------
    df : DataFrame
        Full price series.  Indexed by datetime.
    lookback : int
        Maximum indicator lookback (warm-up).  The first window is the bars
        ``[0 : lookback + chunk_size]``; subsequent windows slide forward by
        ``chunk_size`` and always carry the previous ``lookback`` bars.
    chunk_size : int, optional
        Number of new bars per window.  Defaults to ``max(1, lookback)``.

    Yields
    ------
    DataFrame
        A view of ``df`` containing ``lookback`` warm-up rows followed by up
        to ``chunk_size`` new rows.

    Notes
    -----
    The current ``Backtester`` consumes a full DataFrame; this helper exists
    for *future* streaming integration and for ad-hoc indicator audits.  It is
    a memory pattern, not a performance optimisation — pandas already keeps a
    single materialised view, so iterating windows of the same in-memory frame
    is no cheaper than running on the full frame.  The win comes from
    combining this with :class:`LazyDataset` so the *source* never enters RAM
    in full.
    """
    n = len(df)
    if chunk_size is None:
        chunk_size = max(1, lookback)
    if lookback < 0 or chunk_size <= 0:
        raise ValueError("lookback must be ≥ 0 and chunk_size ≥ 1")

    start = 0
    while start < n:
        end = min(n, start + lookback + chunk_size)
        # First window starts at 0; subsequent windows start `chunk_size` later
        # but always carry `lookback` warm-up bars on the left.
        win_start = max(0, start - lookback) if start > 0 else 0
        yield df.iloc[win_start:end]
        if end >= n:
            return
        start = end
