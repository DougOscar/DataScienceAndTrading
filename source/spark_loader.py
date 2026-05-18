"""PySpark data layer for the 3-market multi-filter system.

The raw market data is MetaTrader-5 M1 exports (tab-separated, ~200-330 MB per
file, 31 forex + 3 crypto + 23 B3 instruments).  Reading and *resampling* every
M1 file with pandas in the driver process would blow RAM and waste cores, so the
heavy ETL is pushed to Spark:

* one ``SparkSession`` (``local[*]``) reads a CSV as a distributed DataFrame,
* the M1 bars are aggregated into the requested timeframe with a tumbling
  ``window`` (true OHLCV: open = first by time, close = last by time,
  high = max, low = min, tick/real volume = sum),
* the (small) resampled frame is written to a parquet cache keyed by the source
  file's mtime so a notebook re-run is instant,
* the backtest / WFO layers consume the cached parquet through a
  :class:`SparkResampledDataset` that mirrors the lazy-load contract used
  elsewhere in :mod:`source` (``.load()`` / ``.using()`` / ``.unload()``).

Spark is an optional dependency: importing this module never imports pyspark.
Every entry point that needs Spark imports it lazily and raises a clear,
actionable error if it (or a JDK) is missing — so ``import source`` keeps
working for the non-Spark notebooks.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import pandas as pd

from .data_loader import DatasetMeta, discover_datasets

try:  # presence check only — do NOT import the heavy submodules at module load
    import importlib.util as _ilu

    PYSPARK_AVAILABLE = _ilu.find_spec("pyspark") is not None
except Exception:  # pragma: no cover - defensive
    PYSPARK_AVAILABLE = False


_INSTALL_HINT = (
    "PySpark is required for the multi-filter portfolio system data layer.\n"
    "  1. pip install -r requirements.txt   (adds pyspark>=3.5, pyarrow)\n"
    "  2. a JDK must be on PATH / JAVA_HOME (java -version should work).\n"
    "If you only need the single-strategy notebooks, those do not use Spark."
)

# pandas offset alias -> Spark tumbling-window duration string.
_TF_TO_SPARK_WINDOW = {
    "5min": "5 minutes",
    "15min": "15 minutes",
    "30min": "30 minutes",
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours",
    "1D": "1 day",
}

# Canonical aliases so callers can write "5m"/"1H"/"1d" etc.
_TF_ALIASES = {
    "5m": "5min", "15m": "15min", "30m": "30min",
    "1H": "1h", "2H": "2h", "4H": "4h",
    "1d": "1D", "d": "1D", "daily": "1D",
}


def canonical_tf(tf: str) -> str:
    """Normalise a timeframe label to the keys used by the Spark loader."""
    tf = _TF_ALIASES.get(tf, tf)
    if tf not in _TF_TO_SPARK_WINDOW:
        raise ValueError(
            f"Unsupported timeframe {tf!r}. Supported: {sorted(_TF_TO_SPARK_WINDOW)}"
        )
    return tf


def require_pyspark() -> None:
    """Raise a helpful ImportError if pyspark is not importable."""
    if not PYSPARK_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


def get_spark(
    app_name: str = "multi_filter_portfolio_system",
    *,
    driver_memory: str = "6g",
    shuffle_partitions: int = 64,
):
    """Return a process-wide local ``SparkSession`` (created on first call).

    ``local[*]`` uses every core for the distributed CSV scan + window
    aggregation.  Arrow is enabled so ``toPandas()`` on the small resampled
    frame is fast and low-overhead.
    """
    require_pyspark()
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _read_mt5_csv(spark, csv_path: str):
    """Read a MetaTrader-5 tab-separated export into a typed Spark DataFrame.

    Header is ``<DATE>\\t<TIME>\\t<OPEN>\\t<HIGH>\\t<LOW>\\t<CLOSE>``
    ``\\t<TICKVOL>\\t<VOL>\\t<SPREAD>``; ``<DATE>`` is ``yyyy.MM.dd`` and
    ``<TIME>`` is ``HH:mm:ss``.  Returns columns
    ``ts, open, high, low, close, tick_vol, volume, spread``.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
    )

    schema = StructType(
        [
            StructField("<DATE>", StringType(), True),
            StructField("<TIME>", StringType(), True),
            StructField("<OPEN>", DoubleType(), True),
            StructField("<HIGH>", DoubleType(), True),
            StructField("<LOW>", DoubleType(), True),
            StructField("<CLOSE>", DoubleType(), True),
            StructField("<TICKVOL>", LongType(), True),
            StructField("<VOL>", LongType(), True),
            StructField("<SPREAD>", LongType(), True),
        ]
    )
    df = (
        spark.read.option("sep", "\t")
        .option("header", "true")
        .schema(schema)
        .csv(csv_path)
    )
    ts = F.to_timestamp(
        F.concat_ws(" ", F.col("<DATE>"), F.col("<TIME>")), "yyyy.MM.dd HH:mm:ss"
    )
    return (
        df.select(
            ts.alias("ts"),
            F.col("<OPEN>").alias("open"),
            F.col("<HIGH>").alias("high"),
            F.col("<LOW>").alias("low"),
            F.col("<CLOSE>").alias("close"),
            F.col("<TICKVOL>").cast("double").alias("tick_vol"),
            F.col("<VOL>").cast("double").alias("volume"),
            F.col("<SPREAD>").cast("double").alias("spread"),
        )
        .where(F.col("ts").isNotNull() & F.col("close").isNotNull())
    )


def spark_eda_summary(spark, csv_path: str) -> dict:
    """Distributed EDA: row count, span, gaps and basic stats — no pandas pull."""
    from pyspark.sql import functions as F

    sdf = _read_mt5_csv(spark, csv_path).cache()
    n = sdf.count()
    agg = sdf.select(
        F.min("ts").alias("start"),
        F.max("ts").alias("end"),
        F.mean("close").alias("close_mean"),
        F.stddev("close").alias("close_std"),
        F.min("low").alias("low_min"),
        F.max("high").alias("high_max"),
        F.mean("tick_vol").alias("tick_vol_mean"),
        F.sum(F.when(F.col("close").isNull(), 1).otherwise(0)).alias("close_nulls"),
        F.sum(F.when(F.col("tick_vol") <= 0, 1).otherwise(0)).alias("zero_vol_bars"),
    ).collect()[0]
    sdf.unpersist()
    return {
        "rows_m1": int(n),
        "start": agg["start"],
        "end": agg["end"],
        "close_mean": float(agg["close_mean"]) if agg["close_mean"] is not None else float("nan"),
        "close_std": float(agg["close_std"]) if agg["close_std"] is not None else float("nan"),
        "low_min": float(agg["low_min"]) if agg["low_min"] is not None else float("nan"),
        "high_max": float(agg["high_max"]) if agg["high_max"] is not None else float("nan"),
        "tick_vol_mean": float(agg["tick_vol_mean"]) if agg["tick_vol_mean"] is not None else float("nan"),
        "close_nulls": int(agg["close_nulls"] or 0),
        "zero_vol_bars": int(agg["zero_vol_bars"] or 0),
    }


def _resample_spark(spark, csv_path: str, tf: str) -> pd.DataFrame:
    """Aggregate M1 -> ``tf`` with a tumbling window; return a clean pandas frame."""
    from pyspark.sql import functions as F

    tf = canonical_tf(tf)
    window = _TF_TO_SPARK_WINDOW[tf]
    sdf = _read_mt5_csv(spark, csv_path)
    grouped = (
        sdf.groupBy(F.window(F.col("ts"), window))
        .agg(
            F.min_by("open", "ts").alias("open"),
            F.max("high").alias("high"),
            F.min("low").alias("low"),
            F.max_by("close", "ts").alias("close"),
            F.sum("tick_vol").alias("tick_vol"),
            F.sum("volume").alias("volume"),
            F.sum("spread").alias("spread"),
        )
        .select(
            F.col("window.start").alias("datetime"),
            "open", "high", "low", "close", "tick_vol", "volume", "spread",
        )
    )
    pdf = grouped.toPandas()
    if pdf.empty:
        return pdf
    pdf = (
        pdf.dropna(subset=["open", "high", "low", "close"])
        .set_index("datetime")
        .sort_index()
    )
    pdf.index = pd.to_datetime(pdf.index)
    return pdf


def _cache_path(cache_dir: Path, group: str, asset: str, tf: str, src_mtime: int) -> Path:
    """Parquet cache path keyed by source mtime (auto-invalidates on data refresh)."""
    return cache_dir / group / f"{asset}_{canonical_tf(tf)}_{src_mtime}.parquet"


def resample_to_cache(
    spark,
    meta: DatasetMeta,
    tf: str,
    cache_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Resample one (asset, tf) to parquet (skip if a fresh cache exists)."""
    src = Path(meta.path)
    src_mtime = int(src.stat().st_mtime)
    out = _cache_path(cache_dir, meta.group or "root", meta.asset, tf, src_mtime)
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf = _resample_spark(spark, str(src), tf)
    # Stale caches for older mtimes of the same (asset, tf) are harmless but
    # waste disk — prune them so the cache dir tracks the live data only.
    for old in out.parent.glob(f"{meta.asset}_{canonical_tf(tf)}_*.parquet"):
        if old != out:
            try:
                old.unlink()
            except OSError:
                pass
    pdf.to_parquet(out)
    return out


@dataclass
class SparkResampledDataset:
    """Lazy accessor over a Spark-resampled parquet cache.

    Mirrors :class:`source.data_loader.LazyDataset` (``.load`` / ``.unload`` /
    ``.using`` / ``.asset`` / ``.group`` / ``.timeframe``) so the portfolio
    backtester and WFO driver consume it with no special-casing.
    """

    asset: str
    group: str
    timeframe: str
    parquet_path: Path
    _cached: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def is_loaded(self) -> bool:
        return self._cached is not None

    def load(self) -> pd.DataFrame:
        if self._cached is None:
            self._cached = pd.read_parquet(self.parquet_path)
        return self._cached

    def unload(self) -> None:
        self._cached = None

    @contextmanager
    def using(self) -> Iterator[pd.DataFrame]:
        try:
            yield self.load()
        finally:
            self.unload()


GroupGrid = Dict[str, Dict[str, Dict[str, SparkResampledDataset]]]


def build_spark_grid(
    data_dir: str | Path,
    group_timeframes: Dict[str, List[str]],
    *,
    cache_dir: str | Path | None = None,
    asset_filter: Dict[str, List[str]] | None = None,
    spark=None,
    force: bool = False,
    progress: bool = True,
) -> GroupGrid:
    """Resample every (group, tf, asset) to a parquet cache via Spark.

    Returns ``{group: {tf: {asset: SparkResampledDataset}}}`` — the same nested
    shape the notebooks already use, but each cell is a lazy parquet reader.

    ``asset_filter`` (optional) restricts the universe per group, e.g.
    ``{"forex": ["EURUSD", "GBPUSD"]}`` — used to keep WFO tractable.
    Spark itself parallelises the scan/aggregation across all cores, so a
    single session resamples the whole grid; the parent never holds a frame.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir / "_spark_cache"
    metas = discover_datasets(data_dir)

    own_spark = spark is None
    if own_spark:
        spark = get_spark()

    grid: GroupGrid = {}
    try:
        jobs: List[Tuple[str, str, DatasetMeta]] = []
        for (asset, _src_tf), meta in metas.items():
            group = meta.group
            if group not in group_timeframes:
                continue
            keep = asset_filter.get(group) if asset_filter else None
            if keep is not None and asset not in keep:
                continue
            for tf in group_timeframes[group]:
                jobs.append((tf, group, meta))

        total = len(jobs)
        for i, (tf, group, meta) in enumerate(jobs, start=1):
            tf_c = canonical_tf(tf)
            pq = resample_to_cache(spark, meta, tf_c, cache_dir, force=force)
            (
                grid.setdefault(group, {})
                .setdefault(tf_c, {})[meta.asset]
            ) = SparkResampledDataset(
                asset=meta.asset, group=group, timeframe=tf_c, parquet_path=pq
            )
            if progress:
                print(f"  spark-resample [{i}/{total}] {group}/{meta.asset}/{tf_c}",
                      end="\n" if i == total else "\r", flush=True)
    finally:
        if own_spark:
            spark.stop()
    return grid
