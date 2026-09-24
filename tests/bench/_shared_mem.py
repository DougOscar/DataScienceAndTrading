"""Shared-memory plumbing for the process-pool throughput benchmark (DESIGN §7:
"No DataFrame pickled per task -- use shared-memory arrays or memory-mapped
Parquet").

The pattern: the parent puts each OHLC/spread/ts column of ``bars``/``m1``
into a ``multiprocessing.shared_memory.SharedMemory`` block *once*; workers
attach to those blocks *once* in a pool ``initializer`` (not per task) and
rebuild a local ``bars``/``m1`` frame from them. Every task submitted after
that only carries a handful of small strategy-parameter floats/ints --
nothing bar-sized ever crosses a task boundary.
"""
from __future__ import annotations

from multiprocessing import shared_memory
from typing import Any

import numpy as np
import polars as pl

# Columns needed to satisfy quantlab.contracts.BAR_COLUMNS + what run_backtest
# actually reads. ts_utc/spread_max/tick_vol are present-but-unused by the
# engine (only checked for column existence), so they are filled with cheap
# placeholders in the worker rather than round-tripped through shared memory.
_SHM_COLUMNS = ("ts", "open", "high", "low", "close", "spread")


def _array_to_shm(arr: np.ndarray) -> tuple[shared_memory.SharedMemory, tuple[str, tuple[int, ...], str]]:
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    view[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


def bars_to_shm(bars: pl.DataFrame) -> tuple[list[shared_memory.SharedMemory], dict[str, Any]]:
    """One-time (per benchmark run, not per task) export of ``bars`` to shared memory.

    Returns ``(handles, spec)``. ``handles`` must be kept alive (and later
    ``.close()``d + ``.unlink()``d) by the caller for as long as any worker
    might still be attaching to them -- SharedMemory segments are OS-level
    resources (``/dev/shm``), not garbage-collected Python objects. ``spec``
    is a small dict of ``(name, shape, dtype_str)`` triples, cheap to pass
    through ``ProcessPoolExecutor(initargs=...)``.
    """
    ts_i8 = bars["ts"].to_physical().to_numpy().astype(np.int64)  # epoch ms, physical repr
    cols = {
        "ts": ts_i8,
        "open": bars["open"].to_numpy().astype(np.float64),
        "high": bars["high"].to_numpy().astype(np.float64),
        "low": bars["low"].to_numpy().astype(np.float64),
        "close": bars["close"].to_numpy().astype(np.float64),
        "spread": bars["spread"].to_numpy().astype(np.float64),
    }
    handles: list[shared_memory.SharedMemory] = []
    spec: dict[str, Any] = {}
    for name in _SHM_COLUMNS:
        shm, meta = _array_to_shm(cols[name])
        handles.append(shm)
        spec[name] = meta
    return handles, spec


def shm_to_bars(spec: dict[str, Any]) -> tuple[pl.DataFrame, list[shared_memory.SharedMemory]]:
    """Worker-side reconstruction (call once per worker, in the pool initializer).

    Copies the shared-memory bytes into this worker's own arrays (one copy,
    at worker start-up) and rebuilds a ``contracts.BAR_COLUMNS``-shaped frame.
    Returns the frame and the attached ``SharedMemory`` handles (keep them
    referenced for the worker's lifetime; closing them here is unnecessary --
    the process exit will release the attachment, and ``unlink()`` is the
    parent's job).
    """
    handles: list[shared_memory.SharedMemory] = []
    arrs: dict[str, np.ndarray] = {}
    for name in _SHM_COLUMNS:
        shm_name, shape, dtype = spec[name]
        shm = shared_memory.SharedMemory(name=shm_name)
        handles.append(shm)
        arrs[name] = np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf).copy()
    n = arrs["ts"].shape[0]
    df = pl.DataFrame({
        "ts": pl.Series(arrs["ts"]).cast(pl.Datetime("ms")),
        "open": arrs["open"], "high": arrs["high"], "low": arrs["low"], "close": arrs["close"],
        "spread": arrs["spread"],
    }).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(0).cast(pl.Int64).alias("tick_vol"),
    )
    return df, handles


def close_and_unlink(handles: list[shared_memory.SharedMemory]) -> None:
    for shm in handles:
        shm.close()
        shm.unlink()
