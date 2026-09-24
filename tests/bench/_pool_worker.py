"""Top-level (picklable-by-reference) worker functions for the process-pool
throughput benchmark. Must live at module scope -- ``ProcessPoolExecutor``
looks callables up by ``module.qualname``, so closures won't do.
"""
from __future__ import annotations

import os
from typing import Any

from _shared_mem import shm_to_bars
from _strategy import sma_cross_atr_signals

from quantlab.engine import run_backtest

# Populated once per worker process by `init` (below), then reused by every
# task that worker services -- this is the "amortise the heavy setup over
# many cheap tasks" half of the no-per-task-DataFrame rule.
_STATE: dict[str, Any] = {}


def init(htf_spec: dict, m1_spec: dict, spec, cost) -> None:
    # Each pool worker is its own OS process; fork() only carries over the
    # calling thread, so this process's Polars/Rayon thread pool does not
    # exist yet and will be lazily created on first use *in this process*.
    # Cap it to 1 before that happens: with N worker processes each also
    # trying to run polars ops on all logical cores, you get N x cores
    # oversubscription, which is what actually caused this benchmark's
    # pool throughput to *fall* going from 8 to 14 workers before this line
    # was added. Numba's njit kernel is unaffected (no parallel=True/prange
    # -> already single-threaded).
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    bars, h1_handles = shm_to_bars(htf_spec)
    m1, m1_handles = shm_to_bars(m1_spec)
    _STATE["bars"] = bars
    _STATE["m1"] = m1
    _STATE["spec"] = spec
    _STATE["cost"] = cost
    _STATE["_handles"] = h1_handles + m1_handles  # keep SharedMemory attachments alive
    run_backtest(bars, sma_cross_atr_signals(bars), spec, cost, m1=m1)  # numba warm-up, once per worker


def run_one(params: dict) -> dict:
    bars, m1, spec, cost = _STATE["bars"], _STATE["m1"], _STATE["spec"], _STATE["cost"]
    sig = sma_cross_atr_signals(bars, **params)
    result = run_backtest(bars, sig, spec, cost, m1=m1)
    t = result.trades
    # Return a small summary, not the trades frame -- an optimisation study
    # only needs a handful of metrics per trial, and shipping the full frame
    # back through the pool's result queue would reintroduce a per-task
    # DataFrame (de)serialisation cost on the way *out* this time.
    return {
        "n_trades": t.height,
        "pnl_points": float(t["pnl_points"].sum()) if t.height else 0.0,
    }
