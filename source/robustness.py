"""Robustness diagnostics: Monte-Carlo trade shuffling, bootstraps, and parameter sweeps.

Two optimisation patterns are used here:

* **Vectorised resampling** — Monte-Carlo and block-bootstrap loops are
  rewritten as single ``numpy`` ops over a (n_trades, n_runs) index matrix.
  This typically lifts 1 000-run jobs from seconds to milliseconds and is
  faster than process-pool parallelism for this shape of work.
* **Parallel parameter sensitivity** — independent (param, value) backtests
  are dispatched via :func:`source.parallel.parallel_map` when ``n_jobs > 1``.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .backtest import Backtester
from .metrics import compute_metrics
from .parallel import parallel_map
from .strategy import SMACrossoverStrategy, StrategyParams

_SimpleResult = namedtuple("_SimpleResult", ["trades", "equity"])


def monte_carlo_trades(trades: pd.DataFrame, n_runs: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Shuffle trade order ``n_runs`` times; return one cumulative equity curve per run.

    Vectorised: a single ``argsort`` of ``rand(n, n_runs)`` produces all
    permutations at once, then ``cumsum`` along axis 0 turns them into
    equity curves.  Equivalent to the loop ``cumsum(rng.permutation(pnls))``
    but ~100× faster on n_runs=1000.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pnls = trades["pnl_points"].to_numpy(dtype=float)
    n = len(pnls)
    # argsort of uniform noise yields independent permutations per column.
    perms = np.argsort(rng.random((n, n_runs)), axis=0)
    curves = np.cumsum(pnls[perms], axis=0)
    return pd.DataFrame(curves)


def monte_carlo_summary(mc_df: pd.DataFrame) -> Dict[str, float]:
    """Reduce the Monte-Carlo cloud to headline statistics."""
    if mc_df is None or mc_df.empty:
        return {}
    final = mc_df.iloc[-1]
    rolling_max = mc_df.cummax(axis=0)
    max_dd = (rolling_max - mc_df).max(axis=0)
    return {
        "mean_final_pnl": float(final.mean()),
        "median_final_pnl": float(final.median()),
        "p05_final_pnl": float(final.quantile(0.05)),
        "p95_final_pnl": float(final.quantile(0.95)),
        "mean_max_drawdown": float(-max_dd.mean()),
        "p95_max_drawdown": float(-max_dd.quantile(0.95)),
        "prob_profitable": float((final > 0).mean()),
    }


def block_bootstrap_trades(
    trades: pd.DataFrame,
    block_size: int | None = None,
    n_runs: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Circular block bootstrap: sample consecutive blocks of trades with replacement.

    Unlike per-trade shuffling, this preserves serial correlation within blocks
    (volatility clustering, regime persistence).  Uses circular wrapping so
    every block is exactly ``block_size`` trades — no edge truncation.

    Vectorised: builds the full (n, n_runs) index matrix in one shot using
    broadcasting + modular arithmetic, then a single ``cumsum``.

    Returns one cumulative equity curve per run (columns), indexed by trade number.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    pnls = trades["pnl_points"].to_numpy(dtype=float)
    n = len(pnls)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)

    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=(n_blocks, n_runs))                # (B, R)
    offsets = np.arange(block_size)[None, :, None]                     # (1, K, 1)
    # idx[b, k, r] = (starts[b, r] + k) mod n
    idx = (starts[:, None, :] + offsets) % n                           # (B, K, R)
    flat = idx.reshape(n_blocks * block_size, n_runs)[:n]              # (n, R)
    curves = np.cumsum(pnls[flat], axis=0)
    return pd.DataFrame(curves)


def subperiod_analysis(trades: pd.DataFrame, freq: str = "YE") -> pd.DataFrame:
    """Compute performance metrics independently for each calendar sub-period.

    Splits trades by ``freq`` (pandas offset alias, default 'YE' = calendar year)
    and runs ``compute_metrics`` on each slice.  Useful for detecting whether
    an edge is concentrated in a single regime or regime-change period.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t = t.sort_values("exit_time").set_index("exit_time")
    rows = []
    for period_ts, group in t.resample(freq):
        if group.empty:
            continue
        g = group.reset_index()
        eq = pd.Series(
            g["pnl_points"].cumsum().values,
            index=pd.to_datetime(g["exit_time"].values),
        )
        m = compute_metrics(_SimpleResult(g, eq))
        label = str(period_ts.year) if hasattr(period_ts, "year") else str(period_ts)[:7]
        rows.append({"period": label, **m})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("period")


def _run_one_sensitivity_point(args: dict) -> dict:
    """Worker: backtest a single (param, value) variation; return a row dict."""
    pname = args["pname"]
    value = args["value"]
    df = args["df"]
    base = args["base_params"]
    strategy_cls = args["strategy_cls"]
    params = base.__class__(**{**base.as_dict(), pname: value})
    res = Backtester(strategy_cls(params)).run(df)
    m = compute_metrics(res)
    return {"param": pname, "value": value, **m}


def parameter_sensitivity(
    df: pd.DataFrame,
    base_params: StrategyParams,
    variations: Dict[str, Iterable],
    strategy_cls=SMACrossoverStrategy,
    *,
    n_jobs: int | str | None = 1,
) -> pd.DataFrame:
    """Sweep one parameter at a time around a baseline; return metrics per point.

    Each (param, value) backtest is independent, so the sweep parallelises
    cleanly: pass ``n_jobs="auto"`` (or any int > 1) to fan out across
    processes.  For small grids on cheap strategies the serial path is faster
    — the default ``n_jobs=1`` matches the legacy behaviour.
    """
    points = [
        {
            "pname": pname,
            "value": value,
            "df": df,
            "base_params": base_params,
            "strategy_cls": strategy_cls,
        }
        for pname, values in variations.items()
        for value in values
    ]
    rows = parallel_map(_run_one_sensitivity_point, points, n_jobs=n_jobs)
    return pd.DataFrame(rows)
