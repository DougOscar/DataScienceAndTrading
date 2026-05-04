"""Robustness diagnostics: Monte-Carlo trade shuffling and parameter sweeps."""

from __future__ import annotations

from collections import namedtuple
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .backtest import Backtester
from .metrics import compute_metrics
from .strategy import SMACrossoverStrategy, StrategyParams

_SimpleResult = namedtuple("_SimpleResult", ["trades", "equity"])


def monte_carlo_trades(trades: pd.DataFrame, n_runs: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Shuffle trade order `n_runs` times; return one equity curve per run (as columns)."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    pnls = trades["pnl_points"].to_numpy(dtype=float)
    n = len(pnls)
    curves = np.empty((n, n_runs))
    for i in range(n_runs):
        curves[:, i] = np.cumsum(rng.permutation(pnls))
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
    (volatility clustering, regime persistence). Uses circular wrapping so every
    block is exactly `block_size` trades — no edge truncation.

    Returns one cumulative equity curve per run (columns), indexed by trade number.
    """
    if trades is None or trades.empty:
        return pd.DataFrame()
    pnls = trades["pnl_points"].to_numpy(dtype=float)
    n = len(pnls)
    if block_size is None:
        block_size = max(1, int(np.sqrt(n)))
    rng = np.random.default_rng(seed)
    # Tile once so any block starting at index s < n is always full length
    pnls_ext = np.concatenate([pnls, pnls])
    n_blocks = int(np.ceil(n / block_size))
    curves = np.empty((n, n_runs))
    for i in range(n_runs):
        starts = rng.integers(0, n, size=n_blocks)
        sampled = np.concatenate([pnls_ext[s : s + block_size] for s in starts])[:n]
        curves[:, i] = np.cumsum(sampled)
    return pd.DataFrame(curves)


def subperiod_analysis(trades: pd.DataFrame, freq: str = "YE") -> pd.DataFrame:
    """Compute performance metrics independently for each calendar sub-period.

    Splits trades by `freq` (pandas offset alias, default 'YE' = calendar year)
    and runs compute_metrics on each slice. Useful for detecting whether an edge
    is concentrated in a single regime or regime-change period.

    Returns a DataFrame with one row per period and standard metric columns.
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


def parameter_sensitivity(
    df: pd.DataFrame,
    base_params: StrategyParams,
    variations: Dict[str, Iterable],
    strategy_cls=SMACrossoverStrategy,
) -> pd.DataFrame:
    """Sweep one parameter at a time around a baseline; return metrics per point."""
    rows = []
    for pname, values in variations.items():
        for v in values:
            params = base_params.__class__(**{**base_params.as_dict(), pname: v})
            res = Backtester(strategy_cls(params)).run(df)
            m = compute_metrics(res)
            rows.append({"param": pname, "value": v, **m})
    return pd.DataFrame(rows)
