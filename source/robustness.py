"""Robustness diagnostics: Monte-Carlo trade shuffling and parameter sweeps."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .backtest import Backtester
from .metrics import compute_metrics
from .strategy import SMACrossoverStrategy, StrategyParams


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
