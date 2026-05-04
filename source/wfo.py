"""Walk-Forward Optimization driver."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List

import numpy as np
import pandas as pd

from .backtest import Backtester
from .metrics import compute_metrics
from .strategy import SMACrossoverStrategy, StrategyParams


@dataclass
class WFOResult:
    windows: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame


def _default_score(m: Dict[str, float]) -> float:
    """Default in-sample score: Sharpe if meaningful, else -inf."""
    val = m.get("sharpe_daily", float("nan"))
    if np.isnan(val):
        val = m.get("sharpe_per_trade", float("nan"))
    if np.isnan(val) or m.get("num_trades", 0) < 5:
        return -np.inf
    return val


def walk_forward(
    df: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    n_splits: int = 5,
    oos_ratio: float = 0.25,
    strategy_cls=SMACrossoverStrategy,
    params_cls=StrategyParams,
    score_fn: Callable[[Dict[str, float]], float] | None = None,
) -> WFOResult:
    """Rolling walk-forward: n equal folds, each split IS / OOS."""

    score_fn = score_fn or _default_score
    total = len(df)
    if total == 0 or n_splits < 1:
        return WFOResult(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    fold_size = total // n_splits
    rows: List[dict] = []
    oos_equities = []
    oos_trades_all = []

    keys = list(param_grid.keys())
    combos = list(product(*[list(param_grid[k]) for k in keys]))

    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else total
        fold = df.iloc[start:end]
        if len(fold) < 50:
            continue
        split = max(1, int(len(fold) * (1 - oos_ratio)))
        is_df = fold.iloc[:split]
        oos_df = fold.iloc[split:]
        if oos_df.empty or is_df.empty:
            continue

        best_score = -np.inf
        best_params = None
        for combo in combos:
            params = params_cls(**dict(zip(keys, combo)))
            res = Backtester(strategy_cls(params)).run(is_df)
            s = score_fn(compute_metrics(res))
            if s > best_score:
                best_score = s
                best_params = params

        if best_params is None:
            best_params = params_cls()

        oos_res = Backtester(strategy_cls(best_params)).run(oos_df)
        m_oos = compute_metrics(oos_res)

        oos_s = m_oos.get("sharpe_daily", np.nan)
        deg = float(oos_s / best_score) if (np.isfinite(best_score) and best_score > 0) else np.nan
        row = {
            "fold": i,
            "is_start": is_df.index[0],
            "is_end": is_df.index[-1],
            "oos_start": oos_df.index[0],
            "oos_end": oos_df.index[-1],
            "is_score": best_score,
            "oos_pnl": m_oos.get("total_pnl", 0.0),
            "oos_sharpe": oos_s,
            "oos_profit_factor": m_oos.get("profit_factor", np.nan),
            "oos_win_rate": m_oos.get("win_rate", np.nan),
            "oos_trades": m_oos.get("num_trades", 0),
            "degradation_ratio": deg,
        }
        row.update({f"param_{k}": getattr(best_params, k) for k in keys})
        rows.append(row)
        oos_equities.append(oos_res.equity)
        if not oos_res.trades.empty:
            oos_trades_all.append(oos_res.trades.assign(fold=i))

    if not rows:
        return WFOResult(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    # Stitch OOS equity curves end-to-end, accumulating PnL across folds.
    stitched = []
    offset = 0.0
    for eq in oos_equities:
        if eq.empty:
            continue
        adj = eq + offset
        stitched.append(adj)
        offset = float(adj.iloc[-1])
    oos_eq = pd.concat(stitched) if stitched else pd.Series(dtype=float)
    oos_trades = (
        pd.concat(oos_trades_all, ignore_index=True) if oos_trades_all else pd.DataFrame()
    )
    return WFOResult(pd.DataFrame(rows), oos_eq, oos_trades)
