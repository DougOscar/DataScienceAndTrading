"""Walk-Forward Optimization driver.

Each fold is independent (its IS grid search and OOS backtest don't depend on
any other fold), so folds can run in parallel.  The orchestration is in
:func:`walk_forward`; the per-fold worker is the top-level function
:func:`_run_fold` so it pickles cleanly across process boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .backtest import Backtester
from .metrics import compute_metrics
from .parallel import parallel_map, resolve_n_jobs
from .strategy import SMACrossoverStrategy, StrategyParams


@dataclass
class WFOResult:
    windows: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame


def default_score(m: Dict[str, float]) -> float:
    """Default in-sample score: Sharpe if meaningful, else -inf.

    Module-level (rather than nested) so it remains picklable for multi-process
    workers when callers don't pass an explicit ``score_fn``.
    """
    val = m.get("sharpe_daily", float("nan"))
    if np.isnan(val):
        val = m.get("sharpe_per_trade", float("nan"))
    if np.isnan(val) or m.get("num_trades", 0) < 5:
        return -np.inf
    return val


# Backward-compatible alias.
_default_score = default_score


def _evaluate_combo(
    combo: tuple,
    keys: List[str],
    is_df: pd.DataFrame,
    strategy_cls,
    params_cls,
    score_fn: Callable[[Dict[str, float]], float],
) -> Tuple[float, tuple]:
    """Run one (param-combo) backtest on the IS slice; return (score, combo)."""
    params = params_cls(**dict(zip(keys, combo)))
    res = Backtester(strategy_cls(params)).run(is_df)
    return score_fn(compute_metrics(res)), combo


def _run_fold(args: dict) -> dict | None:
    """Worker: grid-search on IS, backtest the winner on OOS, return one row dict.

    Returning a single dict (plus the ``oos_result`` payload) keeps the
    serialised payload small enough that process-pool overhead is negligible
    even for short fold lengths.
    """
    fold_idx: int = args["fold_idx"]
    fold: pd.DataFrame = args["fold"]
    oos_ratio: float = args["oos_ratio"]
    keys: List[str] = args["keys"]
    combos: List[tuple] = args["combos"]
    strategy_cls = args["strategy_cls"]
    params_cls = args["params_cls"]
    score_fn = args["score_fn"]

    if len(fold) < 50:
        return None
    split = max(1, int(len(fold) * (1 - oos_ratio)))
    is_df = fold.iloc[:split]
    oos_df = fold.iloc[split:]
    if oos_df.empty or is_df.empty:
        return None

    best_score = -np.inf
    best_combo: tuple | None = None
    for combo in combos:
        score, _ = _evaluate_combo(combo, keys, is_df, strategy_cls, params_cls, score_fn)
        if score > best_score:
            best_score = score
            best_combo = combo

    best_params = params_cls(**dict(zip(keys, best_combo))) if best_combo is not None else params_cls()

    oos_res = Backtester(strategy_cls(best_params)).run(oos_df)
    m_oos = compute_metrics(oos_res)
    oos_s = m_oos.get("sharpe_daily", np.nan)
    deg = float(oos_s / best_score) if (np.isfinite(best_score) and best_score > 0) else np.nan
    row = {
        "fold": fold_idx,
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

    return {
        "row": row,
        "oos_equity": oos_res.equity,
        "oos_trades": (
            oos_res.trades.assign(fold=fold_idx)
            if not oos_res.trades.empty else pd.DataFrame()
        ),
    }


def walk_forward(
    df: pd.DataFrame,
    param_grid: Dict[str, Iterable],
    n_splits: int = 5,
    oos_ratio: float = 0.25,
    strategy_cls=SMACrossoverStrategy,
    params_cls=StrategyParams,
    score_fn: Callable[[Dict[str, float]], float] | None = None,
    *,
    n_jobs: int | str | None = 1,
) -> WFOResult:
    """Rolling walk-forward: ``n_splits`` equal folds, each split IS / OOS.

    Set ``n_jobs="auto"`` (or any positive int) to run folds in parallel.
    With 5 folds the wall-clock speedup is ~4× on an 8-core box; with more
    folds or expensive grids, more.

    The default is ``n_jobs=1`` for backwards-compatible determinism — the
    parallel path produces *identical* results in identical order, but a few
    pipelines pin process counts elsewhere.

    Notes
    -----
    * ``score_fn`` must be picklable when ``n_jobs > 1`` — pass a top-level
      function (not a lambda or closure).  ``None`` uses :func:`default_score`.
    * Fold work is roughly equal, so the simple ``ProcessPoolExecutor`` map
      below is enough — no chunking heuristics needed.
    """

    score_fn = score_fn or default_score
    total = len(df)
    if total == 0 or n_splits < 1:
        return WFOResult(pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame())

    fold_size = total // n_splits
    keys = list(param_grid.keys())
    combos = list(product(*[list(param_grid[k]) for k in keys]))

    fold_args = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else total
        fold_args.append(
            {
                "fold_idx": i,
                "fold": df.iloc[start:end],
                "oos_ratio": oos_ratio,
                "keys": keys,
                "combos": combos,
                "strategy_cls": strategy_cls,
                "params_cls": params_cls,
                "score_fn": score_fn,
            }
        )

    workers = resolve_n_jobs(n_jobs)
    raw = parallel_map(_run_fold, fold_args, n_jobs=workers)

    rows: List[dict] = []
    oos_equities: List[pd.Series] = []
    oos_trades_all: List[pd.DataFrame] = []
    for out in raw:
        if out is None:
            continue
        rows.append(out["row"])
        eq = out["oos_equity"]
        if eq is not None and not eq.empty:
            oos_equities.append(eq)
        ot = out["oos_trades"]
        if not ot.empty:
            oos_trades_all.append(ot)

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
