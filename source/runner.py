"""High-level parallel runners across (group, timeframe, asset) combinations.

These are the functions you reach for in a notebook or pipeline once you have
a :class:`source.data_loader.LazyDataset` per (group, tf, asset) cell of the
grid.  They:

* Load each cell's DataFrame **inside** the worker process — the parent never
  holds more than one frame at a time.
* Run independent jobs (baseline backtest, WFO, parameter sweep) in parallel.
* Return a flat ``{(group, tf, asset): result}`` dict; the notebook reshapes it.

The pattern looks like this::

    from source import build_lazy_grid, run_backtest_grid
    grid = build_lazy_grid(REPO_ROOT / "data", group_timeframes={"forex": ["1h", "4h"]})
    results = run_backtest_grid(grid, baseline_params, n_jobs="auto")

For WFO, swap ``run_backtest_grid`` → ``run_wfo_grid`` with the param grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Tuple

import pandas as pd

from .backtest import Backtester, BacktestResult
from .data_loader import LazyDataset, load_csv, resample_ohlc
from .parallel import parallel_map
from .strategy import SMACrossoverStrategy, StrategyParams
from .wfo import WFOResult, walk_forward


GridKey = Tuple[str, str, str]   # (group, timeframe, asset)


@dataclass
class GridSpec:
    """A flat description of one (group, tf, asset) cell — pickled into workers."""

    group: str
    timeframe: str
    asset: str
    csv_path: str
    source_timeframe: str
    resample_to: str | None = None

    @classmethod
    def from_lazy(cls, group: str, tf: str, asset: str, ds: LazyDataset) -> "GridSpec":
        return cls(
            group=group,
            timeframe=tf,
            asset=asset,
            csv_path=str(ds.meta.path),
            source_timeframe=ds.meta.timeframe,
            resample_to=ds.resample_to,
        )

    @property
    def key(self) -> GridKey:
        return (self.group, self.timeframe, self.asset)

    def load(self) -> pd.DataFrame:
        df = load_csv(self.csv_path)
        if (
            self.resample_to is not None
            and self.source_timeframe.upper() in {"M1", "M5", "M15", "M30"}
        ):
            df = resample_ohlc(df, self.resample_to)
        return df


def _flatten_lazy_grid(
    grid: Dict[str, Dict[str, Dict[str, LazyDataset]]],
) -> list[GridSpec]:
    out: list[GridSpec] = []
    for group, tfs in grid.items():
        for tf, assets in tfs.items():
            for asset, ds in assets.items():
                out.append(GridSpec.from_lazy(group, tf, asset, ds))
    return out


# ---------------------------------------------------------------------------
# Baseline backtest grid
# ---------------------------------------------------------------------------


def _run_one_backtest(args: dict) -> Tuple[GridKey, BacktestResult]:
    spec: GridSpec = args["spec"]
    params = args["params"]
    strategy_cls = args["strategy_cls"]
    slippage = args["slippage"]
    initial_capital = args["initial_capital"]
    df = spec.load()
    bt = Backtester(
        strategy_cls(params),
        slippage_points=slippage,
        initial_capital=initial_capital,
    )
    return spec.key, bt.run(df)


def run_backtests_with_params(
    grid: Dict[str, Dict[str, Dict[str, LazyDataset]]],
    params_by_key: Dict[GridKey, Any],
    *,
    strategy_cls=SMACrossoverStrategy,
    slippage: float = 0.0,
    initial_capital: float = 10_000.0,
    n_jobs: int | str | None = "auto",
    progress: bool = True,
) -> Dict[GridKey, BacktestResult]:
    """Run a backtest per cell with **per-cell** parameters (e.g. WFO-picked).

    ``params_by_key`` is indexed by ``(group, tf, asset)`` tuples and supplies
    a (potentially) different params object per cell — exactly the shape you
    get back from a WFO sweep when each asset chooses its own best params.
    """
    items = []
    for group, tfs in grid.items():
        for tf, assets in tfs.items():
            for asset, ds in assets.items():
                key = (group, tf, asset)
                if key not in params_by_key:
                    continue
                items.append(
                    {
                        "spec": GridSpec.from_lazy(group, tf, asset, ds),
                        "params": params_by_key[key],
                        "strategy_cls": strategy_cls,
                        "slippage": slippage,
                        "initial_capital": initial_capital,
                    }
                )
    pairs = parallel_map(
        _run_one_backtest, items, n_jobs=n_jobs, progress=progress, desc="backtest"
    )
    return dict(pairs)


def run_backtest_grid(
    grid: Dict[str, Dict[str, Dict[str, LazyDataset]]],
    params,
    *,
    strategy_cls=SMACrossoverStrategy,
    slippage: float = 0.0,
    initial_capital: float = 10_000.0,
    n_jobs: int | str | None = "auto",
    progress: bool = True,
) -> Dict[GridKey, BacktestResult]:
    """Backtest one strategy across every cell of a lazy grid in parallel.

    Each worker loads its own CSV, runs the backtest, and returns the
    :class:`BacktestResult`.  The parent never holds a DataFrame.
    """
    specs = _flatten_lazy_grid(grid)
    items = [
        {
            "spec": s,
            "params": params,
            "strategy_cls": strategy_cls,
            "slippage": slippage,
            "initial_capital": initial_capital,
        }
        for s in specs
    ]
    pairs = parallel_map(
        _run_one_backtest, items, n_jobs=n_jobs, progress=progress, desc="backtest"
    )
    return dict(pairs)


# ---------------------------------------------------------------------------
# WFO grid
# ---------------------------------------------------------------------------


def _run_one_wfo(args: dict) -> Tuple[GridKey, WFOResult]:
    spec: GridSpec = args["spec"]
    df = spec.load()
    return spec.key, walk_forward(
        df,
        param_grid=args["param_grid"],
        n_splits=args["n_splits"],
        oos_ratio=args["oos_ratio"],
        strategy_cls=args["strategy_cls"],
        params_cls=args["params_cls"],
        score_fn=args["score_fn"],
        n_jobs=args["inner_n_jobs"],
    )


def run_wfo_grid(
    grid: Dict[str, Dict[str, Dict[str, LazyDataset]]],
    param_grid: Dict[str, Iterable],
    *,
    n_splits: int = 5,
    oos_ratio: float = 0.25,
    strategy_cls=SMACrossoverStrategy,
    params_cls=StrategyParams,
    score_fn: Callable | None = None,
    n_jobs: int | str | None = "auto",
    inner_n_jobs: int | str | None = 1,
    progress: bool = True,
) -> Dict[GridKey, WFOResult]:
    """Run :func:`walk_forward` across every cell of a lazy grid in parallel.

    Two parallelism levels are exposed:

    * **outer** (``n_jobs``) — across (group, tf, asset) cells.  This is the
      big lever: typically 5–30 cells, each independent, each holding only
      its own DataFrame.
    * **inner** (``inner_n_jobs``) — across folds inside each WFO call.  Set
      to ``1`` (the default) when the outer loop already saturates the CPU;
      bump it up only if the outer grid is tiny (≤ 2 cells).
    """
    specs = _flatten_lazy_grid(grid)
    items = [
        {
            "spec": s,
            "param_grid": param_grid,
            "n_splits": n_splits,
            "oos_ratio": oos_ratio,
            "strategy_cls": strategy_cls,
            "params_cls": params_cls,
            "score_fn": score_fn,
            "inner_n_jobs": inner_n_jobs,
        }
        for s in specs
    ]
    pairs = parallel_map(
        _run_one_wfo, items, n_jobs=n_jobs, progress=progress, desc="WFO"
    )
    return dict(pairs)


# ---------------------------------------------------------------------------
# Reshape helpers (flat dict → nested {group: {tf: {asset: result}}})
# ---------------------------------------------------------------------------


def reshape_grid_results(flat: Dict[GridKey, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Turn a flat ``{(group, tf, asset): X}`` into ``{group: {tf: {asset: X}}}``."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for (group, tf, asset), value in flat.items():
        out.setdefault(group, {}).setdefault(tf, {})[asset] = value
    return out
