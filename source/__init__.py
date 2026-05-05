"""Reusable building blocks for the notebooks in this repo."""

from .data_loader import DatasetMeta, load_all, load_csv, resample_ohlc
from .strategy import SMACrossoverStrategy, StrategyParams
from .backtest import Backtester, BacktestResult, Trade
from .metrics import compute_metrics, metrics_table
from .dashboard import (
    plot_backtest_dashboard,
    plot_wfo_dashboard,
    plot_robustness_dashboard,
)
from .wfo import walk_forward, WFOResult
from .robustness import (
    monte_carlo_trades,
    monte_carlo_summary,
    block_bootstrap_trades,
    subperiod_analysis,
    parameter_sensitivity,
)
from .portfolio import correlation_weights, weighted_portfolio

__all__ = [
    "DatasetMeta",
    "load_all",
    "load_csv",
    "resample_ohlc",
    "SMACrossoverStrategy",
    "StrategyParams",
    "Backtester",
    "BacktestResult",
    "Trade",
    "compute_metrics",
    "metrics_table",
    "plot_backtest_dashboard",
    "plot_wfo_dashboard",
    "plot_robustness_dashboard",
    "walk_forward",
    "WFOResult",
    "monte_carlo_trades",
    "monte_carlo_summary",
    "block_bootstrap_trades",
    "subperiod_analysis",
    "parameter_sensitivity",
    "correlation_weights",
    "weighted_portfolio",
]
