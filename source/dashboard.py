"""Matplotlib-based dashboards for backtest, WFO and robustness results."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_backtest_dashboard(result, title: str = "Backtest"):
    """Equity, drawdown, per-trade PnL and rolling win rate."""
    trades = result.trades
    eq = result.equity

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(title, fontsize=14)

    ax = axes[0, 0]
    ax.plot(eq.index, eq.values, color="steelblue", lw=1.2)
    ax.set_title("Cumulative PnL (points)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    dd = eq - eq.cummax()
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown (points)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if trades is not None and not trades.empty:
        ax.hist(trades["pnl_points"], bins=40, color="steelblue", edgecolor="black")
        ax.axvline(0, color="red", ls="--")
        ax.set_title(f"PnL per trade  (n={len(trades)})")
    else:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if trades is not None and len(trades) >= 20:
        wins = (trades["pnl_points"] > 0).astype(float)
        rolling = wins.rolling(20).mean()
        ax.plot(trades["exit_time"], rolling.values, color="darkgreen")
        ax.axhline(0.5, color="gray", ls="--")
        ax.set_ylim(0, 1)
        ax.set_title("Rolling win rate (window=20 trades)")
    else:
        ax.text(0.5, 0.5, "Not enough trades", ha="center", va="center")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_wfo_dashboard(wfo_result, full_equity: pd.Series | None = None):
    """OOS equity, per-fold PnL, and heatmap of selected parameters."""
    windows = wfo_result.windows
    oos_eq = wfo_result.oos_equity

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Walk-Forward Optimization", fontsize=14)

    ax = axes[0, 0]
    if full_equity is not None and not full_equity.empty:
        ax.plot(full_equity.index, full_equity.values, color="lightgray",
                lw=1.0, label="Baseline (single run)")
    if oos_eq is not None and not oos_eq.empty:
        ax.plot(oos_eq.index, oos_eq.values, color="steelblue",
                lw=1.3, label="Stitched OOS")
    ax.set_title("OOS stitched equity (points)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if not windows.empty:
        colors = ["seagreen" if v > 0 else "crimson" for v in windows["oos_pnl"]]
        ax.bar(windows["fold"].astype(str), windows["oos_pnl"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Per-fold OOS PnL")
        ax.set_xlabel("fold")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if not windows.empty:
        param_cols = [c for c in windows.columns if c.startswith("param_")]
        if param_cols:
            heat = windows.set_index("fold")[param_cols]
            im = ax.imshow(heat.values.T, aspect="auto", cmap="viridis")
            ax.set_yticks(range(len(param_cols)))
            ax.set_yticklabels([c.replace("param_", "") for c in param_cols])
            ax.set_xticks(range(len(heat)))
            ax.set_xticklabels(heat.index)
            ax.set_title("Selected parameters per fold")
            fig.colorbar(im, ax=ax, shrink=0.8)
        elif "oos_profit_factor" in windows.columns:
            pf = windows["oos_profit_factor"].fillna(0)
            colors = ["seagreen" if v >= 1 else "crimson" for v in pf]
            ax.bar(windows["fold"].astype(str), pf, color=colors)
            ax.axhline(1.0, color="black", lw=0.8, ls="--")
            ax.set_title("Per-fold OOS profit factor")
            ax.set_xlabel("fold")

    ax = axes[1, 1]
    if not windows.empty:
        ax.bar(windows["fold"].astype(str), windows["oos_sharpe"].fillna(0),
               color="slateblue")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Per-fold OOS Sharpe (daily)")
        ax.set_xlabel("fold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_robustness_dashboard(mc_df: pd.DataFrame,
                              baseline_equity: Iterable[float] | None = None,
                              sensitivity: pd.DataFrame | None = None):
    """Monte-Carlo cone of equity curves and parameter-sensitivity panels."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Robustness", fontsize=14)

    ax = axes[0]
    if mc_df is not None and not mc_df.empty:
        q05 = mc_df.quantile(0.05, axis=1)
        q50 = mc_df.quantile(0.5, axis=1)
        q95 = mc_df.quantile(0.95, axis=1)
        x = np.arange(len(mc_df))
        ax.fill_between(x, q05, q95, color="steelblue", alpha=0.25, label="5-95%")
        ax.plot(x, q50, color="steelblue", lw=1.2, label="median")
        if baseline_equity is not None:
            base = np.asarray(list(baseline_equity))
            ax.plot(np.arange(len(base)), base, color="black", lw=1.0,
                    label="actual order")
        ax.set_title("Monte-Carlo trade-shuffle equity")
        ax.set_xlabel("trade #")
        ax.set_ylabel("cumulative PnL (points)")
        ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    if sensitivity is not None and not sensitivity.empty:
        params = sensitivity["param"].unique()
        for pname in params:
            sub = sensitivity[sensitivity["param"] == pname].sort_values("value")
            ax.plot(sub["value"], sub["total_pnl"], marker="o", label=pname)
        ax.set_title("Parameter sensitivity (total PnL)")
        ax.set_xlabel("parameter value")
        ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig
