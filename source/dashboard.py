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
    """OOS equity, per-fold PnL, parameter heatmap, Sharpe, and degradation ratio."""
    windows = wfo_result.windows
    oos_eq = wfo_result.oos_equity

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Walk-Forward Optimization", fontsize=14)

    # --- row 0 ---
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

    # --- row 1 ---
    ax = axes[1, 0]
    if not windows.empty:
        param_cols = [c for c in windows.columns if c.startswith("param_")]
        if param_cols:
            heat = windows.set_index("fold")[param_cols]
            # param_* columns may be categorical/object (filter names,
            # regimes, bools, None). Build a numeric matrix for coloring:
            # numeric cols pass through, non-numeric cols are factorized.
            numeric = pd.DataFrame(index=heat.index)
            labels = pd.DataFrame(index=heat.index)
            for c in param_cols:
                col = heat[c]
                as_num = pd.to_numeric(col, errors="coerce")
                if as_num.notna().any():
                    numeric[c] = as_num
                    labels[c] = col.map(
                        lambda v: "" if pd.isna(v) else f"{v:g}"
                        if isinstance(v, (int, float)) else str(v))
                else:
                    codes, _ = pd.factorize(col)
                    numeric[c] = pd.Series(codes, index=heat.index).where(
                        codes >= 0)
                    labels[c] = col.astype(str)
            # normalize each param independently so colors are comparable
            mat = numeric.values.T.astype(float)
            rng = np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
            rng[rng == 0] = 1.0
            norm = (mat - np.nanmin(mat, axis=1)[:, None]) / rng[:, None]
            im = ax.imshow(norm, aspect="auto", cmap="viridis",
                           vmin=0, vmax=1)
            ax.set_yticks(range(len(param_cols)))
            ax.set_yticklabels([c.replace("param_", "") for c in param_cols])
            ax.set_xticks(range(len(heat)))
            ax.set_xticklabels(heat.index)
            ax.set_title("Selected parameters per fold")
            for yi, c in enumerate(param_cols):
                for xi in range(len(heat)):
                    txt = labels[c].iloc[xi]
                    if txt:
                        ax.text(xi, yi, txt, ha="center", va="center",
                                fontsize=7,
                                color="white" if norm[yi, xi] < 0.5
                                else "black")
        else:
            wr = windows["oos_win_rate"].fillna(0) if "oos_win_rate" in windows.columns else pd.Series([0] * len(windows))
            bar_colors = ["seagreen" if v >= 0.5 else "crimson" for v in wr]
            ax.bar(windows["fold"].astype(str), wr, color=bar_colors)
            ax.axhline(0.5, color="black", lw=0.8, ls="--")
            ax.set_ylim(0, 1)
            ax.set_title("Per-fold OOS win rate")
            ax.set_xlabel("fold")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if not windows.empty:
        ax.bar(windows["fold"].astype(str), windows["oos_sharpe"].fillna(0),
               color="slateblue")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Per-fold OOS Sharpe (daily)")
        ax.set_xlabel("fold")
    ax.grid(alpha=0.3)

    # --- row 2: IS/OOS degradation diagnostics ---
    ax = axes[2, 0]
    if not windows.empty and "degradation_ratio" in windows.columns:
        dr = windows["degradation_ratio"]
        colors = ["seagreen" if (pd.notna(v) and v >= 0.3) else "crimson" for v in dr]
        ax.bar(windows["fold"].astype(str), dr.fillna(0), color=colors)
        ax.axhline(1.0, color="black", lw=0.8, ls="--", label="no degradation (1.0)")
        ax.axhline(0.3, color="gray", lw=0.8, ls=":", label="0.3 threshold")
        ax.set_title("IS→OOS degradation ratio (oos_sharpe / is_score)")
        ax.set_xlabel("fold")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No degradation data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("IS→OOS degradation ratio")
    ax.grid(alpha=0.3)

    ax = axes[2, 1]
    if not windows.empty and "oos_profit_factor" in windows.columns:
        pf = windows["oos_profit_factor"].fillna(0)
        colors = ["seagreen" if v > 1 else "crimson" for v in pf]
        ax.bar(windows["fold"].astype(str), pf, color=colors)
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_title("Per-fold OOS profit factor")
        ax.set_xlabel("fold")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def _plot_equity_cone(ax, mc_df, baseline_equity, title):
    """Shared helper: draw quantile bands + optional baseline on ax."""
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
        ax.set_xlabel("trade #")
        ax.set_ylabel("cumulative PnL (points)")
        ax.legend()
    ax.set_title(title)
    ax.grid(alpha=0.3)


def plot_robustness_dashboard(
    mc_df: pd.DataFrame,
    baseline_equity: Iterable[float] | None = None,
    sensitivity: pd.DataFrame | None = None,
    block_bootstrap_df: pd.DataFrame | None = None,
    subperiod_df: pd.DataFrame | None = None,
):
    """Four-panel robustness dashboard.

    [0,0] Monte-Carlo trade-shuffle equity cone
    [0,1] Block-bootstrap equity cone (preserves serial correlation)
    [1,0] Sub-period annual Sharpe (regime consistency)
    [1,1] Parameter sensitivity sweep
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Robustness", fontsize=14)

    _plot_equity_cone(axes[0, 0], mc_df, baseline_equity,
                      "Monte-Carlo trade-shuffle equity")

    _plot_equity_cone(axes[0, 1], block_bootstrap_df, baseline_equity,
                      "Block-bootstrap equity (serial correlation preserved)")

    ax = axes[1, 0]
    if subperiod_df is not None and not subperiod_df.empty and "sharpe_daily" in subperiod_df.columns:
        sharpes = subperiod_df["sharpe_daily"].fillna(0)
        colors = ["seagreen" if v > 0 else "crimson" for v in sharpes]
        ax.bar(sharpes.index, sharpes.values, color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Annual Sharpe by sub-period")
        ax.set_xlabel("period")
        ax.tick_params(axis="x", rotation=45)
    else:
        ax.text(0.5, 0.5, "No sub-period data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Annual Sharpe by sub-period")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
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


# ---------------------------------------------------------------------------
# Cross-strategy comparison plots
#
# All helpers consume the long-format frames produced by
# ``source.comparison.run_all_strategies`` and return a Matplotlib Figure.
# ---------------------------------------------------------------------------


def _strategy_color_map(strategies):
    cmap = plt.get_cmap("tab20")
    return {s: cmap(i % cmap.N) for i, s in enumerate(strategies)}


def plot_strategy_leaderboard(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "sharpe_daily",
    top_n: int = 25,
    ascending: bool = False,
):
    """Horizontal bar chart of the top-N ``(strategy, group, tf, asset)`` rows."""
    if metrics_df.empty or metric not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"No data for metric '{metric}'", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    sub = metrics_df.dropna(subset=[metric]).copy()
    sub["label"] = (sub["strategy"] + "  ·  " + sub["group"]
                    + "  ·  " + sub["tf"].astype(str)
                    + "  ·  " + sub["asset"])
    sub = sub.sort_values(metric, ascending=ascending).head(top_n)
    colors = _strategy_color_map(sub["strategy"].unique())
    fig, ax = plt.subplots(figsize=(11, max(4, 0.32 * len(sub))))
    ax.barh(sub["label"], sub[metric],
            color=[colors[s] for s in sub["strategy"]], edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_xlabel(metric)
    ax.set_title(f"Leaderboard — top {len(sub)} by {metric}")
    ax.invert_yaxis()
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return fig


def plot_strategy_heatmap(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "sharpe_daily",
    groups: list[str] | None = None,
    timeframes_per_group: dict[str, list[str]] | None = None,
    cmap: str = "RdYlGn",
):
    """One ``strategy × asset`` heatmap per (group, tf) panel.

    Layout: rows = groups, columns = timeframes. Cells with no row in
    ``metrics_df`` are left blank. Color is centered at zero for Sharpe-like
    metrics; otherwise spans the data range.
    """
    if metrics_df.empty or metric not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"No data for metric '{metric}'", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    groups = groups or sorted(metrics_df["group"].unique())
    if timeframes_per_group is None:
        timeframes_per_group = {
            g: sorted(metrics_df.loc[metrics_df["group"] == g, "tf"].unique())
            for g in groups
        }

    n_rows = len(groups)
    n_cols = max(len(timeframes_per_group.get(g, [])) for g in groups) if groups else 1
    if n_cols == 0:
        n_cols = 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6, 3.4 * n_cols), max(3, 2.4 * n_rows)),
        squeeze=False,
    )

    centered = metric.startswith("sharpe") or metric in {"expectancy", "total_pnl"}
    if centered:
        vmax = float(np.nanmax(np.abs(metrics_df[metric].astype(float))))
        vmin = -vmax if np.isfinite(vmax) and vmax > 0 else None
        vmax = vmax if np.isfinite(vmax) and vmax > 0 else None
    else:
        vmin = vmax = None

    for r, group in enumerate(groups):
        tfs = timeframes_per_group.get(group, [])
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(tfs):
                ax.set_axis_off()
                continue
            tf = tfs[c]
            mat = (metrics_df[(metrics_df["group"] == group) & (metrics_df["tf"] == tf)]
                   .pivot_table(index="strategy", columns="asset",
                                values=metric, aggfunc="first"))
            if mat.empty:
                ax.set_axis_off()
                continue
            im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(mat.shape[1]))
            ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels(mat.index, fontsize=8)
            ax.set_title(f"{group} · {tf}", fontsize=10)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    v = mat.values[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                fontsize=7, color="black")
            fig.colorbar(im, ax=ax, shrink=0.75)

    fig.suptitle(f"Per-asset {metric}", fontsize=13)
    fig.tight_layout()
    return fig


def plot_strategy_equity_overlay(
    equity_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    *,
    group: str,
    tf: str,
    asset: str,
    top_n: int = 5,
    rank_by: str = "sharpe_daily",
):
    """Overlay the equity curves of the top-N strategies on one (group, tf, asset)."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if equity_df.empty or metrics_df.empty:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    rank_sub = metrics_df[(metrics_df["group"] == group)
                          & (metrics_df["tf"] == tf)
                          & (metrics_df["asset"] == asset)].dropna(subset=[rank_by])
    if rank_sub.empty:
        ax.text(0.5, 0.5, f"No metrics for {group}/{tf}/{asset}",
                ha="center", va="center", transform=ax.transAxes)
        return fig
    top_strategies = (rank_sub.sort_values(rank_by, ascending=False)
                      .head(top_n)["strategy"].tolist())

    colors = _strategy_color_map(top_strategies)
    plotted = 0
    for name in top_strategies:
        sub = equity_df[(equity_df["strategy"] == name)
                        & (equity_df["group"] == group)
                        & (equity_df["tf"] == tf)
                        & (equity_df["asset"] == asset)]
        if sub.empty:
            continue
        sharpe = rank_sub.loc[rank_sub["strategy"] == name, rank_by].iloc[0]
        ax.plot(sub["timestamp"], sub["equity"], lw=1.2, color=colors[name],
                label=f"{name}  ({rank_by}={sharpe:.2f})")
        plotted += 1

    if plotted == 0:
        ax.text(0.5, 0.5, "No equity curves to plot", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(f"Equity overlay — top {plotted} by {rank_by}  ·  {group} / {tf} / {asset}")
    ax.set_ylabel("Cumulative PnL (points)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_strategy_metric_distribution(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "sharpe_daily",
    by_group: bool = True,
):
    """Box plot of one metric distributed across (group, tf, asset) per strategy.

    Useful to spot strategies that score high on one combo but are flat or
    negative elsewhere (single-asset overfit risk).
    """
    if metrics_df.empty or metric not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, f"No data for metric '{metric}'",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    sub = metrics_df.dropna(subset=[metric]).copy()
    order = (sub.groupby("strategy")[metric].median()
             .sort_values(ascending=False).index.tolist())
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(order)), 5))

    if by_group and sub["group"].nunique() > 1:
        groups = sorted(sub["group"].unique())
        n_groups = len(groups)
        width = 0.8 / n_groups
        for gi, g in enumerate(groups):
            data = [sub.loc[(sub["strategy"] == s) & (sub["group"] == g), metric].values
                    for s in order]
            positions = [i + (gi - (n_groups - 1) / 2) * width for i in range(len(order))]
            bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                            patch_artist=True, showfliers=True)
            color = plt.get_cmap("Set2")(gi % 8)
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.plot([], [], color=color, lw=8, alpha=0.7, label=g)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.legend(title="group")
    else:
        data = [sub.loc[sub["strategy"] == s, metric].values for s in order]
        ax.boxplot(data, labels=order, patch_artist=True, showfliers=True)
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")

    ax.axhline(0, color="black", lw=0.7)
    ax.set_ylabel(metric)
    ax.set_title(f"Distribution of {metric} across (group, tf, asset)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_strategy_return_correlation(
    daily_returns_df: pd.DataFrame,
    *,
    group: str,
    tf: str,
    asset: str,
    min_overlap_days: int = 30,
):
    """Pairwise correlation of daily PnL across strategies on one benchmark asset.

    ``daily_returns_df`` is the output of
    :func:`source.comparison.daily_returns_from_equity`.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    if daily_returns_df.empty:
        ax.text(0.5, 0.5, "No return data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    sub = daily_returns_df[(daily_returns_df["group"] == group)
                           & (daily_returns_df["tf"] == tf)
                           & (daily_returns_df["asset"] == asset)]
    if sub.empty:
        ax.text(0.5, 0.5, f"No returns for {group}/{tf}/{asset}",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    wide = sub.pivot_table(index="day", columns="strategy",
                           values="daily_return", aggfunc="sum").fillna(0.0)
    if wide.shape[1] < 2 or len(wide) < min_overlap_days:
        ax.text(0.5, 0.5, "Not enough overlapping days", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    corr = wide.corr()
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.values[i, j]:+.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"Daily-PnL correlation  ·  {group} / {tf} / {asset}")
    fig.tight_layout()
    return fig


def plot_strategy_group_breakdown(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "sharpe_daily",
    agg: str = "mean",
):
    """Grouped bar chart of an aggregated metric per (strategy, group)."""
    fig, ax = plt.subplots(figsize=(11, 5))
    if metrics_df.empty or metric not in metrics_df.columns:
        ax.text(0.5, 0.5, f"No data for metric '{metric}'",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    pivot = (metrics_df
             .groupby(["strategy", "group"])[metric]
             .agg(agg)
             .unstack("group"))
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    groups = list(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(groups))
    for gi, g in enumerate(groups):
        ax.bar(x + (gi - (len(groups) - 1) / 2) * width, pivot[g].values,
               width=width, label=g, edgecolor="black", linewidth=0.4)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_ylabel(f"{agg}({metric})")
    ax.set_title(f"Group breakdown — {agg}({metric}) per strategy × group")
    ax.legend(title="group")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
