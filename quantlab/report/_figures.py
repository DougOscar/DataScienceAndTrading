"""Matplotlib figure builders for the tear sheet (DESIGN §7 / report-builder agent file).

Every function takes plain data (arrays / polars frames already pulled off the ledger /
trial store) and returns a ``matplotlib.figure.Figure`` — no ledger or file access here, so
each is independently testable.  One consistent palette (``_style``), labelled axes with units,
transparent surfaces so figures read on both light and dark notebook/Obsidian themes.
"""

from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import polars as pl
from matplotlib import patheffects

from .. import metrics as m
from . import _style as sty


def _fig(w: float = 7.0, h: float = 3.2, **kw) -> tuple[plt.Figure, Any]:
    return plt.subplots(figsize=(w, h), **kw)


# --------------------------------------------------------------------------- equity + underwater
def fig_equity_underwater(daily: pl.DataFrame) -> plt.Figure:
    with sty.style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.0), sharex=True,
                                       gridspec_kw={"height_ratios": [2.2, 1]})
        r = daily.sort("date")
        dates = r["date"].to_list()
        eq = m.equity_curve(r["ret"])
        dd = m.drawdown(r["ret"]) * 100.0
        ax1.plot(dates, eq, color=sty.BLUE, linewidth=1.8)
        ax1.set_ylabel("Equity (x initial, unlevered)")
        ax1.set_title("Equity curve and drawdown")
        ax2.fill_between(dates, dd, 0, color=sty.RED, alpha=0.55, linewidth=0)
        ax2.plot(dates, dd, color=sty.RED, linewidth=0.8)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- yearly Sharpe
def fig_yearly_sharpe(daily: pl.DataFrame, ppy: float) -> plt.Figure:
    with sty.style():
        ps = m.period_sharpe(daily, "1y", ppy)
        years = [d.year if hasattr(d, "year") else str(d) for d in ps["period"].to_list()]
        vals = ps["sharpe"].to_numpy()
        fig, ax = _fig(7.0, 3.0)
        colors = [sty.BLUE if v >= 0 or not np.isfinite(v) else sty.RED for v in vals]
        ax.bar([str(y) for y in years], np.nan_to_num(vals), color=colors, width=0.65)
        ax.axhline(0, color=sty.GRID, linewidth=0.8)
        ax.set_ylabel("Annualised Sharpe")
        ax.set_title(f"Yearly Sharpe (annualised from daily returns, periods/year = {ppy:g})")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- monthly Sharpe + bootstrap band
def fig_monthly_sharpe_band(daily: pl.DataFrame, ppy: float, n_boot: int = 500,
                            rng: np.random.Generator | None = None) -> plt.Figure:
    rng = rng or np.random.default_rng(0)
    with sty.style():
        r = daily.sort("date").with_columns(pl.col("date").cast(pl.Datetime("ms")))
        months = (r.group_by_dynamic("date", every="1mo", label="left")
                 .agg(pl.col("ret").alias("rets")))
        labels, mid, lo, hi = [], [], [], []
        for row in months.iter_rows(named=True):
            x = np.asarray(row["rets"], dtype=float)
            labels.append(str(row["date"])[:7])
            if x.size < 2 or x.std(ddof=1) == 0:
                mid.append(float("nan")); lo.append(float("nan")); hi.append(float("nan"))
                continue
            mid.append(m.sharpe(x, ppy))
            boot = rng.choice(x, size=(n_boot, x.size), replace=True)
            with np.errstate(invalid="ignore", divide="ignore"):
                bsh = boot.mean(axis=1) / boot.std(axis=1, ddof=1) * math.sqrt(ppy)
            bsh = bsh[np.isfinite(bsh)]
            q = np.quantile(bsh, [0.05, 0.95]) if bsh.size else (float("nan"), float("nan"))
            lo.append(q[0]); hi.append(q[1])
        mid, lo, hi = np.array(mid), np.array(lo), np.array(hi)
        fig, ax = _fig(8.0, 3.2)
        x = np.arange(len(labels))
        ax.fill_between(x, lo, hi, color=sty.BLUE, alpha=0.20, linewidth=0, label="5th-95th pctile (bootstrap)")
        ax.plot(x, mid, color=sty.BLUE, marker="o", markersize=3, linewidth=1.0, label="point estimate")
        ax.axhline(0, color=sty.GRID, linewidth=0.8)
        step = max(1, len(labels) // 18)
        ax.set_xticks(x[::step]); ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)],
                                                      rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Annualised Sharpe")
        ax.set_title("Monthly Sharpe with bootstrap band\n"
                     "(~21 daily returns per month: very noisy — read the band, not the dot)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- monthly returns heatmap
def fig_monthly_heatmap(daily: pl.DataFrame) -> plt.Figure:
    with sty.style():
        mo = m.period_returns(daily, "1mo")
        years = sorted({d.year for d in mo["period"].to_list()})
        grid = np.full((len(years), 12), np.nan)
        for d, ret in zip(mo["period"].to_list(), mo["ret"].to_list()):
            grid[years.index(d.year), d.month - 1] = ret * 100.0
        vmax = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1.0
        fig, ax = _fig(8.0, 0.35 * len(years) + 1.4)
        im = ax.imshow(grid, cmap=_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_yticks(range(len(years))); ax.set_yticklabels(years)
        for i in range(len(years)):
            for j in range(12):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:+.1f}", ha="center", va="center", fontsize=6.5, color=sty.INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Monthly return (%)")
        ax.set_title("Monthly returns (%, unlevered)")
        fig.tight_layout()
        return fig


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("div_blue_red", [sty.RED, sty.DIVERGING[1], sty.BLUE])


# --------------------------------------------------------------------------- CPCV path fan
def fig_cpcv_path_fan(cpcv_paths: pl.DataFrame | None) -> plt.Figure | None:
    if cpcv_paths is None or cpcv_paths.is_empty():
        return None
    with sty.style():
        fig, ax = _fig(7.5, 3.4)
        curves = []
        for (_pid,), g in cpcv_paths.group_by("path_id", maintain_order=True):
            g = g.sort("date")
            eq = m.equity_curve(g["ret"])
            ax.plot(range(len(eq)), eq, color=sty.SEQ_BLUE[2], alpha=0.35, linewidth=0.9)
            curves.append(eq)
        if curves:
            n = min(len(c) for c in curves)
            stacked = np.vstack([c[:n] for c in curves])
            med = np.median(stacked, axis=0)
            ax.plot(range(n), med, color=sty.BLUE, linewidth=2.0, label="median path")
        ax.set_xlabel("OOS day index (paths not calendar-aligned)")
        ax.set_ylabel("Equity (x initial)")
        ax.set_title(f"CPCV OOS path fan ({cpcv_paths['path_id'].n_unique()} paths)")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- parameter plateau heatmap
def fig_plateau_heatmap(plateau_detail: dict[str, Any] | None) -> plt.Figure | None:
    """#59: plots the **raw** per-point Sharpe on a diverging (blue<->red) scale centred at 0,
    with the peak (the re-evaluated selected configuration -- not one of the perturbed points,
    so it isn't a cell of its own) marked on the colour scale. The previous version coloured
    ``sharpe / peak_sharpe`` on a ``vmin=0`` sequential scale: on a losing peak (peak <= 0) that
    ratio inverts sign meaning -- a worse (more negative) point divided by a negative peak comes
    out as the *largest* positive ratio and paints the darkest "best" colour, while the peak
    itself (ratio = 1) sits in the middle of the scale, not at either end -- so the figure was
    unreadable exactly when the plateau gate needed reading most."""
    points = (plateau_detail or {}).get("points")
    if not points:
        return None
    with sty.style():
        axes = []
        for p in points:
            if p.get("param") not in axes:
                axes.append(p["param"])
        offsets = sorted({p.get("offset") for p in points},
                        key=lambda o: (0, float(o)) if isinstance(o, (int, float)) else (1, str(o)))
        grid = np.full((len(axes), len(offsets)), np.nan)
        passed = np.zeros_like(grid, dtype=bool)
        peak = (plateau_detail or {}).get("peak_sharpe")
        peak_ok = peak is not None and np.isfinite(peak)
        for p in points:
            i, j = axes.index(p["param"]), offsets.index(p["offset"])
            sr = p.get("sharpe")
            if sr is not None and np.isfinite(sr):
                grid[i, j] = sr                                    # raw Sharpe, never sr / peak
                passed[i, j] = bool(p.get("pass"))
        finite = grid[np.isfinite(grid)]
        vmax = max([1e-6] + ([float(np.max(np.abs(finite)))] if finite.size else [])
                   + ([abs(float(peak))] if peak_ok else []))
        fig, ax = _fig(max(5.0, 1.1 * len(offsets)), max(2.2, 0.55 * len(axes) + 1.0))
        im = ax.imshow(grid, cmap=_diverging_cmap(), vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(offsets))); ax.set_xticklabels([str(o) for o in offsets])
        ax.set_yticks(range(len(axes))); ax.set_yticklabels(axes)
        for i in range(len(axes)):
            for j in range(len(offsets)):
                if np.isfinite(grid[i, j]):
                    mark = "" if passed[i, j] else "x"
                    txt = ax.text(j, i, mark, ha="center", va="center", fontsize=9, color="white",
                                 fontweight="bold")
                    txt.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground=sty.INK)])
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Sharpe (annualised, full dev window)")
        if peak_ok:
            cb.ax.axhline(peak, color=sty.INK, linewidth=1.8)
            trans = mtransforms.blended_transform_factory(cb.ax.transAxes, cb.ax.transData)
            peak_lbl = cb.ax.text(0.5, peak, f"peak {peak:.2f}", transform=trans, va="center", ha="center",
                                  fontsize=7.5, color="white", fontweight="bold")
            peak_lbl.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground=sty.INK)])
        peak_txt = f"peak Sharpe {peak:.2f}" if peak_ok else "peak Sharpe: not available"
        not_assessable = " -- not assessable (peak <= 0)" if peak_ok and peak <= 0 else ""
        ax.set_xlabel("Offset from the selected value (pre-registered plateau scale)")
        ax.set_ylabel("Parameter axis")
        ax.set_title(f"Parameter plateau (judge-run perturbations; x = fails >=50% of peak; "
                    f"{peak_txt}{not_assessable})")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- cost-sensitivity curve
def fig_cost_sensitivity(mults: np.ndarray | None, sharpes: np.ndarray | None,
                         threshold: float | None = None) -> plt.Figure | None:
    if mults is None or sharpes is None or len(mults) == 0:
        return None
    with sty.style():
        fig, ax = _fig(6.5, 3.0)
        ax.plot(mults, sharpes, color=sty.BLUE, marker="o", markersize=4)
        if threshold is not None:
            ax.axhline(threshold, color=sty.RED, linewidth=1.0, linestyle="--",
                      label=f"cost-stress gate ({threshold:g})")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("Spread multiplier (x current dev spread)")
        ax.set_ylabel("Annualised Sharpe")
        ax.set_title("Cost-sensitivity curve")
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- realised vs bootstrapped DD
def fig_dd_distribution(daily: pl.DataFrame, n_boot: int = 1000,
                        rng: np.random.Generator | None = None) -> plt.Figure:
    from .. import stats as st
    rng = rng or np.random.default_rng(0)
    with sty.style():
        boot = st.bootstrap_stats(daily, fns={"max_dd": st.VEC_STATS["max_dd"]}, n_boot=n_boot, rng=rng)
        dd_boot = -boot["max_dd"] * 100.0
        realised = -m.max_drawdown(daily["ret"]) * 100.0
        fig, ax = _fig(6.5, 3.0)
        ax.hist(dd_boot, bins=40, color=sty.BLUE, alpha=0.55, density=True)
        ax.axvline(realised, color=sty.RED, linewidth=1.8, label=f"realised {realised:.1f}%")
        p95 = float(np.quantile(dd_boot, 0.95))
        ax.axvline(p95, color=sty.INK, linewidth=1.0, linestyle="--", label=f"bootstrap p95 {p95:.1f}%")
        ax.set_xlabel("Max drawdown magnitude (%, unlevered)")
        ax.set_ylabel("Density (bootstrap draws)")
        ax.set_title("Realised vs bootstrapped max-drawdown distribution")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig


# --------------------------------------------------------------------------- walk-forward parameter drift
def fig_wfo_param_drift(wfo_params: pl.DataFrame | None) -> plt.Figure | None:
    if wfo_params is None or wfo_params.is_empty():
        return None
    pcols = [c for c in wfo_params.columns if c.startswith("param_")
            and wfo_params[c].dtype.is_numeric()]
    if not pcols:
        return None
    with sty.style():
        w = wfo_params.sort("refit_date") if "refit_date" in wfo_params.columns else wfo_params
        x = w["refit_date"].to_list() if "refit_date" in w.columns else list(range(w.height))
        fig, axes = plt.subplots(len(pcols), 1, figsize=(7.0, 1.5 * len(pcols) + 0.5), sharex=True, squeeze=False)
        for k, c in enumerate(pcols):
            ax = axes[k, 0]
            ax.step(x, w[c].to_numpy(), where="post", color=sty.CATEGORICAL[k % len(sty.CATEGORICAL)],
                    linewidth=1.4, marker="o", markersize=3)
            ax.set_ylabel(c.removeprefix("param_"))
        axes[-1, 0].set_xlabel("Refit date")
        fig.suptitle("Walk-forward parameter drift (selected value per refit)", fontsize=11, y=0.995)
        fig.tight_layout()
        return fig
