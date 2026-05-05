"""Correlation-aware cross-asset portfolio weighting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlation_weights(
    equity_curves: dict[str, pd.Series],
    lookback_days: int = 60,
    method: str = "inv_vol",
) -> dict[str, float]:
    """Compute portfolio weights from per-asset equity curves.

    method:
      "equal"   — 1/n per asset (ignores correlation).
      "inv_vol" — weight ∝ 1/σ_daily; larger for lower-volatility assets.
      "min_var" — minimum-variance weights via Σ^{-1}·1 (long-only clamped).

    Returns a dict of {asset: weight} summing to 1.
    """
    assets = list(equity_curves.keys())
    n = len(assets)
    if n == 0:
        return {}
    if n == 1:
        return {assets[0]: 1.0}

    if method == "equal":
        return {a: 1.0 / n for a in assets}

    # Compute trailing daily returns per asset
    ret_dict = {}
    for asset, eq in equity_curves.items():
        if eq.empty:
            continue
        daily = eq.resample("1D").last().ffill()
        ret_dict[asset] = daily.diff().dropna()

    ret_df = pd.DataFrame(ret_dict).dropna(how="all")
    if ret_df.empty or len(ret_df) < 5:
        return {a: 1.0 / n for a in assets}

    tail = ret_df.tail(lookback_days)
    vols = tail.std().replace(0, np.nan)

    if method == "inv_vol":
        inv_v = (1.0 / vols).fillna(0.0)
        total = inv_v.sum()
        if total == 0:
            return {a: 1.0 / n for a in assets}
        return inv_v.div(total).reindex(assets, fill_value=0.0).to_dict()

    if method == "min_var":
        cov = tail.cov().values
        try:
            cov_inv = np.linalg.inv(cov)
            ones = np.ones(len(tail.columns))
            raw = cov_inv @ ones
            raw = np.maximum(raw, 0.0)          # long-only constraint
            total = raw.sum()
            if total == 0:
                raise np.linalg.LinAlgError("zero total")
            w = {a: float(raw[j] / total) for j, a in enumerate(tail.columns)}
            return {a: w.get(a, 0.0) for a in assets}
        except np.linalg.LinAlgError:
            # Singular covariance — fall back to inv_vol
            inv_v = (1.0 / vols).fillna(0.0)
            total = inv_v.sum()
            if total == 0:
                return {a: 1.0 / n for a in assets}
            return inv_v.div(total).reindex(assets, fill_value=0.0).to_dict()

    return {a: 1.0 / n for a in assets}


def weighted_portfolio(
    per_asset_results: dict,
    weights: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Combine per-asset BacktestResults using given portfolio weights.

    Trades carry two extra columns:
      weight       — the asset's portfolio weight.
      pnl_weighted — pnl_points * weight (the portfolio-level PnL contribution).

    The returned equity series is built from pnl_weighted, so metrics computed
    on it reflect the correlation-adjusted portfolio performance.
    """
    frames = []
    for asset, result in per_asset_results.items():
        if result.trades.empty:
            continue
        w = weights.get(asset, 0.0)
        t = result.trades.copy()
        t["asset"] = asset
        t["weight"] = w
        t["pnl_weighted"] = t["pnl_points"] * w
        frames.append(t)

    if not frames:
        return pd.DataFrame(), pd.Series(dtype=float)

    combined = (
        pd.concat(frames, ignore_index=True)
        .sort_values("exit_time")
        .reset_index(drop=True)
    )
    equity = pd.Series(
        combined["pnl_weighted"].cumsum().values,
        index=pd.to_datetime(combined["exit_time"].values),
        name="equity_weighted",
    )
    return combined, equity
