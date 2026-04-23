"""Performance metrics for a backtest result."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd


def _norm_sf(x: float) -> float:
    """Survival function of the standard normal; used as a p-value proxy."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    streak = cur = 0
    for p in pnls:
        if p < 0:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0
    return streak


def compute_metrics(result, periods_per_year: int = 252) -> Dict[str, float]:
    """Return a dict of standard performance metrics.

    `periods_per_year` is only used to annualize the daily Sharpe.
    """
    trades = result.trades
    eq = result.equity

    out: Dict[str, float] = {
        "num_trades": 0,
        "total_pnl": 0.0,
        "win_rate": float("nan"),
        "avg_win": float("nan"),
        "avg_loss": float("nan"),
        "profit_factor": float("nan"),
        "expectancy": float("nan"),
        "max_drawdown": 0.0,
        "max_consec_losses": 0,
        "sharpe_daily": float("nan"),
        "sharpe_per_trade": float("nan"),
        "t_stat": float("nan"),
        "p_value": float("nan"),
    }
    if trades is None or trades.empty:
        return out

    pnls = trades["pnl_points"].to_numpy(dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    out["num_trades"] = int(len(pnls))
    out["total_pnl"] = float(pnls.sum())
    out["win_rate"] = float(len(wins) / len(pnls))
    out["avg_win"] = float(wins.mean()) if len(wins) else float("nan")
    out["avg_loss"] = float(losses.mean()) if len(losses) else float("nan")
    out["expectancy"] = float(pnls.mean())
    out["profit_factor"] = (
        float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0
        else float("inf") if len(wins) else float("nan")
    )
    out["max_consec_losses"] = int(_max_consecutive_losses(pnls))

    if eq is not None and not eq.empty:
        dd = eq - eq.cummax()
        out["max_drawdown"] = float(dd.min())

        daily_eq = eq.resample("1D").last().ffill()
        daily_ret = daily_eq.diff().dropna()
        if len(daily_ret) > 1 and daily_ret.std() > 0:
            out["sharpe_daily"] = float(
                daily_ret.mean() / daily_ret.std() * np.sqrt(periods_per_year)
            )

    if len(pnls) > 1 and pnls.std(ddof=1) > 0:
        t = pnls.mean() / (pnls.std(ddof=1) / np.sqrt(len(pnls)))
        out["t_stat"] = float(t)
        out["p_value"] = float(2 * _norm_sf(abs(t)))
        out["sharpe_per_trade"] = float(
            pnls.mean() / pnls.std(ddof=1) * np.sqrt(len(pnls))
        )

    return out


def metrics_table(metrics: Dict[str, float]) -> pd.DataFrame:
    """Render a metrics dict as a one-column DataFrame for notebook display."""
    ordered = [
        "num_trades",
        "total_pnl",
        "win_rate",
        "profit_factor",
        "expectancy",
        "avg_win",
        "avg_loss",
        "max_drawdown",
        "max_consec_losses",
        "sharpe_daily",
        "sharpe_per_trade",
        "t_stat",
        "p_value",
    ]
    rows = [(k, metrics.get(k, float("nan"))) for k in ordered if k in metrics]
    return pd.DataFrame(rows, columns=["metric", "value"]).set_index("metric")
