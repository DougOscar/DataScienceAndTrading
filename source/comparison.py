"""Cross-strategy comparison: registry of baseline configurations, batched
runner, and parquet caching for the comparison dashboard notebook.

The registry below mirrors the baseline ``GROUP_TIMEFRAMES`` / ``baseline_params``
used inside each per-strategy notebook on ``main``. Running every entry through
:func:`run_all_strategies` produces three long-format tables — metrics, equity
curves, and trades — keyed by ``(strategy, group, timeframe, asset)``. The
tables are the inputs the comparison dashboard plots consume.

A small parquet cache makes re-runs cheap: deleting ``cache_dir`` is the only
invalidation gesture. See ``DocumentationVault/general/Strategy_Comparison_Dashboard.md``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .data_loader import build_lazy_grid
from .metrics import compute_metrics
from .runner import GridKey, run_backtests_with_params
from .strategy import (
    BBSqueezeParams,
    BBSqueezeStrategy,
    DonchianBreakoutParams,
    DonchianBreakoutStrategy,
    DualThrustParams,
    DualThrustStrategy,
    EMARibbonParams,
    EMARibbonStrategy,
    FFTCycleFilterParams,
    FFTCycleFilterStrategy,
    HMMRegimeFilterParams,
    HMMRegimeFilterStrategy,
    HurstRegimeSwitcherParams,
    HurstRegimeSwitcherStrategy,
    KeltnerReversionParams,
    KeltnerReversionStrategy,
    LinearRegressionChannelParams,
    LinearRegressionChannelStrategy,
    MACDHistogramParams,
    MACDHistogramStrategy,
    RSIMeanReversionParams,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    StochasticDivergenceParams,
    StochasticDivergenceStrategy,
    StrategyParams,
    VWAPReversionParams,
    VWAPReversionStrategy,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyRegistration:
    """One row of the comparison registry — everything needed to run a strategy.

    ``base_kwargs`` populates the params dataclass for the *forex* group.
    ``b3_overrides`` (if set) is merged on top for the *b3* group — typically
    ``session_start=9, session_end=18`` so swing strategies respect B3 hours.

    Intraday strategies (Dual Thrust, VWAP) bake their session window into
    ``base_kwargs`` directly because they are B3-only.
    """

    name: str
    notebook_id: str
    strategy_cls: type
    params_cls: type
    group_timeframes: Dict[str, List[str]]
    base_kwargs: Dict[str, Any] = field(default_factory=dict)
    b3_overrides: Dict[str, Any] = field(default_factory=dict)


STRATEGY_REGISTRY: List[StrategyRegistration] = [
    StrategyRegistration(
        name="SMA Crossover",
        notebook_id="01",
        strategy_cls=SMACrossoverStrategy,
        params_cls=StrategyParams,
        group_timeframes={"forex": ["1h", "4h", "1D"], "b3": ["1min", "5min", "15min", "30min"]},
        base_kwargs=dict(fast=20, slow=50, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="RSI Mean Reversion",
        notebook_id="02",
        strategy_cls=RSIMeanReversionStrategy,
        params_cls=RSIMeanReversionParams,
        group_timeframes={"forex": ["15min", "1h", "4h"]},
        base_kwargs=dict(rsi_period=14, rsi_lower=30.0, rsi_upper=70.0,
                         atr_period=14, sl_atr_mult=1.5, tp_atr_mult=2.0),
    ),
    StrategyRegistration(
        name="BB Squeeze Breakout",
        notebook_id="03",
        strategy_cls=BBSqueezeStrategy,
        params_cls=BBSqueezeParams,
        group_timeframes={"forex": ["1h", "4h", "1D"], "b3": ["1h", "4h", "1D"]},
        base_kwargs=dict(bb_period=20, bb_mult=2.0, squeeze_lookback=20,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="MACD Histogram",
        notebook_id="04",
        strategy_cls=MACDHistogramStrategy,
        params_cls=MACDHistogramParams,
        group_timeframes={"forex": ["1h", "4h", "1D"], "b3": ["1h", "4h", "1D"]},
        base_kwargs=dict(macd_fast=12, macd_slow=26, signal_period=9,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.5),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="Donchian Breakout",
        notebook_id="05",
        strategy_cls=DonchianBreakoutStrategy,
        params_cls=DonchianBreakoutParams,
        group_timeframes={"forex": ["4h", "1D"], "b3": ["1h", "4h"]},
        base_kwargs=dict(dc_entry=20, dc_exit=10, atr_period=14, sl_atr_mult=3.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="EMA Ribbon + RSI",
        notebook_id="06",
        strategy_cls=EMARibbonStrategy,
        params_cls=EMARibbonParams,
        group_timeframes={"forex": ["1h", "4h"], "b3": ["15min", "1h"]},
        base_kwargs=dict(ema_p1=5, ema_p2=8, ema_p3=13, ema_p4=21,
                         rsi_period=14, rsi_overbought=65.0, rsi_oversold=35.0,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="Keltner Reversion",
        notebook_id="07",
        strategy_cls=KeltnerReversionStrategy,
        params_cls=KeltnerReversionParams,
        group_timeframes={"forex": ["1h", "4h"]},
        base_kwargs=dict(kc_period=20, kc_atr_period=10, kc_mult=2.0,
                         atr_period=14, sl_atr_mult=2.5, tp_atr_mult=2.0,
                         adx_period=14, adx_max=25.0, use_adx_filter=True),
    ),
    StrategyRegistration(
        name="Hurst Regime Switcher",
        notebook_id="08",
        strategy_cls=HurstRegimeSwitcherStrategy,
        params_cls=HurstRegimeSwitcherParams,
        group_timeframes={"forex": ["4h", "1D"], "b3": ["1h", "4h"]},
        base_kwargs=dict(hurst_window=128, hurst_step=4,
                         hurst_trend_threshold=0.6, hurst_mean_rev_threshold=0.4,
                         trend_fast=20, trend_slow=50,
                         rsi_period=14, rsi_lower=30.0, rsi_upper=70.0,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="Dual Thrust Intraday",
        notebook_id="09",
        strategy_cls=DualThrustStrategy,
        params_cls=DualThrustParams,
        group_timeframes={"b3": ["5min", "15min"]},
        base_kwargs=dict(dt_lookback=4, k1=0.5, k2=0.5,
                         atr_period=14, sl_atr_mult=1.5, tp_atr_mult=2.5,
                         entry_cutoff_minutes=30),
    ),
    StrategyRegistration(
        name="HMM Regime Filter",
        notebook_id="10",
        strategy_cls=HMMRegimeFilterStrategy,
        params_cls=HMMRegimeFilterParams,
        group_timeframes={"forex": ["4h", "1D"], "b3": ["1h", "4h"]},
        base_kwargs=dict(n_states=3, hmm_train_window=504, hmm_prob_threshold=0.7,
                         rv_window=20, vol_period=20,
                         sma_fast=20, sma_slow=50,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.5),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="VWAP Intraday Reversion",
        notebook_id="11",
        strategy_cls=VWAPReversionStrategy,
        params_cls=VWAPReversionParams,
        group_timeframes={"b3": ["5min", "15min"]},
        base_kwargs=dict(n_sigma=2.0, atr_period=14, sl_atr_mult=2.0, tp_atr_mult=1.5,
                         vwap_warmup_bars=10, entry_cutoff_minutes=30),
    ),
    StrategyRegistration(
        name="Stochastic Divergence",
        notebook_id="12",
        strategy_cls=StochasticDivergenceStrategy,
        params_cls=StochasticDivergenceParams,
        group_timeframes={"forex": ["1h", "4h"], "b3": ["15min", "1h"]},
        base_kwargs=dict(k_period=14, d_period=3,
                         stoch_oversold=20.0, stoch_overbought=80.0,
                         atr_period=14, sl_atr_mult=1.5, tp_atr_mult=2.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="LR Channel",
        notebook_id="13",
        strategy_cls=LinearRegressionChannelStrategy,
        params_cls=LinearRegressionChannelParams,
        group_timeframes={"forex": ["4h", "1D"], "b3": ["1h", "4h"]},
        base_kwargs=dict(lr_period=50, lr_mult=2.0, slope_threshold=0.3,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
                         mode="auto"),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
    StrategyRegistration(
        name="FFT Cycle Filter",
        notebook_id="14",
        strategy_cls=FFTCycleFilterStrategy,
        params_cls=FFTCycleFilterParams,
        group_timeframes={"forex": ["4h", "1D"], "b3": ["1h", "4h"]},
        base_kwargs=dict(fft_window=128, min_cycle_bars=8, max_cycle_bars=64,
                         n_harmonics=0, delta_min_atr=0.1,
                         atr_period=14, sl_atr_mult=2.0, tp_atr_mult=3.0,
                         min_power_fraction=0.0),
        b3_overrides=dict(session_start=9, session_end=18),
    ),
]


def make_params_for_group(reg: StrategyRegistration, group: str):
    """Construct ``reg.params_cls`` for ``group``, merging B3 overrides if present."""
    kw = dict(reg.base_kwargs)
    if group == "b3" and reg.b3_overrides:
        kw.update(reg.b3_overrides)
    return reg.params_cls(**kw)


# ---------------------------------------------------------------------------
# Runner + cache
# ---------------------------------------------------------------------------


def _params_fingerprint(reg: StrategyRegistration) -> str:
    """Stable hash of ``(name, base_kwargs, b3_overrides, group_timeframes)``.

    Used as part of the cache filename so changing baselines auto-invalidates."""
    payload = json.dumps(
        {
            "name": reg.name,
            "base_kwargs": reg.base_kwargs,
            "b3_overrides": reg.b3_overrides,
            "group_timeframes": reg.group_timeframes,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _slug(name: str) -> str:
    return name.lower().replace("+", "plus").replace(" ", "_").replace("/", "_")


def _result_to_equity_rows(strategy: str, key: GridKey, result) -> List[dict]:
    eq = result.equity
    if eq is None or eq.empty:
        return []
    group, tf, asset = key
    return [
        {"strategy": strategy, "group": group, "tf": tf, "asset": asset,
         "timestamp": ts, "equity": float(v)}
        for ts, v in eq.items()
    ]


def _result_to_trade_rows(strategy: str, key: GridKey, result) -> List[dict]:
    t = result.trades
    if t is None or t.empty:
        return []
    group, tf, asset = key
    out = t.copy()
    out.insert(0, "strategy", strategy)
    out.insert(1, "group", group)
    out.insert(2, "tf", tf)
    out.insert(3, "asset", asset)
    return out.to_dict(orient="records")


def _run_single_strategy(
    reg: StrategyRegistration,
    data_dir: Path,
    *,
    n_jobs: int | str | None,
    progress: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Run one registry entry across its full grid, returning three row lists."""
    grid = build_lazy_grid(data_dir, group_timeframes=reg.group_timeframes)
    grid = {g: tfs for g, tfs in grid.items() if g in reg.group_timeframes}

    params_by_key: Dict[GridKey, Any] = {}
    for group, tfs in grid.items():
        params = make_params_for_group(reg, group)
        for tf, assets in tfs.items():
            for asset in assets:
                params_by_key[(group, tf, asset)] = params

    if not params_by_key:
        return [], [], []

    results = run_backtests_with_params(
        grid, params_by_key,
        strategy_cls=reg.strategy_cls,
        n_jobs=n_jobs, progress=progress,
    )

    metric_rows, equity_rows, trade_rows = [], [], []
    for key, result in results.items():
        group, tf, asset = key
        m = compute_metrics(result)
        metric_rows.append({"strategy": reg.name, "group": group, "tf": tf,
                            "asset": asset, **m})
        equity_rows.extend(_result_to_equity_rows(reg.name, key, result))
        trade_rows.extend(_result_to_trade_rows(reg.name, key, result))
    return metric_rows, equity_rows, trade_rows


def run_all_strategies(
    data_dir: str | Path,
    registry: List[StrategyRegistration] | None = None,
    *,
    cache_dir: str | Path | None = None,
    n_jobs: int | str | None = "auto",
    progress: bool = True,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every strategy in ``registry`` across its native group/TF grid.

    Returns ``(metrics_df, equity_df, trades_df)`` — long-format tables keyed
    by ``(strategy, group, tf, asset)``.

    When ``cache_dir`` is set, each strategy's three tables are written to
    ``cache_dir/<slug>__<fingerprint>__{metrics,equity,trades}.parquet`` and
    reused on subsequent runs. Change a strategy's baseline params or grid and
    its cache invalidates automatically (fingerprint changes). To force a full
    rerun, pass ``force=True`` or delete ``cache_dir``.

    Strategies whose grid resolves to zero cells (e.g. group data missing) are
    silently skipped — the caller's notebook can surface the resulting absence.
    """
    if registry is None:
        registry = STRATEGY_REGISTRY
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    metrics_all: List[pd.DataFrame] = []
    equity_all: List[pd.DataFrame] = []
    trades_all: List[pd.DataFrame] = []

    for reg in registry:
        fp = _params_fingerprint(reg)
        slug = _slug(reg.name)
        cache_paths = None
        if cache_dir is not None:
            cache_paths = {
                kind: cache_dir / f"{slug}__{fp}__{kind}.parquet"
                for kind in ("metrics", "equity", "trades")
            }

        if (cache_paths is not None and not force
                and all(p.exists() for p in cache_paths.values())):
            if progress:
                print(f"[cache] {reg.name:<24s}  (fingerprint={fp})")
            m_df = pd.read_parquet(cache_paths["metrics"])
            e_df = pd.read_parquet(cache_paths["equity"])
            t_df = pd.read_parquet(cache_paths["trades"])
        else:
            if progress:
                print(f"[run]   {reg.name:<24s}")
            m_rows, e_rows, t_rows = _run_single_strategy(
                reg, data_dir, n_jobs=n_jobs, progress=progress
            )
            m_df = pd.DataFrame(m_rows)
            e_df = pd.DataFrame(e_rows)
            t_df = pd.DataFrame(t_rows)
            if cache_paths is not None and not m_df.empty:
                m_df.to_parquet(cache_paths["metrics"], index=False)
                # Equity/trades may be empty for some strategies; still write
                # (possibly-empty) parquet so the cache key is fully populated.
                e_df.to_parquet(cache_paths["equity"], index=False)
                t_df.to_parquet(cache_paths["trades"], index=False)

        if not m_df.empty:
            metrics_all.append(m_df)
        if not e_df.empty:
            equity_all.append(e_df)
        if not t_df.empty:
            trades_all.append(t_df)

    metrics_df = pd.concat(metrics_all, ignore_index=True) if metrics_all else pd.DataFrame()
    equity_df = pd.concat(equity_all, ignore_index=True) if equity_all else pd.DataFrame()
    trades_df = pd.concat(trades_all, ignore_index=True) if trades_all else pd.DataFrame()
    if not equity_df.empty:
        equity_df["timestamp"] = pd.to_datetime(equity_df["timestamp"])
    return metrics_df, equity_df, trades_df


# ---------------------------------------------------------------------------
# Aggregations consumed by the dashboard plots
# ---------------------------------------------------------------------------


def daily_returns_from_equity(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Convert long-format equity points to long-format daily returns.

    Equity is in PnL **points** (cumulative). We resample to daily-last,
    forward-fill so non-trading days carry the prior cumulative PnL, then
    diff. The result is one row per (strategy, group, tf, asset, day).
    """
    if equity_df.empty:
        return equity_df
    rows: List[pd.DataFrame] = []
    keys = ["strategy", "group", "tf", "asset"]
    for tup, sub in equity_df.groupby(keys, sort=False):
        s = (sub.set_index("timestamp")["equity"]
                .sort_index()
                .resample("1D").last()
                .ffill())
        ret = s.diff().dropna().rename("daily_return").to_frame()
        for col, val in zip(keys, tup):
            ret[col] = val
        rows.append(ret.reset_index().rename(columns={"timestamp": "day"}))
    return pd.concat(rows, ignore_index=True)


def pivot_metric(
    metrics_df: pd.DataFrame,
    *,
    metric: str,
    group: str | None = None,
    tf: str | None = None,
) -> pd.DataFrame:
    """Pivot a long metrics frame to ``strategy × asset`` for one (group, tf).

    Missing combinations stay ``NaN`` so heatmaps can mask them out.
    """
    sub = metrics_df
    if group is not None:
        sub = sub[sub["group"] == group]
    if tf is not None:
        sub = sub[sub["tf"] == tf]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index="strategy", columns="asset", values=metric, aggfunc="first")


def group_aggregated_metrics(
    metrics_df: pd.DataFrame,
    *,
    metric: str = "sharpe_daily",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate one metric across assets/TFs per (strategy, group)."""
    if metrics_df.empty:
        return metrics_df
    return (metrics_df
            .groupby(["strategy", "group"])[metric]
            .agg(agg)
            .unstack("group"))
