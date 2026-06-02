"""Portfolio-level, multi-asset backtester for the multi-filter system.

The single-asset :class:`source.backtest.Backtester` cannot express the
system's portfolio rules — they are *cross-asset* constraints:

* **<= 2 % account risk per trade** — size = equity x risk_fraction /
  (sl_atr_mult x ATR_entry); the dollar risk to the ATR stop is therefore
  exactly ``equity x risk_fraction`` (0.02 by default).
* **<= 3 concurrent open trades** across the whole portfolio.
* **forex currency exclusion** — while ``EURUSD`` is open, no other pair that
  contains ``EUR`` or ``USD`` may be opened (auto-enabled for ``group="forex"``).
* one shared account: position size scales with the *running* equity, so the
  three rules interact (a binding concurrency cap changes which trades get the
  risk budget).

Signal generation stays per-asset and stateless (``strategy.generate_signals``),
so it parallelises trivially; only the *execution* layer here is cross-asset.

Also provided:

* :func:`consolidated_index` — the single scalar the optimiser maximises
  (Sharpe + win-rate + drawdown-adjusted return + profit factor +
  statistical significance), usable as a WFO ``score_fn``.
* :func:`portfolio_walk_forward` — WFO whose IS grid-search and OOS evaluation
  both run the *portfolio* engine; combos fan out across processes.
* :func:`portfolio_parameter_sensitivity` + :func:`overfitting_report` — the
  §7 robustness / overfitting diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_metrics
from .parallel import parallel_map

# Three-letter ISO currency / metal codes that appear in the forex filenames.
_FX_CODES = {
    "EUR", "USD", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF",
    "XAU", "XAG",  # gold / silver vs USD
}


def parse_currencies(asset: str, group: str) -> frozenset[str] | None:
    """Currencies an instrument exposes, or ``None`` when exclusion is off.

    * forex 6-letter pairs / metals -> ``{base, quote}`` (``EURUSD`` ->
      ``{EUR, USD}``, ``XAUUSD`` -> ``{XAU, USD}``);
    * named metals (``Platinum``/``Palladium``) -> ``{asset}`` (collides only
      with itself);
    * non-forex groups -> ``None`` (no currency-level exclusion; the engine
      still blocks re-opening the *same* asset and still enforces the
      concurrency cap).
    """
    if group != "forex":
        return None
    a = asset.upper()
    if len(a) == 6 and a.isalpha():
        return frozenset({a[:3], a[3:]})
    if a[:3] in _FX_CODES and a[3:6] in _FX_CODES:
        return frozenset({a[:3], a[3:6]})
    return frozenset({a})


@dataclass
class PortfolioBacktestResult:
    trades: pd.DataFrame
    equity: pd.Series                       # cumulative realised PnL (for metrics)
    balance: pd.Series                      # initial_capital + cumulative PnL
    open_positions: pd.Series               # concurrent-open count over time
    params: dict = field(default_factory=dict)
    initial_capital: float = 100_000.0


class PortfolioBacktester:
    """Bar-close, cross-asset execution with the system's portfolio rules.

    Parameters
    ----------
    params
        A ``MultiFilterSystemParams``-like object exposing ``sl_atr_mult``,
        ``tp_atr_mult``, ``use_take_profit``, ``risk_fraction``,
        ``session_start``, ``session_end``.
    group
        Market group ("forex" enables currency exclusion).
    max_concurrent
        Hard cap on simultaneously open trades (default 3).
    initial_capital
        Starting account equity; position size scales with running equity.
    """

    def __init__(
        self,
        params,
        *,
        group: str = "",
        max_concurrent: int = 3,
        initial_capital: float = 100_000.0,
        slippage_points: float = 0.0,
        currency_exclusion: str | bool = "auto",
    ):
        self.params = params
        self.group = group
        self.max_concurrent = max_concurrent
        self.initial_capital = initial_capital
        self.slippage_points = slippage_points
        if currency_exclusion == "auto":
            self.currency_exclusion = group == "forex"
        else:
            self.currency_exclusion = bool(currency_exclusion)

    def run(self, signals_by_asset: Dict[str, pd.DataFrame]) -> PortfolioBacktestResult:
        p = self.params
        sl_mult = float(p.sl_atr_mult)
        tp_mult = float(p.tp_atr_mult)
        use_tp = bool(getattr(p, "use_take_profit", False))
        risk_frac = float(getattr(p, "risk_fraction", 0.02))
        s_start = getattr(p, "session_start", None)
        s_end = getattr(p, "session_end", None)
        has_session = s_start is not None and s_end is not None
        slip = self.slippage_points

        # --- flatten all assets into one time-ordered event stream --------
        frames = []
        assets = sorted(signals_by_asset)
        for code, asset in enumerate(assets):
            df = signals_by_asset[asset]
            if df is None or df.empty:
                continue
            sub = pd.DataFrame(
                {
                    "t": df.index.values.astype("datetime64[ns]"),
                    "code": code,
                    "open": df["open"].to_numpy(float),
                    "high": df["high"].to_numpy(float),
                    "low": df["low"].to_numpy(float),
                    "close": df["close"].to_numpy(float),
                    "atr": df["atr"].to_numpy(float),
                    "sig": df["signal"].to_numpy(float),
                    "hour": df.index.hour.values,
                }
            )
            frames.append(sub)
        empty = PortfolioBacktestResult(
            pd.DataFrame(), pd.Series(dtype=float), pd.Series(dtype=float),
            pd.Series(dtype=float), p.as_dict() if hasattr(p, "as_dict") else {},
            self.initial_capital,
        )
        if not frames:
            return empty

        ev = pd.concat(frames, ignore_index=True)
        ev = ev.sort_values(["t", "code"], kind="stable").reset_index(drop=True)

        t = ev["t"].to_numpy()
        code = ev["code"].to_numpy()
        o = ev["open"].to_numpy()
        hi = ev["high"].to_numpy()
        lo = ev["low"].to_numpy()
        cl = ev["close"].to_numpy()
        atr = ev["atr"].to_numpy()
        sig = ev["sig"].to_numpy()
        hour = ev["hour"].to_numpy()
        n = len(ev)

        currencies = [parse_currencies(a, self.group) for a in assets]
        last_idx_for_code: Dict[int, int] = {}
        for i in range(n):
            last_idx_for_code[code[i]] = i

        # --- per-asset open-position state --------------------------------
        pos = {}  # code -> dict(dir, entry, entry_t, sl, tp, size, entry_i)
        open_ccy: Dict[str, int] = {}  # currency -> open-trade count
        realized = 0.0
        trades: List[dict] = []
        bal_t: List[np.datetime64] = []
        bal_v: List[float] = []
        oc_t: List[np.datetime64] = []
        oc_v: List[int] = []

        def _close(c, exit_price, reason, i):
            nonlocal realized
            st = pos.pop(c)
            pnl = (exit_price - st["entry"]) * st["dir"] * st["size"] \
                - 2 * slip * st["size"]
            realized += pnl
            ccy = currencies[c]
            if self.currency_exclusion and ccy is not None:
                for k in ccy:
                    open_ccy[k] -= 1
                    if open_ccy[k] <= 0:
                        del open_ccy[k]
            trades.append(
                {
                    "asset": assets[c],
                    "entry_time": st["entry_t"],
                    "exit_time": t[i],
                    "direction": int(st["dir"]),
                    "entry": st["entry"],
                    "exit": float(exit_price),
                    "reason": reason,
                    "pnl_points": pnl,
                    "bars_held": i - st["entry_i"],
                    "size": st["size"],
                    "currencies": "/".join(sorted(ccy)) if ccy else assets[c],
                }
            )
            bal_t.append(t[i])
            bal_v.append(realized)

        for i in range(n):
            c = int(code[i])

            # 1) manage an open position for this asset
            if c in pos:
                st = pos[c]
                d = st["dir"]
                exit_price = None
                reason = None
                if d == 1:
                    if lo[i] <= st["sl"]:
                        exit_price, reason = st["sl"], "SL"
                    elif use_tp and hi[i] >= st["tp"]:
                        exit_price, reason = st["tp"], "TP"
                else:
                    if hi[i] >= st["sl"]:
                        exit_price, reason = st["sl"], "SL"
                    elif use_tp and lo[i] <= st["tp"]:
                        exit_price, reason = st["tp"], "TP"
                if exit_price is None and sig[i] == -d:
                    exit_price, reason = cl[i], "SIGNAL"
                if exit_price is None and has_session:
                    cur_in = s_start <= hour[i] < s_end
                    nxt_in = (
                        i + 1 < n and code[i + 1] == c
                        and s_start <= hour[i + 1] < s_end
                    )
                    # close on the asset's last in-session bar
                    if cur_in and (not nxt_in) and i != last_idx_for_code[c]:
                        exit_price, reason = cl[i], "SESSION_END"
                if exit_price is None and i == last_idx_for_code[c]:
                    exit_price, reason = cl[i], "EOD"
                if exit_price is not None:
                    _close(c, exit_price, reason, i)

            # 2) consider a new entry for this asset
            if c not in pos and sig[i] != 0 and not np.isnan(atr[i]) \
                    and atr[i] > 0:
                if has_session and not (s_start <= hour[i] < s_end):
                    pass
                elif len(pos) >= self.max_concurrent:
                    pass
                else:
                    ccy = currencies[c]
                    blocked = False
                    if self.currency_exclusion and ccy is not None:
                        blocked = any(k in open_ccy for k in ccy)
                    if not blocked:
                        equity_now = self.initial_capital + realized
                        dollar_risk = max(equity_now, 0.0) * risk_frac
                        size = dollar_risk / (sl_mult * atr[i])
                        if size > 0:
                            d = int(np.sign(sig[i]))
                            entry = cl[i]
                            pos[c] = {
                                "dir": d,
                                "entry": entry,
                                "entry_t": t[i],
                                "entry_i": i,
                                "sl": entry - d * sl_mult * atr[i],
                                "tp": entry + d * tp_mult * atr[i],
                                "size": size,
                            }
                            if self.currency_exclusion and ccy is not None:
                                for k in ccy:
                                    open_ccy[k] = open_ccy.get(k, 0) + 1

            oc_t.append(t[i])
            oc_v.append(len(pos))

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return PortfolioBacktestResult(
                trades_df, pd.Series(dtype=float), pd.Series(dtype=float),
                pd.Series(oc_v, index=pd.to_datetime(oc_t), name="open_positions"),
                p.as_dict() if hasattr(p, "as_dict") else {}, self.initial_capital,
            )
        trades_df = trades_df.sort_values("exit_time").reset_index(drop=True)
        eq = pd.Series(
            trades_df["pnl_points"].cumsum().to_numpy(),
            index=pd.to_datetime(trades_df["exit_time"].to_numpy()),
            name="equity_points",
        )
        balance = (self.initial_capital + eq).rename("balance")
        open_pos = pd.Series(
            oc_v, index=pd.to_datetime(oc_t), name="open_positions"
        )
        open_pos = open_pos[~open_pos.index.duplicated(keep="last")]
        return PortfolioBacktestResult(
            trades=trades_df,
            equity=eq,
            balance=balance,
            open_positions=open_pos,
            params=p.as_dict() if hasattr(p, "as_dict") else {},
            initial_capital=self.initial_capital,
        )


# ---------------------------------------------------------------------------
# Consolidated optimisation index
# ---------------------------------------------------------------------------

CONSOLIDATED_WEIGHTS = {
    "sharpe": 0.35,     # annualised Sharpe (compute_metrics.sharpe_daily)
    "win": 0.15,        # win rate vs 50 %
    "recovery": 0.25,   # total PnL / |max drawdown|  (DD-adjusted return)
    "pf": 0.10,         # profit factor (>1 good)
    "signif": 0.15,     # statistical significance (|t-stat|)
}
_MIN_TRADES_FOR_SCORE = 10


def consolidated_index(m: Dict[str, float]) -> float:
    """Single scalar the optimiser maximises — higher is better.

    ``m`` is a :func:`source.metrics.compute_metrics` dict.  Combines, with
    :data:`CONSOLIDATED_WEIGHTS`:

    * annualised Sharpe (clipped to [-3, 5]),
    * win-rate edge over 0.5 (scaled),
    * recovery factor = total PnL / |max drawdown| (drawdown-adjusted return),
    * profit factor via ``tanh(pf - 1)``,
    * significance = min(|t-stat|, 4) / 4.

    Returns ``-inf`` when there are too few trades or no usable Sharpe, so
    degenerate combos sort last in the WFO grid search.
    """
    nt = m.get("num_trades", 0) or 0
    if nt < _MIN_TRADES_FOR_SCORE:
        return float("-inf")
    sharpe = m.get("sharpe_daily", float("nan"))
    if not np.isfinite(sharpe):
        sharpe = m.get("sharpe_per_trade", float("nan"))
    if not np.isfinite(sharpe):
        return float("-inf")

    win = m.get("win_rate", float("nan"))
    win = 0.0 if not np.isfinite(win) else win
    total = m.get("total_pnl", 0.0)
    mdd = abs(m.get("max_drawdown", 0.0))
    recovery = (total / mdd) if mdd > 0 else (1.0 if total > 0 else 0.0)
    pf = m.get("profit_factor", float("nan"))
    pf_term = np.tanh((pf - 1.0)) if np.isfinite(pf) else 0.0
    tstat = m.get("t_stat", float("nan"))
    signif = (min(abs(tstat), 4.0) / 4.0) if np.isfinite(tstat) else 0.0

    w = CONSOLIDATED_WEIGHTS
    return float(
        w["sharpe"] * float(np.clip(sharpe, -3.0, 5.0))
        + w["win"] * ((win - 0.5) * 4.0)
        + w["recovery"] * float(np.clip(recovery, -2.0, 5.0))
        + w["pf"] * float(pf_term)
        + w["signif"] * float(signif)
    )


# ---------------------------------------------------------------------------
# Extra metrics required by the §5 backtest report
# ---------------------------------------------------------------------------


def calmar_ratio(result, periods_per_year: int = 252) -> float:
    """Annualised return / |max drawdown| from the realised equity curve."""
    eq = result.equity
    if eq is None or eq.empty:
        return float("nan")
    span_days = (eq.index[-1] - eq.index[0]).days
    if span_days <= 0:
        return float("nan")
    years = span_days / 365.25
    total = float(eq.iloc[-1])
    ann_return = total / years if years > 0 else float("nan")
    mdd = float((eq - eq.cummax()).min())
    return ann_return / abs(mdd) if mdd < 0 else float("nan")


def rolling_annualized_sharpe(
    equity: pd.Series, window_days: int = 90, periods_per_year: int = 252
) -> pd.Series:
    """Trailing annualised Sharpe of daily equity changes (window in days)."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float)
    daily = equity.resample("1D").last().ffill()
    ret = daily.diff()
    mu = ret.rolling(window_days, min_periods=max(5, window_days // 3)).mean()
    sd = ret.rolling(window_days, min_periods=max(5, window_days // 3)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mu / sd) * np.sqrt(periods_per_year)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def annual_profit_factor(trades: pd.DataFrame) -> pd.Series:
    """Profit factor computed independently per calendar year."""
    if trades is None or trades.empty:
        return pd.Series(dtype=float)
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"])
    t["year"] = t["exit_time"].dt.year
    rows = {}
    for yr, g in t.groupby("year"):
        pnl = g["pnl_points"].to_numpy(float)
        wins = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        rows[int(yr)] = (wins / losses) if losses > 0 else np.inf
    return pd.Series(rows, name="annual_profit_factor")


def random_trade_returns(
    trades: pd.DataFrame, n: int = 200, seed: int = 42
) -> np.ndarray:
    """Per-trade PnL of ``n`` randomly sampled trades (for the §5 histogram)."""
    if trades is None or trades.empty:
        return np.array([])
    pnl = trades["pnl_points"].to_numpy(float)
    if len(pnl) <= n:
        return pnl
    rng = np.random.default_rng(seed)
    return pnl[rng.choice(len(pnl), size=n, replace=False)]


def portfolio_metrics(result, periods_per_year: int = 252) -> Dict[str, float]:
    """``compute_metrics`` + Calmar + recovery factor + consolidated index."""
    m = compute_metrics(result, periods_per_year=periods_per_year)
    m["calmar_ratio"] = calmar_ratio(result, periods_per_year)
    mdd = abs(m.get("max_drawdown", 0.0))
    m["recovery_factor"] = (
        m.get("total_pnl", 0.0) / mdd if mdd > 0 else float("nan")
    )
    m["consolidated_index"] = consolidated_index(m)
    return m


# ---------------------------------------------------------------------------
# Helpers shared by the parallel workers (must stay top-level / picklable)
# ---------------------------------------------------------------------------


def _load_window(ds, lo: pd.Timestamp | None, hi: pd.Timestamp | None) -> pd.DataFrame:
    df = ds.load()
    if lo is not None or hi is not None:
        df = df.loc[lo:hi]
    return df


def _signals_for(strategy_cls, params, frames: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    strat = strategy_cls(params)
    out = {}
    for asset, df in frames.items():
        if df is None or df.empty:
            continue
        out[asset] = strat.generate_signals(df)
    return out


def run_portfolio(
    datasets: Dict[str, object],
    params,
    *,
    strategy_cls,
    group: str,
    max_concurrent: int = 3,
    initial_capital: float = 100_000.0,
    slippage: float = 0.0,
    lo: pd.Timestamp | None = None,
    hi: pd.Timestamp | None = None,
) -> PortfolioBacktestResult:
    """Generate signals per asset then run the cross-asset portfolio engine."""
    frames = {a: _load_window(ds, lo, hi) for a, ds in datasets.items()}
    sigs = _signals_for(strategy_cls, params, frames)
    bt = PortfolioBacktester(
        params,
        group=group,
        max_concurrent=max_concurrent,
        initial_capital=initial_capital,
        slippage_points=slippage,
    )
    return bt.run(sigs)


def _pf_cell(args: dict) -> Tuple[Tuple[str, str], PortfolioBacktestResult]:
    """Worker: run one (group, tf) portfolio across all its assets."""
    res = run_portfolio(
        args["datasets"], args["params"], strategy_cls=args["strategy_cls"],
        group=args["group"], max_concurrent=args["max_concurrent"],
        initial_capital=args["initial_capital"], slippage=args["slippage"],
    )
    return (args["group"], args["tf"]), res


def run_portfolio_grid(
    grid: Dict[str, Dict[str, Dict[str, object]]],
    params_by_group: Dict[str, object],
    *,
    strategy_cls,
    default_params=None,
    max_concurrent: int = 3,
    initial_capital: float = 100_000.0,
    slippage: float = 0.0,
    n_jobs: int | str | None = "auto",
    progress: bool = True,
) -> Dict[Tuple[str, str], PortfolioBacktestResult]:
    """Fan out one portfolio backtest per ``(group, tf)`` cell across processes.

    ``grid`` is ``{group: {tf: {asset: dataset}}}`` (e.g. from
    :func:`source.spark_loader.build_spark_grid`).  Each cell is independent —
    its own assets, its own shared account — so cells run in parallel; each
    worker only reads its own parquet slices.  ``params_by_group`` lets B3 use
    its session params while forex/crypto use the 24 h defaults.
    """
    items = []
    for group, tfs in grid.items():
        params = params_by_group.get(group, default_params)
        if params is None:
            continue
        for tf, assets in tfs.items():
            items.append({
                "datasets": assets, "params": params, "tf": tf,
                "strategy_cls": strategy_cls, "group": group,
                "max_concurrent": max_concurrent,
                "initial_capital": initial_capital, "slippage": slippage,
            })
    pairs = parallel_map(_pf_cell, items, n_jobs=n_jobs, progress=progress,
                         desc="portfolio")
    return dict(pairs)


def _pf_eval_combo(args: dict) -> Tuple[float, tuple]:
    """Worker: score one param combo's *portfolio* backtest on the IS window."""
    keys = args["keys"]
    combo = args["combo"]
    params = args["params_cls"](**dict(zip(keys, combo)))
    res = run_portfolio(
        args["datasets"],
        params,
        strategy_cls=args["strategy_cls"],
        group=args["group"],
        max_concurrent=args["max_concurrent"],
        initial_capital=args["initial_capital"],
        lo=args["lo"],
        hi=args["hi"],
    )
    score = args["score_fn"](compute_metrics(res))
    return float(score), combo


# ---------------------------------------------------------------------------
# Portfolio Walk-Forward Optimisation
# ---------------------------------------------------------------------------


@dataclass
class PortfolioWFOResult:
    windows: pd.DataFrame
    oos_equity: pd.Series
    oos_trades: pd.DataFrame
    best_params_per_fold: list = field(default_factory=list)


def _common_timeline(datasets: Dict[str, object]) -> pd.DatetimeIndex:
    idx = None
    for ds in datasets.values():
        di = ds.load().index
        ds.unload() if hasattr(ds, "unload") else None
        idx = di if idx is None else idx.union(di)
    return pd.DatetimeIndex(sorted(idx)) if idx is not None else pd.DatetimeIndex([])


def portfolio_walk_forward(
    datasets: Dict[str, object],
    param_grid: Dict[str, Iterable],
    *,
    strategy_cls,
    params_cls,
    group: str,
    n_splits: int = 5,
    oos_ratio: float = 0.25,
    score_fn: Callable[[Dict[str, float]], float] = consolidated_index,
    max_concurrent: int = 3,
    initial_capital: float = 100_000.0,
    n_jobs: int | str | None = "auto",
    progress: bool = True,
) -> PortfolioWFOResult:
    """Walk-forward where each IS grid-search and OOS test runs the *portfolio*.

    Folds are contiguous slices of the assets' common timeline.  For every
    fold the full ``param_grid`` is grid-searched on the IS slice (combos fan
    out across processes via :func:`source.parallel.parallel_map`), the best
    combo by ``score_fn`` is locked, then evaluated once on the OOS slice.
    OOS equity is stitched fold-to-fold so ``plot_wfo_dashboard`` works.
    """
    timeline = _common_timeline(datasets)
    if len(timeline) == 0:
        return PortfolioWFOResult(pd.DataFrame(), pd.Series(dtype=float),
                                  pd.DataFrame())
    keys = list(param_grid.keys())
    combos = list(product(*[list(param_grid[k]) for k in keys]))
    fold_size = len(timeline) // n_splits

    rows: List[dict] = []
    oos_curves: List[pd.Series] = []
    oos_trades_all: List[pd.DataFrame] = []
    best_per_fold: list = []
    offset = 0.0

    for f in range(n_splits):
        start = f * fold_size
        end = start + fold_size if f < n_splits - 1 else len(timeline)
        fold_idx = timeline[start:end]
        if len(fold_idx) < 50:
            continue
        split = max(1, int(len(fold_idx) * (1 - oos_ratio)))
        is_lo, is_hi = fold_idx[0], fold_idx[split - 1]
        oos_lo, oos_hi = fold_idx[split], fold_idx[-1]

        items = [
            {
                "keys": keys, "combo": combo, "params_cls": params_cls,
                "strategy_cls": strategy_cls, "datasets": datasets,
                "group": group, "max_concurrent": max_concurrent,
                "initial_capital": initial_capital, "score_fn": score_fn,
                "lo": is_lo, "hi": is_hi,
            }
            for combo in combos
        ]
        scored = parallel_map(
            _pf_eval_combo, items, n_jobs=n_jobs, progress=progress,
            desc=f"WFO fold {f} IS",
        )
        best_score, best_combo = max(scored, key=lambda sc: sc[0])
        best_params = params_cls(**dict(zip(keys, best_combo)))
        best_per_fold.append(best_params)

        oos_res = run_portfolio(
            datasets, best_params, strategy_cls=strategy_cls, group=group,
            max_concurrent=max_concurrent, initial_capital=initial_capital,
            lo=oos_lo, hi=oos_hi,
        )
        m_oos = compute_metrics(oos_res)
        oos_s = m_oos.get("sharpe_daily", np.nan)
        deg = (
            float(oos_s / best_score)
            if (np.isfinite(best_score) and best_score > 0) else np.nan
        )
        row = {
            "fold": f,
            "is_start": is_lo, "is_end": is_hi,
            "oos_start": oos_lo, "oos_end": oos_hi,
            "is_score": best_score,
            "oos_pnl": m_oos.get("total_pnl", 0.0),
            "oos_sharpe": oos_s,
            "oos_profit_factor": m_oos.get("profit_factor", np.nan),
            "oos_win_rate": m_oos.get("win_rate", np.nan),
            "oos_trades": m_oos.get("num_trades", 0),
            "oos_consolidated": consolidated_index(m_oos),
            "degradation_ratio": deg,
        }
        row.update({f"param_{k}": getattr(best_params, k) for k in keys})
        rows.append(row)

        if not oos_res.equity.empty:
            adj = oos_res.equity + offset
            oos_curves.append(adj)
            offset = float(adj.iloc[-1])
        if not oos_res.trades.empty:
            oos_trades_all.append(oos_res.trades.assign(fold=f))

    oos_eq = pd.concat(oos_curves) if oos_curves else pd.Series(dtype=float)
    oos_tr = (
        pd.concat(oos_trades_all, ignore_index=True)
        if oos_trades_all else pd.DataFrame()
    )
    return PortfolioWFOResult(
        windows=pd.DataFrame(rows),
        oos_equity=oos_eq,
        oos_trades=oos_tr,
        best_params_per_fold=best_per_fold,
    )


# ---------------------------------------------------------------------------
# Parameter sensitivity + overfitting report (§7)
# ---------------------------------------------------------------------------


def _pf_sensitivity_point(args: dict) -> dict:
    base = args["base_params"]
    pname, value = args["pname"], args["value"]
    params = base.__class__(**{**base.as_dict(), pname: value})
    res = run_portfolio(
        args["datasets"], params, strategy_cls=args["strategy_cls"],
        group=args["group"], max_concurrent=args["max_concurrent"],
        initial_capital=args["initial_capital"],
    )
    m = compute_metrics(res)
    m["consolidated_index"] = consolidated_index(m)
    return {"param": pname, "value": value,
            **{k: m[k] for k in (
                "num_trades", "total_pnl", "win_rate", "profit_factor",
                "max_drawdown", "sharpe_daily", "consolidated_index")}}


def portfolio_parameter_sensitivity(
    datasets: Dict[str, object],
    base_params,
    variations: Dict[str, Iterable],
    *,
    strategy_cls,
    group: str,
    max_concurrent: int = 3,
    initial_capital: float = 100_000.0,
    n_jobs: int | str | None = "auto",
) -> pd.DataFrame:
    """One-at-a-time parameter sweep, scored with the *portfolio* engine.

    Each (param, value) point is an independent portfolio backtest and fans
    out across processes.
    """
    points = [
        {
            "pname": pname, "value": value, "base_params": base_params,
            "datasets": datasets, "strategy_cls": strategy_cls, "group": group,
            "max_concurrent": max_concurrent, "initial_capital": initial_capital,
        }
        for pname, values in variations.items()
        for value in values
    ]
    rows = parallel_map(_pf_sensitivity_point, points, n_jobs=n_jobs,
                        progress=True, desc="sensitivity")
    return pd.DataFrame(rows)


def overfitting_report(
    sensitivity_df: pd.DataFrame,
    wfo_windows: pd.DataFrame | None = None,
    *,
    metric: str = "consolidated_index",
    plateau_tol: float = 0.20,
) -> pd.DataFrame:
    """Flag overfitting from sensitivity curvature + WFO IS->OOS degradation.

    Per swept parameter:

    * **cv** — coefficient of variation of ``metric`` across its values
      (how violently the result moves when the knob is nudged);
    * **best_vs_neighbour** — relative drop from the best value to its better
      adjacent value (a tall isolated spike => fragile / overfit);
    * **plateau_frac** — fraction of swept values within ``plateau_tol`` of the
      best (a wide plateau => robust);
    * **verdict** — ``ROBUST`` / ``SENSITIVE`` / ``FRAGILE``.

    When ``wfo_windows`` is given, the mean OOS/IS ``degradation_ratio`` is
    attached and downgrades the overall verdict if OOS collapses.
    """
    if sensitivity_df is None or sensitivity_df.empty:
        return pd.DataFrame()
    out = []
    for pname, g in sensitivity_df.groupby("param"):
        g = g.sort_values("value")
        vals = g[metric].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        cv = std / abs(mean) if mean != 0 else np.inf
        best = float(np.max(vals))
        order = np.argsort(g[metric].to_numpy(float))
        bi = int(order[-1])
        seq = g[metric].to_numpy(float)
        neigh = [seq[j] for j in (bi - 1, bi + 1) if 0 <= j < len(seq)]
        best_nb = max(neigh) if neigh else best
        drop = (best - best_nb) / abs(best) if best != 0 else 0.0
        rng = (np.nanmax(vals) - np.nanmin(vals)) or 1.0
        plateau_frac = float(np.mean(vals >= best - plateau_tol * rng))
        if cv < 0.25 and plateau_frac >= 0.5:
            verdict = "ROBUST"
        elif drop > 0.5 or cv > 0.75:
            verdict = "FRAGILE"
        else:
            verdict = "SENSITIVE"
        out.append({
            "param": pname, "cv": round(cv, 3),
            "best_vs_neighbour": round(float(drop), 3),
            "plateau_frac": round(plateau_frac, 3),
            "verdict": verdict,
        })
    rep = pd.DataFrame(out).set_index("param") if out else pd.DataFrame()
    if wfo_windows is not None and not wfo_windows.empty \
            and "degradation_ratio" in wfo_windows.columns:
        deg = pd.to_numeric(wfo_windows["degradation_ratio"], errors="coerce")
        mean_deg = float(np.nanmean(deg.to_numpy())) if len(deg) else np.nan
        rep.attrs["mean_oos_is_degradation"] = mean_deg
        rep.attrs["wfo_overfit_flag"] = bool(
            np.isfinite(mean_deg) and mean_deg < 0.5
        )
    return rep
