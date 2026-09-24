"""Position sizing and mark-to-market equity curves (DESIGN §5, §10 Phase 0).

``apply_sizing`` turns the point/price-level trades from
``quantlab.engine.run_backtest`` into currency P&L at a given lot size,
respecting MT5's lot-step rounding (DESIGN §5's "risk-realisation error" for
type-A systems). ``daily_equity`` builds the mark-to-market equity curve that
Sharpe/DD/CAGR are computed from (DESIGN §5: daily returns, sampled at the
last bar of each server day).
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Union

import numpy as np
import polars as pl

from .costs import InstrumentSpec
from .engine import BacktestResult

__all__ = ["apply_sizing", "daily_equity", "floor_to_step"]

_TRADE_REQUIRED_COLUMNS = (
    "entry_ts", "exit_ts", "entry_idx", "exit_idx", "direction", "entry_price", "exit_price",
    "stop_price", "target_price", "pnl_points", "swap_points", "swap_money_per_lot", "commission_per_lot",
)

RateFn = Callable[[str, str, pl.Series], Union[np.ndarray, pl.Series, list]]


def floor_to_step(raw: float, volume_min: float, volume_step: float, volume_max: float) -> float:
    """Round a raw lot size DOWN to the broker's lot grid (never up: risk must not exceed target)."""
    if raw < volume_min - 1e-9:
        return 0.0
    steps = math.floor((raw - volume_min) / volume_step + 1e-9)
    lots = volume_min + steps * volume_step
    lots = min(lots, volume_max)
    return round(lots, 8)


def apply_sizing(
    trades: pl.DataFrame,
    spec: InstrumentSpec,
    *,
    mode: str,
    equity0: float = 100_000.0,
    risk_fraction: Optional[float] = None,
    lots: Optional[float] = None,
    rate_fn: Optional[RateFn] = None,
    account_ccy: str = "USD",
    below_min: str = "skip",
    bars: Optional[pl.DataFrame] = None,
) -> pl.DataFrame:
    """Size each trade and compound equity sequentially (DESIGN §5).

    ``mode="fixed_fraction"`` (type A): ``lots = floor_to_step(equity_before *
    risk_fraction / (stop_distance_points * value_per_point_per_lot_acct))``.
    Every trade must carry a ``stop_price`` in this mode.

    ``mode="fixed_lots"`` (type B): every trade uses ``lots`` (still rounded
    to the symbol's volume grid); no risk target is defined, so
    ``risk_target_ccy``/``risk_realisation_error`` are null.

    ``rate_fn(quote_ccy, account_ccy, ts_utc_series) -> rates`` converts the
    instrument's quote-currency P&L to ``account_ccy``; it is only invoked
    (and only required) when ``spec.quote_ccy != account_ccy`` — the identity
    rate is used otherwise, matching DESIGN §4.3's "convert P&L ... using our
    own cross rates." **It is called with tz-aware UTC timestamps, using the
    time each value is actually known** (red-team M3): once with the *entry
    fill* time (for sizing -- how many lots to buy, decided at entry, so it
    must use the rate known *then*) and once with the *exit* time (for
    converting the realised P&L, booked when the trade actually closes).
    These come from ``bars['ts_utc']`` gathered at ``entry_idx``/``exit_idx``
    (bars carry ``ts_utc`` already tz-aware per the ``contracts`` schema), so
    ``bars`` (the same frame passed to ``run_backtest``) must be supplied
    whenever currency conversion is needed. This matches
    ``quantlab.data.conversion_rate``'s contract: it raises on naive
    timestamps and returns the close of the last M1 bar whose *close* time is
    at or before ``t`` (never a bar that merely opened at or before ``t``).

    ``account_ccy`` is not in the DESIGN task signature verbatim; it was added
    (optional, defaults to "USD") because the account-currency conversion the
    spec calls for has to compare against *something* — see the final report.

    Adds columns: ``lots``, ``pnl_ccy`` (incl. swap -- both points- and
    money-mode, see below -- and commission), ``equity_before``,
    ``equity_after``, ``risk_target_ccy``, ``risk_realised_ccy``,
    ``risk_realisation_error``, ``skipped``, ``value_per_point_acct`` (the
    entry-time quote->account conversion, used for sizing/marking; kept so
    ``daily_equity`` doesn't need ``rate_fn`` again) and
    ``value_per_point_acct_exit`` (the exit-time conversion actually used for
    the realised price P&L, for transparency/debugging).

    **Swap (red-team M4)**: ``swap_points`` (price-unit swap, any symbol whose
    ``swap_mode`` is ``"points"``) is converted like the trade's own P&L,
    at the exit-time rate. ``swap_money_per_lot`` (``swap_mode="money"``) is
    already a money amount per lot -- assumed to be in the instrument's quote
    currency (the MT5 export's "money" modes collapse 3 different currency
    choices into one, see ``costs.normalize_swap_mode``; this is documented,
    not silently guessed) -- so it needs the plain FX rate, not the
    price-value conversion, again at the exit-time rate.
    """
    if mode not in ("fixed_fraction", "fixed_lots"):
        raise ValueError(f"mode must be 'fixed_fraction' or 'fixed_lots', got {mode!r}")
    if mode == "fixed_fraction" and not risk_fraction:
        raise ValueError("mode='fixed_fraction' requires risk_fraction > 0")
    if mode == "fixed_lots" and not lots:
        raise ValueError("mode='fixed_lots' requires lots > 0")
    if below_min not in ("skip", "min"):
        raise ValueError(f"below_min must be 'skip' or 'min', got {below_min!r}")
    missing = [c for c in _TRADE_REQUIRED_COLUMNS if c not in trades.columns]
    if missing:
        raise ValueError(f"trades frame is missing columns: {missing}")

    trades = trades.sort("entry_idx")
    n = trades.height

    if spec.quote_ccy == account_ccy:
        rate_entry = np.ones(n, dtype=np.float64)
        rate_exit = np.ones(n, dtype=np.float64)
    elif rate_fn is not None:
        if bars is None or "ts_utc" not in bars.columns:
            raise ValueError(
                "apply_sizing: rate_fn requires bars=<the same bar frame run_backtest used>, so the "
                "entry/exit fill times can be looked up as tz-aware UTC via bars['ts_utc'] "
                "(entry_idx/exit_idx index into it) -- DESIGN §4.3 / red-team M3."
            )
        entry_ts_utc = bars["ts_utc"].gather(trades["entry_idx"])
        exit_ts_utc = bars["ts_utc"].gather(trades["exit_idx"])
        rate_entry = np.asarray(rate_fn(spec.quote_ccy, account_ccy, entry_ts_utc), dtype=np.float64)
        rate_exit = np.asarray(rate_fn(spec.quote_ccy, account_ccy, exit_ts_utc), dtype=np.float64)
        if rate_entry.shape[0] != n:
            raise ValueError(f"rate_fn returned {rate_entry.shape[0]} entry rates for {n} trades")
        if rate_exit.shape[0] != n:
            raise ValueError(f"rate_fn returned {rate_exit.shape[0]} exit rates for {n} trades")
    else:
        raise ValueError(
            f"apply_sizing: instrument quote currency {spec.quote_ccy!r} != account_ccy "
            f"{account_ccy!r}; pass rate_fn=<callable(quote_ccy, account_ccy, ts_utc_series) -> rates> "
            "and bars=<the bar frame run_backtest used>."
        )
    value_per_point_entry = spec.value_per_point_per_lot * rate_entry
    value_per_point_exit = spec.value_per_point_per_lot * rate_exit

    entry_price = trades["entry_price"].to_numpy()
    stop_price = trades["stop_price"].fill_null(float("nan")).to_numpy()
    pnl_points = trades["pnl_points"].to_numpy()
    swap_points = trades["swap_points"].fill_null(0.0).to_numpy()
    swap_money_per_lot = trades["swap_money_per_lot"].fill_null(0.0).to_numpy()
    commission_per_lot = trades["commission_per_lot"].fill_null(0.0).to_numpy()
    entry_idx_col = trades["entry_idx"].to_numpy()

    with np.errstate(invalid="ignore"):
        stop_dist_points = np.abs(entry_price - stop_price) / spec.point

    lots_arr = np.empty(n, dtype=np.float64)
    pnl_ccy_arr = np.empty(n, dtype=np.float64)
    eq_before_arr = np.empty(n, dtype=np.float64)
    eq_after_arr = np.empty(n, dtype=np.float64)
    risk_target_arr = np.full(n, np.nan, dtype=np.float64)
    risk_real_arr = np.full(n, np.nan, dtype=np.float64)
    risk_err_arr = np.full(n, np.nan, dtype=np.float64)
    skipped_arr = np.zeros(n, dtype=np.bool_)

    equity = equity0
    for i in range(n):
        eq_before = equity
        vpp_e = value_per_point_entry[i]
        vpp_x = value_per_point_exit[i]
        sd = stop_dist_points[i]

        if mode == "fixed_fraction":
            if not np.isfinite(sd) or sd <= 0:
                raise ValueError(
                    "fixed_fraction sizing requires a stop_price on every trade; "
                    f"trade at entry_idx={int(entry_idx_col[i])} has none."
                )
            risk_target_i = eq_before * risk_fraction
            risk_per_lot = sd * vpp_e
            raw_lots = risk_target_i / risk_per_lot if risk_per_lot > 0 else 0.0
            lots_i = floor_to_step(raw_lots, spec.volume_min, spec.volume_step, spec.volume_max)
        else:  # fixed_lots
            risk_target_i = np.nan
            lots_i = floor_to_step(lots, spec.volume_min, spec.volume_step, spec.volume_max)

        below = lots_i < spec.volume_min - 1e-9
        skipped_i = False
        if below:
            if below_min == "skip":
                lots_final = 0.0
                skipped_i = True
            else:
                lots_final = spec.volume_min
        else:
            lots_final = lots_i

        swap_points_ccy = swap_points[i] * vpp_x * lots_final
        swap_money_ccy = swap_money_per_lot[i] * rate_exit[i] * lots_final
        swap_ccy = swap_points_ccy + swap_money_ccy
        commission_ccy = commission_per_lot[i] * lots_final
        pnl_ccy = pnl_points[i] * vpp_x * lots_final + swap_ccy - commission_ccy
        eq_after = eq_before + pnl_ccy
        equity = eq_after

        if mode == "fixed_fraction":
            risk_target_arr[i] = risk_target_i
            if skipped_i:
                risk_real_arr[i] = 0.0
                risk_err_arr[i] = -1.0
            else:
                risk_real_arr[i] = sd * vpp_e * lots_final
                risk_err_arr[i] = (risk_real_arr[i] / risk_target_i - 1.0) if risk_target_i > 0 else np.nan
        elif np.isfinite(sd):
            risk_real_arr[i] = sd * vpp_e * lots_final

        lots_arr[i] = lots_final
        pnl_ccy_arr[i] = pnl_ccy
        eq_before_arr[i] = eq_before
        eq_after_arr[i] = eq_after
        skipped_arr[i] = skipped_i

    return trades.with_columns(
        pl.Series("lots", lots_arr),
        pl.Series("pnl_ccy", pnl_ccy_arr),
        pl.Series("equity_before", eq_before_arr),
        pl.Series("equity_after", eq_after_arr),
        pl.Series("risk_target_ccy", risk_target_arr),
        pl.Series("risk_realised_ccy", risk_real_arr),
        pl.Series("risk_realisation_error", risk_err_arr),
        pl.Series("skipped", skipped_arr),
        pl.Series("value_per_point_acct", value_per_point_entry),
        pl.Series("value_per_point_acct_exit", value_per_point_exit),
    )


def daily_equity(
    bars: pl.DataFrame,
    result: BacktestResult,
    sized_trades: pl.DataFrame,
    *,
    equity0: float = 100_000.0,
) -> pl.DataFrame:
    """Mark-to-market equity, sampled at the last bar of each server day (DESIGN §5).

    Between trades, equity is the running ``equity_after`` of the last closed
    trade. While a position is open, it is marked at that bar's own close:
    Bid for longs, Ask (``close + spread*point``, using the bar's *raw*
    spread column — this is a mark, not a fill, so no cost-model multiplier
    is applied) for shorts. Swap/commission for the currently open trade are
    only booked at its close (already reflected in ``sized_trades``), so the
    intraday MTM path is price-only — a documented simplification.
    """
    spec = result.meta["spec"]
    point = spec.point
    n = bars.height
    close = bars["close"].cast(pl.Float64).to_numpy()
    spread = bars["spread"].cast(pl.Float64).to_numpy()

    exit_idx = sized_trades["exit_idx"].to_numpy()
    eq_after = sized_trades["equity_after"].to_numpy()
    order = np.argsort(exit_idx)
    exit_idx_sorted = exit_idx[order]
    eq_after_sorted = eq_after[order]
    step_pos = np.searchsorted(exit_idx_sorted, np.arange(n), side="right") - 1
    closed_eq = np.where(step_pos >= 0, eq_after_sorted[np.clip(step_pos, 0, None)], equity0)

    entry_idx = sized_trades["entry_idx"].to_numpy()
    lots = sized_trades["lots"].to_numpy()
    direction = sized_trades["direction"].to_numpy()
    entry_price = sized_trades["entry_price"].to_numpy()
    vpp = sized_trades["value_per_point_acct"].to_numpy()

    active = np.full(n, -1, dtype=np.int64)
    for k in range(sized_trades.height):
        if exit_idx[k] > entry_idx[k]:
            active[entry_idx[k]:exit_idx[k]] = k

    open_pnl = np.zeros(n, dtype=np.float64)
    mask = active >= 0
    if mask.any():
        tk = active[mask]
        is_long = direction[tk] == 1
        mark = np.where(is_long, close[mask], close[mask] + spread[mask] * point)
        pnl_pts = np.where(is_long, (mark - entry_price[tk]) / point, (entry_price[tk] - mark) / point)
        open_pnl[mask] = pnl_pts * vpp[tk] * lots[tk]

    equity_curve = closed_eq + open_pnl

    df = bars.select(pl.col("ts")).with_columns(
        pl.Series("equity", equity_curve),
        pl.col("ts").dt.date().alias("date"),
    )
    daily = (
        df.group_by("date", maintain_order=True)
        .agg(pl.col("equity").last())
        .sort("date")
        .with_columns(pl.col("equity").shift(1).alias("_prev_equity"))
    )
    daily = daily.with_columns(
        pl.when(pl.col("_prev_equity").is_null())
        .then(pl.col("equity") / equity0 - 1.0)
        .otherwise(pl.col("equity") / pl.col("_prev_equity") - 1.0)
        .alias("ret")
    ).drop("_prev_equity")
    return daily
