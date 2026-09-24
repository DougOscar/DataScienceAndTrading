"""quantlab.sizing: lot rounding, risk-realisation error, MTM daily equity.

Every hand-computed test lays out the arithmetic in comments.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from quantlab.costs import CostModel, InstrumentSpec
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing, daily_equity, floor_to_step

POINT = 0.0001
T0 = datetime(2024, 1, 2)  # a Tuesday


def _spec(**over):
    base = dict(
        symbol="EURUSD", digits=4, point=POINT, contract_size=100_000.0, tick_size=POINT,
        volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
        swap_mode="points", swap_long=-3.0, swap_short=0.5, swap_3day=2,
        commission_per_lot_rt=7.0, stops_level=0.0, calibrated=True,
    )
    base.update(over)
    return InstrumentSpec(**base)


_TRADE_SCHEMA = {
    "entry_ts": pl.Datetime("ms"), "exit_ts": pl.Datetime("ms"),
    "entry_idx": pl.Int64, "exit_idx": pl.Int64, "direction": pl.Int8,
    "entry_price": pl.Float64, "exit_price": pl.Float64,
    "stop_price": pl.Float64, "target_price": pl.Float64,
    "exit_reason": pl.Utf8, "bars_held": pl.Int64, "pnl_points": pl.Float64,
    "spread_cost_points": pl.Float64, "mae_points": pl.Float64, "mfe_points": pl.Float64,
    "nights": pl.Int64, "swap_points": pl.Float64, "swap_money_per_lot": pl.Float64,
    "commission_per_lot": pl.Float64,
}


def _trade(**over):
    """One synthetic engine-shaped trade row; only the fields sizing.py reads matter."""
    base = dict(
        entry_ts=T0, exit_ts=T0 + timedelta(hours=1), entry_idx=0, exit_idx=1,
        direction=1, entry_price=1.1007, exit_price=1.1037, stop_price=1.0984,
        target_price=None, exit_reason="eod", bars_held=1, pnl_points=30.0,
        spread_cost_points=2.0, mae_points=0.0, mfe_points=5.0, nights=0,
        swap_points=0.0, swap_money_per_lot=0.0, commission_per_lot=7.0,
    )
    base.update(over)
    return pl.DataFrame([base], schema=_TRADE_SCHEMA)


def _ts_utc_bars(timestamps: list[datetime]) -> pl.DataFrame:
    """Minimal bars-shaped frame carrying only ``ts_utc`` -- all apply_sizing
    reads from ``bars`` when converting currency (DESIGN §4.3 / red-team M3)."""
    return pl.DataFrame({"ts_utc": pl.Series(timestamps).cast(pl.Datetime("ms", time_zone="UTC"))})


# --------------------------------------------------------------------------- floor_to_step

def test_floor_to_step_hand_computed():
    # (8.695652... - 0.01) / 0.01 = 868.565... -> floor 868 -> 0.01 + 8.68 = 8.69
    assert floor_to_step(200.0 / 23.0, 0.01, 0.01, 100.0) == pytest.approx(8.69)
    assert floor_to_step(0.0, 0.01, 0.01, 100.0) == 0.0
    assert floor_to_step(0.005, 0.01, 0.01, 100.0) == 0.0       # below volume_min
    assert floor_to_step(1000.0, 0.01, 0.01, 5.0) == 5.0        # clipped to volume_max


# --------------------------------------------------------------------------- fixed_fraction

def test_fixed_fraction_lot_rounding_and_risk_realisation_error():
    spec = _spec()  # value_per_point_per_lot = 100_000 * 0.0001 = 10.0 (USD, == account ccy)
    # stop_dist_points = |1.1007-1.0984|/0.0001 = 23.0
    trades = _trade(entry_price=1.1007, stop_price=1.0984, pnl_points=30.0, commission_per_lot=7.0)

    out = apply_sizing(trades, spec, mode="fixed_fraction", equity0=100_000.0, risk_fraction=0.02)
    r = out.row(0, named=True)

    # risk_target = 100_000*0.02 = 2000 ; risk_per_lot = 23*10 = 230 ; raw_lots = 2000/230 = 8.695652...
    # floor_to_step -> 8.69
    assert r["lots"] == pytest.approx(8.69)
    assert r["risk_target_ccy"] == pytest.approx(2000.0)
    # risk_realised = 23*10*8.69 = 1998.7
    assert r["risk_realised_ccy"] == pytest.approx(1998.7)
    # error = 1998.7/2000 - 1 = -0.00065
    assert r["risk_realisation_error"] == pytest.approx(-0.00065, abs=1e-9)
    # pnl_ccy = 30*10*8.69 + 0(swap) - 7*8.69 = 2607.0 - 60.83 = 2546.17
    assert r["pnl_ccy"] == pytest.approx(2546.17)
    assert r["equity_before"] == pytest.approx(100_000.0)
    assert r["equity_after"] == pytest.approx(102_546.17)
    assert r["skipped"] is False
    assert r["value_per_point_acct"] == pytest.approx(10.0)


def test_fixed_fraction_requires_stop_price():
    spec = _spec()
    trades = _trade(stop_price=None)
    with pytest.raises(ValueError, match="requires a stop_price"):
        apply_sizing(trades, spec, mode="fixed_fraction", risk_fraction=0.02)


def test_fixed_fraction_below_min_skip_vs_min():
    spec = _spec()
    # risk_target = 100_000*0.000005 = 0.5 ; risk_per_lot = 20*10 = 200 ; raw_lots = 0.5/200 = 0.0025
    # 0.0025 < volume_min(0.01) -> below-min
    trades = _trade(entry_price=1.1000, stop_price=1.0980, pnl_points=30.0, commission_per_lot=7.0)

    skipped = apply_sizing(trades, spec, mode="fixed_fraction", risk_fraction=0.000005,
                            below_min="skip").row(0, named=True)
    assert skipped["lots"] == 0.0
    assert skipped["skipped"] is True
    assert skipped["pnl_ccy"] == pytest.approx(0.0)
    assert skipped["equity_after"] == pytest.approx(100_000.0)
    assert skipped["risk_realised_ccy"] == pytest.approx(0.0)
    assert skipped["risk_realisation_error"] == pytest.approx(-1.0)

    floored_up = apply_sizing(trades, spec, mode="fixed_fraction", risk_fraction=0.000005,
                               below_min="min").row(0, named=True)
    assert floored_up["lots"] == pytest.approx(0.01)
    assert floored_up["skipped"] is False
    # risk_realised = 20*10*0.01 = 2.0 ; target was 0.5 -> error = 2.0/0.5 - 1 = 3.0 (300% over target)
    assert floored_up["risk_realised_ccy"] == pytest.approx(2.0)
    assert floored_up["risk_realisation_error"] == pytest.approx(3.0)
    # pnl_ccy = 30*10*0.01 - 7*0.01 = 3.0 - 0.07 = 2.93
    assert floored_up["pnl_ccy"] == pytest.approx(2.93)


# --------------------------------------------------------------------------- fixed_lots

def test_fixed_lots_mode_has_no_risk_target():
    spec = _spec()
    trades = _trade(entry_price=1.1000, stop_price=1.0950, pnl_points=20.0,
                     swap_points=-1.0, commission_per_lot=5.0)

    out = apply_sizing(trades, spec, mode="fixed_lots", lots=0.5).row(0, named=True)
    assert out["lots"] == pytest.approx(0.5)
    # swap_ccy = -1*10*0.5 = -5.0 ; commission_ccy = 5*0.5 = 2.5
    # pnl_ccy = 20*10*0.5 - 5.0 - 2.5 = 100 - 5 - 2.5 = 92.5
    assert out["pnl_ccy"] == pytest.approx(92.5)
    assert out["risk_target_ccy"] is None or np.isnan(out["risk_target_ccy"])
    assert out["risk_realisation_error"] is None or np.isnan(out["risk_realisation_error"])
    # risk_realised is still reported (informational) since a stop exists: 50*10*0.5=250
    assert out["risk_realised_ccy"] == pytest.approx(250.0)


def test_money_mode_swap_is_included_in_pnl_ccy():
    """Regression (red-team M4): apply_sizing must add swap_money_per_lot * lots
    (converted to account ccy) to pnl_ccy -- the pre-fix code only ever read
    swap_points, so money-mode swap (swap_mode="money") was silently dropped."""
    spec = _spec()  # quote_ccy == account_ccy == USD -> identity conversion
    trades = _trade(entry_price=1.1000, stop_price=1.0950, pnl_points=30.0,
                     swap_points=0.0, swap_money_per_lot=-15.0, commission_per_lot=0.0)

    out = apply_sizing(trades, spec, mode="fixed_lots", lots=2.0).row(0, named=True)

    # price pnl = 30*10*2 = 600 ; swap_ccy = -15 * rate(1.0) * 2 lots = -30 ; commission = 0
    assert out["pnl_ccy"] == pytest.approx(570.0)


def test_points_and_money_mode_swap_both_contribute():
    """Both swap representations can be non-zero at once and simply add."""
    spec = _spec()
    trades = _trade(entry_price=1.1000, stop_price=1.0950, pnl_points=0.0,
                     swap_points=-4.0, swap_money_per_lot=-2.5, commission_per_lot=0.0)

    out = apply_sizing(trades, spec, mode="fixed_lots", lots=1.0).row(0, named=True)
    # swap_ccy = (-4*10*1) + (-2.5*1.0*1) = -40 - 2.5 = -42.5
    assert out["pnl_ccy"] == pytest.approx(-42.5)


# --------------------------------------------------------------------------- currency conversion

def test_currency_mismatch_without_rate_fn_raises():
    spec = _spec(quote_ccy="EUR")
    trades = _trade()
    with pytest.raises(ValueError, match="rate_fn"):
        apply_sizing(trades, spec, mode="fixed_lots", lots=1.0, account_ccy="USD")


def test_currency_mismatch_with_rate_fn_converts():
    spec = _spec(quote_ccy="EUR")
    trades = _trade(entry_price=1.1000, stop_price=1.0950, pnl_points=10.0,
                     swap_points=0.0, commission_per_lot=0.0)
    bars = _ts_utc_bars([datetime(2024, 1, 2, tzinfo=timezone.utc), datetime(2024, 1, 2, 1, tzinfo=timezone.utc)])

    def rate_fn(quote_ccy, account_ccy, ts_series):
        assert quote_ccy == "EUR" and account_ccy == "USD"
        assert ts_series.dtype.time_zone is not None, "rate_fn must receive tz-aware timestamps"
        return np.full(ts_series.len(), 1.10)

    out = apply_sizing(trades, spec, mode="fixed_lots", lots=1.0, account_ccy="USD",
                        rate_fn=rate_fn, bars=bars).row(0, named=True)
    # value_per_point_acct = 10.0(EUR/point/lot) * 1.10 = 11.0
    assert out["value_per_point_acct"] == pytest.approx(11.0)
    # pnl_ccy = 10*11.0*1.0 = 110.0
    assert out["pnl_ccy"] == pytest.approx(110.0)


def test_currency_mismatch_rate_fn_without_bars_raises():
    """Regression (red-team M3): apply_sizing must not silently fall back to
    trades['entry_ts'] (naive server time) when rate_fn needs real timestamps
    -- it must demand bars=<the run_backtest frame> so it can look up
    bars['ts_utc'] (tz-aware) at entry_idx/exit_idx instead."""
    spec = _spec(quote_ccy="EUR")
    trades = _trade()

    def rate_fn(quote_ccy, account_ccy, ts_series):
        return np.ones(ts_series.len())

    with pytest.raises(ValueError, match="bars"):
        apply_sizing(trades, spec, mode="fixed_lots", lots=1.0, account_ccy="USD", rate_fn=rate_fn)


def test_apply_sizing_uses_entry_time_rate_for_sizing_and_exit_time_rate_for_pnl():
    """Regression (red-team M3): risk/lot sizing at entry must use the rate as of
    the entry FILL time; P&L conversion at exit must use the rate as of the exit
    time. The pre-fix code called rate_fn once, with trades['entry_ts'] (naive
    server time), and reused that single rate for both -- so a rate_fn whose
    value depends on the (tz-aware) hour of day looks identical at entry and
    exit under the old code, but must differ here."""
    spec = _spec(quote_ccy="EUR")
    # 2 bars: entry bar at 06:00 UTC (rate regime "morning"), exit bar at
    # 18:00 UTC (rate regime "evening") -- far enough apart that mixing them up
    # is unmistakable.
    bars = _ts_utc_bars([datetime(2024, 1, 2, 6, 0, tzinfo=timezone.utc),
                         datetime(2024, 1, 2, 18, 0, tzinfo=timezone.utc)])
    trades = _trade(entry_idx=0, exit_idx=1, entry_price=1.1000, stop_price=1.0950,
                     pnl_points=10.0, swap_points=0.0, commission_per_lot=0.0)

    seen_hours: list[list[int]] = []

    def rate_fn(quote_ccy, account_ccy, ts_series):
        assert ts_series.dtype.time_zone is not None
        hours = [t.hour for t in ts_series.to_list()]
        seen_hours.append(hours)
        return [1.0 if h < 12 else 2.0 for h in hours]

    out = apply_sizing(trades, spec, mode="fixed_lots", lots=1.0, account_ccy="USD",
                        rate_fn=rate_fn, bars=bars).row(0, named=True)

    assert seen_hours == [[6], [18]]  # one call for entry timestamps, one for exit timestamps
    # sizing/marking uses the ENTRY-time rate (06:00 -> 1.0): value_per_point_acct = 10*1.0 = 10
    assert out["value_per_point_acct"] == pytest.approx(10.0)
    assert out["value_per_point_acct_exit"] == pytest.approx(20.0)
    # realised P&L uses the EXIT-time rate (18:00 -> 2.0): pnl_ccy = 10*20.0*1 = 200 (not 100)
    assert out["pnl_ccy"] == pytest.approx(200.0)


# --------------------------------------------------------------------------- daily_equity

def test_daily_equity_mtm_on_open_multi_day_position():
    days = [T0, T0, T0, T0 + timedelta(days=1), T0 + timedelta(days=1), T0 + timedelta(days=1)]
    hours = [0, 1, 2, 0, 1, 2]
    ts = [d + timedelta(hours=h) for d, h in zip(days, hours)]
    rows = [
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # entry fill bar
        dict(open=1.1006, high=1.1010, low=1.1002, close=1.1008, spread=2),  # day1 last bar
        dict(open=1.1008, high=1.1012, low=1.1004, close=1.1010, spread=2),
        dict(open=1.1010, high=1.1015, low=1.1006, close=1.1012, spread=2),
        dict(open=1.1012, high=1.1016, low=1.1008, close=1.1014, spread=2),  # day2 last bar, eod
    ]
    bars = pl.DataFrame({
        "ts": ts, "open": [r["open"] for r in rows], "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows], "close": [r["close"] for r in rows],
        "spread": [float(r["spread"]) for r in rows],
    }, schema_overrides={"ts": pl.Datetime("ms")}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )
    signals = pl.DataFrame({
        "signal": pl.Series([1, None, None, None, None, None], dtype=pl.Int8),
        "stop_dist": pl.Series([0.0500, None, None, None, None, None], dtype=pl.Float64),
        "target_dist": pl.Series([0.0500, None, None, None, None, None], dtype=pl.Float64),
    })
    spec = _spec(swap_long=-3.0)
    result = run_backtest(bars, signals, spec, CostModel())
    sized = apply_sizing(result.trades, spec, mode="fixed_lots", lots=1.0, equity0=100_000.0)

    # sanity on the underlying trade: entry=1.1005+0.0002=1.1007, eod exit=close[5]=1.1014
    t = sized.row(0, named=True)
    assert t["entry_idx"] == 1 and t["exit_idx"] == 5
    # pnl_points = (1.1014-1.1007)/0.0001 = 7.0
    assert t["pnl_points"] == pytest.approx(7.0)
    # entry Tue Jan2 01:00 -> exit Wed Jan3 02:00 crosses exactly 1 midnight (Wed 00:00), Tue night -> weight 1
    assert t["nights"] == 1
    assert t["swap_points"] == pytest.approx(-3.0)
    # pnl_ccy = 7*10*1 + swap(-3*10*1=-30) - commission(7*1) = 70 - 30 - 7 = 33.0
    assert t["pnl_ccy"] == pytest.approx(33.0)
    equity_final = 100_000.0 + 33.0

    daily = daily_equity(bars, result, sized, equity0=100_000.0)
    assert daily.height == 2

    day1 = daily.row(0, named=True)
    day2 = daily.row(1, named=True)

    # Day1 last bar = idx2 (still open): mark at close[2]=1.1008 (Bid, long)
    # pnl_pts = (1.1008-1.1007)/0.0001 = 1.0 ; open_pnl = 1.0*10*1 = 10.0 ; closed_eq = equity0 (no exit yet)
    expected_day1_equity = 100_000.0 + 1.0 * 10.0 * 1.0
    assert day1["equity"] == pytest.approx(expected_day1_equity)
    assert day1["ret"] == pytest.approx(expected_day1_equity / 100_000.0 - 1.0)

    # Day2 last bar = idx5 = the exit bar itself: fully realised equity, no open_pnl left
    assert day2["equity"] == pytest.approx(equity_final)
    assert day2["ret"] == pytest.approx(equity_final / expected_day1_equity - 1.0)
