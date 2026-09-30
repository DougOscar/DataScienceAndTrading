"""quantlab.engine: fill/stop/target loop, hand-computed against DESIGN §4.3.

Every test lays out the exact arithmetic in comments. Prices use point=0.0001
throughout (a round, easy-to-check unit -- the real per-symbol point comes
from costs.load_instrument / data.infer_point, not from engine.py).
"""

from __future__ import annotations

import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab.costs import CostModel, InstrumentSpec
from quantlab.engine import _nights_weighted, run_backtest

POINT = 0.0001
T0 = datetime(2024, 1, 1, 0, 0)  # a Monday


def _spec(**over):
    base = dict(
        symbol="EURUSD", digits=4, point=POINT, contract_size=100_000.0, tick_size=POINT,
        volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
        swap_mode="points", swap_long=-2.0, swap_short=0.5, swap_3day=2,
        commission_per_lot_rt=7.0, stops_level=0.0, calibrated=True,
    )
    base.update(over)
    return InstrumentSpec(**base)


def _bars(rows: list[dict], start: datetime = T0, step: timedelta = timedelta(hours=1)) -> pl.DataFrame:
    """``spread_max`` mirrors ``spread`` unless a row overrides it -- used by the
    minor-10 tests below to give a bar a ``spread_max`` that differs from its
    own open ``spread`` (as real resampled bars do)."""
    ts = [start + i * step for i in range(len(rows))]
    df = pl.DataFrame({
        "ts": ts,
        "open": [float(r["open"]) for r in rows],
        "high": [float(r["high"]) for r in rows],
        "low": [float(r["low"]) for r in rows],
        "close": [float(r["close"]) for r in rows],
        "spread": [float(r["spread"]) for r in rows],
        "spread_max": [float(r.get("spread_max", r["spread"])) for r in rows],
    }, schema_overrides={"ts": pl.Datetime("ms")})
    return df.with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )


def _signals(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame({
        "signal": pl.Series([r.get("signal") for r in rows], dtype=pl.Int8),
        "stop_dist": pl.Series([r.get("stop_dist") for r in rows], dtype=pl.Float64),
        "target_dist": pl.Series([r.get("target_dist") for r in rows], dtype=pl.Float64),
    })


def _row(t):
    """Extract trade k=0 as a plain dict for easy assertions."""
    return t.row(0, named=True)


# --------------------------------------------------------------------------- basic fills / eod

def test_long_entry_at_next_open_ask_and_eod_exit():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1006, high=1.1012, low=1.1002, close=1.1006, spread=2),  # eod bar
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0050, target_dist=0.0100),
        dict(),
        dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)

    # entry = open[1] + spread(2)*point = 1.1005 + 0.0002 = 1.1007 (Ask)
    assert t["entry_price"] == pytest.approx(1.1007)
    assert t["entry_idx"] == 1 and t["exit_idx"] == 2 and t["bars_held"] == 1
    # neither SL (1.1007-0.005=1.0957) nor TP (1.1007+0.01=1.1107) is touched -> eod at close[2]
    assert t["exit_reason"] == "eod"
    assert t["exit_price"] == pytest.approx(1.1006)  # raw Bid, no spread on a long's exit
    # pnl_points = (1.1006-1.1007)/0.0001 = -1.0
    assert t["pnl_points"] == pytest.approx(-1.0)
    assert t["spread_cost_points"] == pytest.approx(2.0)  # fixed at entry
    # mae: max((entry-low)/point) over bars 1,2 = max((1.1007-1.1000)/1e-4, (1.1007-1.1002)/1e-4) = max(7,5) = 7
    assert t["mae_points"] == pytest.approx(7.0)
    # mfe: max((high-entry)/point) = max((1.1010-1.1007)/1e-4, (1.1012-1.1007)/1e-4) = max(3,5) = 5
    assert t["mfe_points"] == pytest.approx(5.0)
    assert result.trades.height == 1
    assert list(result.position) == [0, 1, 0]


def test_short_entry_at_bid_pays_spread_on_exit_not_entry():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1005, spread=2),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=3),  # eod bar, different spread
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0050, target_dist=0.0100),
        dict(),
        dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)

    assert t["entry_price"] == pytest.approx(1.1005)  # raw Bid, no spread on a short's entry
    assert t["exit_reason"] == "eod"
    # exit = close[2] + spread[2]*point = 1.1006 + 3*0.0001 = 1.1009 (Ask, uses the EXIT bar's spread)
    assert t["exit_price"] == pytest.approx(1.1009)
    # pnl_points = (entry-exit)/point = (1.1005-1.1009)/0.0001 = -4.0
    assert t["pnl_points"] == pytest.approx(-4.0)
    # spread cost is charged at exit for shorts, using bar 2's spread (3), not bar 1's (2)
    assert t["spread_cost_points"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- stop / target

def test_stop_loss_hit_no_gap():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.0980, close=1.1000, spread=2),  # SL touched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    cost = CostModel(slippage_points=1.0)
    result = run_backtest(bars, signals, _spec(), cost)
    t = _row(result.trades)

    # entry (red-team M2: entry now also slips 1pt adverse -- long buys higher) =
    #   1.1005 + 2*0.0001 + 1*0.0001 = 1.1008 ; SL = 1.1008-0.0020 = 1.0988
    # bar2 opens at 1.1005 (> SL, no gap), low 1.0980 <= SL -> touch, not gap
    # fill = SL - slippage_price = 1.0988 - 1*0.0001 = 1.0987
    assert t["entry_price"] == pytest.approx(1.1008)
    assert t["stop_price"] == pytest.approx(1.0988)
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(1.0987)
    # pnl_points = (1.0987-1.1008)/0.0001 = -21.0 -- unchanged from the pre-M2 value: the
    # stop is a fixed *distance* from entry, so entry's own +1pt slippage shifts both the
    # stop level and the fill by the same amount and cancels out of pnl (only the stop's
    # own, unrelated 1pt of slippage -- present before this fix too -- still shows up).
    assert t["pnl_points"] == pytest.approx(-21.0)


def test_target_hit_no_gap_no_slippage():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1060, low=1.1000, close=1.1050, spread=2),  # TP touched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0500, target_dist=0.0050),
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)

    # entry=1.1007 ; TP=1.1007+0.0050=1.1057 ; bar2 high 1.1060>=TP, open 1.1005<TP -> no gap
    # target fills exactly at TP, never gets slippage
    assert t["target_price"] == pytest.approx(1.1057)
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == pytest.approx(1.1057)
    # pnl_points = (1.1057-1.1007)/0.0001 = 50.0
    assert t["pnl_points"] == pytest.approx(50.0)


def test_gap_through_stop_fills_at_open_plus_slippage():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.0950, high=1.0960, low=1.0940, close=1.0955, spread=2),  # gap down through SL
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    cost = CostModel(slippage_points=1.0)
    result = run_backtest(bars, signals, _spec(), cost)
    t = _row(result.trades)

    # entry (red-team M2: +1pt adverse) = 1.1005+0.0002+0.0001 = 1.1008 ; SL = 1.0988
    # bar2 opens at 1.0950, already through SL -> gap
    # fill = open - slippage_price = 1.0950 - 0.0001 = 1.0949 (gap fill uses the actual gap
    # open, independent of SL/entry, so it is unaffected by entry's own slippage)
    assert t["entry_price"] == pytest.approx(1.1008)
    assert t["exit_reason"] == "gap_stop"
    assert t["exit_price"] == pytest.approx(1.0949)
    # pnl_points = (1.0949-1.1008)/0.0001 = -59.0 (one point worse than pre-M2's -58.0,
    # entirely from entry's own new +1pt of adverse slippage)
    assert t["pnl_points"] == pytest.approx(-59.0)


# --------------------------------------------------------------------------- market-fill slippage scope
# (red-team M2): entries, signal exits, session-flatten exits and eod exits all pay adverse
# slippage now, not just stops. Every case below uses a no-stop/no-target (risk-type-C-style)
# signal so the *only* source of any price shift is slippage_points itself.

def test_long_entry_and_signal_exit_both_pay_adverse_slippage():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=3),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=3),  # entry bar
        dict(open=1.1010, high=1.1015, low=1.1005, close=1.1012, spread=3),  # signal-exit bar
        dict(open=1.1012, high=1.1020, low=1.1008, close=1.1015, spread=3),  # untouched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=None, target_dist=None),
        dict(signal=0, stop_dist=None, target_dist=None),
        dict(), dict(),
    ])
    cost = CostModel(slippage_points=2.0)  # slip_price = 2*0.0001 = 0.0002
    result = run_backtest(bars, signals, _spec(), cost)
    t = _row(result.trades)

    # entry (long, BUYS -> higher): 1.1005 + 3*0.0001 + 0.0002 = 1.1010
    assert t["entry_price"] == pytest.approx(1.1010)
    # signal exit (long, SELLS -> lower, no spread on a long's exit): 1.1010 - 0.0002 = 1.1008
    assert t["exit_reason"] == "signal"
    assert t["exit_price"] == pytest.approx(1.1008)
    # pnl_points = (1.1008-1.1010)/0.0001 = -2.0 (pure round-trip slippage, 1pt*2 legs*2pips)
    assert t["pnl_points"] == pytest.approx(-2.0)


def test_short_entry_and_signal_exit_both_pay_adverse_slippage():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=3),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=3),  # entry bar
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1002, spread=3),  # signal-exit bar
        dict(open=1.0998, high=1.1005, low=1.0994, close=1.1000, spread=3),  # untouched
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=None, target_dist=None),
        dict(signal=0, stop_dist=None, target_dist=None),
        dict(), dict(),
    ])
    cost = CostModel(slippage_points=2.0)  # slip_price = 0.0002
    result = run_backtest(bars, signals, _spec(), cost)
    t = _row(result.trades)

    # entry (short, SELLS -> lower, raw Bid otherwise): 1.1005 - 0.0002 = 1.1003
    assert t["entry_price"] == pytest.approx(1.1003)
    # signal exit (short, BUYS BACK -> higher, plus that leg's own spread):
    #   1.1000 + 3*0.0001 + 0.0002 = 1.1005
    assert t["exit_reason"] == "signal"
    assert t["exit_price"] == pytest.approx(1.1005)
    # pnl_points = (entry-exit)/point = (1.1003-1.1005)/0.0001 = -2.0
    assert t["pnl_points"] == pytest.approx(-2.0)


def test_force_exit_and_eod_exit_pay_adverse_slippage_too():
    """Session-flatten (force_exit) and end-of-data exits are market orders too (red-team M2) --
    same adverse-slippage convention as a signal exit, for both directions."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),   # long entry bar
        dict(open=1.1006, high=1.1010, low=1.1002, close=1.1030, spread=2),   # force-closed here
        dict(open=1.1030, high=1.1040, low=1.1020, close=1.1035, spread=2),   # short entry bar
        dict(open=1.1035, high=1.1040, low=1.1030, close=1.1038, spread=2),   # eod bar
    ])
    signals = _signals([
        dict(signal=1, stop_dist=None, target_dist=None),
        dict(),
        dict(signal=-1, stop_dist=None, target_dist=None),  # fires at bar3's open, after the force-close
        dict(),
        dict(),
    ])
    force_exit = [False, False, True, False, False]
    cost = CostModel(slippage_points=1.0)  # slip_price = 0.0001
    result = run_backtest(bars, signals, _spec(), cost, force_exit=force_exit)

    assert result.trades.height == 2
    long_leg = result.trades.row(0, named=True)
    short_leg = result.trades.row(1, named=True)

    assert long_leg["exit_reason"] == "force"
    # force exit (long, close - slippage): 1.1030 - 0.0001 = 1.1029
    assert long_leg["exit_price"] == pytest.approx(1.1029)

    assert short_leg["entry_idx"] == 3 and short_leg["exit_reason"] == "eod"
    # short entry: 1.1030 - 0.0001 = 1.1029 (raw Bid minus slippage)
    assert short_leg["entry_price"] == pytest.approx(1.1029)
    # eod exit (short, buys back at close + that bar's spread + slippage):
    #   1.1038 + 2*0.0001 + 0.0001 = 1.1041
    assert short_leg["exit_price"] == pytest.approx(1.1041)


def test_target_fills_never_slip_even_when_cost_has_slippage():
    """Regression guard: slippage_points must never touch a target/limit fill (red-team M2's
    scope is entries + non-target exits + stops -- targets are the one fill type explicitly
    excluded, both before and after this fix)."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1070, low=1.1000, close=1.1050, spread=2),  # TP touched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0500, target_dist=0.0050),
        dict(), dict(),
    ])
    cost = CostModel(slippage_points=5.0)
    result = run_backtest(bars, signals, _spec(), cost)
    t = _row(result.trades)

    # entry = 1.1005+0.0002(spread)+0.0005(slippage) = 1.1012 ; TP = 1.1012+0.0050 = 1.1062
    # (bar2's high raised to 1.1070 so it still reaches this slippage-shifted TP)
    assert t["entry_price"] == pytest.approx(1.1012)
    assert t["target_price"] == pytest.approx(1.1062)
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == pytest.approx(1.1062)  # exactly TP, no slippage


def test_stressed_no_stop_system_no_longer_a_slippage_no_op():
    """p09-style regression (red-team M2): pre-fix, CostModel.stressed()'s +1pt slippage only
    ever reached stop fills, so a no-stop (risk-type-C-style) system's stressed P&L was
    bit-identical to 'spread x1.5 only' -- the cost-stress gate's slippage term was a complete
    no-op for exactly the systems (no stop, or signal/time-exit driven) it most needs to
    stress. It must now differ."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # long entry bar
        dict(open=1.1010, high=1.1015, low=1.1005, close=1.1012, spread=2),  # signal-exit bar
        dict(open=1.1012, high=1.1020, low=1.1008, close=1.1015, spread=2),  # untouched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=None, target_dist=None),   # no stop: risk-type-C style
        dict(signal=0, stop_dist=None, target_dist=None),
        dict(), dict(),
    ])
    base = CostModel()
    spread_only = CostModel(spread_multiplier=1.5)
    stressed = base.stressed()  # x1.5 spread + 1pt slippage (no spec -> legacy 1pt fallback)

    r_spread_only = run_backtest(bars, signals, _spec(), spread_only)
    r_stressed = run_backtest(bars, signals, _spec(), stressed)
    assert r_spread_only.trades.height == 1 and r_stressed.trades.height == 1

    pnl_spread_only = r_spread_only.trades["pnl_points"][0]
    pnl_stressed = r_stressed.trades["pnl_points"][0]
    # Pre-fix these were bit-identical (the whole point of the M2 bug). Now the entry (+1pt)
    # and the signal exit (+1pt) each pay slippage on top of the spread stress.
    assert pnl_stressed < pnl_spread_only
    assert pnl_spread_only - pnl_stressed == pytest.approx(2.0)


# --------------------------------------------------------------------------- same-bar SL/TP ambiguity

def _ambiguous_setup():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1030, low=1.0980, close=1.1010, spread=2),  # both SL & TP touched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0020, target_dist=0.0020),
        dict(), dict(),
    ])
    # entry=1.1007 ; SL=1.0987 ; TP=1.1027
    return bars, signals


def test_ambiguous_same_bar_without_m1_assumes_stop():
    bars, signals = _ambiguous_setup()
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(1.0987)


def test_ambiguous_same_bar_with_m1_resolves_by_time_order():
    bars, signals = _ambiguous_setup()
    # M1 path inside HTF bar 2: target is touched first (t=0), stop only later (t=1) ->
    # the walk breaks at the first hit, so the target wins despite the HTF-aggregated
    # bar looking ambiguous.
    m1_rows = [
        dict(open=1.1005, high=1.1030, low=1.1000, close=1.1025, spread=2),  # TP hit here first
        dict(open=1.1025, high=1.1030, low=1.0980, close=1.0985, spread=2),  # SL only, never reached
    ]
    m1 = _bars(m1_rows, start=bars["ts"][2], step=timedelta(minutes=20))
    result = run_backtest(bars, signals, _spec(), m1=m1, timeframe="H1")
    t = _row(result.trades)
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == pytest.approx(1.1027)
    assert result.meta["m1_used"] is True


# --------------------------------------------------------------------------- B1 (BLOCKER): per-direction
# sentinels -- a short with no stop and/or no target must not gap-exit on its entry bar.

_QUIET_ROW = dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=2)


@pytest.mark.parametrize("use_m1", [False, True])
@pytest.mark.parametrize("stop_dist,target_dist", [
    (None, None), (0.0050, None), (None, 0.0050), (0.0050, 0.0050),
])
@pytest.mark.parametrize("direction", [1, -1])
def test_missing_stop_or_target_never_gap_exits_on_the_entry_bar(direction, stop_dist, target_dist, use_m1):
    """Regression (red-team B1, BLOCKER). Pre-fix, ``_run_core`` used the same
    sentinel pair (-1e18 stop / +1e18 target) for both directions. That's safe
    for a long (whose checks compare against raw Bid low/high) but wrong for a
    short (whose checks compare against Ask high/low, always positive prices):
    ``ask_h >= -1e18`` and ``ask_l <= +1e18`` are trivially true, so every
    short with no hard stop and/or no target was "gap-stopped"/"gap-targeted"
    at -spread on its own entry bar. All 4 {stop, no stop} x {target, no
    target} combinations, both directions, with and without ``m1``, must run
    a genuinely quiet market all the way to ``eod`` -- never exit on the entry
    bar via a sentinel that should never be reachable.
    """
    n = 5
    bars = _bars([dict(_QUIET_ROW) for _ in range(n)])
    signals = _signals(
        [dict(signal=direction, stop_dist=stop_dist, target_dist=target_dist)] + [dict()] * (n - 1)
    )
    m1 = _bars([dict(_QUIET_ROW) for _ in range(n)]) if use_m1 else None

    result = run_backtest(bars, signals, _spec(), m1=m1, timeframe="H1")
    t = _row(result.trades)

    assert t["direction"] == direction
    assert t["entry_idx"] == 1
    assert t["exit_idx"] == n - 1
    assert t["exit_reason"] == "eod"
    assert t["exit_reason"] not in ("gap_stop", "gap_target")


def test_short_stop_only_still_triggers_a_real_stop_ignoring_the_missing_target():
    """A short with a stop but no target must still stop out normally when the
    real level is touched (B1's fix must not turn "no target" into "no stop
    either")."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1005, spread=2),  # fill bar
        dict(open=1.1005, high=1.1060, low=1.1000, close=1.1050, spread=2),  # price rises through SL
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)

    # entry = 1.1005 (raw Bid) ; SL = 1.1005+0.0020 = 1.1025
    # bar2: ask_o=1.1005+2pt=1.10070 (<SL, not gap) ; ask_h=1.1060+2pt=1.10620 (>=SL) -> stop, no slippage
    assert t["exit_reason"] == "stop"
    assert t["stop_price"] == pytest.approx(1.1025)
    assert t["exit_price"] == pytest.approx(1.1025)
    assert np.isnan(t["target_price"])


# --------------------------------------------------------------------------- M1 window bounds (red-team M1):
# the M1 window for HTF bar j must be [ts_j, ts_j + timeframe), never ts[j+1] and never len(m1).

def test_map_m1_bounds_last_bar_does_not_leak_past_its_own_span():
    """Last-bar case (red-team p02 case B): an m1 frame that runs well past
    the last HTF bar's own window must not all be mapped to it."""
    from quantlab.engine import _map_m1_bounds

    htf_ts = pl.Series([T0, T0 + timedelta(hours=1)]).cast(pl.Datetime("ms"))
    # 12 M1 rows, 20 minutes apart, spanning 4h past the last HTF bar's own open
    m1_ts = pl.Series(
        [T0 + timedelta(hours=1) + timedelta(minutes=20 * k) for k in range(12)]
    ).cast(pl.Datetime("ms"))

    starts, ends = _map_m1_bounds(htf_ts, m1_ts, np.timedelta64(60, "m"))

    # last HTF bar's window is [T0+1h, T0+2h) -- only offsets 0/20/40 min qualify
    assert starts[-1] == 0
    assert ends[-1] == 3


def test_map_m1_bounds_clips_at_bar_span_across_a_session_gap():
    """Gapped/session-filtered HTF frame (red-team p02 case C): a bar right
    before a large gap (weekend, session filter, holiday) must not have that
    entire gap's M1 rows mapped to it just because ts[j+1] is far away."""
    from quantlab.engine import _map_m1_bounds

    htf_ts = pl.Series([T0, T0 + timedelta(hours=46)]).cast(pl.Datetime("ms"))
    # M1 has no gap at all here (e.g. the caller loaded M1 without the same
    # session/day filter that produced the gapped HTF frame)
    m1_ts = pl.Series([T0 + timedelta(minutes=k) for k in range(46 * 60)]).cast(pl.Datetime("ms"))

    starts, ends = _map_m1_bounds(htf_ts, m1_ts, np.timedelta64(60, "m"))

    # bar0's own window is only its first hour: 60 one-minute rows, not 46h of them
    assert ends[0] - starts[0] == 60


def test_map_m1_bounds_d1_intraday_end_excludes_the_dropped_partial_day():
    """D1 with an intraday end (red-team p02 case A): ``load_bars`` drops a
    trailing partial D1 day but keeps its minutes in the M1 series; the last
    D1 bar's window must still be exactly one day, not day+partial-day."""
    from quantlab.engine import _map_m1_bounds

    d1_ts = pl.Series([T0, T0 + timedelta(days=1)]).cast(pl.Datetime("ms"))
    # day 2 in full (1440 minutes) plus half of day 3 (720 minutes), 30 min apart
    minutes = range(0, 1440 + 720, 30)
    m1_ts = pl.Series(
        [T0 + timedelta(days=1) + timedelta(minutes=k) for k in minutes]
    ).cast(pl.Datetime("ms"))

    starts, ends = _map_m1_bounds(d1_ts, m1_ts, np.timedelta64(1440, "m"))

    assert ends[-1] - starts[-1] == 48  # 1440 minutes / 30 == 48, none of day 3


# --------------------------------------------------------------------------- N3/N4 (re-verify
# 2026-09-24): timeframe is required whenever m1 is given, and an M1/HTF OHLC consistency
# check catches a wrong-but-plausible explicit declaration.

def test_run_backtest_requires_timeframe_when_m1_given():
    """Regression (red-team N3): there is no min-gap-based inference fallback
    left at all -- m1= without an explicit timeframe= must raise a clear
    error, not silently guess a span from bars' own smallest ts gap (which
    overstates the true span on a sparse/session-filtered frame)."""
    bars = _bars([dict(_QUIET_ROW) for _ in range(3)])
    signals = _signals(
        [dict(signal=1, stop_dist=0.0050, target_dist=0.0100)] + [dict()] * 2
    )
    m1 = _bars([dict(_QUIET_ROW) for _ in range(3)])
    with pytest.raises(ValueError, match="timeframe"):
        run_backtest(bars, signals, _spec(), m1=m1)


def test_run_backtest_sparse_daily_frame_requires_timeframe_instead_of_inferring():
    """Regression (red-team N3, report's sparse-frame case): a frame that
    keeps only one bar per day would have its 1-day ts-gap mistaken for the
    true (sub-day) timeframe under the old inference rule. Now it must raise
    instead of silently mapping a whole day's worth of M1 to a single bar."""
    bars = _bars([dict(_QUIET_ROW) for _ in range(3)], step=timedelta(days=1))
    signals = _signals(
        [dict(signal=1, stop_dist=0.0050, target_dist=0.0100)] + [dict()] * 2
    )
    m1 = _bars([dict(_QUIET_ROW) for _ in range(3)], step=timedelta(days=1))
    with pytest.raises(ValueError, match="timeframe"):
        run_backtest(bars, signals, _spec(), m1=m1)


def test_run_backtest_wrong_explicit_timeframe_raises_on_ohlc_mismatch():
    """Regression (red-team N4): a mis-declared (but explicitly given, so the
    N3 check alone can't catch it) timeframe must no longer be accepted
    silently. D1 bars declared as timeframe="H1" restrict each day's mapped
    M1 window to just its first hour; that truncated window's own max/min
    then disagrees with the D1 bar's real (whole-day) high/low, and
    run_backtest must raise naming the first offending bar."""
    d1_bars = _bars([
        dict(open=1.1000, high=1.1200, low=1.0900, close=1.1050, spread=2),  # day 1: full-day range
        dict(open=1.1050, high=1.1080, low=1.1020, close=1.1060, spread=2),  # day 2
    ], step=timedelta(days=1))
    signals = _signals([
        dict(signal=1, stop_dist=0.0500, target_dist=None),
        dict(),
    ])
    # Only 1 M1 row exists, inside day 1's *wrongly*-declared 1-hour window -- its own
    # max/min (1.1005/1.0995) disagrees with the D1 bar's real high/low (1.1200/1.0900).
    m1 = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
    ], start=d1_bars["ts"][0], step=timedelta(hours=1))
    with pytest.raises(ValueError, match="inconsistent"):
        run_backtest(d1_bars, signals, _spec(), m1=m1, timeframe="H1")


def test_find_m1_inconsistency_returns_the_first_offending_bar():
    """Hand-computed unit test of the consistency check itself, isolated from
    run_backtest's other validation."""
    from quantlab.engine import _M1_CONSISTENCY_TOL, _find_m1_inconsistency

    h = np.array([1.1010, 1.1050, 1.1030])
    l = np.array([1.0990, 1.1000, 1.1010])
    # bar0's window [0,1): max=1.1005/min=1.0995 vs its own declared 1.1010/1.0990 -> mismatch
    m1_h = np.array([1.1005, 1.1010, 1.1200, 1.1030])
    m1_l = np.array([1.0995, 1.1000, 1.1100, 1.1010])
    m1_start = np.array([0, 1, 3], dtype=np.int64)
    m1_end = np.array([1, 3, 4], dtype=np.int64)

    j, mx, mn = _find_m1_inconsistency(h, l, m1_h, m1_l, m1_start, m1_end, _M1_CONSISTENCY_TOL)
    assert j == 0
    assert mx == pytest.approx(1.1005)
    assert mn == pytest.approx(1.0995)


def test_find_m1_inconsistency_skips_bars_with_no_m1_coverage_and_returns_minus_one():
    """A bar with an empty M1 window (start == end) must be skipped outright
    (its own declared high/low can be anything -- there is nothing to check
    against), and an otherwise fully-consistent frame must return -1."""
    from quantlab.engine import _M1_CONSISTENCY_TOL, _find_m1_inconsistency

    h = np.array([1.1005, 1.1200, 5.0])     # bar2's h/l are nonsense but uncovered -> ignored
    l = np.array([1.0995, 1.1050, -5.0])
    m1_h = np.array([1.1005, 1.1100, 1.1200])
    m1_l = np.array([1.0995, 1.1050, 1.1100])
    m1_start = np.array([0, 1, 3], dtype=np.int64)
    m1_end = np.array([1, 3, 3], dtype=np.int64)  # bar2: empty window

    j, mx, mn = _find_m1_inconsistency(h, l, m1_h, m1_l, m1_start, m1_end, _M1_CONSISTENCY_TOL)
    assert j == -1


def test_run_backtest_walk_forward_slice_ignores_m1_from_the_next_fold():
    """Walk-forward slice of bars with a full M1 frame (red-team M1's stated
    failure scenario): a fold's HTF bars end mid-series but the caller passes
    the FULL M1 frame it loaded once for every fold. The still-open trade on
    the slice's last bar must resolve using only that bar's own M1 window --
    not M1 data belonging to the next fold, even though it is chronologically
    contiguous and would otherwise auto-map via ts[j+1] (which doesn't exist
    for the last bar) or via a mis-inferred span."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fold's last bar
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0500, target_dist=0.0050),
        dict(),
    ])
    m1 = _bars([
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),   # bar1's own minute: no hit
        dict(open=1.1006, high=1.1200, low=1.1006, close=1.1150, spread=2),   # NEXT fold: would hit target
    ], start=bars["ts"][1], step=timedelta(hours=1))

    result = run_backtest(bars, signals, _spec(), m1=m1, timeframe="H1")
    t = _row(result.trades)

    assert t["exit_reason"] == "eod"  # not "target" -- the next-fold M1 row must be excluded
    assert t["exit_price"] == pytest.approx(1.1006)


# --------------------------------------------------------------------------- M5: MAE/MFE must only reflect
# prices up to the exit, and shorts' excursions must be measured on Ask.

def test_mae_mfe_capped_at_fill_price_on_exit_bar_without_m1():
    """The report's synthetic case: a rally after a stop, in the SAME bar,
    must not inflate mfe_points; the exit bar's own low must not inflate
    mae_points beyond the fill distance either (we only know a hit happened
    *somewhere* in the bar's range)."""
    spec = _spec(point=1e-5)
    bars = _bars([
        dict(open=1.1, high=1.1001, low=1.0999, close=1.1, spread=10),
        dict(open=1.1, high=1.1002, low=1.0998, close=1.1, spread=10),      # fill bar
        dict(open=1.1, high=1.1100, low=1.0980, close=1.1090, spread=10),   # stop hit, THEN rallies to 1.1100
        dict(open=1.109, high=1.109, low=1.109, close=1.109, spread=10),
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0011, target_dist=None),
        dict(), dict(), dict(),
    ])
    result = run_backtest(bars, signals, spec)
    t = _row(result.trades)

    # ep = 1.1 + 10*1e-5 = 1.1001 ; SL = 1.1001-0.0011 = 1.0990
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(1.0990)
    assert t["pnl_points"] == pytest.approx(-110.0)
    assert t["mae_points"] == pytest.approx(110.0)   # capped at the fill distance, not the raw low (-> 210)
    assert t["mfe_points"] == pytest.approx(10.0)    # must NOT see the post-stop rally (uncapped: 990)


def test_mae_mfe_walks_m1_sub_bars_up_to_the_hit_sub_bar_only():
    """With M1, excursions are accumulated sub-bar by sub-bar in order: every
    sub-bar *before* the hit contributes its own full range, the hit sub-bar
    contributes only the fill-capped excursion (re-verify 2026-09-24, M5
    residual -- see the ECB-minute-style test below for the case where the
    hit sub-bar's own range is what would otherwise inflate MAE), and a rally
    in a *later* sub-bar of the same HTF bar must never be walked at all."""
    spec = _spec(point=1e-5)
    bars = _bars([
        dict(open=1.1, high=1.1001, low=1.0999, close=1.1, spread=10),
        dict(open=1.1, high=1.1002, low=1.0998, close=1.1, spread=10),      # fill bar
        dict(open=1.1, high=1.1100, low=1.0980, close=1.1090, spread=10),   # stop hit sub-bar 0, rally sub-bar 1
        dict(open=1.109, high=1.109, low=1.109, close=1.109, spread=10),
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0011, target_dist=None),
        dict(), dict(), dict(),
    ])
    m1_rows = [
        dict(open=1.1, high=1.1002, low=1.0998, close=1.1),          # bar1's own minute
        dict(open=1.1, high=1.1, low=1.0980, close=1.0985),          # bar2 minute 0: stop hit here
        dict(open=1.0985, high=1.1100, low=1.0985, close=1.1090),    # bar2 minute 1: rally, must be ignored
    ]
    m1_ts = [bars["ts"][1], bars["ts"][2], bars["ts"][2] + timedelta(minutes=1)]
    m1 = pl.DataFrame({
        "ts": m1_ts,
        "open": [r["open"] for r in m1_rows], "high": [r["high"] for r in m1_rows],
        "low": [r["low"] for r in m1_rows], "close": [r["close"] for r in m1_rows],
        "spread": [10.0] * 3,
    }, schema_overrides={"ts": pl.Datetime("ms")}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )

    result = run_backtest(bars, signals, spec, m1=m1, timeframe="H1")
    t = _row(result.trades)

    # ep=1.1001, SL=1.0990 (see the arithmetic above); sub-bar0 (bar2 minute0) touches SL,
    # filling at the level (1.0990, no slippage) -- capped excursion = (1.1001-1.0990)/1e-5 = 110
    # (re-verify 2026-09-24, M5 residual: pre-fix this was 210, sub-bar0's own uncapped low 1.0980)
    assert t["exit_reason"] == "stop"
    assert t["mae_points"] == pytest.approx(110.0)   # capped at the fill, not sub-bar0's raw low (-> 210)
    assert t["mfe_points"] == pytest.approx(10.0)    # the rally sub-bar is never walked


def test_short_m1_exit_minute_excursion_capped_at_fill_ecb_style_spike():
    """Regression (red-team M5 residual, re-verify 2026-09-24). The report's
    real worst case was a 2019-09-12 ECB-announcement-minute short with MAE
    446 against a realised loss of only 159: the exit minute's OWN full range
    (well beyond the level that actually triggered the stop) was still
    counted in full, even though a resting stop order only ever fills once
    and price action after that fill -- even inside the very same minute --
    never happened to an already-closed position. This is the same bug as
    the whole-bar-path fix already covers, one level deeper: on the M1 path
    it must apply *inside* the single sub-bar that triggers the exit too."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),  # fill bar
        dict(open=1.1000, high=1.1200, low=1.0995, close=1.1150, spread=2),  # "ECB" spike bar
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    m1 = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),  # bar1's own minute
        dict(open=1.1000, high=1.1200, low=1.0995, close=1.1150, spread=2),  # bar2's own minute: the spike
    ], start=bars["ts"][1], step=timedelta(hours=1))

    result = run_backtest(bars, signals, _spec(), m1=m1, timeframe="H1")
    t = _row(result.trades)

    # entry=1.1000 (raw Bid, short) ; SL=1.1000+0.0020=1.1020
    # bar2's only M1 sub-bar IS the whole spike minute: ask_o=1.1000+2pt=1.1002 (<SL -> not a
    # gap) ; ask_h=1.1200+2pt=1.1202 (>=SL -> touch), fills at the level: 1.1020
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(1.1020)
    assert t["pnl_points"] == pytest.approx(-20.0)
    # pre-fix (buggy): mae would be (ask_h-entry)/point = (1.1202-1.1000)/0.0001 = 202.0 -- the
    # spike's full range counted even though it happened only AFTER the stop had already filled
    assert t["mae_points"] == pytest.approx(20.0)   # capped at the realised loss, matching pnl
    assert t["mfe_points"] == pytest.approx(3.0)    # from bar1's (non-hit) sub-bar, unaffected by the cap


def test_short_excursions_measured_on_ask_not_bid():
    """Regression (red-team M5): a short's MAE/MFE must be measured on Ask
    (high/low + that bar's own effective spread), matching the price its
    stop/target actually triggers on -- not raw Bid."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1005, spread=2),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=4),  # eod bar, wider spread
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0500, target_dist=0.0500),  # far away, never hit
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)

    # bar1 (entry bar): ask_h=1.1010+2pt=1.1012, ask_l=1.1000+2pt=1.1002
    #   adverse=(1.1012-1.1005)/1e-4=7 ; favorable=(1.1005-1.1002)/1e-4=3
    # bar2 (eod bar, spread=4): ask_h=1.1010+4pt=1.1014, ask_l=1.1000+4pt=1.1004
    #   adverse=(1.1014-1.1005)/1e-4=9 ; favorable=(1.1005-1.1004)/1e-4=1
    # (raw-Bid mae/mfe would both be 5/5 -- the same on every bar here, since only spread changes)
    assert t["mae_points"] == pytest.approx(9.0)
    assert t["mfe_points"] == pytest.approx(3.0)


# --------------------------------------------------------------------------- minor-1: stop_fill knob

def test_stop_fill_bar_extreme_fills_at_the_worst_of_the_bar_no_m1():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.0950, close=1.1000, spread=2),  # SL touched, trades much lower
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    level = run_backtest(bars, signals, _spec(), CostModel(stop_fill="level"))
    extreme = run_backtest(bars, signals, _spec(), CostModel(stop_fill="bar_extreme"))

    # SL = 1.1007-0.0020 = 1.0987 ; bar2 opens 1.1005 (no gap), low 1.0950 <= SL
    assert _row(level.trades)["exit_price"] == pytest.approx(1.0987)         # default: fills at the level
    assert _row(extreme.trades)["exit_price"] == pytest.approx(1.0950)      # stress knob: fills at the bar low


def test_stop_fill_bar_extreme_does_not_affect_targets():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1005, high=1.1200, low=1.1000, close=1.1150, spread=2),  # TP touched, trades much higher
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0500, target_dist=0.0050),
        dict(), dict(),
    ])
    extreme = run_backtest(bars, signals, _spec(), CostModel(stop_fill="bar_extreme"))
    # TP = 1.1007+0.0050 = 1.1057, a resting limit order -- must still fill exactly at TP
    assert _row(extreme.trades)["exit_price"] == pytest.approx(1.1057)


def test_cost_model_rejects_unknown_stop_fill():
    with pytest.raises(ValueError, match="stop_fill"):
        CostModel(stop_fill="worst_case")


# --------------------------------------------------------------------------- minor-10: spread choice for
# eod/force short exits, and for no-M1 short SL/TP touch checks.

def test_eod_short_exit_uses_spread_max_without_m1():
    """Without M1, an eod/force short exit must use the bar's spread_max (a
    conservative stand-in for "the spread somewhere in this bar"), not that
    bar's own open spread (which can be a rollover/session-open spike)."""
    bars = _bars([
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=2),
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=50, spread_max=8),  # eod bar
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0500, target_dist=0.0500),
        dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)
    assert t["exit_reason"] == "eod"
    # exit = close + spread_max(8)*point = 1.1000 + 0.0008 = 1.1008 (NOT +0.0050 from the open spread)
    assert t["exit_price"] == pytest.approx(1.1008)
    assert t["spread_cost_points"] == pytest.approx(8.0)


def test_force_exit_short_uses_spread_max_without_m1():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),   # fill bar
        dict(open=1.1006, high=1.1010, low=1.1002, close=1.1010, spread=50, spread_max=8),  # force-closed here
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0500, target_dist=0.0500),
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec(), force_exit=[False, False, True])
    t = _row(result.trades)
    assert t["exit_reason"] == "force"
    assert t["exit_price"] == pytest.approx(1.1010 + 8 * POINT)
    assert t["spread_cost_points"] == pytest.approx(8.0)


def test_eod_short_exit_uses_last_m1_spread_when_m1_given():
    bars = _bars([
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=2),
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=50, spread_max=8),  # eod bar
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0500, target_dist=0.0500),
        dict(),
    ])
    # One M1 "minute" per HTF bar (still exercises the M1 code path/mapping;
    # a single sub-bar is trivially its own *last* sub-bar).
    m1 = _bars([
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=2),   # bar0's own minute
        dict(open=1.1000, high=1.1000, low=1.1000, close=1.1000, spread=6),   # bar1's own minute: this wins
    ], step=timedelta(hours=1))
    result = run_backtest(bars, signals, _spec(), m1=m1, timeframe="H1")
    t = _row(result.trades)
    assert t["exit_reason"] == "eod"
    assert t["exit_price"] == pytest.approx(1.1000 + 6 * POINT)
    assert t["spread_cost_points"] == pytest.approx(6.0)


def test_no_m1_short_sl_tp_touch_uses_spread_max_conservatively():
    """A no-M1 short's intrabar *touch* test must use spread_max (conservative
    in both directions: it can only make a stop trigger more easily and a
    target trigger less easily), while gap detection/pricing at the bar's own
    open still uses the real open spread."""
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),   # fill bar
        # bar2's raw high (1.1010) never reaches SL with the bar-open spread (2pt: ask_h=1.1012),
        # but does with spread_max (30pt: ask_h=1.1040) -- the open itself stays far from SL (no gap)
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1008, spread=2, spread_max=30),
    ])
    signals = _signals([
        dict(signal=-1, stop_dist=0.0020, target_dist=None),
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    t = _row(result.trades)
    # entry=1.1005 (Bid) ; SL=1.1005+0.0020=1.1025 ; with the (wrong) open spread this bar never
    # touches SL at all (would run to "eod"); with spread_max it does -> "stop", filled at the level
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(1.1025)  # fills at the level (stop_fill="level" default)


def test_tight_stop_warns_once():
    """Minor-10's closing note: warn when a stop distance is narrower than the
    entry bar's effective spread + spec.stops_level, since MT5 would reject
    such an order. De-duplicated per symbol per process (not per-call, which
    would defeat Python's own warning de-dup and spam stderr on every
    run_backtest call in a tight loop) -- so a second run for the same
    symbol must stay silent."""
    from quantlab.engine import _WARNED_TIGHT_STOP

    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=10),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=10),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=10),
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.00005, target_dist=None),  # 0.5 pts < 10pt spread + 0 stops_level
        dict(), dict(),
    ])
    spec = _spec(stops_level=0.0)
    _WARNED_TIGHT_STOP.discard(spec.symbol)  # isolate from any earlier test/process state
    try:
        with pytest.warns(UserWarning, match="stop distance narrower"):
            run_backtest(bars, signals, spec)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            run_backtest(bars, signals, spec)  # second call, same symbol -> must not warn again
    finally:
        _WARNED_TIGHT_STOP.discard(spec.symbol)


# --------------------------------------------------------------------------- reversal

def test_reversal_exits_then_enters_at_same_open():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # long fill bar
        dict(open=1.1020, high=1.1030, low=1.1010, close=1.1020, spread=2),  # reversal bar
        dict(open=1.1020, high=1.1025, low=1.1000, close=1.1005, spread=4),  # eod bar
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0050, target_dist=0.0100),
        dict(signal=-1, stop_dist=0.0030, target_dist=0.0060),
        dict(), dict(),
    ])
    result = run_backtest(bars, signals, _spec())
    trades = result.trades
    assert trades.height == 2

    long_leg = trades.row(0, named=True)
    short_leg = trades.row(1, named=True)

    assert long_leg["direction"] == 1
    assert long_leg["entry_idx"] == 1 and long_leg["exit_idx"] == 2
    assert long_leg["exit_reason"] == "signal"
    assert long_leg["exit_price"] == pytest.approx(1.1020)  # raw open, no spread on a long's exit
    # pnl_points = (1.1020-1.1007)/0.0001 = 13.0  (entry = 1.1005+0.0002)
    assert long_leg["pnl_points"] == pytest.approx(13.0)

    assert short_leg["direction"] == -1
    assert short_leg["entry_idx"] == 2 and short_leg["exit_idx"] == 3
    assert short_leg["entry_price"] == pytest.approx(1.1020)  # raw open, no spread on a short's entry
    assert short_leg["exit_reason"] == "eod"
    # exit = close[3] + spread[3]*point = 1.1005 + 4*0.0001 = 1.1009
    assert short_leg["exit_price"] == pytest.approx(1.1009)
    # pnl_points = (1.1020-1.1009)/0.0001 = 11.0
    assert short_leg["pnl_points"] == pytest.approx(11.0)


# --------------------------------------------------------------------------- force_exit

def test_force_exit_closes_at_close_and_blocks_new_entries():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=2),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=2),  # fill bar
        dict(open=1.1006, high=1.1010, low=1.1002, close=1.1030, spread=2),  # force-closed here
        dict(open=1.1030, high=1.1040, low=1.1020, close=1.1035, spread=2),  # blocked re-entry bar
        dict(open=1.1035, high=1.1040, low=1.1030, close=1.1038, spread=2),  # session resumes
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0050, target_dist=0.0100),
        dict(),
        dict(signal=1, stop_dist=0.0050, target_dist=0.0100),  # fires at bar3's open -> must be blocked
        dict(),
        dict(),
    ])
    force_exit = [False, False, True, True, False]
    result = run_backtest(bars, signals, _spec(), force_exit=force_exit)

    assert result.trades.height == 1
    t = _row(result.trades)
    assert t["exit_reason"] == "force"
    assert t["exit_idx"] == 2
    assert t["exit_price"] == pytest.approx(1.1030)  # raw close, no spread on a long's exit
    assert list(result.position) == [0, 1, 0, 0, 0]  # blocked re-entry never opens


# --------------------------------------------------------------------------- spread multiplier / stress gate

def test_spread_multiplier_and_stressed_shift_entry_and_stop_fill():
    bars = _bars([
        dict(open=1.1000, high=1.1005, low=1.0995, close=1.1000, spread=4),
        dict(open=1.1005, high=1.1010, low=1.1000, close=1.1006, spread=4),  # fill bar
        dict(open=1.1005, high=1.1010, low=1.0970, close=1.1000, spread=4),  # SL touched
    ])
    signals = _signals([
        dict(signal=1, stop_dist=0.0030, target_dist=None),
        dict(), dict(),
    ])
    cost_a = CostModel(version_tag="fbs-v1")  # spread x1, no slippage, stop_fill="level"
    cost_b = cost_a.stressed()                # DESIGN §4.2 default: x1.5 spread, +1pt slippage,
                                               # stop_fill="bar_extreme" (red-team N5)

    result_a = run_backtest(bars, signals, _spec(), cost_a)
    result_b = run_backtest(bars, signals, _spec(), cost_b)
    a = _row(result_a.trades)
    b = _row(result_b.trades)

    # entry_a = 1.1005 + 4*1*0.0001      = 1.1009 ; SL_a = 1.1009-0.0030 = 1.0979
    # entry_b (red-team M2: entry now also slips) = 1.1005 + 4*1.5*0.0001 + 1*0.0001 = 1.1012
    #   SL_b = 1.1012-0.0030 = 1.0982
    assert a["entry_price"] == pytest.approx(1.1009)
    assert b["entry_price"] == pytest.approx(1.1012)
    # neither gaps (open 1.1005 > either SL); fill_a = SL_a, no bar_extreme, no slippage.
    # fill_b: stop_fill="bar_extreme" now applies too (N5) -> fills at bar2's own low (0.0970)
    # instead of SL_b, minus 1pt slippage: 1.0970-0.0001 = 1.0969 (not the pre-N5 "SL_b - 1pt").
    # This stop-fill leg is unaffected by M2 (unchanged _chk_long slippage handling).
    assert a["exit_price"] == pytest.approx(1.0979)
    assert b["exit_price"] == pytest.approx(1.0970 - 0.0001)
    # version (red-team N5): deterministic encoding of every non-default knob, fixed order.
    assert cost_b.version == "fbs-v1+spread_mult1.5+slippage1+stop_fill=bar_extreme"
    assert cost_a.version == "fbs-v1"  # stressed() never mutates the base model


# --------------------------------------------------------------------------- swap / nights

def test_nights_weighted_wednesday_triple():
    entry = datetime(2024, 1, 1, 10, 0)  # Monday
    exit_ = datetime(2024, 1, 4, 10, 0)  # Thursday: holds Mon, Tue, Wed nights
    raw, weighted = _nights_weighted(entry, exit_, swap_3day=2)
    # Mon(1) + Tue(1) + Wed(3, triple) = 5
    assert raw == 3
    assert weighted == pytest.approx(5.0)


def test_nights_weighted_friday_to_monday_counts_one_rollover():
    entry = datetime(2024, 1, 5, 10, 0)  # Friday
    exit_ = datetime(2024, 1, 8, 10, 0)  # Monday
    raw, weighted = _nights_weighted(entry, exit_, swap_3day=2)
    assert raw == 1
    assert weighted == pytest.approx(1.0)


def test_nights_weighted_same_day_and_non_positive_duration():
    d = datetime(2024, 1, 1, 9, 0)
    assert _nights_weighted(d, datetime(2024, 1, 1, 15, 0), 2) == (0, 0.0)
    assert _nights_weighted(d, d, 2) == (0, 0.0)
    assert _nights_weighted(d, d - timedelta(hours=1), 2) == (0, 0.0)


def test_engine_swap_integration_over_a_full_week():
    # D1 bars, weekends skipped (as real FX data has no weekend bars).
    # entry_ts = Tue Jan2 00:00 ; exit_ts (eod) = Mon Jan8 00:00
    # nights: Tue(1) Wed(3,triple) Thu(1) Fri(1) = 4 raw, 6.0 weighted (see
    # test_nights_weighted_wednesday_triple / _friday_to_monday for the isolated rule).
    days = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
            datetime(2024, 1, 4), datetime(2024, 1, 5), datetime(2024, 1, 8)]
    rows = [
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),  # fill bar
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1005, high=1.1015, low=1.0995, close=1.1010, spread=2),  # eod bar
    ]
    bars = pl.DataFrame({
        "ts": days,
        "open": [r["open"] for r in rows], "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows], "close": [r["close"] for r in rows],
        "spread": [float(r["spread"]) for r in rows],
    }, schema_overrides={"ts": pl.Datetime("ms")}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )
    signals = _signals([
        dict(signal=1, stop_dist=0.0100, target_dist=0.0100),
        dict(), dict(), dict(), dict(), dict(),
    ])
    spec = _spec(swap_long=-2.0)
    result = run_backtest(bars, signals, spec, CostModel())
    t = _row(result.trades)
    assert t["entry_ts"] == datetime(2024, 1, 2)
    assert t["exit_ts"] == datetime(2024, 1, 8)
    assert t["exit_reason"] == "eod"
    assert t["nights"] == 4
    # swap_points = weighted_nights(6.0) * swap_long(-2.0) * swap_multiplier(1.0) = -12.0
    assert t["swap_points"] == pytest.approx(-12.0)

    # swap sensitivity band (DESIGN §4.3): +-50% multiplier scales swap_points linearly
    result_lo = run_backtest(bars, signals, spec, CostModel(swap_multiplier=0.5))
    result_hi = run_backtest(bars, signals, spec, CostModel(swap_multiplier=1.5))
    assert _row(result_lo.trades)["swap_points"] == pytest.approx(-6.0)
    assert _row(result_hi.trades)["swap_points"] == pytest.approx(-18.0)


def test_nights_weighted_swap_every_day_counts_weekend_nights():
    """Regression (red-team minor-4): crypto CFDs at FBS-like brokers are
    charged swap every night of the week, including the two nights FX/metals
    skip (Friday->Saturday, Saturday->Sunday)."""
    entry = datetime(2024, 1, 5, 10, 0)  # Friday
    exit_ = datetime(2024, 1, 8, 10, 0)  # Monday
    raw_fx, weighted_fx = _nights_weighted(entry, exit_, swap_3day=2, swap_every_day=False)
    raw_crypto, weighted_crypto = _nights_weighted(entry, exit_, swap_3day=2, swap_every_day=True)
    assert (raw_fx, weighted_fx) == (1, pytest.approx(1.0))  # only Friday's night (default/FX rule)
    assert raw_crypto == 3                                    # Fri, Sat, Sun
    assert weighted_crypto == pytest.approx(3.0)


def test_engine_swap_every_day_spec_counts_weekend_swap():
    """Same D1-bar week as test_engine_swap_integration_over_a_full_week, but
    with a crypto-like spec (``swap_every_day=True``): the Fri->Mon weekend
    gap inside the hold must now also accrue swap for Sat and Sun nights."""
    days = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
            datetime(2024, 1, 4), datetime(2024, 1, 5), datetime(2024, 1, 8)]
    row = dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2)
    rows = [row] * 5 + [dict(open=1.1005, high=1.1015, low=1.0995, close=1.1010, spread=2)]
    bars = pl.DataFrame({
        "ts": days,
        "open": [r["open"] for r in rows], "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows], "close": [r["close"] for r in rows],
        "spread": [float(r["spread"]) for r in rows],
    }, schema_overrides={"ts": pl.Datetime("ms")}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )
    signals = _signals([
        dict(signal=1, stop_dist=0.0100, target_dist=0.0100),
        dict(), dict(), dict(), dict(), dict(),
    ])
    spec = _spec(swap_long=-2.0, swap_every_day=True)
    result = run_backtest(bars, signals, spec, CostModel())
    t = _row(result.trades)
    # FX rule (swap_every_day=False) gives nights=4, weighted=6.0 (see
    # test_engine_swap_integration_over_a_full_week); the crypto rule adds Fri->Sat and
    # Sat->Sun -> 6 raw nights, weighted 6.0 (minor-4 follow-up, re-verify 2026-09-24: an
    # every-day symbol is never ALSO tripled on swap_3day, so Wed counts as a flat 1.0 here
    # too -- pre-fix this was weighted 8.0, over-charging Wednesday's extra x2)
    assert t["nights"] == 6
    assert t["swap_points"] == pytest.approx(-12.0)


def test_nights_weighted_swap_every_day_does_not_also_triple_wednesday():
    """Regression (minor-4 follow-up, re-verify 2026-09-24): swap_every_day
    must not ALSO apply the FX/metals Wednesday x3 -- that multiplier exists
    specifically to compensate FX/metals for the two nights it skips
    (Fri->Sat, Sat->Sun); a symbol already charged every single night has no
    gap left to compensate for, so tripling Wednesday on top would
    double-count the weekend. A full Mon-open -> next-Mon-close week must
    come out to exactly 7 weighted nights (one per calendar night), not 9."""
    entry = datetime(2024, 1, 1, 10, 0)  # Monday
    exit_ = datetime(2024, 1, 8, 10, 0)  # next Monday: a full 7-night week (Jan 3 is Wednesday)
    raw, weighted = _nights_weighted(entry, exit_, swap_3day=2, swap_every_day=True)
    assert raw == 7
    assert weighted == pytest.approx(7.0)  # not 9.0 (pre-fix: 6 nights x1.0 + Wed x3.0)


def test_swap_mode_interest_uses_entry_price_and_annual_rate():
    """Hand-computed test (red-team N1): swap_mode="interest" (MT5's
    SYMBOL_SWAP_MODE_INTEREST_CURRENT/_OPEN) charges an annual percentage of
    *price* per night: swap per night = price * contract_size * rate/100/360
    per lot, approximated with the trade's own entry price for every night
    held. Same D1-bar week as test_engine_swap_integration_over_a_full_week
    (weighted_nights=6.0), so the only new arithmetic is the interest formula
    itself."""
    days = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3),
            datetime(2024, 1, 4), datetime(2024, 1, 5), datetime(2024, 1, 8)]
    rows = [
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),  # fill bar
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1000, high=1.1010, low=1.0990, close=1.1005, spread=2),
        dict(open=1.1005, high=1.1015, low=1.0995, close=1.1010, spread=2),  # eod bar
    ]
    bars = pl.DataFrame({
        "ts": days,
        "open": [r["open"] for r in rows], "high": [r["high"] for r in rows],
        "low": [r["low"] for r in rows], "close": [r["close"] for r in rows],
        "spread": [float(r["spread"]) for r in rows],
    }, schema_overrides={"ts": pl.Datetime("ms")}).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )
    signals = _signals([
        dict(signal=1, stop_dist=0.0100, target_dist=0.0100),
        dict(), dict(), dict(), dict(), dict(),
    ])
    spec = _spec(swap_mode="interest", swap_long=5.0, swap_short=-3.0)  # annual % of price
    result = run_backtest(bars, signals, spec, CostModel())
    t = _row(result.trades)

    # entry_price = open[1] + spread[1]*point = 1.1000 + 2*0.0001 = 1.1002 ; nights weighted = 6.0
    # (see test_engine_swap_integration_over_a_full_week for the isolated night-counting arithmetic)
    assert t["entry_price"] == pytest.approx(1.1002)
    assert t["nights"] == 4
    assert t["swap_points"] == pytest.approx(0.0)  # interest mode never touches swap_points
    # swap_money_per_lot = weighted_nights(6.0) * entry_price(1.1002) * contract_size(100_000)
    #                      * annual_rate(5.0) / 100 / 360
    expected = 6.0 * 1.1002 * spec.contract_size * 5.0 / 100.0 / 360.0
    assert t["swap_money_per_lot"] == pytest.approx(expected)

    # swap sensitivity band (DESIGN §4.3) still scales it linearly, same as the other modes
    result_hi = run_backtest(bars, signals, spec, CostModel(swap_multiplier=1.5))
    assert _row(result_hi.trades)["swap_money_per_lot"] == pytest.approx(expected * 1.5)


# --------------------------------------------------------------------------- performance

@pytest.mark.slow
def test_backtest_3_7m_m1_bars_under_1_5s_after_warmup():
    n = 3_700_000
    rng = np.random.default_rng(0)

    step = rng.normal(0, 0.00005, n).cumsum()
    open_ = 1.1000 + step
    high = open_ + np.abs(rng.normal(0, 0.00003, n))
    low = open_ - np.abs(rng.normal(0, 0.00003, n))
    close = open_ + rng.normal(0, 0.00001, n)
    spread = rng.integers(1, 4, n).astype(np.float64)

    ts_start = datetime(2020, 1, 1)
    ts = pl.datetime_range(ts_start, ts_start + timedelta(minutes=n - 1), interval="1m",
                            eager=True, time_unit="ms")
    assert ts.len() == n

    bars = pl.DataFrame({
        "ts": ts, "open": open_, "high": high, "low": low, "close": close, "spread": spread,
    }).with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc"),
        pl.col("spread").alias("spread_max"),
        pl.lit(100).cast(pl.Int64).alias("tick_vol"),
    )

    prob = 0.0005
    r = rng.random(n)
    has_signal = r < prob
    sig_val = np.where(r < prob / 2, 1, -1).astype(np.int8)
    signals = pl.DataFrame({"_sig": sig_val, "_has": has_signal}).select(
        pl.when(pl.col("_has")).then(pl.col("_sig")).otherwise(None).cast(pl.Int8).alias("signal"),
        pl.when(pl.col("_has")).then(0.0050).otherwise(None).alias("stop_dist"),
        pl.when(pl.col("_has")).then(0.0100).otherwise(None).alias("target_dist"),
    )

    spec = _spec()
    cost = CostModel()

    run_backtest(bars, signals, spec, cost)  # warm-up: triggers numba compilation

    t0 = time.perf_counter()
    result = run_backtest(bars, signals, spec, cost)
    elapsed = time.perf_counter() - t0

    assert result.position.shape[0] == n
    assert elapsed < 1.5, f"backtest took {elapsed:.3f}s for {n} bars (budget: 1.5s post-warmup)"
