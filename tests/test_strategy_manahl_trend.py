"""Tests for ``quantlab.strategies.manahl_trend`` (Python port of MT5 ``ManAhl_EA`` v2).

Indicator values are checked against literal loop ports of the MQL5 code (``SigmaAt``,
``ScoreAt``, ``EffRatio.OnCalculate``, ``iATR``); simulator behaviour is checked on hand-built
daily paths where every expected fill is derivable by hand.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab.contracts import BAR_COLUMNS
from quantlab.costs import InstrumentSpec
from quantlab.engine import run_backtest
from quantlab.strategies import manahl_trend as mt
from quantlab.strategies.manahl_trend import ManAhlParams, ManAhlTrend, SymbolInput, simulate, target_fraction
from quantlab.testing import assert_no_lookahead, assert_strategy_source_clean

PT = 1e-5


def _spec(**kw) -> InstrumentSpec:
    base = dict(symbol="EURUSD", digits=5, point=PT, contract_size=100_000.0, tick_size=PT,
                volume_min=0.01, volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD",
                swap_mode="points", swap_long=0.0, swap_short=0.0)
    base.update(kw)
    return InstrumentSpec(**base)


def _weekdays(n: int, start: datetime = datetime(2021, 1, 4)) -> list[datetime]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() <= 4:
            out.append(d)
        d += timedelta(days=1)
    return out


def _bars(close: list[float], *, low: list[float] | None = None, spread: float = 10.0) -> pl.DataFrame:
    """Daily bars: open = previous close, high/low = max/min(open, close) +/- 5 points."""
    c = np.asarray(close, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + 5 * PT
    lo = np.minimum(o, c) - 5 * PT if low is None else np.asarray(low, float)
    ts = _weekdays(len(c))
    df = pl.DataFrame({"ts": ts, "open": o, "high": h, "low": lo, "close": c,
                       "spread": [spread] * len(c), "spread_max": [spread] * len(c),
                       "tick_vol": [1] * len(c)}).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    df = df.with_columns((pl.col("ts").dt.replace_time_zone("UTC") - timedelta(hours=2)).alias("ts_utc"))
    return df.select(list(BAR_COLUMNS))


def _sym(bars: pl.DataFrame, spec: InstrumentSpec | None = None) -> SymbolInput:
    ones = np.ones(bars.height)
    return SymbolInput("EURUSD", bars, spec or _spec(), ones, ones)


def _random_bars(n: int = 600, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.006, n)))
    return _bars(list(np.round(close, 5)))


# --------------------------------------------------------------------------- MQL5 loop references
def _ref_features(bars: pl.DataFrame, p: ManAhlParams) -> dict[str, list]:
    c = bars["close"].to_numpy()
    h = bars["high"].to_numpy()
    l = bars["low"].to_numpy()
    n = len(c)
    lbs = p.resolved_lookbacks()
    first = max(max(lbs), p.vol_period)
    sigma, score, er, atr = [None] * n, [None] * n, [None] * n, [None] * n
    for i in range(n):
        if i >= p.vol_period:                       # SigmaAt
            r = [math.log(c[j] / c[j - 1]) for j in range(i - p.vol_period + 1, i + 1)]
            m = len(r)
            var = (sum(x * x for x in r) - sum(r) ** 2 / m) / (m - 1)
            sigma[i] = math.sqrt(var) if var > 0 else 0.0
        if i >= first:                              # ScoreAt
            s = 0
            for L in lbs:
                ref = c[i - L]
                if p.thresh_mode == "pct":
                    chg = 100.0 * (c[i] - ref) / ref
                    s += 1 if (chg > 0 and chg >= p.pct_up) else (-1 if (chg < 0 and -chg >= p.pct_down) else 0)
                else:
                    if sigma[i] <= 0:
                        continue
                    z = math.log(c[i] / ref) / (sigma[i] * math.sqrt(L))
                    s += 1 if (z > 0 and z >= p.z_min) else (-1 if (z < 0 and -z >= p.z_min) else 0)
            score[i] = s
        k = p.er_period
        if i >= k:                                  # EffRatio
            den = sum(abs(c[j] - c[j - 1]) for j in range(i - k + 1, i + 1))
            er[i] = 100.0 * (c[i] - c[i - k]) / den if den > 0 else None
        if i >= p.atr_period:                       # iATR: SMA of true range (TR needs a prev close)
            tr = [max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1]))
                  for j in range(i - p.atr_period + 1, i + 1)]
            atr[i] = sum(tr) / p.atr_period
    return dict(sigma=sigma, score=score, er=er, atr=atr)


@pytest.mark.parametrize("p", [
    ManAhlParams(lookback_base=5, vol_period=20),
    ManAhlParams(lookback_base=5, vol_period=20, z_min=0.5),
    ManAhlParams(lookback_base=5, vol_period=20, thresh_mode="pct", pct_up=0.5, pct_down=0.3),
])
def test_features_match_mql5_loops(p):
    bars = _random_bars()
    got = mt.features(bars, p)
    ref = _ref_features(bars, p)
    for col in ("sigma", "score", "er"):
        g, r = got[col].to_list(), ref[col]
        for i, (a, b) in enumerate(zip(g, r)):
            if b is None:
                assert a is None, (col, i)
            else:
                assert a is not None and abs(a - b) < 1e-9, (col, i, a, b)
    # ATR: compare where both defined (row 0's TR has no previous close in the reference)
    g, r = got["atr"].to_list(), ref["atr"]
    for i in range(p.atr_period, len(r)):
        assert abs(g[i] - r[i]) < 1e-12


def test_score_bounds_and_er_bounds():
    f = mt.features(_random_bars(seed=3), ManAhlParams(lookback_base=5, vol_period=20))
    assert f["score"].drop_nulls().is_between(-4, 4).all()
    assert f["er"].drop_nulls().abs().max() <= 100.0 + 1e-9


@pytest.mark.parametrize("score,cur,mode,expected", [
    (4, 0.0, "scaled", 1.0), (3, 0.0, "scaled", 0.5), (2, 0.0, "scaled", 0.5), (1, 0.0, "scaled", 0.0),
    (1, 1.0, "scaled", 0.5), (1, 0.5, "scaled", 0.5), (0, 1.0, "scaled", 0.0), (-2, 1.0, "scaled", -0.5),
    (-4, 0.5, "scaled", -1.0),
    (2, 0.0, "binary_full", 0.0), (4, 0.0, "binary_full", 1.0), (1, 1.0, "binary_full", 1.0),
    (0, 1.0, "binary_full", 0.0), (-1, 1.0, "binary_full", 0.0), (-4, 1.0, "binary_full", -1.0),
    (2, 0.0, "binary_half", 1.0), (-3, 0.0, "binary_half", -1.0),
])
def test_target_fraction_table(score, cur, mode, expected):
    assert target_fraction(score, cur, mode) == expected


# --------------------------------------------------------------------------- causality
def test_source_clean():
    assert_strategy_source_clean(mt.__file__)


def test_no_lookahead_signals():
    p = ManAhlParams(lookback_base=5, vol_period=20, position_mode="binary_half", stop_mode="atr",
                     sizing_mode="fixed", er_mode="directional", er_min=20.0)
    assert_no_lookahead(lambda: ManAhlTrend(p), _random_bars(400, seed=7), n_checks=15, min_history=100)


def test_features_truncation_invariant():
    bars = _random_bars(400, seed=11)
    p = ManAhlParams(lookback_base=5, vol_period=20)
    full = mt.features(bars, p)
    for t in (100, 177, 250, 399):
        part = mt.features(bars.slice(0, t + 1), p)
        assert part.equals(full.slice(0, t + 1))


def test_simulate_matches_engine_without_stops():
    """Binary mode, no stop, fixed lots: identical fills to quantlab.engine."""
    bars = _random_bars(700, seed=5)
    p = ManAhlParams(lookback_base=5, vol_period=20, position_mode="binary_full", stop_mode="none",
                     sizing_mode="fixed", fixed_lots=1.0)
    eng = run_backtest(bars, ManAhlTrend(p).signals(bars), _spec()).trades.sort("entry_ts")
    sim = simulate([_sym(bars)], p).trades.sort("entry_ts")
    assert eng.height == sim.height > 5
    for col in ("entry_ts", "exit_ts", "entry_price", "exit_price"):
        assert eng[col].cast(sim[col].dtype).equals(sim[col]), col
    assert (eng["direction"].cast(pl.Int8) == sim["direction"]).all()


# --------------------------------------------------------------------------- hand-built scenarios
# lookback_base=1 -> lookbacks 1/3/6/12; pct mode with 0 thresholds -> each vote is the sign of
# c_t - c_{t-L}; score valid from row 12. ATR(3) on a +100-point/day ramp with 5-point wicks.
_SMALL = dict(lookback_base=1, vol_period=5, thresh_mode="pct", er_mode="off", atr_period=3,
              sizing_mode="fixed", fixed_lots=1.0)


def _ramp(n: int, start: float = 1.10000, step: float = 0.00100) -> list[float]:
    return [round(start + step * i, 5) for i in range(n)]


def test_long_entry_fill_and_stop_then_wait_after_stop():
    close = _ramp(20)
    low = None
    bars = _bars(close)
    # bar 16 crashes through any 1-ATR stop: low far below
    lows = bars["low"].to_numpy().copy()
    lows[16] = close[15] - 0.02
    bars = bars.with_columns(pl.Series("low", lows))
    p = ManAhlParams(**_SMALL, position_mode="binary_full", stop_mode="atr", stop_atr_mult=1.0)
    res = simulate([_sym(bars)], p)
    tr = res.trades.sort("entry_ts")
    first = tr.row(0, named=True)
    # score(row 12) = +4 -> long at the open of row 13 = close[12], paying the 10-point spread
    assert first["entry_ts"] == bars["ts"][13]
    assert first["entry_price"] == pytest.approx(close[12] + 10 * PT)
    # stop = bid - 1 x ATR(3) at row 12; every TR on the ramp is 100 + 10 points
    atr12 = 0.00110
    assert first["stop"] == pytest.approx(math.floor((close[12] - atr12) / PT) * PT)
    assert first["reason"] == "stop" and first["exit_ts"] == bars["ts"][16]
    assert first["exit_price"] == pytest.approx(first["stop"])        # touch, not a gap
    # score stays +4 after the stop: blocked, no re-entry (only the stop trade exists)
    assert tr.height == 1

    p2 = ManAhlParams(**_SMALL, position_mode="binary_full", stop_mode="atr", stop_atr_mult=1.0,
                      wait_after_stop=False)
    tr2 = simulate([_sym(bars)], p2).trades.sort("entry_ts")
    assert tr2.height == 2 and tr2["entry_ts"][1] == bars["ts"][17]  # re-enters next open


def test_gap_through_stop_fills_at_open():
    close = _ramp(20)
    close[16] = close[15] - 0.03          # open of row 17 = this close, far below the stop
    bars = _bars(close)
    p = ManAhlParams(**_SMALL, position_mode="binary_full", stop_mode="atr", stop_atr_mult=1.0)
    tr = simulate([_sym(bars)], p).trades.sort("entry_ts")
    stop_row = tr.filter(pl.col("reason") == "stop").row(0, named=True)
    # row 16 itself trades down through the stop intrabar (open = close[15], low = close[16]-5pt)
    assert stop_row["exit_ts"] == bars["ts"][16]
    assert stop_row["exit_price"] == pytest.approx(stop_row["stop"])


def test_scaled_mode_scales_down_on_half_signal():
    close = _ramp(16)
    close.append(round(close[-1] - 0.00050, 5))   # row 16: below c[15] only -> score +2
    close += _ramp(3, start=close[-1] + 0.00200)  # rows 17-19: new highs -> score +4 again
    bars = _bars(close)
    p = ManAhlParams(**_SMALL, position_mode="scaled", stop_mode="none")
    res = simulate([_sym(bars)], p)
    tr = res.trades.sort("exit_ts")
    down = tr.filter(pl.col("reason") == "scale_down")
    assert down.height == 1 and down["lots"][0] == pytest.approx(0.5)
    assert down["exit_ts"][0] == bars["ts"][17]          # decided on row 16's close
    pos = res.positions.sort("date")
    lots = dict(zip(pos["date"].to_list(), pos["lots"].to_list()))
    assert lots[bars["ts"][13].date()] == pytest.approx(1.0)
    assert lots[bars["ts"][17].date()] == pytest.approx(0.5)
    assert lots[bars["ts"][18].date()] == pytest.approx(1.0)   # back to full on score +4


def test_swap_points_weekday_weights():
    close = _ramp(30)
    bars = _bars(close)
    spec = _spec(swap_long=-5.0, swap_3day=2)
    p = ManAhlParams(**_SMALL, position_mode="binary_full", stop_mode="none")
    res = simulate([_sym(bars, spec)], p)
    tr = res.trades
    assert tr.height == 1 and tr["reason"][0] == "eod"
    # held from the open of row 13 to the close of row 29 (eod): nights = closes of rows 13..28
    nights = [bars["ts"][i].weekday() for i in range(13, 29)]
    weight = sum(3.0 if wd == 2 else 1.0 for wd in nights)
    expected = -5.0 * PT * 100_000 * 1.0 * weight
    assert tr["swap"][0] == pytest.approx(expected)
    assert res.daily["equity"][-1] == pytest.approx(100_000 + tr["pnl"].sum())


def test_vol_target_sizing_formula():
    bars = _random_bars(400, seed=9)
    p = ManAhlParams(lookback_base=5, vol_period=20, stop_mode="none", er_mode="off",
                     position_mode="binary_half", portfolio_vol_pct=10.0, max_leverage=100.0)
    res = simulate([_sym(bars)], p)
    first = res.trades.sort("entry_ts").row(0, named=True)
    t = bars["ts"].to_list().index(first["entry_ts"])
    sigma = mt.features(bars, p)["sigma"][t - 1]
    entry = first["entry_price"]
    raw = 100_000 * 0.10 / (100_000 * entry * sigma * math.sqrt(260))
    assert first["lots"] == pytest.approx(math.floor(raw / 0.01 + 1e-9) * 0.01)


def test_positions_keep_fractional_lots():
    """Regression: schema inference from the leading flat rows once typed lots as Int64 (0.47 -> 0)."""
    bars = _random_bars(500)
    res = simulate([_sym(bars)], ManAhlParams(lookback_base=5, vol_period=20))
    assert res.positions.schema["lots"] == pl.Float64
    nz = res.positions.filter(pl.col("lots") != 0)
    assert nz.height > 0 and (nz["lots"].abs() < 1).any()


def test_per_symbol_cost_override():
    """SymbolInput.cost overrides the simulate()-level cost model for that symbol only."""
    from dataclasses import replace
    from quantlab.costs import CostModel
    bars = _random_bars(500)
    p = ManAhlParams(lookback_base=5, vol_period=20)
    a = simulate([replace(_sym(bars), cost=CostModel.frictionless())], p, CostModel())
    b = simulate([_sym(bars)], p, CostModel.frictionless())
    assert a.daily["equity"].equals(b.daily["equity"])


def test_params_validation():
    with pytest.raises(ValueError):
        ManAhlParams(sizing_mode="risk", stop_mode="none").validate()
    with pytest.raises(ValueError):
        ManAhlParams(er_mode="sometimes").validate()
