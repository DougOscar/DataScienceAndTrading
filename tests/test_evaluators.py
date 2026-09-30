"""Tests for quantlab.evaluators (RuleEvaluator on real dev data + SyntheticEvaluator)."""

from __future__ import annotations

import pickle
import warnings
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl
import pytest

from quantlab import config, data, evaluators
from quantlab.contracts import Outcome, Params, RiskType
from quantlab.costs import CostModel
from quantlab.evaluators import RuleEvaluator, SyntheticEvaluator


# --------------------------------------------------------------------------- tiny test strategy
@dataclass(frozen=True)
class SmaParams(Params):
    fast: int = 10
    slow: int = 40
    atr_mult: float = 2.0


class SmaCross:
    """Minimal causal SMA-cross with an ATR stop (risk type A).  Test-only."""

    name = "sma_cross_test"
    risk_type = RiskType.A
    params_cls = SmaParams

    def __init__(self, params: SmaParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        c = pl.col("close")
        tr = pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - c.shift(1)).abs(),
                               (pl.col("low") - c.shift(1)).abs())
        atr = tr.rolling_mean(14)
        state = pl.when(c.rolling_mean(p.fast) > c.rolling_mean(p.slow)).then(1).otherwise(-1)
        return bars.select(state.alias("s"), atr.alias("atr"), c.rolling_mean(p.slow).alias("ss")).select(
            pl.when(pl.col("atr").is_null() | pl.col("ss").is_null()).then(None)
            .when(pl.col("s") != pl.col("s").shift(1)).then(pl.col("s")).otherwise(None)
            .cast(pl.Int8).alias("signal"),
            (pl.col("atr") * p.atr_mult).alias("stop_dist"),
            pl.lit(None, dtype=pl.Float64).alias("target_dist"),
        )


class SmaCrossB(SmaCross):
    risk_type = RiskType.B


class SmaCrossD(SmaCross):
    risk_type = RiskType.D


def _have_eurusd() -> bool:
    try:
        cat = data.catalog("FBS")
        return cat.filter(pl.col("symbol") == "EURUSD").height > 0
    except Exception:
        return False


real = pytest.mark.skipif(not _have_eurusd(), reason="EURUSD data not available")
START, END = "2019-01-01", "2020-01-01"


@pytest.fixture(scope="module")
def ev() -> RuleEvaluator:
    warnings.filterwarnings("ignore")
    return RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start=START, end=END)


# --------------------------------------------------------------------------- RuleEvaluator (real data)
@real
def test_rule_evaluator_outcome_and_warmup(ev):
    out = ev({"fast": 10, "slow": 40})
    assert isinstance(out, Outcome)
    d = out.daily
    assert d.columns == ["date", "ret"] and d["date"].dtype == pl.Date
    assert d["date"].is_sorted() and d["date"].n_unique() == d.height
    # returns begin at `start` (warm-up bars are only used for indicator state) …
    assert d["date"][0] >= date(2019, 1, 1) and d["date"][0] <= date(2019, 1, 3)
    assert d["date"][-1] < date(2020, 1, 1)
    # … and no trade is entered before `start`
    assert out.trades["entry_ts"].min() >= datetime(2019, 1, 1)
    for k in ("sharpe", "max_dd", "n_trades", "win_rate", "profit_factor", "expectancy_points",
              "avg_hold_bars", "hold_days_max"):
        assert k in out.metrics
    assert out.metrics["n_trades"] == out.trades.filter(~pl.col("skipped")).height > 20
    assert ev.periods_per_year == 260.0
    assert ev.requires_refit is False


@real
def test_rule_evaluator_uses_warmup_history(ev):
    """Signals are computed with history before `start`: the cached signal frame starts
    `warmup_bars` before the first evaluation bar, and a 0-bar warm-up gives a
    different (later-starting) trade list."""
    ev({"fast": 10, "slow": 40})
    d = evaluators._CACHE[ev._key()]
    assert d.offset == ev.warmup_bars
    assert d.bars_sig["ts"][d.offset] == d.bars_eval["ts"][0] >= datetime(2019, 1, 1)
    cold = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start=START, end=END, warmup_bars=0)
    a = ev({"fast": 10, "slow": 150}).trades["entry_ts"][0]
    b = cold({"fast": 10, "slow": 150}).trades["entry_ts"][0]
    assert b > a   # cold start has to wait ~150 bars for the slow SMA inside the window


@real
def test_rule_evaluator_no_leak_through_end(ev):
    """Evaluating a longer window must not change the returns inside the shorter one
    (except the last day, where the short run force-closes open positions at 'eod')."""
    short = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start=START, end="2019-07-01")
    a = short({"fast": 10, "slow": 40}).daily
    b = ev({"fast": 10, "slow": 40}).daily.filter(pl.col("date") < date(2019, 7, 1))
    assert a.height == b.height
    np.testing.assert_allclose(a["ret"].to_numpy()[:-1], b["ret"].to_numpy()[:-1], rtol=0, atol=1e-12)


@real
def test_rule_evaluator_cost_override(ev):
    base = ev({"fast": 10, "slow": 40})
    stressed = ev({"fast": 10, "slow": 40}, cost=CostModel().stressed())
    assert stressed.metrics["sharpe"] < base.metrics["sharpe"]
    assert not np.allclose(base.daily["ret"].to_numpy(), stressed.daily["ret"].to_numpy())
    again = ev({"fast": 10, "slow": 40})       # override is per call only
    np.testing.assert_array_equal(base.daily["ret"].to_numpy(), again.daily["ret"].to_numpy())


@real
def test_rule_evaluator_picklable_and_deterministic(ev):
    blob = pickle.dumps(ev)
    assert len(blob) < 5_000          # config only — no bars/M1 on the instance
    ev2 = pickle.loads(blob)
    a, b = ev({"fast": 15, "slow": 50}), ev2({"fast": 15, "slow": 50})
    assert a.daily.equals(b.daily)


@real
def test_rule_evaluator_never_requests_holdout(monkeypatch):
    calls = []
    real_load = data.load_bars

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(data, "load_bars", spy)
    evaluators.clear_cache()
    e = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start=START, end=END)
    e({"fast": 10, "slow": 40})
    assert calls, "load_bars was not called"
    hs = config.get_book("FBS").holdout_start
    for kw in calls:
        assert not kw.get("include_holdout", False)
        assert kw.get("end") is not None and kw["end"] <= hs
    with pytest.raises(ValueError, match="dev-window only"):
        RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", end="2025-06-01")
    # default end = holdout start (exclusive)
    assert RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1").end_dt == hs
    evaluators.clear_cache()


@real
def test_rule_evaluator_sizing_by_risk_type():
    assert RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1")._mode() == "fixed_fraction"
    assert RuleEvaluator(SmaCrossB, symbol="EURUSD", timeframe="H1")._mode() == "fixed_lots"
    assert RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", sizing="fixed_lots")._mode() == "fixed_lots"
    with pytest.raises(NotImplementedError):
        RuleEvaluator(SmaCrossD, symbol="EURUSD", timeframe="H1")


@real
@pytest.mark.parametrize("symbol,start", [("EURJPY", "2023-01-02"), ("EURJPY", None), ("GBPAUD", "2020-06-01")])
def test_rule_evaluator_cross_at_week_open_and_data_start(symbol, start):
    """m2 (Phase 1 red team, p08b): a non-USD-quoted cross whose first evaluation bar follows a
    weekend gap (Monday start) or is the data start used to raise in the conversion lookup."""
    warnings.filterwarnings("ignore")
    evaluators.clear_cache()
    try:
        e = RuleEvaluator(SmaCross, symbol=symbol, timeframe="D1", start=start, end="2024-07-01", use_m1=False)
        out = e({"fast": 20, "slow": 50})
        assert out.daily.height > 300 and np.isfinite(out.daily["ret"].to_numpy()).all()
        assert out.metrics["n_trades"] > 0
    finally:
        evaluators.clear_cache()


# --------------------------------------------------------------------------- SyntheticEvaluator
def test_synthetic_evaluator_ground_truth():
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, bumps=({"center": {"a": 5}, "height": 2.0, "width": 0.1},),
                            n_days=5000, rho=1.0)
    assert ev.true_sharpe({"a": 5}) == pytest.approx(2.0)
    assert ev.true_sharpe({"a": 0}) == pytest.approx(0.0, abs=1e-4)
    s = [ev({"a": a}).metrics["sharpe"] for a in range(11)]
    assert int(np.argmax(s)) == 5                    # rho=1: measured Sharpe is monotone in truth
    a1, a2 = ev({"a": 3}), pickle.loads(pickle.dumps(ev))({"a": 3})
    assert a1.daily.equals(a2.daily)
    assert a1.metrics["n_trades"] == a1.trades.height > 0
    assert ev({"a": 5}, cost=1.0).metrics["sharpe"] < ev({"a": 5}).metrics["sharpe"]
    bad = SyntheticEvaluator(bounds={"a": (0, 10)}, error_when=({"a": 2},))
    with pytest.raises(RuntimeError):
        bad({"a": 2})


def test_synthetic_evaluator_regimes():
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, bumps=({"center": {"a": 2}, "height": 2.0, "width": 0.1},),
                            regimes=({"from": 0.5, "bumps": ({"center": {"a": 8}, "height": 2.0, "width": 0.1},)},))
    assert ev.true_sharpe({"a": 2}, 0.0) == pytest.approx(2.0)
    assert ev.true_sharpe({"a": 2}, 0.7) == pytest.approx(0.0, abs=1e-6)
    assert ev.true_sharpe({"a": 8}, 0.7) == pytest.approx(2.0)


# --------------------------------------------------------------------------- L1: input-availability clamp
@real
def test_usdchf_h1_evaluates_from_dev_start_with_effective_start_recorded():
    """L1: USDCHF's M1 (and so its CHF→USD conversion) starts at 00:01 while its first H1 bar
    opens at 00:00.  The evaluation start is clamped to the first bar with every input
    available (no rate from a bar that opens after t) and the effective start is recorded."""
    warnings.filterwarnings("ignore")
    evaluators.clear_cache()
    try:
        e = RuleEvaluator(SmaCross, symbol="USDCHF", timeframe="H1", start="2016-05-02", end="2016-10-01")
        out = e({"fast": 10, "slow": 40})
        assert out.metrics["eval_start_effective"] == "2016-05-02 01:00:00"
        assert out.metrics["eval_start_bars_dropped"] == 1.0
        d = evaluators._CACHE[e._key()]
        assert d.bars_eval["ts"][0] == datetime(2016, 5, 2, 1)
        assert d.eval_start_clamp["binding"] == ["conversion_CHFUSD"]
        assert out.metrics["n_trades"] > 10 and np.isfinite(out.daily["ret"].to_numpy()).all()
        assert out.trades["entry_ts"].min() >= datetime(2016, 5, 2, 1)
        assert e.eval_start_effective() == "2016-05-02 01:00:00"
        # the dropped bar would have needed a CHF rate before any CHF series had opened
        assert data.conversion_available_from("CHF", "USD", book="FBS") == datetime(2016, 5, 2, 0, 1)
        # a symbol whose inputs all exist at its first bar is not clamped
        eu = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start="2016-05-02", end="2016-07-01")
        o2 = eu({"fast": 10, "slow": 40})
        assert o2.metrics["eval_start_bars_dropped"] == 0.0
        assert o2.metrics["eval_start_effective"] == "2016-05-02 00:00:00"
    finally:
        evaluators.clear_cache()


@real
def test_usdchf_h1_study_records_eval_start_effective(tmp_path):
    from quantlab import ledger, opt
    warnings.filterwarnings("ignore")
    evaluators.clear_cache()
    try:
        e = RuleEvaluator(SmaCross, symbol="USDCHF", timeframe="H1", start="2016-05-02", end="2017-01-01")
        sp = opt.SearchSpace([opt.IntParam("fast", 10, 20, 10, plateau_scale="relative")])
        res = opt.run_study(e, sp, book="FBS", system="l1", issue=1, attempt=1, study_id="l1-usdchf",
                            n_jobs=1, wfo=None, min_trades=1, ledger_dir=tmp_path / "ledger",
                            studies_dir=tmp_path / "studies")
        assert res.meta["n_error"] == 0 and res.meta["n_ok"] == 2
        assert res.meta["eval_start_effective"] == "2016-05-02 01:00:00"
        assert (res.trials["m_eval_start_bars_dropped"] == 1.0).all()
        sel = [ev for ev in ledger.study_events("l1-usdchf", ledger_dir=tmp_path / "ledger")
               if ev.get("event") == "selection"][0]
        assert sel["eval_start_effective"] == "2016-05-02 01:00:00"
    finally:
        evaluators.clear_cache()


@real
def test_data_load_error_is_cached_not_reloaded(monkeypatch):
    """L1 perf: a failing data load is cached per process, so later trials re-raise at once
    instead of reloading (M1) data per trial."""
    calls = []
    real_load = data.load_bars

    def spy(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(data, "load_bars", spy)
    evaluators.clear_cache()
    try:
        e = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start="2030-01-01", end="2020-01-01")
        with pytest.raises(ValueError, match="no EURUSD H1 bars"):
            e({"fast": 10, "slow": 40})
        n = len(calls)
        assert n >= 1
        with pytest.raises(ValueError, match="no EURUSD H1 bars"):
            e({"fast": 20, "slow": 40})
        assert len(calls) == n
    finally:
        evaluators.clear_cache()


# --------------------------------------------------------------------------- L4: hold in rows, data gaps
def test_trade_stats_hold_in_rows_across_a_data_gap():
    # 5 weekdays, a 30-weekday hole, then 5 weekdays
    d0 = np.busday_offset(np.datetime64("2016-06-20"), np.arange(5))
    d1 = np.busday_offset(np.datetime64("2016-06-20"), np.arange(36, 41))
    dates = np.concatenate([d0, d1])
    assert evaluators.data_gaps(dates) == [(str(d0[-1]), str(d1[0]), 31)]
    tr = pl.DataFrame({
        "entry_ts": [datetime(2016, 6, 23, 10), datetime(2016, 6, 20, 10)],
        "exit_ts": [datetime(2016, 8, 10, 10), datetime(2016, 6, 21, 10)],   # 1st straddles the gap
        "pnl_points": [1.0, -1.0], "pnl_ccy": [1.0, -1.0], "bars_held": [10, 2], "skipped": [False, False],
    })
    assert str(d1[0]) == "2016-08-09"
    s = evaluators.trade_stats(tr, dates=dates)
    # 2016-06-23 is row 3, 2016-08-10 is row 6 → 3 rows, although 48 calendar days
    assert s["hold_days_max"] == 3.0
    assert s["hold_calendar_days_max"] == 48.0
    assert s["trades_across_data_gap"] == 1.0
    legacy = evaluators.trade_stats(tr)                  # no index → weekdays (old behaviour)
    assert legacy["hold_days_max"] == 34.0 and "trades_across_data_gap" not in legacy


class GapHold:
    """Test-only (risk type A): long from the first bar on/after ``enter`` until the first bar
    on/after ``exit_day`` — used to straddle the 2016 XAUUSD M1 hole."""

    name = "gap_hold_test"
    risk_type = RiskType.A

    def __init__(self, exit_day: int = 10, enter: str = "2016-06-20"):
        self.exit_day, self.enter = exit_day, datetime.fromisoformat(enter)

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        ts = pl.col("ts")
        ex = datetime(2016, 11, self.exit_day)
        on = (ts >= self.enter) & (ts < ex)
        return bars.select(
            pl.when(on & ~on.shift(1, fill_value=False)).then(1)
            .when(~on & on.shift(1, fill_value=False)).then(0)
            .otherwise(None).cast(pl.Int8).alias("signal"),
            pl.lit(500.0).alias("stop_dist"),
            pl.lit(None, dtype=pl.Float64).alias("target_dist"),
        )


def _have_xau() -> bool:
    try:
        return data.catalog("FBS").filter(pl.col("symbol") == "XAUUSD").height > 0
    except Exception:
        return False


@pytest.mark.skipif(not _have_xau(), reason="XAUUSD data not available")
def test_xauusd_h4_gap_trade_does_not_cap_the_embargo(tmp_path):
    """L4: XAUUSD M1 has a 133-day hole (2016-06-24 → 2016-11-04).  A trade across it used to
    set m_hold_days_max ≈ 95 weekdays → auto embargo capped → CPCV gates skipped.  Holding is
    now measured in rows of the daily index, and the trade is flagged as gap-straddling."""
    from quantlab import opt
    warnings.filterwarnings("ignore")
    evaluators.clear_cache()
    try:
        # 3 dev years so a CPCV group (≈ 60 rows) is wide enough that the cap (¼ group) is not
        # binding for any honest short-hold system
        e = RuleEvaluator(GapHold, symbol="XAUUSD", timeframe="H4", start="2016-05-02", end="2019-05-01")
        out = e({"exit_day": 10})
        m = out.metrics
        assert m["n_trades"] == 1 and m["trades_across_data_gap"] == 1.0
        assert m["hold_calendar_days_max"] > 130
        assert m["hold_days_max"] <= 10
        sp = opt.SearchSpace([opt.IntParam("exit_day", 8, 10, 1, plateau_scale="relative")])
        res = opt.run_study(e, sp, book="FBS", system="l4", issue=1, attempt=1, study_id="l4-xau",
                            n_jobs=1, wfo=None, min_trades=1, cv=opt.CPCVConfig(10, 2),
                            ledger_dir=tmp_path / "ledger", studies_dir=tmp_path / "studies")
        assert res.meta["embargo_capped"] is False, res.meta["cpcv"]
        assert res.meta["cpcv"]["embargo_days"] <= 10
        assert res.meta["m_trades_across_data_gap"] == 1
        assert any(g[0] <= "2016-06-24" and g[1] >= "2016-11-04" for g in res.meta["data_gaps"])
    finally:
        evaluators.clear_cache()
