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
