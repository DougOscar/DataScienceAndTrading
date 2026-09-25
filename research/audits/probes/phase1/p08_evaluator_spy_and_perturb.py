"""P08 — RuleEvaluator: holdout spy, future-perturbation (look-ahead) test, warm-up boundary,
sizing mode by risk type, currency conversion source.

Runs EURJPY H4 (JPY quote -> USD conversion exercised) with the probe SMA/ATR strategy on
2023-01-03 .. 2024-07-01 (dev only; a Monday-00:00 start raises in conversion_rate, see p08b), warm-up 500 bars.
 1) spy on data.load_bars / data.conversion_rate: every call's end <= holdout_start, never
    include_holdout=True;
 2) scale every bar (HTF and M1) by 1.07 and the conversion rate by 1.5 for ts >= 2024-01-02 and check
    that daily returns strictly before 2024-01-02 are bit-identical;
 3) engine starts flat: no trade entry before the first evaluation bar; first daily ret = 0
    unless a trade fills that day;
 4) sizing mode per risk type.
"""
import sys
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import polars as pl

sys.path.insert(0, "/home/douglaso/Finances/DataScienceAndTrading/research/audits/probes")
from sma_strategy import SmaCross, SmaParams  # noqa: E402
from quantlab import data, evaluators, config  # noqa: E402
from quantlab.contracts import RiskType  # noqa: E402


class Sma(SmaCross):
    params_cls = SmaParams


class SmaB(Sma):
    risk_type = RiskType.B


HS = config.get_book("FBS").holdout_start
calls = []
_lb, _cr = data.load_bars, data.conversion_rate


def spy_lb(*a, **k):
    calls.append(("load_bars", a, k))
    return _lb(*a, **k)


def spy_cr(*a, **k):
    calls.append(("conversion_rate", a[:2], {kk: v for kk, v in k.items()}))
    return _cr(*a, **k)


data.load_bars, data.conversion_rate = spy_lb, spy_cr
CUT = datetime(2024, 1, 2)


def make(**kw):
    return evaluators.RuleEvaluator(Sma, "EURJPY", "H4", start="2023-01-03", end="2024-07-01", **kw)


ev = make()
out = ev({"fast": 20, "slow": 50})
print("1) spy calls:")
bad = 0
for name, a, k in calls:
    end = k.get("end")
    inc = k.get("include_holdout", False)
    print(f"   {name} args={a} end={end} include_holdout={inc}")
    if inc or (end is not None and end > HS):
        bad += 1
print(f"   violations: {bad}")

# 2) perturbation
evaluators.clear_cache()
rng = np.random.default_rng(0)


def pert_lb(symbol, timeframe="M1", **k):
    b = _lb(symbol, timeframe, **k)
    m = (b["ts"] >= CUT).to_numpy()
    f = np.where(m, 1.07, 1.0)   # constant scale keeps HTF == resample(M1) consistent
    o = b["open"].to_numpy() * f
    c = b["close"].to_numpy() * f
    h = np.maximum(b["high"].to_numpy() * f, np.maximum(o, c))
    l = np.minimum(b["low"].to_numpy() * f, np.minimum(o, c))
    return b.with_columns(pl.Series("open", o), pl.Series("high", h), pl.Series("low", l), pl.Series("close", c))


def pert_cr(ccy, acct, ts, **k):
    s = _cr(ccy, acct, ts, **k)
    m = (ts.dt.replace_time_zone(None) >= CUT).to_numpy() if ts.dtype.time_zone else (ts >= CUT).to_numpy()
    return s * np.where(m, 1.5, 1.0)


data.load_bars, data.conversion_rate = pert_lb, pert_cr
out_p = make()({"fast": 20, "slow": 50})
data.load_bars, data.conversion_rate = spy_lb, spy_cr
a = out.daily.filter(pl.col("date") < CUT.date())
b = out_p.daily.filter(pl.col("date") < CUT.date())
same = a.equals(b)
diff_after = not out.daily.filter(pl.col("date") >= CUT.date()).equals(out_p.daily.filter(pl.col("date") >= CUT.date()))
print(f"2) pre-cut daily identical after perturbing the future: {same} (rows {a.height}); post-cut changed: {diff_after}")
if not same:
    j = a.join(b, on="date", suffix="_p").filter(pl.col("ret") != pl.col("ret_p"))
    print(j.head(10))
# trades closed before the cut must be identical too
ta = out.trades.filter(pl.col("exit_ts") < CUT).select("entry_ts", "exit_ts", "pnl_ccy")
tb = out_p.trades.filter(pl.col("exit_ts") < CUT).select("entry_ts", "exit_ts", "pnl_ccy")
print(f"   trades closed pre-cut identical: {ta.equals(tb)} ({ta.height} trades)")

# 3) warm-up / flat start
t = out.trades
first_eval = datetime(2023, 1, 3)
print(f"3) first daily row {out.daily['date'][0]} ret={out.daily['ret'][0]:.6g}; earliest entry {t['entry_ts'].min()}; "
      f"entries before eval start: {t.filter(pl.col('entry_ts') < first_eval).height}")

# 4) sizing modes
print(f"4) risk type A -> {make()._mode()}, B -> {evaluators.RuleEvaluator(SmaB, 'EURJPY', 'H4', start='2023-01-02', end='2024-07-01')._mode()}")
print(f"   metrics sharpe {out.metrics['sharpe']:.3f}, n_trades {out.metrics['n_trades']:.0f}, hold_days_max {out.metrics['hold_days_max']}")
