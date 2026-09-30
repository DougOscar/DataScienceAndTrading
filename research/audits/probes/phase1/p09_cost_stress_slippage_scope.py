"""P09 — What does the cost-stress gate's '+1 pt slippage' actually do?
engine applies slippage_points only to STOP fills (slip_price in _chk_long/_chk_short); market
entries at next-bar open and signal exits get none.  1 point = instrument.point (EURUSD 1e-5 =
0.1 pip).  Compare per-trade fill prices / Sharpe for base vs slippage +1 / +10 pt, with and
without a stop (no-stop = risk type C style: stop_dist null)."""
import sys, warnings
from dataclasses import replace
warnings.simplefilter("ignore")
import polars as pl
sys.path.insert(0, "/home/douglaso/Finances/DataScienceAndTrading/research/audits/probes")
from sma_strategy import SmaCross, SmaParams
from quantlab import evaluators
from quantlab.costs import CostModel
from quantlab.contracts import RiskType

class Sma(SmaCross):
    params_cls = SmaParams

class SmaNoStop(Sma):
    risk_type = RiskType.C
    def signals(self, bars):
        return super().signals(bars).with_columns(pl.lit(None, pl.Float64).alias("stop_dist"))

base = CostModel()
for cls in (Sma, SmaNoStop):
    ev = evaluators.RuleEvaluator(cls, "EURUSD", "H4", start="2022-01-04", end="2024-07-01")
    print(f"{cls.__name__}: point={ev._data().spec.point}")
    for name, c in (("base", base), ("slip+1pt", replace(base, slippage_points=1.0)),
                    ("slip+10pt", replace(base, slippage_points=10.0)), ("stressed()", base.stressed()),
                    ("spread x1.5 only", replace(base, spread_multiplier=1.5))):
        o = ev({"fast": 20, "slow": 50}, cost=c)
        t = o.trades
        print(f"   {name:18s} sharpe {o.metrics['sharpe']:+.4f}  sum pnl {t['pnl_ccy'].sum():10.2f}  trades {t.height}")
