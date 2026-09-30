"""R1-P09 (M2) — cost stress after the fix: slippage on every market fill, 1 pip via spec.
(1) EURUSD H4 SMA 20/50, with ATR stop and with no stop (type C): base / stressed() (no spec: 1 pt) /
    stressed(spec=) (1 pip = 10 pt) / spread x1.5 only.  The no-stop row must now move with slippage.
(2) The gate's cost-stress row on a RuleEvaluator (spec resolved via load_instrument).
(3) Per-symbol stress slippage vs the median bar spread (dev data 2024-01..06): where 1 'pip'
    is a no-op or dominates the spread."""
import sys, warnings
from dataclasses import replace
warnings.simplefilter("ignore")
import polars as pl
sys.path.insert(0, "/home/douglaso/Finances/DataScienceAndTrading/research/audits/probes")
from sma_strategy import SmaCross, SmaParams
from quantlab import evaluators, data
from quantlab.costs import CostModel, load_instrument, pip_points
from quantlab.contracts import RiskType

class Sma(SmaCross):
    params_cls = SmaParams

class SmaNoStop(Sma):
    risk_type = RiskType.C
    def signals(self, bars):
        return super().signals(bars).with_columns(pl.lit(None, pl.Float64).alias("stop_dist"))

base = CostModel()
spec = load_instrument("EURUSD")
for cls in (Sma, SmaNoStop):
    ev = evaluators.RuleEvaluator(cls, "EURUSD", "H4", start="2022-01-04", end="2024-07-01")
    print(f"{cls.__name__}:")
    for name, c in (("base", base), ("stressed() no spec (1pt)", base.stressed()),
                    ("stressed(spec) (1 pip=10pt)", base.stressed(spec=spec)),
                    ("spread x1.5 only", replace(base, spread_multiplier=1.5)),
                    ("slip 10pt only", replace(base, slippage_points=10.0))):
        o = ev({"fast": 20, "slow": 50}, cost=c)
        t = o.trades
        print(f"   {name:28s} sharpe {o.metrics['sharpe']:+.4f}  sum pnl {t['pnl_ccy'].sum():10.2f}  trades {t.height}")

print("(3) stress slippage per fill vs median D1 spread (2024-01..06)")
for s, b in (("EURUSD", "FBS"), ("USDJPY", "FBS"), ("XAUUSD", "FBS"), ("XAGUSD", "FBS"), ("BTCUSD", "FBS"),
             ("ETHUSD", "FBS"), ("WIN", "B3"), ("WDO", "B3")):
    sp = load_instrument(s, book=b)
    try:
        bars = data.load_bars(s, "D1", book=b, start="2024-01-02", end="2024-06-01")
        spr = float(bars["spread"].median()); px = float(bars["close"].median())
    except Exception as e:
        spr, px = float("nan"), float("nan")
    pp = pip_points(sp)
    print(f"   {s:7s} point={sp.point:g} tick={getattr(sp, 'tick_size', None)} 1 pip={pp:g} pts = {pp * sp.point:g} price "
          f"({pp * sp.point / px * 1e4 if px == px else float('nan'):.3f} bp of price); median spread {spr:g} pts -> "
          f"slippage/spread = {pp / spr if spr == spr and spr else float('nan'):.3f}; slippage/tick = "
          f"{pp * sp.point / sp.tick_size if getattr(sp, 'tick_size', None) else float('nan'):.3f}")
