"""P08b — RuleEvaluator on a non-USD-quoted symbol fails when the first evaluation bar opens
right after a gap > 5 min (week open, or data start): data._asof_closes loads the cross only from
min(ts) - 5 min, so the backward as-of has nothing to match for the first timestamp."""
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, "/home/douglaso/Finances/DataScienceAndTrading/research/audits/probes")
from sma_strategy import SmaCross, SmaParams
from quantlab import evaluators

class Sma(SmaCross):
    params_cls = SmaParams

for sym, tf, start in (("EURJPY", "D1", "2023-01-02"), ("EURJPY", "D1", None), ("EURJPY", "H4", "2023-01-03"),
                       ("GBPAUD", "D1", "2020-06-01")):
    evaluators.clear_cache()
    ev = evaluators.RuleEvaluator(Sma, sym, tf, start=start, end="2024-07-01", use_m1=False)
    try:
        o = ev({"fast": 20, "slow": 50})
        print(f"{sym} {tf} start={start}: OK ({o.daily.height} days)")
    except Exception as e:
        print(f"{sym} {tf} start={start}: {type(e).__name__}: {str(e)[:110]}")
