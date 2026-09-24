"""Probe 14: magnitude of the naive-entry_ts -> conversion_rate look-ahead on a USDJPY run (USD account)."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import polars as pl
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing
from sma_strategy import SmaCross, SmaParams

spec = load_instrument("USDJPY")
bars = data.load_bars("USDJPY", "H1", start="2019-01-01", end="2021-01-01")
tr = run_backtest(bars, SmaCross(SmaParams(far_target=True)).signals(bars), spec).trades
naive_fn = lambda q, a, ts: data.conversion_rate(q, a, ts)                                   # what apply_sizing hands over
aware_fn = lambda q, a, ts: data.conversion_rate(q, a, ts.dt.replace_time_zone("Europe/Helsinki", ambiguous="earliest").dt.convert_time_zone("UTC"))
s_n = apply_sizing(tr, spec, mode="fixed_fraction", risk_fraction=0.01, rate_fn=naive_fn)
s_a = apply_sizing(tr, spec, mode="fixed_fraction", risk_fraction=0.01, rate_fn=aware_fn)
print(f"trades={tr.height}; final equity naive-ts={s_n['equity_after'][-1]:.2f} aware-ts={s_a['equity_after'][-1]:.2f}; "
      f"lots differ on {(s_n['lots'] != s_a['lots']).sum()} trades; risk_realisation_error>0 (over-risk vs entry-time rate) "
      f"when recomputed at true entry rate: {( (s_n['lots']*(s_n['stop_price'] - s_n['entry_price']).abs()/spec.point*s_a['value_per_point_acct']) > s_n['risk_target_ccy']*1.0000001).sum()}")
