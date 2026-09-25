"""R2-P09 (48e696f) -- N6 / M2 re-run: per-asset-class stress slippage (stress_slippage_points) vs the
median D1 spread (dev data 2024-01..06 only), plus: unknown class raises? gate display text."""
import sys, warnings
warnings.simplefilter("ignore")
from quantlab import data
from quantlab.costs import CostModel, load_instrument, pip_points, stress_slippage_points
from dataclasses import replace

print("symbol  point  tick   stress_pts  = price   bp_of_px  med_spread_pts  slip/spread  slip/tick  pip_points(display)")
for s, b in (("EURUSD", "FBS"), ("USDJPY", "FBS"), ("XAUUSD", "FBS"), ("XAGUSD", "FBS"), ("BTCUSD", "FBS"),
             ("ETHUSD", "FBS"), ("WIN", "B3"), ("WDO", "B3")):
    sp = load_instrument(s, book=b)
    bars = data.load_bars(s, "D1", book=b, start="2024-01-02", end="2024-06-01")
    spr = float(bars["spread"].median()); px = float(bars["close"].median())
    try:
        sl = stress_slippage_points(sp, book=b)
    except Exception as e:
        print(f"{s:7s} raises {type(e).__name__}: {str(e)[:80]}"); continue
    st = CostModel().stressed(spec=sp, book=b)
    print(f"{s:7s} {sp.point:<6g} {sp.tick_size:<6g} {sl:10.4g}  {sl * sp.point:<8.4g} {sl * sp.point / px * 1e4:8.3f}  "
          f"{spr:14g}  {sl / spr:11.3f}  {sl * sp.point / sp.tick_size:9.2f}  {pip_points(sp):g}   "
          f"(stressed().slippage_points={st.slippage_points:g})")
try:
    fake = replace(load_instrument("EURUSD"), symbol="ZZZUNKNOWN")
    print("unknown symbol:", stress_slippage_points(fake, median_spread_points=None))
except Exception as e:
    print("unknown symbol ->", type(e).__name__, str(e)[:100])
print("no spec: CostModel().stressed().slippage_points =", CostModel().stressed().slippage_points)
