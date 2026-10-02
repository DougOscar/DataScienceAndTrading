"""S2 code red-team probe: independent loop reference vs toy_tsmom signals on real dev EURUSD H4.
Counts ROC==0 bars, same-direction consecutive crosses and the time-exit clock restarts they cause."""
from datetime import datetime
import numpy as np, polars as pl
from quantlab import data
from quantlab.strategies.toy_tsmom import ToyTsmom, ToyTsmomParams

bars = data.load_bars("EURUSD", "H4", book="FBS", end=datetime(2025, 5, 15))
c = bars["close"].to_numpy(); h = bars["high"].to_numpy(); l = bars["low"].to_numpy(); n = len(c)

def ref_wilder(period=14):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.where(np.isnan(pc), h - l, np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)]))
    a = np.full(n, np.nan); a[period-1] = tr[:period].mean()
    for i in range(period, n): a[i] = (a[i-1]*(period-1) + tr[i]) / period
    sma = np.full(n, np.nan)
    cs = np.cumsum(tr)
    sma[period-1:] = (cs[period-1:] - np.r_[0, cs[:-period]]) / period
    return a, sma, tr

def ref_signals(lb, hold):
    """Card-literal loop, ignoring stops (not observable at signal level)."""
    roc = np.full(n, np.nan); roc[lb:] = c[lb:] / c[:-lb] - 1
    sig = [None]*n; pos = 0; start = None; restarts = 0
    for t in range(lb+1, n):
        if t < 13: continue
        rp, r = roc[t-1], roc[t]
        x = 1 if (rp <= 0 and r > 0) else (-1 if (rp >= 0 and r < 0) else 0)
        if x != 0:
            if x == pos: restarts += 1       # same-direction cross while (assumed) in position: card says no new clock
            else: pos = x; start = t
            sig[t] = x; continue
        if pos != 0 and start is not None and t - start == hold:
            sig[t] = 0; pos = 0
    return sig, roc, restarts

a, sma, tr = ref_wilder()
code_atr = ToyTsmom(ToyTsmomParams()).signals(bars)["stop_dist"].to_numpy() / 2.5
m = ~np.isnan(a)
print("Wilder ref vs code max|diff|:", np.nanmax(np.abs(code_atr[m] - a[m])))
rel = np.abs(a[m][200:] - sma[m][200:]) / a[m][200:]
print(f"Wilder vs SMA-ATR (MT5 iATR) rel diff: median {np.median(rel):.3%}, p95 {np.quantile(rel,0.95):.3%}, max {rel.max():.2%}")
print(f"median ATR14 pips {np.nanmedian(a)*1e4:.1f}; median TR {np.median(tr)*1e4:.1f}")

for lb in [10, 30, 60, 90, 120]:
    for hold in [6, 30, 60]:
        s = ToyTsmom(ToyTsmomParams(lookback=lb, stop_mult=2.5, hold=hold)).signals(bars)["signal"].to_list()
        r, roc, restarts = ref_signals(lb, hold)
        zeros = int(np.sum(roc[lb:] == 0))
        crosses = [i for i, v in enumerate(s) if v in (1, -1)]
        same = sum(1 for a_, b_ in zip(crosses, crosses[1:]) if s[a_] == s[b_])
        mism = [i for i in range(n) if s[i] != r[i]]
        print(f"lb={lb:3d} hold={hold:2d}: ROC==0 bars {zeros:3d}, crosses {len(crosses)} ({len(crosses)/9.04:.0f}/yr), "
              f"same-dir consecutive {same}, restarts-in-position(ref) {restarts}, rows differing code vs ref {len(mism)}")
