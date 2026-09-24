"""Probe 08: independent hand-check of 5 trades from p07 (far_target=True, m1) against RAW M1
parquet rows read with pyarrow/numpy -- no quantlab code in the verification path."""
import json
from datetime import datetime, timedelta
import numpy as np
import pyarrow.parquet as pq

man = json.load(open("data/manifest.json"))
path = [k for k, v in man.items() if v["symbol"] == "EURUSD" and v["timeframe"] == "M1"][0]
t = pq.read_table(path, columns=["ts", "open", "high", "low", "close", "spread"],
                  filters=[("ts", ">=", datetime(2018, 12, 1)), ("ts", "<", datetime(2020, 1, 1))]).to_pandas()
t = t.sort_values("ts").reset_index(drop=True)
ts = t["ts"].values.astype("datetime64[m]")
P = 1e-5

# independent H1 build (floor to hour) and SMA/ATR on the 2019 window exactly as load_bars(start=2019-01-01)
t19 = t[t["ts"] >= datetime(2019, 1, 1)].copy()
t19["h"] = t19["ts"].dt.floor("h")
H = t19.groupby("h").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                         close=("close", "last"), spread=("spread", "first")).reset_index()
c = H["close"].to_numpy(); hi = H["high"].to_numpy(); lo = H["low"].to_numpy()
f = H["close"].rolling(20).mean().to_numpy(); s = H["close"].rolling(50).mean().to_numpy()
pc = np.r_[np.nan, c[:-1]]
tr = np.nanmax(np.c_[hi - lo, np.abs(hi - pc), np.abs(lo - pc)], axis=1); tr[0] = hi[0] - lo[0]
atr = H.assign(tr=tr)["tr"].rolling(14).mean().to_numpy()
Hts = H["h"].to_numpy().astype("datetime64[m]")

def hidx(x): return int(np.where(Hts == np.datetime64(x, "m"))[0][0])

def check(entry, direction, expected):
    j = hidx(entry); i = j - 1
    above = f[i] > s[i]; above_prev = f[i - 1] > s[i - 1]
    cross = (above and not above_prev) if direction == 1 else (not above and above_prev)
    sd = 2 * atr[i]
    k0 = int(np.searchsorted(ts, np.datetime64(entry, "m")))
    o, spr = t["open"][k0], t["spread"][k0]
    ep = o + spr * P if direction == 1 else o
    sl = ep - sd if direction == 1 else ep + sd
    print(f"\n--- {entry} dir={direction:+d}: signal bar {Hts[i]} fast={f[i]:.6f} slow={s[i]:.6f} prev fast={f[i-1]:.6f} prev slow={s[i-1]:.6f} cross={cross}")
    print(f"  ATR14@{Hts[i]}={atr[i]:.7f} -> stop_dist={sd:.7f}")
    print(f"  entry M1 {ts[k0]} open={o} spread={spr} -> fill {'Ask' if direction==1 else 'Bid'} = {ep:.5f}; stop = {sl:.7f}")
    # walk M1 forward: stop check vs opposite cross signal
    k = k0
    while True:
        cur_h = ts[k].astype("datetime64[h]").astype("datetime64[m]")
        jh = hidx(cur_h)
        if direction == 1 and t["low"][k] <= sl:
            gap = t["open"][k] <= sl
            px = t["open"][k] if gap else sl
            reason = "gap_stop" if gap else "stop"; break
        if direction == -1 and t["high"][k] + t["spread"][k] * P >= sl:
            gap = t["open"][k] + t["spread"][k] * P >= sl
            px = (t["open"][k] + t["spread"][k] * P) if gap else sl
            reason = "gap_stop" if gap else "stop"; break
        # next M1 starts a new H1 bar -> check opposite signal on the just-closed H1 bar
        nxt = ts[k + 1].astype("datetime64[h]").astype("datetime64[m]")
        if nxt != cur_h:
            a = f[jh] > s[jh]; ap = f[jh - 1] > s[jh - 1]
            opp = (not a and ap) if direction == 1 else (a and not ap)
            if opp:
                ko = k + 1
                px = t["open"][ko] if direction == 1 else t["open"][ko] + t["spread"][ko] * P
                reason = "signal"; k = ko; break
        k += 1
    pnl = (px - ep) / P if direction == 1 else (ep - px) / P
    print(f"  exit at M1 {ts[k]} (H1 {ts[k].astype('datetime64[h]')}) reason={reason} price={px:.7f} "
          f"(M1 o/h/l/spr = {t['open'][k]}/{t['high'][k]}/{t['low'][k]}/{t['spread'][k]})")
    print(f"  pnl_points = {pnl:.6f}")
    print(f"  ENGINE: {expected}")

import csv
rows = list(csv.DictReader(open("research/audits/probes/r2_p07_trades_m1.csv")))
pick = ["2019-01-04T07:00:00.000000", "2019-01-09T02:00:00.000000", "2019-01-11T09:00:00.000000",
        "2019-01-25T23:00:00.000000", "2019-02-15T09:00:00.000000", "2019-04-05T00:00:00.000000"]
for r in rows:
    if r["entry_ts"] in pick:
        check(r["entry_ts"][:16], int(r["direction"]),
              {k: r[k] for k in ("exit_ts", "entry_price", "exit_price", "stop_price", "exit_reason", "pnl_points", "nights", "mae_points", "mfe_points")})
