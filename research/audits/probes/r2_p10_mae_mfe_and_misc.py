"""Probe 10: MAE/MFE include price action AFTER the exit inside the exit bar; short MAE is measured
on Bid while its stop triggers on Ask; nights for crypto weekends; eod short uses the last bar's OPEN spread."""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from datetime import datetime
import polars as pl
from quantlab.engine import run_backtest, _nights_weighted
from quantlab.costs import InstrumentSpec

spec = InstrumentSpec(symbol="SYN", digits=5, point=1e-5, contract_size=1e5, tick_size=1e-5, volume_min=0.01,
                      volume_step=0.01, volume_max=100, base_ccy="EUR", quote_ccy="USD", swap_mode="points",
                      swap_long=0, swap_short=0)
def mk(rows, t0=datetime(2020, 1, 6)):
    from datetime import timedelta
    df = pl.DataFrame(rows, schema=["open", "high", "low", "close"], orient="row")
    n = df.height
    return df.with_columns(pl.Series("ts", [t0 + timedelta(hours=i) for i in range(n)]).cast(pl.Datetime("ms")),
                           pl.lit(10.0).alias("spread"), pl.lit(10.0).alias("spread_max"), pl.lit(1, pl.Int64).alias("tick_vol")
                           ).with_columns(pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc")).select(
        "ts", "ts_utc", "open", "high", "low", "close", "spread", "spread_max", "tick_vol")

# (a) long; bar 1 = entry bar (quiet); bar 2: stop at 1.0990 hit, then a rally to 1.1100 in the same bar
bars = mk([(1.1, 1.1001, 1.0999, 1.1), (1.1, 1.1002, 1.0998, 1.1), (1.1, 1.1100, 1.0980, 1.1090), (1.109, 1.109, 1.109, 1.109)])
sig = pl.DataFrame({"signal": [1, None, None, None], "stop_dist": [0.0011] * 4, "target_dist": [None] * 4},
                   schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
m1_rows, m1_ts = [], []
from datetime import timedelta
for j, (o, h, l, c) in enumerate(bars.select("open", "high", "low", "close").rows()):
    base = bars["ts"][j]
    if j == 2:   # stop first (minute 0-1), rally afterwards
        path = [(1.1, 1.1, 1.0980, 1.0985), (1.0985, 1.1100, 1.0985, 1.1090)]
    else:
        path = [(o, h, l, c)]
    for k, p in enumerate(path):
        m1_rows.append(p); m1_ts.append(base + timedelta(minutes=k))
m1 = pl.DataFrame(m1_rows, schema=["open", "high", "low", "close"], orient="row").with_columns(
    pl.Series("ts", m1_ts).cast(pl.Datetime("ms")), pl.lit(10.0).alias("spread"), pl.lit(10.0).alias("spread_max"),
    pl.lit(1, pl.Int64).alias("tick_vol")).with_columns(pl.col("ts").dt.replace_time_zone("UTC").alias("ts_utc")).select(bars.columns)
t = run_backtest(bars, sig, spec, m1=m1, timeframe="H1").trades.row(0, named=True)
print(f"(a) long stopped at {t['exit_price']:.5f} ({t['exit_reason']}), pnl={t['pnl_points']:.0f} pts, but mfe_points={t['mfe_points']:.0f}"
      f" (post-exit rally to 1.1100 counted) and mae_points={t['mae_points']:.0f} vs realised loss {-t['pnl_points']:.0f}")

# (b) real data: how often does MAE exceed the realised loss on stop exits by > 20%?
tr = pl.read_csv(os.path.join(os.path.dirname(__file__), "r2_p07_trades_m1.csv"))
st = tr.filter(pl.col("exit_reason") == "stop")
ratio = st["mae_points"] / (-st["pnl_points"])
print(f"(b) EURUSD H1 2019 stop exits: {st.height}; MAE > 1.2x realised loss in {(ratio > 1.2).sum()} "
      f"({(ratio > 1.2).mean()*100:.0f}%), median MAE/loss={ratio.median():.2f}, max={ratio.max():.2f}")
sig_ex = tr.filter(pl.col("exit_reason") == "signal")
print(f"    signal exits: {sig_ex.height} (exit at open of bar -> no post-exit contamination on those)")

# (c) nights: crypto weekend hold (Fri 12:00 -> Mon 12:00) and a Sat/Sun-only hold
print("(c) nights Fri 12:00 -> Mon 12:00:", _nights_weighted(datetime(2020, 1, 10, 12), datetime(2020, 1, 13, 12), 2),
      "| Sat 01:00 -> Sun 23:00:", _nights_weighted(datetime(2020, 1, 11, 1), datetime(2020, 1, 12, 23), 2),
      "| Sat 01:00 -> Mon 01:00 :", _nights_weighted(datetime(2020, 1, 11, 1), datetime(2020, 1, 13, 1), 2))
print("    entry 00:00 Mon -> exit 00:00 Tue:", _nights_weighted(datetime(2020, 1, 6, 0), datetime(2020, 1, 7, 0), 2),
      "| entry 23:00 Mon -> exit 00:00 Tue (signal at open):", _nights_weighted(datetime(2020, 1, 6, 23), datetime(2020, 1, 7, 0), 2))
print("    Christmas: Tue 2019-12-24 12:00 -> Thu 12-26 12:00:", _nights_weighted(datetime(2019, 12, 24, 12), datetime(2019, 12, 26, 12), 2))

# (d) short that is closed by eod: uses c[n-1] + spread of bar n-1 (the spread at that bar's OPEN)
bars2 = mk([(1.1, 1.1, 1.1, 1.1), (1.1, 1.1, 1.1, 1.1), (1.1, 1.1, 1.1, 1.1)]).with_columns(pl.Series("spread", [10.0, 10.0, 50.0]))
sig2 = pl.DataFrame({"signal": [-1, None, None], "stop_dist": [0.01] * 3, "target_dist": [0.01] * 3},
                    schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
t2 = run_backtest(bars2, sig2, spec).trades.row(0, named=True)
print(f"(d) short eod exit price {t2['exit_price']:.5f} = close + spread of last bar's open ({t2['spread_cost_points']:.0f} pts)")

# (e) stop_dist smaller than the spread -> long stopped at entry open
sig3 = pl.DataFrame({"signal": [1, None, None], "stop_dist": [0.00005] * 3, "target_dist": [None] * 3},
                    schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
t3 = run_backtest(mk([(1.1, 1.1, 1.1, 1.1)] * 3), sig3, spec).trades.row(0, named=True)
print(f"(e) stop_dist 5pts < spread 10pts: exit_reason={t3['exit_reason']} at bar {t3['exit_idx']} price {t3['exit_price']} (entry {t3['entry_price']})")
