"""Probe 01: a short with no stop and/or no target must NOT exit on its entry bar.

Engine uses sentinels sl_c=-1e18 / tp_c=+1e18 for 'no level'. For a short the
comparisons are ask_h >= sl and ask_l <= tp, so the sentinels satisfy them.
"""
import warnings
warnings.filterwarnings("ignore")
import polars as pl
from quantlab.engine import run_backtest
from quantlab.costs import InstrumentSpec
from quantlab.testing import synthetic_bars

spec = InstrumentSpec(symbol="SYN", digits=5, point=1e-5, contract_size=1e5, tick_size=1e-5,
                      volume_min=0.01, volume_step=0.01, volume_max=100, base_ccy="EUR",
                      quote_ccy="USD", swap_mode="points", swap_long=0, swap_short=0)
bars = synthetic_bars(50, seed=1)
n = bars.height
for label, sd, td in [("long  no stop/no target", None, None), ("short no stop/no target", None, None),
                      ("short stop only", 0.05, None), ("short target only", None, 0.05),
                      ("short stop+target", 0.05, 0.05)]:
    d = -1 if label.startswith("short") else 1
    sig = pl.DataFrame({"signal": [d] + [None] * (n - 1),
                        "stop_dist": [sd] * n, "target_dist": [td] * n},
                       schema={"signal": pl.Int8, "stop_dist": pl.Float64, "target_dist": pl.Float64})
    r = run_backtest(bars, sig, spec)
    t = r.trades.row(0, named=True)
    print(f"{label:26s} entry_idx={t['entry_idx']} exit_idx={t['exit_idx']} reason={t['exit_reason']:10s} "
          f"entry={t['entry_price']:.5f} exit={t['exit_price']:.5f} pnl_pts={t['pnl_points']:.1f}")
