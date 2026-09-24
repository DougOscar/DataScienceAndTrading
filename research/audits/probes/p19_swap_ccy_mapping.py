"""Probe 19 (round 2, N1): swap currency mapping from the MT5 export's raw swap-mode enum, end to end through
_load_from_tsv -> run_backtest -> apply_sizing. Uses a FAKE broker TSV in a scratch dir (config.BROKER_DIR
monkeypatched); real export still absent."""
import sys, os, warnings, tempfile
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from pathlib import Path
import polars as pl
from quantlab import config, data, costs
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing
from sma_strategy import SmaCross, SmaParams

tmp = Path(tempfile.mkdtemp(prefix="fakebroker_"))
config.BROKER_DIR = tmp
rows = []
def row(sym, mode, base, profit, margin, sl, ss, cs=100000.0, point=1e-5, digits=5):
    return dict(symbol=sym, digits=digits, point=point, contract_size=cs, tick_size=point, volume_min=0.01, volume_step=0.01,
                volume_max=500.0, currency_base=base, currency_profit=profit, currency_margin=margin, swap_mode=mode,
                swap_long=sl, swap_short=ss, swap_3day="WEDNESDAY", stops_level=0)
cases = [
    ("USDJPY", "SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT", "USD", "JPY", "USD", -5.0, -5.0, 100000.0, 0.001, 3),
    ("EURUSD", "SYMBOL_SWAP_MODE_CURRENCY_SYMBOL", "EUR", "USD", "EUR", -5.0, -5.0, 100000.0, 1e-5, 5),
    ("EURJPY", "SYMBOL_SWAP_MODE_CURRENCY_MARGIN", "EUR", "JPY", "EUR", -5.0, -5.0, 100000.0, 0.001, 3),
    ("GBPUSD", "SYMBOL_SWAP_MODE_INTEREST_CURRENT", "GBP", "USD", "GBP", -2.0, -2.0, 100000.0, 1e-5, 5),
    ("AUDUSD", "SYMBOL_SWAP_MODE_POINTS", "AUD", "USD", "AUD", -5.0, -5.0, 100000.0, 1e-5, 5),
]
pl.DataFrame([row(*c[:6], c[6], cs=c[7], point=c[8], digits=c[9]) for c in cases]).write_csv(tmp / "FBS_symbols.tsv", separator="\t")
pl.DataFrame({"symbol": [c[0] for c in cases], "commission_per_lot_roundtrip": [0.0] * len(cases)}).write_csv(tmp / "FBS_commissions.tsv", separator="\t")
rate_fn = lambda q, a, ts: data.conversion_rate(q, a, ts)
for c in cases:
    sym, mode = c[0], c[1]
    spec = costs.load_instrument(sym)
    bars = data.load_bars(sym, "H1", start="2019-01-01", end="2019-04-01")
    tr = run_backtest(bars, SmaCross(SmaParams(far_target=True)).signals(bars), spec).trades
    held = tr.filter(pl.col("nights") > 0).head(1)
    s = apply_sizing(held, spec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=bars)
    s0 = apply_sizing(held.with_columns(pl.lit(0.0).alias("swap_money_per_lot"), pl.lit(0.0).alias("swap_points")),
                      spec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=bars)
    r = held.row(0, named=True)
    booked = s["pnl_ccy"][0] - s0["pnl_ccy"][0]
    print(f"{sym} {mode.replace('SYMBOL_SWAP_MODE_', ''):17s} -> swap_mode={spec.swap_mode:8s} swap_ccy={spec.swap_ccy:7s} "
          f"| trade dir={r['direction']:+d} nights={r['nights']} swap_money/lot={r['swap_money_per_lot']:.2f} swap_pts={r['swap_points']:.1f} "
          f"entry={r['entry_price']} -> booked USD {booked:.2f}")
# account currency != USD with DEPOSIT
spec = costs.load_instrument("USDJPY")
bars = data.load_bars("USDJPY", "H1", start="2019-01-01", end="2019-04-01")
tr = run_backtest(bars, SmaCross(SmaParams(far_target=True)).signals(bars), spec).trades.filter(pl.col("nights") > 0).head(1)
s = apply_sizing(tr, spec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=bars, account_ccy="EUR")
s0 = apply_sizing(tr.with_columns(pl.lit(0.0).alias("swap_money_per_lot")), spec, mode="fixed_lots", lots=1.0, rate_fn=rate_fn, bars=bars, account_ccy="EUR")
print(f"USDJPY DEPOSIT on a EUR account: swap_money/lot={tr['swap_money_per_lot'][0]:.2f} -> booked EUR {s['pnl_ccy'][0]-s0['pnl_ccy'][0]:.2f} (identity, correct: deposit ccy = account ccy)")
print("crypto spec (fallback) swap_every_day:", costs.load_instrument("BTCUSD").swap_every_day, "| can the export override it? field not read from TSV:",
      "swap_every_day" not in open(costs.__file__).read().split("def _load_from_tsv")[1].split("swap_every_day=_is_crypto")[0])
