"""Probe 09: can leaky strategies slip past quantlab.testing.assert_no_lookahead?
(stop and target both 500 pips so the short-without-level bug of p01 does not interfere)"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import dataclass
from typing import ClassVar
import polars as pl
from quantlab import data
from quantlab.contracts import Params, RiskType
from quantlab.testing import assert_no_lookahead
from quantlab.engine import run_backtest
from quantlab.costs import load_instrument

bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
spec = load_instrument("EURUSD")

@dataclass(frozen=True)
class P(Params):
    pass

def frame(sig, stop=None):
    return pl.DataFrame({"signal": sig}).with_columns(pl.col("signal").cast(pl.Int8),
        pl.lit(0.05, pl.Float64).alias("stop_dist"), pl.lit(0.05, pl.Float64).alias("target_dist"))

class SparsePeek:
    """SMA cross, but only take the cross if the NEXT bar closes in its favour (1-bar look-ahead on ~1% of rows)."""
    name: ClassVar[str] = "sparse_peek"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        f, s = pl.col("close").rolling_mean(20), pl.col("close").rolling_mean(50)
        a = f > s
        up = a & ~a.shift(1).fill_null(True); dn = ~a & a.shift(1).fill_null(False) & s.is_not_null()
        nxt = pl.col("close").shift(-1) - pl.col("close")          # <-- future
        sig = pl.when(up & (nxt > 0)).then(1).when(dn & (nxt < 0)).then(-1).when(up | dn).then(0).otherwise(None)
        return frame(b.select(sig.alias("s"))["s"])

class ExternalD1:
    """Loads D1 bars itself (full dev period) and joins today's D1 bar onto each H1 bar by date."""
    name: ClassVar[str] = "external_d1"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        d1 = data.load_bars("EURUSD", "D1", start="2018-12-01").select(pl.col("ts").dt.date().alias("d"), (pl.col("close") > pl.col("open")).alias("up"))
        j = b.select(pl.col("ts").dt.date().alias("d")).join(d1, on="d", how="left", maintain_order="left")
        sig = pl.when(pl.col("up")).then(1).otherwise(-1)
        return frame(j.select(sig.cast(pl.Int8).alias("s"))["s"])

class SelfResampleD1:
    """Resamples the bars it is given to D1 and joins the day's close back (control: should be caught)."""
    name: ClassVar[str] = "self_d1"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        d1 = b.group_by_dynamic("ts", every="1d").agg(pl.col("open").first().alias("o"), pl.col("close").last().alias("c")).select(pl.col("ts").dt.date().alias("d"), (pl.col("c") > pl.col("o")).alias("up"))
        j = b.select(pl.col("ts").dt.date().alias("d")).join(d1, on="d", how="left", maintain_order="left")
        return frame(j.select(pl.when(pl.col("up")).then(1).otherwise(-1).cast(pl.Int8).alias("s"))["s"])

class GlobalZ:
    """Full-sample z-score (control: should be caught by truncation; NOT by permutation, which preserves the mean)."""
    name: ClassVar[str] = "global_z"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        z = (pl.col("close") - pl.col("close").mean()) / pl.col("close").std()
        return frame(b.select(pl.when(z > 0).then(-1).otherwise(1).cast(pl.Int8).alias("s"))["s"])

for S in (SparsePeek, ExternalD1, SelfResampleD1, GlobalZ):
    passes = 0
    for seed in range(20):
        try:
            assert_no_lookahead(S(), bars, seed=seed); passes += 1
        except AssertionError:
            pass
    sig = S().signals(bars)
    tr = run_backtest(bars, sig, spec).trades
    print(f"{S.name:14s} passed assert_no_lookahead in {passes}/20 seeds | non-null signal rows={sig['signal'].is_not_null().sum()} "
          f"| backtest trades={tr.height} sum pnl_points={tr['pnl_points'].sum():.0f} win%={(tr['pnl_points']>0).mean()*100:.1f}")
