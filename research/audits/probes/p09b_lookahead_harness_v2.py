"""Probe 09b (re-verification of M2 + attacks on the new auditor). Every strategy here is leaky; the
auditor should reject all of them. Backtests use stop=target=500 pips so exits are signal-driven."""
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
import leaky_imports

bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
spec = load_instrument("EURUSD")
EURUSD_FILE = data.catalog().filter((pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "M1"))["file"][0]

@dataclass(frozen=True)
class P(Params):
    pass

def frame(sig):
    return pl.DataFrame({"signal": sig}).with_columns(pl.col("signal").cast(pl.Int8),
        pl.lit(0.05, pl.Float64).alias("stop_dist"), pl.lit(0.05, pl.Float64).alias("target_dist"))

def cross_cols(b):
    f, s = pl.col("close").rolling_mean(20), pl.col("close").rolling_mean(50)
    a = f > s
    return a & ~a.shift(1).fill_null(True), ~a & a.shift(1).fill_null(False) & s.is_not_null()

class SparsePeek:          # pre-fix p09 strategy (should now be caught by forced cut points)
    name: ClassVar[str] = "sparse_peek"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        up, dn = cross_cols(b); nxt = pl.col("close").shift(-1) - pl.col("close")
        sig = pl.when(up & (nxt > 0)).then(1).when(dn & (nxt < 0)).then(-1).when(up | dn).then(0).otherwise(None)
        return frame(b.select(sig.alias("s"))["s"])

class VetoPeek:            # take every cross UNLESS the future says it loses -> vetoed rows are NULL
    name: ClassVar[str] = "veto_peek"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        up, dn = cross_cols(b)
        fut = pl.col("close").shift(-6) - pl.col("close")           # 6-bar future move (null near the end)
        bad_up = (fut < 0).fill_null(False); bad_dn = (fut > 0).fill_null(False)
        sig = pl.when(up & ~bad_up).then(1).when(dn & ~bad_dn).then(-1).otherwise(None)
        return frame(b.select(sig.alias("s"))["s"])

class ExitPeek:            # honest entries; exits (signal 0) fired one bar before an adverse 20-pip move
    name: ClassVar[str] = "exit_peek"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        up, dn = cross_cols(b)
        drop = ((pl.col("close").shift(-1) - pl.col("close")) < -0.0020).fill_null(False)
        sig = pl.when(up).then(1).when(dn).then(-1).when(drop).then(0).otherwise(None)
        return frame(b.select(sig.alias("s"))["s"])

def d1_join(b, d1):
    d1 = d1.select(pl.col("ts").dt.date().alias("d"), (pl.col("close") > pl.col("open")).alias("up"))
    j = b.select(pl.col("ts").dt.date().alias("d")).join(d1, on="d", how="left", maintain_order="left")
    return frame(j.select(pl.when(pl.col("up")).then(1).otherwise(-1).cast(pl.Int8).alias("s"))["s"])

class ExternalD1:          # pre-fix p09: data.load_bars via module attribute (should be sandboxed)
    name: ClassVar[str] = "external_d1_attr"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        return d1_join(b, data.load_bars("EURUSD", "D1", start="2018-12-01"))

class ImportedLoadBars:    # `from quantlab.data import load_bars` bound at import time
    name: ClassVar[str] = "imported_load_bars"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        return d1_join(b, leaky_imports.imported_load_bars("EURUSD", "D1", start="2018-12-01"))

class PrivateHelper:       # quantlab.data._load_resampled (private, unguarded, not patched)
    name: ClassVar[str] = "private_helper"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        from pathlib import Path
        d1 = data._load_resampled(Path(EURUSD_FILE), "M1", "D1").filter(pl.col("ts") >= pl.datetime(2018, 12, 1)).collect()
        return d1_join(b, d1)

class RawParquet:          # polars reading the catalog file directly
    name: ClassVar[str] = "raw_parquet"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def signals(self, b):
        m1 = pl.scan_parquet(EURUSD_FILE).filter(pl.col("ts").is_between(pl.datetime(2018, 12, 1), pl.datetime(2020, 1, 1))).collect()
        d1 = m1.group_by_dynamic("ts", every="1d").agg(pl.col("open").first(), pl.col("close").last())
        return d1_join(b, d1)

class ClosureCache:        # data fetched once OUTSIDE signals() (e.g. in __init__) and closed over
    name: ClassVar[str] = "closure_cache"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def __init__(self):
        self.d1 = data.load_bars("EURUSD", "D1", start="2018-12-01")
    def signals(self, b):
        return d1_join(b, self.d1)

import time
small = bars.filter(pl.col("ts") < pl.datetime(2019, 2, 15))   # ~600 bars: dense-signal strategies force a cut at EVERY row
for S in (SparsePeek, VetoPeek, ExitPeek, ExternalD1, ImportedLoadBars, PrivateHelper, RawParquet, ClosureCache):
    strat = S()
    passes, msg = 0, ""
    dense = S in (ImportedLoadBars, PrivateHelper, RawParquet, ClosureCache, ExternalD1)
    audit_bars = small if dense else bars
    seeds = range(1) if dense else range(5)
    t0 = time.time()
    for seed in seeds:
        try:
            assert_no_lookahead(strat, audit_bars, seed=seed); passes += 1
        except AssertionError as ex:
            msg = str(ex)[:90]
    sig = strat.signals(bars)
    tr = run_backtest(bars, sig, spec).trades
    print(f"{S.name:18s} passed auditor {passes}/{len(seeds)} seeds | backtest trades={tr.height} "
          f"pnl={tr['pnl_points'].sum():.0f} win%={(tr['pnl_points'] > 0).mean()*100:.1f} | audit on {audit_bars.height} bars took {time.time()-t0:.0f}s | {msg}", flush=True)
