"""Probe 21 (round 2, N6): forced cut points are now 'signal changes', capped at 200 evenly spaced.
A dense, honest-looking strategy (1-bar momentum, flips often) with a 1-bar peek on a sparse subset of rows:
how often does it pass the auditor?"""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
from dataclasses import dataclass
from typing import ClassVar
import polars as pl
from quantlab import data
from quantlab.contracts import Params, RiskType
from quantlab.testing import assert_no_lookahead, _forced_cut_points
from quantlab.engine import run_backtest
from quantlab.costs import load_instrument

bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2020-01-01")
spec = load_instrument("EURUSD")

@dataclass(frozen=True)
class P(Params):
    every: int = 200

class DenseWithSparsePeek:
    name: ClassVar[str] = "dense_sparse_peek"; risk_type: ClassVar[RiskType] = RiskType.C
    def __init__(self, every): self.params = P(every)
    def signals(self, b):
        mom = pl.col("close") - pl.col("close").shift(1)
        nxt = pl.col("close").shift(-1) - pl.col("close")
        leak_row = (pl.int_range(pl.len()) % self.params.every == (self.params.every // 2 + 7)) & nxt.is_not_null()
        sig = pl.when(leak_row).then(pl.when(nxt > 0).then(1).otherwise(-1)).otherwise(pl.when(mom > 0).then(1).otherwise(-1))
        return b.select(sig.cast(pl.Int8).alias("signal"), pl.lit(0.05).alias("stop_dist"), pl.lit(0.05).alias("target_dist"))

for every in (50, 200, 600):
    s = DenseWithSparsePeek(every)
    full = s.signals(bars)
    changes = int(full["signal"].ne_missing(full["signal"].shift(1)).sum())
    forced = _forced_cut_points(full, bars.height)
    n_leak = bars.height // every
    passes = 0; t0 = time.time()
    for seed in range(10):
        try:
            assert_no_lookahead(s, bars, seed=seed); passes += 1
        except AssertionError:
            pass
    print(f"leak every {every} rows ({n_leak} leak rows): signal changes={changes}, forced cuts used={len(forced)} (cap 200); "
          f"auditor passed {passes}/10 seeds [{time.time()-t0:.0f}s]", flush=True)
