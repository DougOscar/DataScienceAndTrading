"""S6 red team diagnostic: re-run the card's 840-config grid under a cost variant / other symbol.

DIAGNOSTIC ONLY -- not trials of fbs-0023-a1, nothing is logged. Writes the daily-returns matrix
to the scratch dir given as argv[3]. Usage: s6_grid_diag.py SYMBOL {gross|net} OUTDIR
Every config counts as one diagnostic evaluation (840 per call).
"""
import itertools, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, polars as pl

warnings.filterwarnings("ignore")
from quantlab.costs import CostModel
from quantlab.evaluators import RuleEvaluator
from quantlab.strategies.toy_tsmom import ToyTsmom

GRID = [dict(lookback=l, stop_mult=s, hold=h) for l, s, h in itertools.product(
    range(10, 121, 10), [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], range(6, 61, 6))]
EV = None
COST = None


def _init(sym, variant):
    global EV, COST
    warnings.filterwarnings("ignore")
    EV = RuleEvaluator(ToyTsmom, sym, "H4", book="FBS")
    EV.prepare()
    COST = CostModel(spread_multiplier=0.0) if variant == "gross" else None


def _run(i):
    o = EV(GRID[i], cost=COST)
    t = o.trades
    n = int(t.height - (t["skipped"].sum() if "skipped" in t.columns else 0))
    return i, o.daily.with_columns(pl.col("date").cast(pl.Date)), n


if __name__ == "__main__":
    sym, variant, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    cols, ntr = {}, {}
    with ProcessPoolExecutor(8, initializer=_init, initargs=(sym, variant)) as ex:
        for i, d, n in ex.map(_run, range(len(GRID)), chunksize=10):
            cols[i] = d.rename({"ret": f"t{i}"}); ntr[i] = n
    m = cols[0]
    for i in range(1, len(GRID)):
        m = m.join(cols[i], on="date", how="full", coalesce=True)
    m = m.sort("date").fill_null(0.0)
    m.write_parquet(out / f"{sym}_{variant}.parquet")
    pl.DataFrame({"trial": list(ntr), "n_trades": list(ntr.values()),
                  **{k: [GRID[i][k] for i in ntr] for k in ("lookback", "stop_mult", "hold")}}
                 ).write_parquet(out / f"{sym}_{variant}_trials.parquet")
    print(sym, variant, "evals", len(GRID), "wall", round(time.perf_counter() - t0, 1), "s")
