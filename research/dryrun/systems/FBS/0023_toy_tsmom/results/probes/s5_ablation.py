"""S5 mechanism ablation (card: random entry side, same schedule), dry run #23, study fbs-0023-a1.

Evidence only: no gate / ledger write (the mechanism review is the user's decision).

A. Card's test as written (biased, red-team S2 M1): the selected config (best of 840 on the full dev
   window) vs >= 1000 random-side draws of that one fixed config.
B. Unbiased variant (M1 option a): the walk-forward OOS series (the procedure's own OOS output,
   stored artifact wfo_oos) vs the same splice with random sides — for each draw every refit's
   selected config is run with random sides and its test-window days are spliced exactly as
   opt.walk_forward did (selected trial's daily return on each OOS date).
Sanity: side_seed=-1 must reproduce the trial store (selected column and the stored WFO splice).
"""
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

N_DRAWS = 1000
SID = "fbs-0023-a1"
PPY = 260.0
_EV = None


def _init():
    global _EV
    from quantlab.evaluators import RuleEvaluator
    from s5_randside import ToyTsmomRandSide
    _EV = RuleEvaluator(ToyTsmomRandSide, "EURUSD", "H4", book="FBS")


def _run(args):
    params, seed = args
    o = _EV({**params, "side_seed": int(seed)})
    d = o.daily.select(pl.col("date").cast(pl.Date), "ret")
    return seed, params, d["date"].cast(pl.Int32).to_numpy(), d["ret"].to_numpy()


def _series(dates_i32, ret, want_i32):
    m = dict(zip(dates_i32.tolist(), ret.tolist()))
    return np.array([m.get(int(x), 0.0) for x in want_i32])


if __name__ == "__main__":
    for k in ("POLARS_MAX_THREADS", "NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    from quantlab import ledger, metrics, stats

    t0 = time.perf_counter()
    rets = ledger.load_trial_returns(SID).with_columns(pl.col("date").cast(pl.Date))
    dates = rets["date"].cast(pl.Int32).to_numpy()
    trials = ledger.load_trials(SID)
    pmap = {int(t): json.loads(p) for t, p in zip(trials["trial_id"], trials["params"])}
    sel = ledger.study_events(SID, "selection")[-1]
    sel_tid = int(sel["selection"]["trial_id"])
    wfo = ledger.load_study_artifact(SID, "wfo_oos").sort("date")
    wp = ledger.load_study_artifact(SID, "wfo_params")
    rid2tid = dict(zip(wp["refit_id"].to_list(), wp["selected_trial"].to_list()))
    w_dates = wfo["date"].cast(pl.Int32).to_numpy()
    w_tid = np.array([rid2tid[r] for r in wfo["refit_id"].to_list()])
    wfo_tids = sorted(set(w_tid.tolist()))
    real_full = rets[f"t{sel_tid}"].fill_null(0.0).to_numpy()
    real_wfo = wfo["ret"].to_numpy()
    sr_full_real = metrics.sharpe(real_full, PPY)
    sr_wfo_real = metrics.sharpe(real_wfo, PPY)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(8, mp_context=ctx, initializer=_init) as ex:
        # sanity: seed -1 reproduces the store
        chk = list(ex.map(_run, [(pmap[t], -1) for t in sorted(set(wfo_tids) | {sel_tid})]))
        rep = {}
        for _, p, di, r in chk:
            tid = next(t for t, q in pmap.items() if q == p)
            rep[tid] = _series(di, r, dates)
        max_err_sel = float(np.max(np.abs(rep[sel_tid] - real_full)))
        splice_real = np.array([rep[t][np.searchsorted(dates, d)] for t, d in zip(w_tid, w_dates)])
        max_err_wfo = float(np.max(np.abs(splice_real - real_wfo)))
        print(f"sanity: max|err| selected {max_err_sel:.2e}, WFO splice {max_err_wfo:.2e}")

        # A: fixed selected config, random sides
        jobs_a = [(pmap[sel_tid], 10_000_000 + d) for d in range(N_DRAWS)]
        null_a = np.array([metrics.sharpe(_series(di, r, dates), PPY) for _, _, di, r in ex.map(_run, jobs_a, chunksize=8)])
        print(f"A done {time.perf_counter() - t0:.0f}s")

        # B: WFO splice with random sides (independent coins per config per draw)
        jobs_b = [(pmap[t], 20_000_000 + 1000 * d + t) for d in range(N_DRAWS) for t in wfo_tids]
        by_draw: dict[int, dict[int, np.ndarray]] = {}
        for seed, p, di, r in ex.map(_run, jobs_b, chunksize=8):
            d, t = divmod(seed - 20_000_000, 1000)
            by_draw.setdefault(d, {})[t] = _series(di, r, dates)
        idx = np.searchsorted(dates, w_dates)
        null_b = np.array([metrics.sharpe(np.array([by_draw[d][t][i] for t, i in zip(w_tid, idx)]), PPY)
                           for d in range(N_DRAWS)])
        print(f"B done {time.perf_counter() - t0:.0f}s")

    def summ(real, null):
        return {"real_sharpe": float(real), "null_median": float(np.median(null)),
                "null_p95": float(np.quantile(null, 0.95)), "null_p05": float(np.quantile(null, 0.05)),
                "null_sd": float(np.std(null, ddof=1)), "p_value_one_sided": stats.empirical_pvalue(real, null),
                "real_minus_median": float(real - np.median(null)),
                "contradiction": bool(stats.empirical_pvalue(real, null) >= 0.05 or real - np.median(null) <= 0),
                "n_draws": int(null.size)}

    out = {"A_card_full_dev_selected": {**summ(sr_full_real, null_a), "config": pmap[sel_tid], "trial": sel_tid},
           "B_wfo_oos_splice": {**summ(sr_wfo_real, null_b), "configs": {t: pmap[t] for t in wfo_tids},
                                "n_days": int(real_wfo.size)},
           "sanity_max_abs_err": {"selected": max_err_sel, "wfo_splice": max_err_wfo},
           "runtime_s": time.perf_counter() - t0}
    (HERE / "s5_ablation.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))
