"""Re-verification round 1 (e64efcf) shared harness.

Same ExploitEval as e2e_common (imported), but every study AND every evaluate_gates call uses a
fresh tmp ledger (evaluate_gates now reads the ledger for prior trials -- M4 -- so the default
would read research/ledger).  `pmap` runs jobs over <= 4 spawn workers.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from e2e_common import ExploitEval, mech_true, recent_sharpe  # noqa: E402,F401
from quantlab import gates as G, opt  # noqa: E402

MAX_WORKERS = 4


def tmpdirs(prefix="rt_r1_"):
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    return tmp / "ledger", tmp / "studies"


def run_and_gate(ev, space, sid, *, method="grid", n_trials=None, wfo=opt.WFOConfig(), seed=0,
                 system="probe", attempt=1, gate_kw=None, dirs=None, **kw):
    ld, sd = dirs or tmpdirs()
    res = opt.run_study(ev, space, book="FBS", system=system, issue=9999, attempt=attempt, study_id=sid,
                        method=method, n_trials=n_trials, n_jobs=1, wfo=wfo, seed=seed,
                        ledger_dir=ld, studies_dir=sd, **kw)
    rep = G.evaluate_gates(res, ev, periods_per_year=260.0, mechanism_check=mech_true, n_boot=500,
                           ledger_dir=ld, **(gate_kw or {}))
    return res, rep


def pmap(fn, args, workers=MAX_WORKERS):
    with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        return list(ex.map(fn, args))


def ann_sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * math.sqrt(260)) if x.size > 2 and x.std() > 0 else float("nan")
