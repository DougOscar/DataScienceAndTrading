"""R1-F1 — spawn pool: worker thread caps, parent env restoration, bit-identical results for
n_jobs=4 vs 1, and resume (Sobol n=16 -> resume n=32 on 4 workers) vs a fresh n=32 run."""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, polars as pl

def main():
    from r1_common import tmpdirs
    from quantlab import opt
    from quantlab.evaluators import SyntheticEvaluator
    os.environ["POLARS_MAX_THREADS"] = "3"          # parent value that must be restored
    os.environ.pop("OMP_NUM_THREADS", None)
    before = {k: os.environ.get(k) for k in opt.WORKER_THREAD_ENV}
    ev = SyntheticEvaluator(bounds={"a": (0, 1), "b": (0, 1), "c": (0, 1), "d": (0, 1)},
                            bumps=({"center": {"a": .5, "b": .5}, "height": 1.5, "width": .3},), rho=0.5, seed=4)
    space = opt.SearchSpace([opt.FloatParam(n, 0.0, 1.0) for n in "abcd"])
    def run(sid, n, jobs, dirs, resume=False):
        ld, sd = dirs
        return opt.run_study(ev, space, book="FBS", system="f1", issue=9999, attempt=1, study_id=sid, method="sobol",
                             n_trials=n, n_jobs=jobs, seed=3, ledger_dir=ld, studies_dir=sd, resume=resume,
                             checkpoint_every=5)
    r1 = run("f1-a", 32, 1, tmpdirs())
    r4 = run("f1-b", 32, 4, tmpdirs())
    after = {k: os.environ.get(k) for k in opt.WORKER_THREAD_ENV}
    print("start method:", r4.meta["mp_start_method"], "| worker diag:", r4.meta["worker_threads"])
    print("parent env restored:", before == after, before)
    same = lambda a, b: a.returns.equals(b.returns) and a.trials.drop([c for c in a.trials.columns if "time" in c or c.startswith("m_runtime")]).equals(
        b.trials.drop([c for c in b.trials.columns if "time" in c or c.startswith("m_runtime")]))
    print("n_jobs=4 vs 1: returns identical", r1.returns.equals(r4.returns), "| selection identical",
          r1.selected_params == r4.selected_params, "| wfo identical", r1.wfo_oos.equals(r4.wfo_oos),
          "| cpcv identical", r1.cpcv_paths.equals(r4.cpcv_paths))
    d = tmpdirs()
    run("f1-c", 16, 4, d)
    rr = run("f1-c", 32, 4, d, resume=True)
    print("resume 16->32 on 4 workers vs fresh 32: returns identical", r1.returns.equals(rr.returns),
          "| params identical", r1.trials.select([c for c in r1.trials.columns if c.startswith("param_")]).equals(
              rr.trials.select([c for c in rr.trials.columns if c.startswith("param_")])),
          "| selection identical", r1.selected_params == rr.selected_params,
          "| wfo n_trades identical", r1.wfo_oos["n_trades"].equals(rr.wfo_oos["n_trades"]))

if __name__ == "__main__":
    main()
