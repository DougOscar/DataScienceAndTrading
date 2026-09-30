"""S6 red team: is the S5 read-only StudyResult rebuild faithful?

Runs on a SCRATCH COPY of the dry-run ledger + trial store (env vars point there; the caller
copies them first). Rebuilds the StudyResult two ways -- (a) the S5 loader, (b) the library's own
run_study(resume=True) with the notebook's STUDY_KW -- and compares the objects and the
evaluate_gates(log=False) values of both. Nothing is written to research/.
Evaluations: 0 trials re-run (resume on a complete store); gates re-evaluate plateau (21) + cost
stress (1) per call => 2 x 22 judge-run evaluations of already-logged points (diagnostic).
"""
import os, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
assert "scratch" in os.environ["QUANTLAB_LEDGER_DIR"], "must run on a scratch copy"
sys.path.insert(0, str(Path(__file__).parent))
from s5_load_study import SID, load_study, make_evaluator  # noqa
import numpy as np, polars as pl
from quantlab import gates, opt, ledger

def main():
    ev = make_evaluator()
    a = load_study()
    space = opt.SearchSpace((
        opt.IntParam("lookback", 10, 120, step=10, plateau_scale="relative"),
        opt.FloatParam("stop_mult", 1.0, 4.0, step=0.5, plateau_scale="relative"),
        opt.IntParam("hold", 6, 60, step=6, plateau_scale="relative"),
    ))
    KW = dict(book="FBS", system="toy_tsmom", issue=23, attempt=1, method="auto", seed=0, n_jobs=8,
              cv=opt.CPCVConfig(n_groups=10, k_test=2),
              wfo=opt.WFOConfig(refit_every="3mo", window="anchored", min_train="2y"),
              objective=opt.Objective(kind="block_sharpe", n_blocks=4, lam=0.5), min_trades=200,
              plateau_radius=0.20, checkpoint_every=100, notes="scratch resume (S6 faithfulness probe)")
    b = opt.run_study(ev, space, resume=True, **KW)
    print("selected", a.selected_params, b.selected_params)
    print("selection equal:", {k: (a.selection.get(k), b.selection.get(k)) for k in a.selection if a.selection.get(k) != b.selection.get(k)})
    ta = a.trials.sort("trial_id"); tb = b.trials.sort("trial_id")
    common = [c for c in tb.columns if c in ta.columns]
    print("trials cols only in resumed:", sorted(set(tb.columns) - set(ta.columns)), "only in loader:", sorted(set(ta.columns) - set(tb.columns)))
    for c in common:
        try:
            if not ta[c].equals(tb[c], check_names=False) and not ta[c].cast(pl.Utf8).equals(tb[c].cast(pl.Utf8)):
                print("  trials column differs:", c)
        except Exception as e:
            print("  compare error", c, e)
    ra, rb = a.returns, b.returns
    print("returns shape", ra.shape, rb.shape, "null cells resumed:", sum(rb[c].null_count() for c in rb.columns))
    cols = [c for c in rb.columns if c != "date"]
    A = ra.select(cols).to_numpy(); B = rb.select(cols).fill_null(0.0).to_numpy()
    print("returns max|diff|", float(np.nanmax(np.abs(A - B))), "dates equal", ra["date"].cast(pl.Date).equals(rb["date"].cast(pl.Date)))
    for n in ("cpcv_paths", "wfo_oos", "wfo_params"):
        x, y = getattr(a, n), getattr(b, n)
        print(n, x.shape, y.shape, "equal:", x.equals(y))
    print("meta keys missing from loader:", sorted(set(b.meta) - set(a.meta)))
    ga = gates.evaluate_gates(a, ev, periods_per_year=ev.periods_per_year, n_jobs=8, log=False)
    gb = gates.evaluate_gates(b, ev, periods_per_year=ev.periods_per_year, n_jobs=8, log=False)
    for x, y in zip(ga.rows, gb.rows):
        print(f"{x.gate:22s} loader={x.value!s:28s} resumed={y.value!s:28s} {x.status}/{y.status}")
    print("scratch ledger events:", [e["event"] for e in ledger.study_events(SID)])


if __name__ == "__main__":
    main()
