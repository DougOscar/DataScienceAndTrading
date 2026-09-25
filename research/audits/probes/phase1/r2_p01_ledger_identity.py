"""R2-P01 (48e696f) -- ledger as source of truth: N1, N2, N3, N7, N10 re-runs + new bypass attempts.

All ledgers/studies in fresh tmp dirs.  Synthetic evaluators only (no data files read).
Sections:
  N1  TPE -> resume sobol (the r1 laundering) ; variants.
  N2  r1_p11 relabel games ; plus: prior counts only studies CREATED BEFORE -> re-gate a1 after a2..a5.
  N3  plateau_radius kwarg / floor / resume change.
  N7  spec symbol mismatch.
  ID  in-memory StudyResult edits the ledger checks do not see:
      (a) drop trial rows -> N_study shrinks ; (b) the shrunk count is then written by log_gates as
      n_trials_study and becomes the ledger's count for later attempts ; (c) swap study_id to a clean
      study + drop the 'source' column -> TPE study gated as data-independent ; (d) pop meta keys.
  AB  abort_on_uniform_errors: trials recorded? counted?
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import polars as pl
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G, ledger
from quantlab.evaluators import SyntheticEvaluator


def ev(seed=1, bump=True):
    b = ({"center": {"a": 3, "b": 3}, "height": 1.6, "width": 0.4},) if bump else ()
    return ExploitEval(SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)}, bumps=b, rho=0.5, seed=seed))


space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])


def gate(res, ld, label, e=None, **kw):
    try:
        rep = G.evaluate_gates(res, e or ev(1), periods_per_year=260, mechanism_check=mech_true, n_boot=200,
                               ledger_dir=ld, **kw)
        et = rep.effective_trials
        st = rep.statuses()
        print(f"  {label:60s} N_study={et['n_trials_study']:.0f} prior={et['n_trials_prior']:.0f} DSR={et['dsr']:.3f} "
              f"oos/wfo/cscv={st['oos_sharpe']}/{st['wfo_oos']}/{st['cscv_oos_loss']}")
        return rep
    except Exception as ex:
        print(f"  {label:60s} RAISES {type(ex).__name__}: {str(ex)[:150]}")


def run(ld, sd, sid, issue=7, attempt=1, system="sma", seed=1, **kw):
    kw.setdefault("wfo", None)
    return opt.run_study(ev(seed), space, book="FBS", system=system, issue=issue, attempt=attempt, study_id=sid,
                         n_jobs=1, ledger_dir=ld, studies_dir=sd, **kw)


def tryrun(label, fn):
    try:
        r = fn()
        print(f"  {label:60s} OK (meta data_dependent={r.meta.get('candidate_set_data_dependent')})")
        return r
    except Exception as ex:
        print(f"  {label:60s} RAISES {type(ex).__name__}: {str(ex)[:150]}")


print("== N1: TPE then resume as sobol")
ld, sd = tmpdirs()
fspace = opt.SearchSpace([opt.FloatParam("a", 0.0, 6.0), opt.FloatParam("b", 0.0, 6.0)])
def frun(sid, **kw):
    return opt.run_study(ev(1), fspace, book="FBS", system="n1", issue=11, attempt=1, study_id=sid, n_jobs=1,
                         wfo=None, ledger_dir=ld, studies_dir=sd, **kw)
tryrun("tpe 30", lambda: frun("n1-a1", method="tpe", n_trials=30, seed=0))
tryrun("resume same id method=sobol", lambda: frun("n1-a1", method="sobol", n_trials=30, seed=0, resume=True))
tryrun("resume same id method=random", lambda: frun("n1-a1", method="random", n_trials=30, seed=0, resume=True))
r_t = tryrun("resume same id method=tpe n_trials=40", lambda: frun("n1-a1", method="tpe", n_trials=40, seed=0, resume=True))
ev_rows = [e["event"] for e in ledger.study_events("n1-a1", ledger_dir=ld)]
print("  events:", ev_rows)
gate(r_t, ld, "gate tpe study", e=ev(1))

print("== N2: prior-trial relabel games (ledger: sma a1..a3 on issue 7, 49 trials each)")
ld, sd = tmpdirs()
for a in (1, 2, 3):
    r = run(ld, sd, f"fbs-0007-a{a}", attempt=a)
    G.log_gates(gate(r, ld, f"honest attempt {a}"), ledger_dir=ld)
gate(run(ld, sd, "fbs-0008-a1", issue=8, attempt=1), ld, "(b) new issue 8, same system, attempt=1")
gate(run(ld, sd, "fbs-0009-a1", issue=9, system="sma_v2"), ld, "(c) system 'sma_v2', new issue 9")
gate(run(ld, sd, "fbs-0010-a1", issue=10, system="S.M.A"), ld, "(c2) system 'S.M.A', new issue 10")
gate(run(ld, sd, "fbs-0012-a1", issue=12, system="sma_cross_fast"), ld, "(c3) system 'sma_cross_fast', new issue 12")
gate(run(ld, sd, "fbs-0007-x9", issue=7, attempt=1, system="other"), ld, "(d) issue 7, system 'other', attempt=1")
r4 = ledger  # placeholder
try:
    tryrun("(f) resume a3 with system='other', attempt=1",
           lambda: run(ld, sd, "fbs-0007-a3", issue=7, attempt=1, system="other", resume=True))
except Exception:
    pass
r8 = run(ld, sd, "fbs-0008-a1", issue=8, attempt=1, resume=True)
gate(r8, ld, "(e) explicit prior_trials=0", prior_trials=0)
ld2, _ = tmpdirs()
gate(r8, ld2, "(g) gated against an EMPTY ledger_dir")

print("== N2b: prior = studies created BEFORE -> explore a1..a5, then re-gate / promote a1")
ld, sd = tmpdirs()
studs = {}
for a in range(1, 6):
    studs[a] = run(ld, sd, f"fbs-0020-a{a}", issue=20, attempt=a, system="brk", seed=a)
    rep = gate(studs[a], ld, f"attempt {a} gated right after creation")
    G.log_gates(rep, ledger_dir=ld)
rep1 = gate(studs[1], ld, "attempt 1 RE-gated after a2..a5 exist")
print(f"  -> a1 prior after 4 later attempts: {rep1.effective_trials['n_trials_prior']:.0f} (total explored "
      f"{5 * 49}); a5 prior {ledger.related_prior_trials('fbs-0020-a5', ledger_dir=ld)[0]:.0f}")

print("== N3: plateau radius")
ld, sd = tmpdirs()
for rad in (1e-6, 0.05, 0.10, {"a": 0.10, "b": 0.5}):
    tryrun(f"run_study(plateau_radius={rad})", lambda rad=rad: run(ld, sd, f"n3-{len(str(rad))}", issue=30, system=f"n3{len(str(rad))}", plateau_radius=rad))
rr = run(ld, sd, "n3-base", issue=31, system="n3base")
try:
    G.evaluate_gates(rr, ev(1), periods_per_year=260, ledger_dir=ld, plateau_radius=0.5)
    print("  evaluate_gates(plateau_radius=0.5) accepted (BAD)")
except TypeError as ex:
    print("  evaluate_gates(plateau_radius=0.5) -> TypeError:", str(ex)[:80])
tryrun("resume with plateau_radius=0.5 (ledger 0.2)", lambda: run(ld, sd, "n3-base", issue=31, system="n3base",
                                                                    plateau_radius=0.5, resume=True))
print("  ledger radius of n3-base:", ledger.created_row("n3-base", ledger_dir=ld).get("plateau_radius"))

print("== N7: spec symbol mismatch")
from quantlab.costs import load_instrument
class SymEval(ExploitEval):
    symbol = "EURUSD"
e7 = SymEval(ev(1).inner)
e7.inner.is_synthetic = False
try:
    rep = G.evaluate_gates(rr, e7, periods_per_year=260, ledger_dir=ld, spec=load_instrument("BTCUSD"), n_boot=100)
    print("  spec=BTCUSD for EURUSD evaluator accepted:", rep.row("cost_stress_sharpe").status)
except Exception as ex:
    print("  spec=BTCUSD for EURUSD evaluator ->", type(ex).__name__, str(ex)[:90])

print("== ID: in-memory StudyResult edits")
ld, sd = tmpdirs()
big = opt.SearchSpace([opt.IntParam("a", 0, 20), opt.IntParam("b", 0, 20)])
e_big = ExploitEval(SyntheticEvaluator(bounds={"a": (0, 20), "b": (0, 20)},
                                       bumps=({"center": {"a": 10, "b": 10}, "height": 1.6, "width": 0.3},), rho=0.5, seed=4))
rb = opt.run_study(e_big, big, book="FBS", system="idsys", issue=40, attempt=1, study_id="id-a1", n_jobs=1, wfo=None,
                   ledger_dir=ld, studies_dir=sd)
print(f"  ledger trials event n_trials = {ledger.studies(ld)['id-a1'].get('n_trials')}")
honest = gate(rb, ld, "(a0) honest 441-trial study", e=e_big)
sel = rb.selection["trial_id"]
keep = [sel] + [t for t in rb.trials["trial_id"].to_list() if t != sel][:9]
rb2 = rb.__class__(**{**rb.__dict__})
rb2.trials = rb.trials.filter(pl.col("trial_id").is_in(keep))
rb2.returns = rb.returns.select(["date"] + [f"t{t}" for t in keep])
cut = gate(rb2, ld, "(a) same study, trials/returns cut to 10 in memory", e=e_big)
G.log_gates(cut, ledger_dir=ld)
print(f"  (b) after log_gates: ledger study_trial_count(id-a1) = {ledger.study_trial_count(ledger.studies(ld)['id-a1'])}")
ra2 = opt.run_study(e_big, big, book="FBS", system="idsys", issue=40, attempt=2, study_id="id-a2", n_jobs=1, wfo=None,
                    ledger_dir=ld, studies_dir=sd, seed=1)
gate(ra2, ld, "(b) attempt 2 prior (true prior 441)", e=e_big)

# (c) TPE study relabelled as a clean study
ld, sd = tmpdirs()
tp = opt.run_study(ev(1), fspace, book="FBS", system="c", issue=41, attempt=1, study_id="c-tpe", method="tpe",
                   n_trials=40, n_jobs=1, wfo=opt.WFOConfig(), ledger_dir=ld, studies_dir=sd)
cl = opt.run_study(ev(1), fspace, book="FBS", system="c", issue=41, attempt=2, study_id="c-sobol", method="sobol",
                   n_trials=8, n_jobs=1, wfo=opt.WFOConfig(), ledger_dir=ld, studies_dir=sd)
gate(tp, ld, "(c0) TPE study as is")
tp2 = tp.__class__(**{**tp.__dict__})
tp2.study_id = "c-sobol"
tp2.trials = tp.trials.drop("source")
tp2.meta = {k: v for k, v in tp.meta.items() if k not in ("method", "candidate_set", "candidate_set_data_dependent",
                                                           "seed", "attempt", "n_by_source")}
gate(tp2, ld, "(c) TPE StudyResult relabelled study_id='c-sobol', source col + meta keys dropped")
