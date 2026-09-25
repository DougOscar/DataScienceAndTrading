"""R1-P11 (M4 + new) — ledger-derived prior trials: can a caller shrink N without an error?

Ledger: system 'sma' has attempts a1, a2, a3 (49 trials each, 7x7 grid, weak planted edge).
Then we gate a 4th study under several labelings and report n_trials_prior / DSR / error."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G, ledger
from quantlab.evaluators import SyntheticEvaluator

space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
def ev(seed):
    return ExploitEval(SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                       bumps=({"center": {"a": 3, "b": 3}, "height": 1.6, "width": 0.4},), rho=0.5, seed=seed))
ld, sd = tmpdirs()
def study(sid, issue, attempt, system="sma", seed=1, **kw):
    return opt.run_study(ev(seed), space, book="FBS", system=system, issue=issue, attempt=attempt, study_id=sid,
                         n_jobs=1, wfo=None, ledger_dir=ld, studies_dir=sd, **kw)
def gate(res, label, **kw):
    try:
        rep = G.evaluate_gates(res, ev(1), periods_per_year=260, mechanism_check=mech_true, n_boot=200,
                               ledger_dir=ld, **kw)
        et = rep.effective_trials
        print(f"{label:62s} N_study={et['n_trials_study']:.0f} prior={et['n_trials_prior']:.0f} "
              f"DSR={et['dsr']:.3f} hurdle={et['sr0_annual']:.2f} src={et.get('prior_source')}")
        return rep
    except Exception as e:
        print(f"{label:62s} RAISES {type(e).__name__}: {str(e)[:110]}")

for a in (1, 2, 3):
    r = study(f"fbs-0007-a{a}", 7, a)
    G.log_gates(gate(r, f"honest attempt {a} (issue 7)"), ledger_dir=ld)

r4 = study("fbs-0008-a1", 8, 1)
gate(r4, "(b) new issue 8, same system, attempt=1")
r5 = study("fbs-0009-a1", 9, 1, system="sma_v2")
gate(r5, "(c) same idea renamed system='sma_v2', attempt=1")
r6 = study("fbs-0010-a1", 10, 1, system="SMA")
gate(r6, "(c2) system='SMA' (case change)")
r7 = study("fbs-0007-x9", 7, 1)
gate(r7, "(d) issue 7 study with explicit id, attempt=1 label")
gate(study("fbs-0007-a3r", 7, 3, seed=1), "(e0) attempt=3 honest label, custom id")
gate(r4, "(e) explicit prior_trials=0 on a study", prior_trials=0)
r8 = opt.run_study(ev(1), space, book="FBS", system="sma", issue=7, attempt=3, study_id="fbs-0007-a3",
                   n_jobs=1, wfo=None, ledger_dir=ld, studies_dir=sd, resume=True)
gate(r8, "(f0) resume a3 honestly")
r9 = opt.run_study(ev(1), space, book="FBS", system="other", issue=7, attempt=1, study_id="fbs-0007-a3",
                   n_jobs=1, wfo=None, ledger_dir=ld, studies_dir=sd, resume=True)
print(f"    resumed a3 with system='other', attempt=1 -> meta system={r9.meta['system']} attempt={r9.meta['attempt']}; "
      f"ledger row system={ledger.studies(ld)['fbs-0007-a3']['system']} attempt={ledger.studies(ld)['fbs-0007-a3']['attempt']}")
gate(r9, "(f) resume a3 passing system='other', attempt=1")
ld2, _ = tmpdirs()
try:
    rep = G.evaluate_gates(r8, ev(1), periods_per_year=260, mechanism_check=mech_true, n_boot=200, ledger_dir=ld2)
    print(f"(g) a3 gated against an EMPTY ledger_dir: prior={rep.effective_trials['n_trials_prior']}")
except Exception as e:
    print(f"(g) a3 gated against an EMPTY ledger_dir: RAISES {type(e).__name__}")
r4.meta.pop("attempt"); r4.study_id = "fbs-0011-a1"
gate(r4, "(h) meta attempt removed + id suffix a1")
