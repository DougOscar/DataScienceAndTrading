"""R3-P01 (30b23b6) -- re-run of r2_p01 / r2_p03 (R2-4, N2b, prior residuals) + the in-memory residuals.

Fresh tmp ledgers/studies; synthetic evaluators only.  evaluate_gates now fixes n_boot (2000) / seed.
Sections:
  N2   prior-trial relabel games (new name + new issue residual).
  N2b  re-gate a1 after a2..a5 exist.
  ID   (a) truncate trials/returns in memory; (b) ledger count after; (c) TPE relabelled as a clean
       study; (d) evaluator swap (different describe) and (d2) same describe, different behaviour.
  MEM  fields still read from memory only: cpcv_paths, wfo_oos, selection/selected_params,
       and the GateReport handed to log_gates.
"""
import sys, time, warnings
from dataclasses import replace
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import polars as pl
from r1_common import ExploitEval, tmpdirs, mech_true
from quantlab import opt, gates as G, ledger
from quantlab.evaluators import SyntheticEvaluator

# harness: make ExploitEval report the same cost version from both sources (run_study reads
# evaluator.cost_version, the gates read evaluator.cost.version)
ExploitEval.cost_version = property(lambda self: self.cost.version)


def ev(seed=1, bump=True, **kw):
    b = ({"center": {"a": 3, "b": 3}, "height": 1.6, "width": 0.4},) if bump else ()
    return ExploitEval(SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)}, bumps=b, rho=0.5, seed=seed), **kw)


space = opt.SearchSpace([opt.IntParam("a", 0, 6, plateau_step=2), opt.IntParam("b", 0, 6, plateau_step=2)])


def gate(res, ld, label, e=None, quiet=False, **kw):
    try:
        t0 = time.perf_counter()
        rep = G.evaluate_gates(res, e or ev(1), periods_per_year=260, mechanism_check=mech_true, ledger_dir=ld, **kw)
        dt = time.perf_counter() - t0
        et = rep.effective_trials
        st = rep.statuses()
        if not quiet:
            print(f"  {label:62s} N_study={et['n_trials_study']:.0f} prior={et['n_trials_prior']:.0f} "
                  f"DSR={et['dsr']:.3f} oos/wfo/cscv={st['oos_sharpe']}/{st['wfo_oos']}/{st['cscv_oos_loss']} "
                  f"verdict={rep.verdict} ({dt:.1f}s)")
        return rep
    except Exception as ex:
        print(f"  {label:62s} RAISES {type(ex).__name__}: {str(ex)[:140]}")


def run(ld, sd, sid, issue=7, attempt=1, system="sma", seed=1, e=None, **kw):
    kw.setdefault("wfo", None)
    return opt.run_study(e or ev(seed), space, book="FBS", system=system, issue=issue, attempt=attempt,
                         study_id=sid, n_jobs=1, ledger_dir=ld, studies_dir=sd, seed=seed, **kw)


print("== N2: prior-trial relabel games (sma a1..a3 on issue 7, 49 trials each)")
ld, sd = tmpdirs("rt_r3_")
for a in (1, 2, 3):
    r = run(ld, sd, f"fbs-0007-a{a}", attempt=a)
    G.log_gates(gate(r, ld, f"honest attempt {a}"), ledger_dir=ld)
gate(run(ld, sd, "fbs-0009-a1", issue=9, system="sma_v2"), ld, "(c) system 'sma_v2', new issue 9")
gate(run(ld, sd, "fbs-0010-a1", issue=10, system="S.M.A"), ld, "(c2) system 'S.M.A', new issue 10")
gate(run(ld, sd, "fbs-0012-a1", issue=12, system="sma_cross_fast"), ld, "(c3) system 'sma_cross_fast', new issue 12")

print("== N2b: explore a1..a5, then re-gate a1")
ld, sd = tmpdirs("rt_r3_")
studs = {}
for a in range(1, 6):
    studs[a] = run(ld, sd, f"fbs-0020-a{a}", issue=20, attempt=a, system="brk", seed=a)
    G.log_gates(gate(studs[a], ld, f"attempt {a} gated right after creation", e=ev(a)), ledger_dir=ld)
rep1 = gate(studs[1], ld, "attempt 1 RE-gated after a2..a5 exist", e=ev(1))

print("== ID: in-memory StudyResult edits (R2-4)")
ld, sd = tmpdirs("rt_r3_")
big = opt.SearchSpace([opt.IntParam("a", 0, 20, plateau_step=4), opt.IntParam("b", 0, 20, plateau_step=4)])
e_big = ExploitEval(SyntheticEvaluator(bounds={"a": (0, 20), "b": (0, 20)},
                                       bumps=({"center": {"a": 10, "b": 10}, "height": 1.6, "width": 0.3},),
                                       rho=0.5, seed=4))
rb = opt.run_study(e_big, big, book="FBS", system="idsys", issue=40, attempt=1, study_id="id-a1", n_jobs=1,
                   wfo=opt.WFOConfig(), ledger_dir=ld, studies_dir=sd)
honest = gate(rb, ld, "(a0) honest 441-trial study", e=e_big)
sel = rb.selection["trial_id"]
keep = [sel] + [t for t in rb.trials["trial_id"].to_list() if t != sel][:9]
rb2 = replace(rb, trials=rb.trials.filter(pl.col("trial_id").is_in(keep)),
              returns=rb.returns.select(["date"] + [f"t{t}" for t in keep]))
gate(rb2, ld, "(a) trials/returns cut to 10 in memory", e=e_big)
rb3 = replace(rb, trials=rb.trials.filter(pl.col("trial_id").is_in(keep)))
gate(rb3, ld, "(a') trials cut, returns intact", e=e_big)
st_ = ledger.studies(ld)["id-a1"]
print(f"  (b) ledger study_trial_count(id-a1) = {ledger.study_trial_count(st_)} "
      f"(logged_trial_count {ledger.logged_trial_count('id-a1', ledger_dir=ld)})")

ld_c, sd_c = tmpdirs("rt_r3_")
fspace = opt.SearchSpace([opt.FloatParam("a", 0.0, 6.0, plateau_step=1.0), opt.FloatParam("b", 0.0, 6.0, plateau_step=1.0)])
tp = opt.run_study(ev(1), fspace, book="FBS", system="c", issue=41, attempt=1, study_id="c-tpe", method="tpe",
                   n_trials=40, n_jobs=1, wfo=opt.WFOConfig(), ledger_dir=ld_c, studies_dir=sd_c)
cl = opt.run_study(ev(1), fspace, book="FBS", system="c", issue=41, attempt=2, study_id="c-sobol", method="sobol",
                   n_trials=8, n_jobs=1, wfo=opt.WFOConfig(), ledger_dir=ld_c, studies_dir=sd_c)
tp2 = replace(tp, study_id="c-sobol", trials=tp.trials.drop("source"),
              meta={k: v for k, v in tp.meta.items() if k not in ("method", "candidate_set", "candidate_set_data_dependent",
                                                                 "seed", "attempt", "n_by_source")})
gate(tp2, ld_c, "(c) TPE StudyResult relabelled as 'c-sobol'")

# (d) evaluator swap
sp1 = opt.SearchSpace([opt.IntParam("a", 1, 101, 5, plateau_scale="relative")])
spiky = ExploitEval(SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 51}, "height": 2.5, "width": 0.02},),
                                       base_sharpe=0.3, rho=1.0, seed=3))
broad = ExploitEval(SyntheticEvaluator(bounds={"a": (1, 101)}, bumps=({"center": {"a": 51}, "height": 2.5, "width": 0.5},),
                                       base_sharpe=0.3, rho=1.0, seed=3), drag_per_unit=0.0)
same_desc_nodrag = ExploitEval(spiky.inner, drag_per_unit=0.0)
ld_d, sd_d = tmpdirs("rt_r3_")
rs = opt.run_study(spiky, sp1, book="FBS", system="swap", issue=88, attempt=1, study_id="sw-a1", n_jobs=1, wfo=None,
                   ledger_dir=ld_d, studies_dir=sd_d)
for lab, e in (("(d0) study's own evaluator", spiky), ("(d) broad + zero-drag (different describe)", broad),
               ("(d2) same describe(), zero cost drag", same_desc_nodrag)):
    rep = gate(rs, ld_d, lab, e=e, quiet=True)
    if rep is not None:
        print(f"  {lab:62s} plateau {rep.row('plateau').value:.2f} {rep.row('plateau').status}; "
              f"cost_stress {rep.row('cost_stress_sharpe').value:.2f} {rep.row('cost_stress_sharpe').status}")

print("== MEM: fields the gates still take from memory")
# a dead-edge study: the planted bump sits in the first half only -> procedure OOS weak
ld_m, sd_m = tmpdirs("rt_r3_")
null_ev = ExploitEval(SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)}, bumps=(), rho=0.5, seed=21))
rn = run(ld_m, sd_m, "mem-a1", issue=70, system="memsys", e=null_ev, wfo=opt.WFOConfig())
r0 = gate(rn, ld_m, "(m0) zero-edge study, honest", e=null_ev)
print("     honest values:", {k: (None if v is None else round(v, 2)) for k, v in r0.values().items()})
selcol = f"t{rn.selection['trial_id']}"
# (m1) CPCV paths replaced by the in-sample best column (full-sample selection leak) -> "OOS"
is_best = max((c for c in rn.returns.columns if c != "date"),
              key=lambda c: rn.returns[c].mean() / rn.returns[c].std())
pids = rn.cpcv_paths["path_id"].unique().to_list()
fake_paths = pl.concat([rn.returns.select("date", pl.col(is_best).fill_null(0.0).alias("ret"))
                        .with_columns(pl.lit(p).cast(rn.cpcv_paths["path_id"].dtype).alias("path_id"))
                        .select("date", "path_id", "ret") for p in pids])
fake_wfo = rn.wfo_oos.with_columns(pl.col("ret") + 0.0)
d_is = dict(zip(rn.returns["date"].to_list(), rn.returns[is_best].fill_null(0.0).to_list()))
fake_wfo = rn.wfo_oos.with_columns(pl.col("date").replace_strict(d_is, default=0.0).cast(pl.Float64).alias("ret"))
r1 = gate(replace(rn, cpcv_paths=fake_paths), ld_m, "(m1) cpcv_paths := IS-best trial (in-sample)", e=null_ev)
r2 = gate(replace(rn, wfo_oos=fake_wfo), ld_m, "(m2) wfo_oos := IS-best trial on the WFO dates", e=null_ev)
r3 = gate(replace(rn, cpcv_paths=fake_paths, wfo_oos=fake_wfo), ld_m, "(m3) both", e=null_ev)
if r3 is not None:
    print("     tampered values:", {k: (None if v is None else round(v, 2)) for k, v in r3.values().items()})
# (m4) selection moved to the IS-best trial (dsr / plateau / cost stress follow it)
tid_best = int(is_best[1:])
row_best = rn.trials.filter(pl.col("trial_id") == tid_best)
sp_best = {n: row_best[f"param_{n}"][0] for n in rn.param_names}
r4 = gate(replace(rn, selection={**rn.selection, "trial_id": tid_best}, selected_params=sp_best), ld_m,
          "(m4) selection := IS-best trial (not the plateau pick)", e=null_ev)
print(f"     ledger selection event trial: {ledger.study_events('mem-a1', 'selection', ledger_dir=ld_m)[-1]['selection']['trial_id']}"
      f" vs in-memory {tid_best}")
# (m5) GateReport forged before log_gates
rep = gate(rn, ld_m, "(m5) honest report, then forged before log_gates", e=null_ev, quiet=True)
rep.rows = [replace(r, status="PASS") if r.status != "SKIPPED" else r for r in rep.rows]
rep.verdict = "PASS"
hb = dict(rep.holdout_band_run or rep.holdout_band)
hb.update(sharpe_lo=-9.0, p_pass_zero_edge=0.05)
rep.holdout_band_run = hb
rep.holdout_band = hb
try:
    ev_ = G.log_gates(rep, ledger_dir=ld_m)
    reg = ledger.registered_holdout_band("mem-a1", ledger_dir=ld_m)
    print(f"  (m5) log_gates ACCEPTED forged report: logged verdict={ev_['verdict']}, "
          f"registered band sharpe_lo={reg['sharpe_lo']}, p_pass_zero_edge={reg['p_pass_zero_edge']}")
except Exception as ex:
    print(f"  (m5) log_gates refused: {type(ex).__name__}: {str(ex)[:140]}")

print("== determinism of the band (needed for any recompute-and-compare fix)")
a = gate(rn, ld_m, "run A", e=null_ev, quiet=True).holdout_band_run
b = gate(rn, ld_m, "run B", e=null_ev, quiet=True).holdout_band_run
import json
print("  band identical across two evaluate_gates runs:", json.dumps(a, sort_keys=True, default=str) ==
      json.dumps(b, sort_keys=True, default=str))
