"""R2-P05 (48e696f) -- B2 + M3/N5 re-run, with the new PASS / FAIL / NOT_DECISIVE exam.

Same worlds as r1_p05 (SyntheticEvaluator 7x7, broad bump H=3.0, rho=0.5; edge exactly 0 from FROM of
dev; default WFO and min_train=1y; 'None' = live control).  For every gate-PASS study, a ZERO-edge
holdout process (dev vol 0.005, 100 trades/yr Poisson) is judged:
  * 1y exam (the gates' band, 260 d): P(criteria pass), band p_pass_zero_edge, P(decisive PASS);
  * v1.1-level band (alpha 0.10 marginal) P(criteria pass) for comparison (N5);
  * honest NOT_DECISIVE chain: exams at 260 / 390 / 520 / 650 days, each on the FULL span with a band
    built for that horizon; FAIL stops, NOT_DECISIVE continues -> P(eventual PASS).
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

HORIZONS = (260, 390, 520, 650)


def chain(res, bands, n_sim=400, seed=7, vol=0.005, tpy=100):
    from quantlab import stats as st
    rng = np.random.default_rng(seed)
    out = {"PASS": 0, "FAIL": 0, "PENDING": 0}
    first_pass = 0
    for _ in range(n_sim):
        h = rng.standard_normal(HORIZONS[-1]) * vol
        tr = np.cumsum(rng.poisson(tpy / 260.0, HORIZONS[-1]))
        verdict = "PENDING"
        for H, b in zip(HORIZONS, bands):
            r = st.holdout_check(b, h[:H], float(tr[H - 1]), 260.0, check_horizon=False)
            if r["status"] == "FAIL":
                verdict = "FAIL"; break
            if r["status"] == "PASS":
                verdict = "PASS"; break
        out[verdict] += 1
    return {k: v / n_sim for k, v in out.items()}


def one(job):
    import warnings; warnings.filterwarnings("ignore")
    from r1_common import ExploitEval, run_and_gate, tmpdirs
    from e2e_common import dead_holdout_pass_rate
    from quantlab import opt, stats as st, gates as G
    from quantlab.evaluators import SyntheticEvaluator
    FROM, sched, seed = job
    space = opt.SearchSpace([opt.IntParam("a", 0, 6), opt.IntParam("b", 0, 6)])
    reg = () if FROM is None else ({"from": FROM, "bumps": (), "base_sharpe": 0.0},)
    inner = SyntheticEvaluator(bounds={"a": (0, 6), "b": (0, 6)},
                               bumps=({"center": {"a": 3, "b": 3}, "height": 3.0, "width": 0.35},),
                               rho=0.5, seed=100 + seed, trades_per_year=100, regimes=reg)
    ev = ExploitEval(inner)
    wfo = opt.WFOConfig() if sched == "default" else opt.WFOConfig(min_train="1y")
    dirs = tmpdirs()
    res, rep = run_and_gate(ev, space, f"fbs-9999-r2dead-{FROM}-{sched}-{seed}", wfo=wfo, dirs=dirs)
    stt = rep.statuses()
    out = {"verdict": rep.verdict, "st": stt, "wfo": rep.diagnostics.get("wfo_oos")}
    if rep.verdict == "PASS":
        hb = rep.holdout_band
        out["p_crit_1y"] = dead_holdout_pass_rate(res, rep)
        out["p0_1y"] = hb["p_pass_zero_edge"]
        out["decisive_1y"] = hb["p_pass_zero_edge"] <= 0.30
        orig = st.joint_tail_level
        st.joint_tail_level = lambda *a, **k: (0.10, float("nan"))
        try:
            b11 = G._build_holdout_band(res, 260, 260.0, n_boot=500, seed=12345)
        finally:
            st.joint_tail_level = orig
        class _R: pass
        r11 = _R(); r11.holdout_band = b11.as_dict()
        out["p_crit_v11"] = dead_holdout_pass_rate(res, r11)
        bands = [hb] + [G._build_holdout_band(res, H, 260.0, n_boot=500, seed=12345).as_dict() for H in HORIZONS[1:]]
        out["p0_by_h"] = [b["p_pass_zero_edge"] for b in bands]
        out["chain"] = chain(res, bands)
    return job, out


if __name__ == "__main__":
    from r1_common import pmap
    from collections import defaultdict
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    jobs = [(f, s, k) for f in (0.55, 0.70, None) for s in ("default", "min_train=1y") for k in range(NS)]
    res = pmap(one, jobs)
    grp = defaultdict(list)
    for (f, s, k), o in res:
        grp[(f, s)].append(o)
    allpass = defaultdict(list)
    for (f, s), os_ in grp.items():
        n = len(os_)
        ps = [o for o in os_ if o["verdict"] == "PASS"]
        wf = sum(o["st"]["wfo_oos"] != "PASS" and all(v == "PASS" for g, v in o["st"].items() if g != "wfo_oos") for o in os_)
        rec = [o["wfo"]["sharpe_recent"] for o in os_ if o["wfo"]]
        print(f"dead_from={f} sched={s:13s}: S5 verdict PASS {len(ps)}/{n}; wfo_oos was the only failing gate in {wf}; "
              f"median recent-third WFO SR {np.median(rec):+.2f}")
        if ps:
            allpass["live" if f is None else "dead"] += ps
            print(f"   1y band p_pass_zero_edge {np.round([o['p0_1y'] for o in ps], 2).tolist()}")
            print(f"   1y P(criteria pass | zero edge): joint {np.round([o['p_crit_1y'] for o in ps], 2).tolist()} | "
                  f"v1.1-level {np.round([o['p_crit_v11'] for o in ps], 2).tolist()}")
            print(f"   1y P(decisive PASS | zero edge) {np.round([o['p_crit_1y'] if o['decisive_1y'] else 0.0 for o in ps], 2).tolist()}")
            print(f"   p_pass_zero_edge by horizon {HORIZONS}: " +
                  "; ".join(str(np.round(o['p0_by_h'], 2).tolist()) for o in ps))
            print(f"   chain (<=4 exams) zero-edge: P(PASS) {np.round([o['chain']['PASS'] for o in ps], 2).tolist()} "
                  f"P(FAIL) {np.round([o['chain']['FAIL'] for o in ps], 2).tolist()} "
                  f"P(still pending) {np.round([o['chain']['PENDING'] for o in ps], 2).tolist()}")
    for k, ps in allpass.items():
        print(f"POOLED {k} S5-passing studies (n={len(ps)}): mean P(zero-edge holdout -> PASS) 1y-decisive-only "
              f"{np.mean([o['p_crit_1y'] if o['decisive_1y'] else 0.0 for o in ps]):.3f}, chain {np.mean([o['chain']['PASS'] for o in ps]):.3f}, "
              f"max chain {np.max([o['chain']['PASS'] for o in ps]):.3f}; old 1y criteria-pass {np.mean([o['p_crit_1y'] for o in ps]):.3f}, "
              f"v1.1-level {np.mean([o['p_crit_v11'] for o in ps]):.3f}")
