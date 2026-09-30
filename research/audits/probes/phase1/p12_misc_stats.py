"""P12 — stationary bootstrap, Politis-White, return_at_dd_budget, SPA/RC size, BH, CUSUM ARL."""
import math
import numpy as np
from quantlab import stats as st

rng = np.random.default_rng(0)
# --- stationary bootstrap: block lengths geometric with the requested mean; indices uniform
idx = st.stationary_bootstrap_indices(1000, 10.0, 200, rng)
runs = []
for row in idx[:50]:
    br = np.flatnonzero(np.diff(row) != 1) + 1
    lens = np.diff(np.r_[0, br, row.size])
    runs += list(lens[1:-1])
print(f"SB mean block (want ~10, wrap makes a few look continuous): {np.mean(runs):.2f}; index histogram cv {np.bincount(idx.ravel(), minlength=1000).std()/200:.3f}")
# AR(1) phi=0.5: long-run var of the mean = sigma^2/(1-phi)^2/n ; bootstrap var of the mean should match
phi, n = 0.5, 4000
e = rng.standard_normal(n + 100)
x = np.zeros_like(e)
for t in range(1, e.size):
    x[t] = phi * x[t - 1] + e[t]
x = x[100:]
b = st.optimal_block_length(x)
bs = st.bootstrap_stats(x, {"m": lambda a: a.mean(axis=1)}, n_boot=2000, rng=rng)["m"]
print(f"PW block for AR(1) 0.5, n=4000: {b:.1f}; boot var(mean) {bs.var():.3e} vs theory {1/(1-phi)**2/n:.3e}")
# --- return_at_dd_budget: root achieved, scale property
r = rng.standard_normal(2340) * 0.005 + 0.0003
br = st.return_at_dd_budget(r, 0.10, n_boot=1000, rng=np.random.default_rng(1))
br2 = st.return_at_dd_budget(r / 2, 0.10, n_boot=1000, rng=np.random.default_rng(1))
print(f"budget: k={br.leverage:.4f}, k(r/2)={br2.leverage:.4f} (ratio {br2.leverage/br.leverage:.4f}, want 2); monthly {br.mean_monthly:.4%}")
# --- SPA / RC size under the null (K=20 zero-mean, correlated), and power with one real strategy
def sim(mu_best, reps=150):
    pc, pr = [], []
    for s in range(reps):
        g = np.random.default_rng(1000 + s)
        z0 = g.standard_normal((1000, 1))
        d = 0.6 * z0 + 0.8 * g.standard_normal((1000, 20))
        d[:, 0] += mu_best
        out = st.spa_test(d, n_boot=300, rng=g, mean_block=1.0)
        rc = st.white_reality_check(d, n_boot=300, rng=g, mean_block=1.0)
        pc.append(out["p_consistent"]); pr.append(rc["p_value"])
    return np.mean(np.array(pc) <= 0.05), np.mean(np.array(pr) <= 0.05)
print("SPA_c / RC rejection rate at 5%, null:", sim(0.0))
print("SPA_c / RC rejection rate at 5%, one strategy with mean 0.1 sd:", sim(0.1))
# --- BH vs brute force
p = rng.uniform(size=30) ** 3
res = st.bh_fdr(p, 0.1)
srt = np.sort(p); k = max([i + 1 for i in range(30) if srt[i] <= 0.1 * (i + 1) / 30] or [0])
print(f"BH rejections lib {res['reject'].sum()} vs brute {k}")
# --- CUSUM in-control ARL
k_, h_ = 0.25, st.cusum_threshold(0.25, 500)
rl = []
for s in range(300):
    z = np.random.default_rng(s).standard_normal(20000)
    out = st.cusum_decay(z, 0.0, 1.0, k=k_, h=h_)
    rl.append(out["alarm_index"] + 1 if out["alarm"] else 20000)
print(f"CUSUM k=0.25 h={h_:.3f} target ARL0 500: simulated mean run length {np.mean(rl):.0f}")
# default k for a realistic edge: SR 1.0 annual -> mu/sd = 0.062 -> k = 0.031
print(f"default k at SR 1.0 ann: {0.5*1/math.sqrt(260):.4f}; h for ARL0 2600: {st.cusum_threshold(0.5/math.sqrt(260), 2600):.2f}")
# detection delay if the edge dies completely (SR 1.0 ann -> 0)
mu, sd = 1.0 / math.sqrt(260) * 0.005, 0.005
dl = []
for s in range(200):
    z = np.random.default_rng(10 + s).standard_normal(20000) * sd
    o = st.cusum_decay(z, mu, sd)
    dl.append(o["alarm_index"] + 1 if o["alarm"] else 20000)
print(f"CUSUM default settings: days to alarm after an SR-1.0 edge dies: median {np.median(dl):.0f} (~{np.median(dl)/260:.1f} y)")
