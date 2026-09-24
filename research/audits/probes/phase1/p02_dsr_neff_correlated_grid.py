"""P02 — Is DSR calibrated under the null when the trial grid is highly correlated?

Zero-edge world: daily log-returns are iid (GARCH-free) N(0, 0.6%) -- a random walk.
Trial grid = SMA crossover (always-in, long/short), fast in 5..50 step 5, slow in 20..200
step 20 (fast < slow), cost 0.5 bp per position change (cost is tiny -> near zero-edge).
Plus an equicorrelated synthetic grid (rho=0.8) where the exact answer is known.

For each seed: select the argmax-Sharpe trial (what DSR is designed for), compute the
library's dsr_from_matrix (n_eff = max(eigen, cluster)), and report
  * n_eff by method, raw N
  * P(DSR >= 0.95) across seeds  (must be <= ~0.05 if calibrated)
  * P(DSR_rawN >= 0.95)
  * the realised mean of max SR vs the sr0 hurdle.
"""
import math, sys
import numpy as np
from quantlab import stats as st

T = 2340
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def sma_grid_returns(rng):
    r = rng.standard_normal(T + 250) * 0.006
    p = np.cumsum(r)
    cs = np.concatenate([[0], np.cumsum(p)])
    def sma(n):
        out = np.full(p.size, np.nan)
        out[n - 1:] = (cs[n:] - cs[:-n]) / n
        return out
    cols = []
    for f in range(5, 55, 5):
        for s in range(20, 220, 20):
            if f >= s:
                continue
            pos = np.sign(sma(f) - sma(s))
            pos = np.nan_to_num(pos)
            ret = np.zeros_like(r)
            ret[1:] = pos[:-1] * r[1:] - 0.00005 * np.abs(np.diff(pos))
            cols.append(ret[250:])
    return np.column_stack(cols)


def equicorr(rng, n=100, rho=0.8):
    z0 = rng.standard_normal((T, 1))
    z = rng.standard_normal((T, n))
    return (math.sqrt(rho) * z0 + math.sqrt(1 - rho) * z) * 0.005


for name, gen in (("sma_grid", sma_grid_returns), ("equicorr_0.8", equicorr)):
    rng = np.random.default_rng(7)
    res = []
    for s in range(NSEED):
        m = gen(rng)
        srs = st.sharpe_per_period(m, axis=0)
        j = int(np.nanargmax(srs))
        d = st.dsr_from_matrix(m, j, methods=("eigen", "cluster"))
        liji = st.effective_n_trials(m, "liji")
        # DSR at Li-Ji and at raw N using the same var_sr
        dl = st.psr(d.sr, st.expected_max_sharpe(liji, d.var_sr), d.n_obs, d.skew, d.kurt)
        res.append((d.n_eff_by_method["eigen"], d.n_eff_by_method["cluster"], liji, m.shape[1],
                    d.dsr, d.dsr_raw_n, dl, d.sr, d.sr0, math.sqrt(d.var_sr), d.psr0))
    a = np.array(res)
    print(f"== {name}: {NSEED} null worlds, N raw = {int(a[0,3])}")
    print(f"   n_eff eigen median {np.median(a[:,0]):.2f}  cluster median {np.median(a[:,1]):.2f}  liji median {np.median(a[:,2]):.1f}")
    print(f"   P(DSR>=0.95) lib(max eigen,cluster) = {np.mean(a[:,4]>=0.95):.3f}   "
          f"Li-Ji = {np.mean(a[:,6]>=0.95):.3f}   raw N = {np.mean(a[:,5]>=0.95):.3f}   "
          f"PSR(0) (no deflation) = {np.mean(a[:,10]>=0.95):.3f}")
    print(f"   mean selected SR (per-period) {a[:,7].mean():.4f} ; mean sr0 hurdle (lib) {a[:,8].mean():.4f} ;"
          f" cross-sectional sd(SR) {a[:,9].mean():.4f} ; 1/sqrt(T) {1/math.sqrt(T):.4f}")
    print(f"   annualised: max SR mean {a[:,7].mean()*math.sqrt(260):.2f}, hurdle {a[:,8].mean()*math.sqrt(260):.2f}")
