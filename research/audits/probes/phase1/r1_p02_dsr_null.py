"""R1-P02 (B1) — DSR gate (v1.2: V0 = 1/(T-1), raw N) under the null on correlated grids.
Same worlds as p02 (SMA-crossover grid on a random walk; equicorrelated rho=0.8, N=100) plus
iid N=100 and a 'few trials, heavy tails' world (t3 returns) to probe the normal-moment V0.
Selection = argmax Sharpe.  P(DSR >= 0.95) must be <= ~0.05."""
import math, sys
import numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
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


from quantlab import stats as st

NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def iid(rng, n=100):
    return rng.standard_normal((T, n)) * 0.005

def t3(rng, n=20):
    return rng.standard_t(3, (T, n)) * 0.003

def ar_vol(rng, n=50):
    # GARCH-ish volatility clustering shared by all trials (common vol regime), zero mean
    h = np.exp(np.cumsum(rng.standard_normal(T) * 0.05)); h /= h.mean()
    return rng.standard_normal((T, n)) * 0.005 * h[:, None]

for name, gen in (("sma_grid", sma_grid_returns), ("equicorr_0.8", equicorr), ("iid_100", iid),
                  ("t3_20", t3), ("volclust_50", ar_vol)):
    rng = np.random.default_rng(7)
    a = []
    for s in range(NSEED):
        m = gen(rng)
        srs = st.sharpe_per_period(m, axis=0)
        j = int(np.nanargmax(srs))
        d = st.dsr_from_matrix(m, j)
        a.append((d.dsr, d.dsr_neff_cross, d.sr_annual, d.sr0_annual, m.shape[1]))
    a = np.array(a)
    print(f"{name:13s} N={int(a[0,4]):3d}: P(DSR_gate>=0.95)={np.mean(a[:,0]>=0.95):.3f}  "
          f"(v1.1-style {np.mean(a[:,1]>=0.95):.3f})  mean max SR {a[:,2].mean():.2f} ann vs hurdle {a[:,3].mean():.2f}")
