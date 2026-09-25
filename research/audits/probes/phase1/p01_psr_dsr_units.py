"""P01 — independent reproduction of PSR / expected-max-Sharpe / MinTRL / DSR units.

Checks:
 a) PSR against an independent implementation (raw kurtosis, per-period SR) and against a
    Monte-Carlo coverage test (PSR(SR*=true SR) should be ~Uniform -> P(PSR>=0.95) ~ 5%).
 b) expected_max_sharpe vs Monte-Carlo E[max of N iid N(0, V)].
 c) MinTRL inversion: PSR at n = MinTRL equals prob.
 d) Unit trap: what DSR does if a caller passes ANNUALISED var_sr / sr (no guard?).
"""
import math
import numpy as np
from scipy import stats as sps
from quantlab import stats as st, metrics

rng = np.random.default_rng(1)

# ---- a) PSR independent formula
def psr_indep(sr, sr0, n, g3, g4_raw):
    return sps.norm.cdf((sr - sr0) * math.sqrt(n - 1) / math.sqrt(1 - g3 * sr + (g4_raw - 1) / 4 * sr * sr))

for sr, sr0, n, g3, g4 in [(0.08, 0.0, 2340, -0.5, 6.0), (0.05, 0.02, 1000, 0.3, 3.0), (0.2, 0.1, 500, -2, 12)]:
    print("PSR lib vs indep", st.psr(sr, sr0, n, g3, g4), psr_indep(sr, sr0, n, g3, g4))

# coverage: true SR per period 0.05, fat tails (t5), n=2340 -> P(PSR(true) >= 0.95) should be ~0.05
hits = []
for _ in range(4000):
    x = rng.standard_t(5, 2340) / math.sqrt(5 / 3)
    x = x + 0.05
    sr = st.sharpe_per_period(x)
    sk, ku = metrics.skew_kurt(x)
    hits.append(st.psr(sr, 0.05, x.size, sk, ku) >= 0.95)
print("PSR coverage (target 0.05):", np.mean(hits))

# ---- b) expected max sharpe vs MC
for N in (2, 5, 10, 50, 100, 1000):
    V = 1.0
    mc = rng.standard_normal((20000, N)).max(axis=1).mean()
    print(f"E[max] N={N}: formula {st.expected_max_sharpe(N, V):.4f}  MC {mc:.4f}")

# ---- c) MinTRL inversion
for sr, tgt, g3, g4 in [(0.06, 0.0, -0.4, 5.0), (0.1, 0.03, 0.0, 3.0)]:
    m = st.min_trl(sr, tgt, g3, g4)["periods"]
    print("PSR at MinTRL (want 0.95):", st.psr(sr, tgt, m, g3, g4))

# ---- d) unit trap: annualised inputs to dsr() are silently accepted
x = rng.standard_normal(2340) * 0.005 + 0.0003  # ~ SR 0.95 annual
srs_pp = rng.standard_normal(100) * (1 / math.sqrt(2340))
print("dsr per-period var:", st.dsr(x, trial_sharpes=srs_pp, n_eff=100))
print("dsr annualised trial_sharpes (unit bug, no error raised):",
      st.dsr(x, trial_sharpes=srs_pp * math.sqrt(260), n_eff=100))
