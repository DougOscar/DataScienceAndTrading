"""P03 — PBO (CSCV) can be driven to ~0 by padding the grid with consistently-bad configs.

PBO measures the IS-best config's OOS *relative rank*.  If a share of the grid is reliably
worse than everything else (e.g. fast SMA pairs that churn and bleed spread, or an
"invert signal" flag in a trending sample), the IS-best -- drawn from the non-bleeding
group -- always ranks above them OOS, so its rank is >= that share even with ZERO edge.

World: 40 zero-edge configs (random walk, correlation 0.5) + K "bleeders"
(same noise, mean -X bp/day).  PBO vs K.  Also verify the rank convention on a toy case.
"""
import math
import numpy as np
from quantlab import stats as st

T = 2340
rng = np.random.default_rng(3)

# toy check of the rank/logit convention: 2 strategies, IS-best is always the OOS-worse one
bl = T // 16
sgn = np.repeat(np.where(np.arange(16) % 2 == 0, 1.0, -1.0), bl)
a = 1e-3 * sgn + 1e-3 * rng.standard_normal(sgn.size)
b = -1e-3 * sgn + 1e-3 * rng.standard_normal(sgn.size)
print("toy anti-persistence (expect PBO ~1):", st.pbo_cscv(np.column_stack([a, b]), 16).pbo)

for bleed_bp in (1.0, 3.0):
    for K in (0, 10, 40, 80, 160):
        pbos = []
        for s in range(30):
            z0 = rng.standard_normal((T, 1))
            good = (math.sqrt(0.5) * z0 + math.sqrt(0.5) * rng.standard_normal((T, 40))) * 0.005
            bad = (math.sqrt(0.5) * z0 + math.sqrt(0.5) * rng.standard_normal((T, K))) * 0.005 - bleed_bp * 1e-4
            m = np.column_stack([good, bad]) if K else good
            pbos.append(st.pbo_cscv(m, 16, max_combos=2000, rng=np.random.default_rng(s)).pbo)
        print(f"bleed {bleed_bp:.0f}bp/day  K={K:4d} bleeders + 40 null: mean PBO {np.mean(pbos):.3f}  "
              f"P(PBO<0.30) {np.mean(np.array(pbos) < 0.30):.2f}")
