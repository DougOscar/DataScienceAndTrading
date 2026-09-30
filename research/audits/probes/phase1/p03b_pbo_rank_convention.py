"""P03b — PBO rank/logit convention check.  Two strategies with block-alternating sign:
whenever IS holds more 'even' blocks than 'odd', the IS-best must be the OOS-worse (overfit
fraction 1.0).  Balanced splits (k=4) are decided by the full-sample noise mean, so they
are NOT expected to be 50/50 -- that is correct CSCV behaviour on a single realisation."""
import numpy as np
from itertools import combinations
from quantlab import stats as st
rng = np.random.default_rng(3)
S, bl = 16, 146
T = S * bl
sgn = np.repeat(np.where(np.arange(S) % 2 == 0, 1.0, -1.0), bl)
a = 1e-3 * sgn + 1e-3 * rng.standard_normal(T)
b = -1e-3 * sgn + 1e-3 * rng.standard_normal(T)
r = st.pbo_cscv(np.column_stack([a, b]), 16)
print("PBO", r.pbo)
combos = list(combinations(range(S), S // 2))
ev = np.array([sum(1 for g in c if g % 2 == 0) for c in combos])
for k in range(9):
    m = ev == k
    print(f"IS even-blocks={k}: n={m.sum():5d}  share logit<=0 = {np.mean(r.logits[m] <= 0):.3f}")
