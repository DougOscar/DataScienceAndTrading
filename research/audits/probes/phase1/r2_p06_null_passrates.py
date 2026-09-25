"""R2-P06 (48e696f) -- r1_p06 re-run on the new code: per-gate pass rates of zero-edge worlds (10x10 grid,
default WFO, v1.2 gates, judge-run plateau).  Same worlds and harness as r1_p06."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from r1_p06_null_passrates import FAMS, one

if __name__ == "__main__":
    from r1_common import pmap
    NS = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows = pmap(one, [(f, s) for f in FAMS for s in range(NS)])
    for fam in FAMS:
        rs = [r for r in rows if r[0] == fam]
        gates = list(rs[0][1])
        print(f"== {fam}: {len(rs)} zero-edge worlds; verdict PASS {np.mean([r[2] == 'PASS' for r in rs]):.2f}")
        pos = [r for r in rs if r[3] > 0]
        for g in gates:
            print(f"   {g:20s} pass {np.mean([r[1][g] == 'PASS' for r in rs]):.2f}   | among {len(pos)} with IS SR>0: "
                  f"{np.mean([r[1][g] == 'PASS' for r in pos]) if pos else float('nan'):.2f}")
