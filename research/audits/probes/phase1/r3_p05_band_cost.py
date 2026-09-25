"""R3-P05 (30b23b6) -- cost of the fixed band construction (n_boot 2000 / n_power 1000) per evaluate_gates
call, versus the rest of the gate run.  Synthetic study (make_study), WFO series of 1500-2340 days.
Measured while a separate 8-worker calibration was running on the box (timings are upper-bound-ish)."""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))
from test_gates import make_study, register
from quantlab import gates as G, stats as S

study, trades = make_study(seed=4)
ld = register(study)
for nb, npw in ((2000, 1000), (500, 300), (200, 0)):
    t0 = time.perf_counter()
    for _ in range(3):
        G._build_holdout_band(study, 262, 260.0, n_boot=nb, seed=1, selected_trades=trades)
    print(f"  band n_boot={nb:4d} (n_power {S.HOLDOUT_BAND_N_POWER} fixed inside _build_holdout_band): "
          f"{(time.perf_counter() - t0) / 3:.2f} s")
for npw in (1000, 300, 0):
    t0 = time.perf_counter()
    S.holdout_band(study, 262, 260.0, n_boot=2000, n_power=npw, seed=1, trades_per_day=0.4)
    print(f"  stats.holdout_band n_boot=2000 n_power={npw:4d}: {time.perf_counter() - t0:.2f} s")
t0 = time.perf_counter()
G.evaluate_gates(study, None, periods_per_year=260.0, selected_trades=trades, ledger_dir=ld)
print(f"  full evaluate_gates (no evaluator -> no plateau / cost stress): {time.perf_counter() - t0:.2f} s")
