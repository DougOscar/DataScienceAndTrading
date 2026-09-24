"""Smoke version of the Phase 1 exit test ("validate the validator", DESIGN §10).

Runs a handful of studies from research/calibration/phase1_validator_calibration.py through
the real pipeline (run_study → evaluate_gates) and asserts the headline properties:

* null systems (random entries on real EURUSD H1, best of 144 data-mined configs) fail;
* a correlated-grid null (shared entry schedule, only exits vary, ρ ≈ 0.8 — red-team B1) fails,
  and its DSR (V0, raw N) fails even though the v1.1 N_eff-based DSR would be lenient;
* a strong planted edge (noisy oracle, true net Sharpe ≈ 2.9) passes every statistical gate
  of DESIGN §4.2 v1.2 (``STAT_GATES``);
* synthetic zero-edge / isolated-spike / early-regime-only surfaces fail.

The full calibration (≈ 1,800 studies) is the script itself; see
research/calibration/2026-09-24_phase1_calibration_v12.md.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "calibration" / "phase1_validator_calibration.py"

pytestmark = pytest.mark.slow


def _load():
    spec = importlib.util.spec_from_file_location("phase1_validator_calibration", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    return mod


def _have(symbol: str) -> bool:
    try:
        from quantlab import data
        return data.catalog("FBS").filter(pl.col("symbol") == symbol).height > 0
    except Exception:
        return False


@pytest.fixture(scope="module")
def cal():
    warnings.filterwarnings("ignore")
    return _load()


def _run(cal, task, tmp_path):
    r = cal.run_task(task, str(tmp_path))
    assert r.get("error") is None, r.get("traceback")
    return r


@pytest.mark.skipif(not _have("EURUSD"), reason="EURUSD data not available")
def test_null_real_data_fails(cal, tmp_path):
    for s in range(3):
        r = _run(cal, {"task_id": f"smoke-null-{s}", "experiment": "null", "kind": "null",
                       "setup": "EURUSD_H1", "salt": 91000 + s}, tmp_path)
        assert r["stat_pass"] is False
        assert r["verdict"] == "FAIL"
        # the data-mined best config is still deflated away
        assert r["gate_status"]["dsr"] == "FAIL"


@pytest.mark.skipif(not _have("EURUSD"), reason="EURUSD data not available")
def test_correlated_grid_null_fails(cal, tmp_path):
    r = _run(cal, {"task_id": "smoke-corr-shared-0", "experiment": "corr", "family": "shared", "kind": "null",
                   "space": "exits", "setup": "EURUSD_H1", "cost": "zero", "salt": 91500}, tmp_path)
    assert r["mean_rho"] > 0.7                      # the grid really is highly correlated
    assert r["n_eff_eigen"] < 5                     # … and the N_eff estimators collapse (B1)
    assert r["n_trials_dsr"] == 126                 # the gate deflates by the raw trial count
    assert r["gate_status"]["dsr"] == "FAIL"
    assert r["stat_pass"] is False


@pytest.mark.skipif(not _have("EURUSD"), reason="EURUSD data not available")
def test_planted_strong_edge_passes(cal, tmp_path):
    for s in range(2):
        r = _run(cal, {"task_id": f"smoke-oracle-{s}", "experiment": "planted", "kind": "oracle",
                       "setup": "EURUSD_H1", "p": 0.72, "salt": 92000 + s}, tmp_path)
        assert r["true_sharpe_mean_trials"] > 2.0
        assert r["stat_pass"] is True, r["gate_status"]
        # mechanism is MANUAL, so the formal verdict is INCOMPLETE, never PASS without a human
        assert r["verdict"] == "INCOMPLETE"
        assert r["holdout_band"]["source"] == "wfo_oos"


def test_synthetic_surfaces(cal, tmp_path):
    z = _run(cal, {"task_id": "smoke-syn-zero", "experiment": "synthetic", "scenario": "zero",
                   "salt": 93000}, tmp_path)
    assert z["stat_pass"] is False
    sp = _run(cal, {"task_id": "smoke-syn-spike", "experiment": "synthetic", "scenario": "spike",
                    "height": 3.0, "salt": 93001}, tmp_path)
    # Rejected overall (plateau selection rarely centres on an isolated point, so the pick is
    # noise and fails DSR / OOS / trade count; see the v1.2 calibration report).
    assert sp["stat_pass"] is False
    rg = _run(cal, {"task_id": "smoke-syn-regime", "experiment": "synthetic", "scenario": "regime",
                    "height": 3.0, "salt": 93002}, tmp_path)
    # Rejected overall.  NB (calibration finding): the time-stability gates alone catch only
    # ~40% of these early-regime surfaces (an edge alive for 30% of 9 years keeps each year
    # under the 40% PnL share); DSR / OOS Sharpe / wfo_oos do most of the work.
    assert rg["stat_pass"] is False


def test_gate_names_match_library(cal):
    from quantlab import gates
    assert set(cal.STAT_GATES) == set(gates.GATE_ORDER) - {"mechanism"}


def test_clopper_pearson(cal):
    lo, hi = cal.clopper_pearson(0, 40)
    assert lo == 0.0 and abs(hi - 0.0881) < 1e-3
    lo, hi = cal.clopper_pearson(40, 40)
    assert hi == 1.0 and abs(lo - 0.9119) < 1e-3
