"""Phase 1 exit test — "validate the validator" (DESIGN §10 Phase 1), gate set DESIGN §4.2 v1.2.

Runs the REAL pipeline (``opt.run_study`` → ``gates.evaluate_gates``) many times on systems
whose truth is known, to measure

1. NULL      the false-pass rate of the §4.2 gates on real data with no edge
             (random entries + ATR stop/target, the best of 144 seeds×holds is data-mined);
2. PLANTED   the power curve: P(PASS) vs true net Sharpe for a *noisy oracle*;
3. SYNTHETIC SyntheticEvaluator surfaces (zero edge, broad plateau, isolated spike, regime shift);
4. HOLDOUT   the coverage / zero-edge power of the pre-registered holdout band (§4.4 v1.2):
             calibrate on 2016-05→2023-05-15, run the frozen WFO procedure on
             2023-05-15→2024-05-15 (1 year) and 2023-05-15→2025-05-15 (2 years), for the live
             edge and for the same edge killed at 2023-05-15.  The real locked holdout
             (≥ 2025-05-15) is never touched (RuleEvaluator refuses end > holdout start).
5. CORR      correlated-grid nulls (red-team B1): (a) random entries whose entry schedule and
             sides are SHARED by every configuration (one draw per study, schedule spaced by
             48 bars) — only the stop / target multiples and the hold (36/48) vary, ρ ≈ 0.8; (b) an SMA-crossover grid (fast × slow × stop)
             whose direction is multiplied by a random ±1 per calendar month (shared by all
             configs → the grid keeps SMA's realistic correlation, expected gross edge exactly 0);
             (c) descriptive: the plain SMA grid on 8 real H1 symbols (truth unknown).
6. DEAD      a planted oracle edge that dies at 45 % / 55 % of the dev window (red-team B2).
7. SOBOL     4-parameter random-entry / oracle studies with ``method="sobol"`` (B3).

!!! ``NoisyOracle`` reads FUTURE prices on purpose.  It is a calibration device that plants an
!!! edge of known size; it is NOT a strategy and must never be used as one.

Every study uses its own temporary ledger + studies directory (never ``research/ledger``).
Studies run in parallel on a process pool (≤ 8 workers, one study per worker, ``n_jobs=1``
inside ``run_study``).  Each finished study appends one JSON line to the checkpoint file, so
an interrupted run resumes where it stopped.  The v1.1 baseline results are archived read-only
at ``output/baseline_a741fa2/phase1_results.jsonl``; the v1.2 run writes
``output/phase1_v12_results.jsonl``.  Task ids and salts of the baseline families are unchanged,
so every v1.2 row can be paired with its v1.1 row.

Usage::

    PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py \
        --experiments null,planted,synthetic,holdout,corr,dead,sobol --workers 8
    PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --summarise

Headline definitions
--------------------
* ``stat_pass`` = every *statistical* gate of DESIGN §4.2 v1.2 (``STAT_GATES``: dsr,
  cscv_oos_loss, oos_sharpe, wfo_oos, cost_stress_sharpe, plateau, positive_years,
  max_year_share, trade_count) is PASS.  The mechanism gate is MANUAL by construction
  (hypothesis-specific ablation) and is treated as passed — the most lenient reading, so the
  measured null false-pass rate is an upper bound for the full gate set.
* ``true_sharpe`` of a study = mean full-dev net Sharpe across ALL its trials (every trial of a
  planted study carries the same edge; the seed is irrelevant), i.e. the Sharpe of a randomly
  drawn configuration — the edge a genuinely skilled system would have.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Thread caps must be set BEFORE polars/numba are imported; pool workers inherit the env.
if __name__ == "__main__":
    os.environ.setdefault("POLARS_MAX_THREADS", "1")
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantlab import evaluators, gates, metrics, opt  # noqa: E402
from quantlab import stats as st  # noqa: E402
from quantlab.contracts import Params, RiskType  # noqa: E402
from quantlab.costs import CostModel  # noqa: E402

OUT_DIR = ROOT / "research" / "calibration" / "output"
CHECKPOINT = OUT_DIR / "phase1_v12_results.jsonl"
BASELINE = OUT_DIR / "baseline_a741fa2" / "phase1_results.jsonl"      # v1.1 run, read-only
DEV_START = "2016-05-02"
DEV_END = "2025-05-15"            # exclusive: holdout start (evaluator default)
SPLIT_CAL_END = "2023-05-15"      # holdout-band experiment: calibration end (exclusive)
SPLIT_HO_END = "2024-05-15"       # 1-year pseudo-holdout end (exclusive) — inside dev
SPLIT_HO2_END = DEV_END           # 2-year pseudo-holdout end (exclusive) — still inside dev
MAX_WORKERS = 8
STAT_GATES = ("dsr", "cscv_oos_loss", "oos_sharpe", "wfo_oos", "cost_stress_sharpe", "plateau",
              "positive_years", "max_year_share", "trade_count")
# v1.1 baseline gate names (for reading the archived rows)
STAT_GATES_V11 = ("dsr", "pbo", "oos_sharpe", "cost_stress_sharpe", "plateau", "positive_years",
                  "max_year_share", "trade_count")
N_SOBOL = 144

# symbol/timeframe setups: entry rate per bar and the hold grid (bars)
SETUPS = {
    "EURUSD_H1": {"symbol": "EURUSD", "timeframe": "H1", "rate": 1 / 24, "holds": (12, 24, 48)},
    "XAUUSD_H4": {"symbol": "XAUUSD", "timeframe": "H4", "rate": 1 / 6, "holds": (3, 6, 12)},
}
SMA_SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "EURJPY", "GBPJPY", "XAUUSD")
for _s in SMA_SYMBOLS:
    SETUPS.setdefault(f"{_s}_H1", {"symbol": _s, "timeframe": "H1", "rate": 1 / 24, "holds": (12, 24, 48)})
N_SEEDS = 48                       # 48 seeds × 3 holds = 144 trials per study


def frac_date(frac: float, start: str = DEV_START, end: str = DEV_END) -> str:
    a, b = datetime.fromisoformat(start), datetime.fromisoformat(end)
    return (a + timedelta(days=round(frac * (b - a).days))).date().isoformat()


# =========================================================================== deterministic noise
_M64 = (1 << 64) - 1


def _uniform(ts_ms: np.ndarray, key: int, stream: int) -> np.ndarray:
    """U(0,1) per bar from a splitmix64 hash of (bar minute, key, stream).

    Keyed on the bar *timestamp*, not its row position, so the draw for a bar never depends
    on how much history was loaded (warm-up / window length) — the noise is identical in every
    window that contains the bar (needed for the holdout-split experiment)."""
    mix = np.uint64((key * 0x9E3779B97F4A7C15 + stream * 0xD1B54A32D192ED03 + 0x632BE59BD9B4E019) & _M64)
    with np.errstate(over="ignore"):
        z = (ts_ms.astype(np.int64) // 60000).astype(np.uint64) ^ mix
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def _atr(bars: pl.DataFrame, n: int) -> np.ndarray:
    c = pl.col("close")
    tr = pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - c.shift(1)).abs(),
                           (pl.col("low") - c.shift(1)).abs())
    return bars.select(tr.rolling_mean(n).alias("a"))["a"].to_numpy()


def _schedule(cand: np.ndarray, hold: int, n: int) -> list[int]:
    """Greedy non-overlapping entries: accept a candidate only after the previous trade's
    time exit row (entry + hold) has passed."""
    out, last_exit = [], -1
    for i in cand:
        if i > last_exit:
            out.append(int(i))
            last_exit = i + hold
    return out


def _frame(sig: np.ndarray, sig_valid: np.ndarray, stop: np.ndarray, target: np.ndarray | None) -> pl.DataFrame:
    """Signal frame: ``signal`` is null (keep state) except on entry / time-exit rows."""
    tgt = target if target is not None else np.full(stop.size, np.nan)
    return pl.DataFrame({"s": sig.astype(np.int8), "v": sig_valid, "stop_dist": stop, "target_dist": tgt}).select(
        pl.when(pl.col("v")).then(pl.col("s")).cast(pl.Int8).alias("signal"),
        pl.col("stop_dist").fill_nan(None),
        pl.col("target_dist").fill_nan(None),
    )


def _death_mask(bars: pl.DataFrame, death: str | None) -> np.ndarray:
    """True on bars at/after the (bar-timezone, naive) ``death`` date."""
    if death is None:
        return np.zeros(bars.height, bool)
    d = datetime.fromisoformat(death)
    ts = bars["ts"]
    lit = pl.lit(d).cast(ts.dtype) if ts.dtype.time_zone is None else pl.lit(d).dt.replace_time_zone(ts.dtype.time_zone)
    return bars.select((pl.col("ts") >= lit).alias("m"))["m"].to_numpy()


# =========================================================================== NULL strategy
@dataclass(frozen=True)
class RandomEntryParams(Params):
    seed: int = 0
    hold: int = 24
    salt: int = 0              # study master seed (fixed per study, not searched)
    rate: float = 1 / 24       # entry probability per bar (fixed)
    stop_mult: float = 2.0     # ATR multiples (fixed in the seed×hold grid; searched in "exits"/"sobol")
    target_mult: float = 3.0
    atr_len: int = 14
    sched_hold: int | None = None   # spacing of the entry schedule (None → hold); fixed → shared entries


class RandomEntryATR:
    """Zero-information entries: side and timing are coin flips; ATR stop/target + time exit.

    Causal (row i uses bars ≤ i: ATR and a timestamp hash).  Risk type A (fixed-fraction risk
    on the ATR stop).  Any Sharpe it shows is luck ± costs, so selecting the best seed is pure
    data-mining — the null the gates must reject.  With ``seed`` fixed and only the exits
    searched, every configuration shares one entry schedule (correlated-grid null)."""

    name = "calib_random_entry_atr"
    risk_type = RiskType.A
    params_cls = RandomEntryParams

    def __init__(self, params: RandomEntryParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        n = bars.height
        hold = int(p.hold)
        ts = bars["ts"].dt.epoch("ms").to_numpy()
        atr = _atr(bars, p.atr_len)
        key = p.salt * 100_003 + int(p.seed)
        u_entry, u_side = _uniform(ts, key, 1), _uniform(ts, key, 2)
        ok = np.isfinite(atr) & (atr > 0)
        cand = np.flatnonzero(ok & (u_entry < p.rate))
        sig = np.zeros(n, dtype=np.int8)
        valid = np.zeros(n, dtype=bool)
        for i in _schedule(cand, int(p.sched_hold or hold), n):
            sig[i] = 1 if u_side[i] < 0.5 else -1
            valid[i] = True
            if i + hold < n:
                sig[i + hold] = 0
                valid[i + hold] = True
        atr0 = np.where(ok, atr, np.nan)
        return _frame(sig, valid, atr0 * float(p.stop_mult), atr0 * float(p.target_mult))


# =========================================================================== PLANTED edge (look-ahead!)
@dataclass(frozen=True)
class OracleParams(Params):
    seed: int = 0
    hold: int = 24
    salt: int = 0
    p: float = 0.55            # accuracy of the oracle (fixed per study)
    rate: float = 1 / 24
    stop_mult: float = 3.0     # safety stop only; exit is by time (or target, if set)
    target_mult: float | None = None
    atr_len: int = 14
    death: str | None = None   # edge dies at this (bar-timezone) date: accuracy 0.5 from then on


class NoisyOracle:
    """CALIBRATION DEVICE — USES FUTURE DATA ON PURPOSE.  NOT A STRATEGY.

    Same random entry schedule as :class:`RandomEntryATR`, but the side is the sign of the
    price change over the trade's own life (open of bar i+1 → open of bar i+hold+1, the
    engine's actual entry/time-exit fills), reported correctly with probability ``p`` and
    flipped otherwise.  Every configuration carries the same planted edge (the seed is an
    irrelevant parameter), so a search still happens but the truth is a flat plateau.
    ``death``: entries on/after that date get accuracy 0.5 (a dead edge; net of costs < 0)."""

    name = "calib_noisy_oracle_LOOKAHEAD"
    risk_type = RiskType.A
    params_cls = OracleParams

    def __init__(self, params: OracleParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        n = bars.height
        hold = int(p.hold)
        ts = bars["ts"].dt.epoch("ms").to_numpy()
        op = bars["open"].to_numpy()
        atr = _atr(bars, p.atr_len)
        dead = _death_mask(bars, p.death)
        key = p.salt * 100_003 + int(p.seed)
        u_entry, u_flip = _uniform(ts, key, 1), _uniform(ts, key, 3)
        ok = np.isfinite(atr) & (atr > 0)
        ok[max(0, n - hold - 1):] = False                     # need the future exit fill
        cand = np.flatnonzero(ok & (u_entry < p.rate))
        sig = np.zeros(n, dtype=np.int8)
        valid = np.zeros(n, dtype=bool)
        for i in _schedule(cand, hold, n):
            move = op[i + hold + 1] - op[i + 1]                # LOOK-AHEAD (deliberate)
            if move == 0:
                continue
            side = 1 if move > 0 else -1
            if u_flip[i] >= (0.5 if dead[i] else p.p):
                side = -side
            sig[i] = side
            valid[i] = True
            sig[i + hold] = 0
            valid[i + hold] = True
        atr0 = np.where(ok | valid, atr, np.nan)
        tgt = atr0 * float(p.target_mult) if p.target_mult is not None else None
        return _frame(sig, valid, atr0 * float(p.stop_mult), tgt)


# =========================================================================== correlated SMA grid
@dataclass(frozen=True)
class SMAParams(Params):
    fast: int = 10
    slow: int = 100
    stop_mult: float = 3.0
    salt: int = 0
    flip: bool = True          # multiply the direction by a random ±1 per calendar month
    atr_len: int = 14


class SignRandomSMA:
    """SMA crossover (desired position = sign(SMA_fast − SMA_slow)), ATR safety stop.

    ``flip=True``: the direction is multiplied by a ±1 drawn per calendar month from the study
    salt, identical for every configuration — the configurations keep the realistic correlation
    of an SMA grid but the sign is independent of prices, so the expected gross edge is exactly
    zero (a correlated-grid NULL).  ``flip=False``: the plain SMA crossover (truth unknown).
    Signals are emitted only when the desired position changes (engine contract: after a stop
    the system waits for the next change)."""

    name = "calib_sma_signflip"
    risk_type = RiskType.A
    params_cls = SMAParams

    def __init__(self, params: SMAParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        n = bars.height
        c = bars["close"]
        f = c.rolling_mean(int(p.fast)).to_numpy()
        s = c.rolling_mean(int(p.slow)).to_numpy()
        atr = _atr(bars, p.atr_len)
        ok = np.isfinite(f) & np.isfinite(s) & np.isfinite(atr) & (atr > 0)
        d = np.where(ok, np.sign(f - s), 0.0).astype(np.int8)
        if p.flip:
            month = (bars["ts"].dt.year().cast(pl.Int64) * 12 + bars["ts"].dt.month().cast(pl.Int64)).to_numpy()
            u = _uniform(month * 60000, p.salt * 100_003 + 7, 5)
            d = (d * np.where(u < 0.5, 1, -1)).astype(np.int8)
        prev = np.concatenate([[0], d[:-1]])
        valid = ok & (d != prev)
        atr0 = np.where(ok, atr, np.nan)
        return _frame(d, valid, atr0 * float(p.stop_mult), None)


# =========================================================================== study plumbing
def _space(task: dict[str, Any]) -> opt.SearchSpace:
    sp = task.get("space", "seedhold")
    if sp == "seedhold":
        return opt.SearchSpace((opt.IntParam("seed", 0, N_SEEDS - 1),
                                opt.CategoricalParam("hold", SETUPS[task["setup"]]["holds"], ordered=True)))
    if sp == "exits":            # shared entry schedule; only exits vary (9 × 7 × 2 = 126; pilot ρ ≈ 0.79)
        return opt.SearchSpace((opt.FloatParam("stop_mult", 2.0, 4.0, step=0.25),
                                opt.FloatParam("target_mult", 3.0, 6.0, step=0.5),
                                opt.IntParam("hold", 36, 48, step=12)))
    if sp == "sma":              # 7 × 7 × 3 = 147 (pilot ρ ≈ 0.57, a typical SMA grid)
        return opt.SearchSpace((opt.IntParam("fast", 10, 40, step=5), opt.IntParam("slow", 120, 240, step=20),
                                opt.FloatParam("stop_mult", 2.0, 4.0, step=1.0)))
    if sp == "sobol4":
        return opt.SearchSpace((opt.IntParam("seed", 0, N_SEEDS - 1), opt.IntParam("hold", 12, 48),
                                opt.FloatParam("stop_mult", 1.5, 4.0), opt.FloatParam("target_mult", 1.5, 6.0)))
    if sp == "sobol4_oracle":
        return opt.SearchSpace((opt.IntParam("seed", 0, N_SEEDS - 1), opt.IntParam("hold", 12, 48),
                                opt.FloatParam("stop_mult", 2.0, 5.0), opt.FloatParam("target_mult", 2.0, 8.0)))
    raise ValueError(sp)


def _cost(name: str | None) -> CostModel:
    """``None``/"base": the book's base cost model.  "zero": no spread, no swap — a frictionless
    run used ONLY for the exact-zero-edge real-data null (the gross Sharpe of random entries is 0;
    with costs on it is negative, which makes the null too easy)."""
    if name in (None, "base"):
        return CostModel()
    if name == "zero":
        return CostModel(version_tag="calib-frictionless", spread_multiplier=0.0, swap_multiplier=0.0)
    raise ValueError(name)


def _rule_evaluator(task: dict[str, Any], end: str, death: str | None = "task") -> evaluators.RuleEvaluator:
    s = SETUPS[task["setup"]]
    kind = task["kind"]
    fixed: dict[str, Any] = {"salt": task["salt"]}
    if kind in ("null", "oracle"):
        fixed["rate"] = s["rate"]
    cls: Any = RandomEntryATR
    if kind == "oracle":
        cls = NoisyOracle
        fixed.update(p=task["p"], death=task.get("death") if death == "task" else death)
    elif kind == "sma":
        cls = SignRandomSMA
        fixed["flip"] = bool(task.get("flip", True))
    if task.get("space") == "exits":
        fixed.update(seed=0, sched_hold=48)       # one entry schedule + sides for every configuration
    return evaluators.RuleEvaluator(cls, symbol=s["symbol"], timeframe=s["timeframe"], book="FBS",
                                    start=DEV_START, end=end, risk_fraction=0.01, fixed_params=fixed,
                                    cost=_cost(task.get("cost")))


def _json(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    return str(x)


def _trial_sharpes(study, date_lt: str | None = None, date_ge: str | None = None) -> np.ndarray:
    R = study.returns
    if date_lt is not None:
        R = R.filter(pl.col("date") < pl.lit(datetime.fromisoformat(date_lt).date()))
    if date_ge is not None:
        R = R.filter(pl.col("date") >= pl.lit(datetime.fromisoformat(date_ge).date()))
    M = R.drop("date").to_numpy()
    return np.asarray(st.sharpe_per_period(M, axis=0), float) * math.sqrt(260.0)


def _mean_rho(study) -> float | None:
    """Mean pairwise correlation of the trial return columns (usable columns only)."""
    M = study.returns.drop("date").to_numpy().astype(float)
    M = np.nan_to_num(M)
    M = M[:, M.std(axis=0) > 0]
    if M.shape[1] < 2:
        return None
    C = np.corrcoef(M.T)
    iu = np.triu_indices_from(C, 1)
    return float(np.nanmean(C[iu]))


def _matrix_plateau(study, work: Path, ppy: float) -> dict[str, Any]:
    """The pre-round-2 matrix-based ±radius box plateau over the recorded trials (diagnostic
    only, for comparison with the judge-run gate).  Radius and space from the study's ledger row."""
    try:
        ctx = gates.ledger_context(study, work / "ledger")
        pm = gates.plateau_score(study, ppy, radius=ctx["plateau_radius"], space=ctx.get("space") or {})
        sc = pm["plateau_score"]
        status = "SKIPPED" if pm["n_neighbours"] == 0 else ("PASS" if sc >= gates.GATE_THRESHOLDS["plateau"][1]
                                                            else "FAIL")
        return {"score": sc, "n_neighbours": pm["n_neighbours"], "status": status}
    except Exception as e:  # noqa: BLE001 — diagnostic only
        return {"error": repr(e)[:200]}


def _summarise_report(rep: gates.GateReport) -> dict[str, Any]:
    stat = {g: rep.row(g).status for g in STAT_GATES}
    et = rep.effective_trials
    return {
        "verdict": rep.verdict,
        "stat_pass": all(s == "PASS" for s in stat.values()),
        "gate_status": stat,
        "gate_values": {g: rep.row(g).value for g in STAT_GATES},
        "wfo_diag": rep.diagnostics.get("wfo_oos"),
        "plateau_diag": rep.diagnostics.get("plateau"),
        "plateau_optimizer": rep.diagnostics.get("plateau_optimizer"),
        # DSR: gate (V0, raw N) + diagnostics
        "n_trials_dsr": et.get("n_trials"), "sr0_annual": et.get("sr0_annual"),
        "sr_selected": et.get("sr_annual"), "psr0": et.get("psr0"),
        "dsr_neff_cross": et.get("dsr_neff_cross"), "dsr_raw_cross": et.get("dsr_raw_cross"),
        "sr0_neff_cross_annual": et.get("sr0_neff_cross_annual"),
        "var_sr_cross": et.get("var_sr"), "n_obs": et.get("n_obs"),
        "n_eff": et.get("n_eff"), "n_eff_eigen": et.get("eigen"), "n_eff_cluster": et.get("cluster"),
        "n_eff_liji": et.get("liji"),
        "holdout_band": rep.holdout_band,
        "holdout_horizon_note": rep.diagnostics.get("holdout_horizon_note"),
        "cpcv_path_sharpe": rep.diagnostics.get("cpcv_path_sharpe"),
        "pbo_detail": rep.diagnostics.get("pbo_detail"),
        "cost_stress_by_swap_mult": rep.diagnostics.get("cost_stress_by_swap_mult"),
        "cost_stress_pip_points": rep.diagnostics.get("cost_stress_pip_points"),
        "min_trl_days": rep.diagnostics.get("min_trl_days"),
    }


def _run_real_study(task: dict[str, Any], work: Path) -> tuple[dict[str, Any], Any, Any, Any]:
    end = task.get("end", DEV_END)
    ev = _rule_evaluator(task, end)
    space = _space(task)
    method = task.get("method", "grid")
    t0 = time.perf_counter()
    study = opt.run_study(ev, space, book="FBS", system=f"calib_{task['kind']}", issue=0, attempt=1,
                          method=method, n_trials=N_SOBOL if method == "sobol" else None, n_jobs=1,
                          seed=task["salt"], study_id=task["task_id"],
                          cv=opt.CPCVConfig(10, 2), wfo=opt.WFOConfig(),
                          ledger_dir=work / "ledger", studies_dir=work / "studies")
    t_study = time.perf_counter() - t0
    t1 = time.perf_counter()
    # final run: the holdout-band horizon comes from the data manifest (DESIGN §4.4, 2026-09-24;
    # metadata only); the v1.2 run used an explicit 260-day horizon.
    hkw = {} if task.get("manifest_horizon") else {"holdout_days": 260}
    rep = gates.evaluate_gates(study, ev, periods_per_year=ev.periods_per_year, n_boot=2000,
                               seed=task["salt"], ledger_dir=work / "ledger", **hkw)
    t_gates = time.perf_counter() - t1
    srs = _trial_sharpes(study)
    row: dict[str, Any] = {"plateau_matrix": _matrix_plateau(study, work, ev.periods_per_year)}
    if "param_hold" in study.trials.columns and task.get("space", "seedhold") == "seedhold":
        tids = [int(c[1:]) for c in study.returns.columns if c != "date"]
        hold_of = dict(zip(study.trials["trial_id"].to_list(), study.trials["param_hold"].to_list()))
        by_hold = {str(h): float(np.nanmean([s for t, s in zip(tids, srs) if hold_of.get(t) == h]))
                   for h in SETUPS[task["setup"]]["holds"]}
        sel_hold = str(study.selected_params.get("hold")) if study.selected_params else None
        row["true_sharpe_by_hold"] = by_hold
        row["true_sharpe_selected_hold"] = by_hold.get(sel_hold) if sel_hold else None
    if task.get("death"):
        row["true_sharpe_alive"] = float(np.nanmean(_trial_sharpes(study, date_lt=task["death"])))
        row["true_sharpe_dead"] = float(np.nanmean(_trial_sharpes(study, date_ge=task["death"])))
    row.update({
        **_summarise_report(rep),
        "selected_params": study.selected_params,
        "true_sharpe_mean_trials": float(np.nanmean(srs)),
        "trial_sharpe_sd": float(np.nanstd(srs, ddof=1)),
        "trial_sharpe_max": float(np.nanmax(srs)),
        "n_trial_columns": int(srs.size),
        "mean_rho": _mean_rho(study),
        "median_trades": study.meta.get("median_trades_ok"),
        "embargo_days": study.meta["cpcv"]["embargo_days"],
        "embargo_capped": bool(study.meta.get("embargo_capped") or study.meta["cpcv"].get("embargo_capped")),
        "method": study.meta.get("method"),
        "runtime_study_s": t_study, "runtime_gates_s": t_gates,
    })
    return row, study, ev, rep


def _grid_matrices(ev, trials: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Re-evaluate every evaluated trial with ``ev``: (returns, exit-date counts, entry-date counts)."""
    cols: dict[str, Any] = {}
    xcols: dict[str, Any] = {}
    ecols: dict[str, Any] = {}
    dates = None
    for row in trials.iter_rows(named=True):
        if row["status"] not in ("ok", "low_trades"):
            continue
        out = ev(json.loads(row["params"]))
        d = out.daily.sort("date")
        if dates is None:
            dates = d["date"]
        k = f"t{row['trial_id']}"
        cols[k] = d["ret"].to_numpy()
        tr = out.trades.filter(~pl.col("skipped")) if "skipped" in out.trades.columns else out.trades
        grid = pl.DataFrame({"date": dates})
        for tcol, sink in (("exit_ts", xcols), ("entry_ts", ecols)):
            cnt = tr.group_by(pl.col(tcol).dt.date().alias("date")).agg(pl.len().alias("n"))
            sink[k] = grid.join(cnt, on="date", how="left").fill_null(0)["n"].to_numpy().astype(float)
    return (pl.DataFrame({"date": dates, **cols}), pl.DataFrame({"date": dates, **xcols}),
            pl.DataFrame({"date": dates, **ecols}))


def _crit(chk: dict[str, Any]) -> bool:
    """All four band criteria met (``pass`` before 2026-09-24; ``pass`` now means a decisive PASS)."""
    return bool(chk.get("criteria_pass", chk.get("pass")))


def _exam(R, X, E, trials, space, bands: dict[str, dict], ends: dict[str, str]) -> dict[str, Any]:
    """Frozen WFO procedure on the extended matrix; holdout_check per horizon."""
    wfo_oos, _wp, _ = opt.walk_forward(R, trials, space, opt.WFOConfig(), trade_counts=X, entry_counts=E,
                                       periods_per_year=260.0, method="grid")
    ho_start = datetime.fromisoformat(SPLIT_CAL_END).date()
    out: dict[str, Any] = {}
    for h, end in ends.items():
        e = datetime.fromisoformat(end).date()
        ho = wfo_oos.filter((pl.col("date") >= ho_start) & (pl.col("date") < e)).sort("date")
        n_tr = float(ho["n_trades"].fill_null(0).sum()) if "n_trades" in ho.columns else float("nan")
        band = bands.get(h)
        chk = st.holdout_check(band, ho.select("ret"), n_tr, 260.0) if band else None
        Rh = R.filter((pl.col("date") >= ho_start) & (pl.col("date") < e))
        out[h] = {"days": ho.height, "sharpe": metrics.sharpe(ho.select("ret"), 260.0), "trades": n_tr,
                  "check": chk,
                  "true_sharpe_mean_trials": float(np.nanmean(
                      np.asarray(st.sharpe_per_period(Rh.drop("date").to_numpy(), axis=0)) * math.sqrt(260.0)))}
    return out


def _holdout_exam(task: dict[str, Any], study, rep) -> dict[str, Any]:
    """Run the frozen WFO procedure over [SPLIT_CAL_END, +1y) and [SPLIT_CAL_END, +2y) and
    apply the pre-registered bands (1-year band from the gates; 2-year band built the same way
    with horizon 520 days).  For oracle studies the exam is repeated with the SAME system whose
    edge dies at SPLIT_CAL_END (identical on the calibration window → same study and band)."""
    space = _space(task)
    trials = study.trials
    tpd, tsrc = gates.wfo_trades_per_day(study)
    b2 = st.holdout_band(study, 520, 260.0, n_boot=2000, trades_per_day=tpd, seed=task["salt"],
                         trades_source=tsrc, target_coverage=gates.HOLDOUT_TARGET_COVERAGE,
                         max_zero_edge_pass=gates.HOLDOUT_MAX_ZERO_EDGE_PASS).as_dict()
    bands = {"1y": rep.holdout_band, "2y": b2}
    ends = {"1y": SPLIT_HO_END, "2y": SPLIT_HO2_END}
    out: dict[str, Any] = {"holdout_band_2y": b2}
    ev_live = _rule_evaluator(task, SPLIT_HO2_END)
    out["exam_live"] = _exam(*_grid_matrices(ev_live, trials), trials, space, bands, ends)
    if task["kind"] == "oracle":
        ev_dead = _rule_evaluator(task, SPLIT_HO2_END, death=SPLIT_CAL_END)
        out["exam_dead"] = _exam(*_grid_matrices(ev_dead, trials), trials, space, bands, ends)
    return out


# ---------------------------------------------------------------- synthetic scenarios
def _synthetic_evaluator(task: dict[str, Any]) -> evaluators.SyntheticEvaluator:
    sc, h, seed = task["scenario"], task.get("height", 0.0), task["salt"]
    bounds = {"x": (0.0, 11.0), "y": (0.0, 11.0)}
    common = dict(bounds=bounds, n_days=2340, vol=0.005, rho=task.get("rho", 0.3), seed=seed,
                  trades_per_year=150.0, hold_days=2)
    if sc == "zero":
        return evaluators.SyntheticEvaluator(**common, base_sharpe=0.0)
    if sc == "plateau":          # broad true plateau: box of half-width 0.35 (≈ 8×8 of the 12×12 grid)
        return evaluators.SyntheticEvaluator(**common, base_sharpe=0.0, bumps=(
            {"center": {"x": 5.5, "y": 5.5}, "height": h, "width": 0.35, "kind": "box"},))
    if sc == "spike":            # a single grid point carries the edge
        return evaluators.SyntheticEvaluator(**common, base_sharpe=0.0, bumps=(
            {"center": {"x": 7.0, "y": 3.0}, "height": h, "width": 0.04, "kind": "box"},))
    if sc == "regime":           # broad edge, but only in the first 30 % of the sample
        return evaluators.SyntheticEvaluator(**common, base_sharpe=0.0, bumps=(
            {"center": {"x": 5.5, "y": 5.5}, "height": h, "width": 0.35, "kind": "box"},),
            regimes=({"from": 0.30, "bumps": (), "base_sharpe": 0.0},))
    raise ValueError(sc)


def _run_synthetic(task: dict[str, Any], work: Path) -> dict[str, Any]:
    ev = _synthetic_evaluator(task)
    space = opt.SearchSpace((opt.IntParam("x", 0, 11), opt.IntParam("y", 0, 11)))
    t0 = time.perf_counter()
    study = opt.run_study(ev, space, book="FBS", system=f"calib_syn_{task['scenario']}", issue=0, attempt=1,
                          method="grid", n_jobs=1, seed=task["salt"], study_id=task["task_id"],
                          cv=opt.CPCVConfig(10, 2), wfo=opt.WFOConfig(),
                          ledger_dir=work / "ledger", studies_dir=work / "studies")
    t_study = time.perf_counter() - t0
    # SyntheticEvaluator has no cost model: pass a neutral CostModel so the cost-stress gate runs
    # (stress has no effect on a synthetic series → cost-stress Sharpe = base Sharpe).
    # SyntheticEvaluator trades carry no P&L: give the gate per-trade returns (compounded daily
    # returns over each trade's life) so the per-trade MinTRL condition is actually tested.
    o = ev(study.selected_params)
    d = o.daily.with_columns(pl.col("date").cast(pl.Date))
    g = np.cumprod(1.0 + d["ret"].to_numpy())
    dd = d["date"].to_numpy().astype("datetime64[D]")
    ei = np.searchsorted(dd, o.trades["entry_ts"].dt.date().to_numpy().astype("datetime64[D]"))
    xi = np.searchsorted(dd, o.trades["exit_ts"].dt.date().to_numpy().astype("datetime64[D]"))
    tr = o.trades.with_columns(pl.Series("ret", g[xi] / g[ei] - 1.0))
    rep = gates.evaluate_gates(study, ev, periods_per_year=260.0, base_cost=CostModel(), n_boot=2000,
                               seed=task["salt"], selected_trades=tr, ledger_dir=work / "ledger",
                               holdout_days=260)  # pseudo-holdout: 1 y after SPLIT_CAL_END, not the manifest
    sel = study.selected_params
    return {**_summarise_report(rep), "selected_params": sel,
            "plateau_matrix": _matrix_plateau(study, work, 260.0),
            "true_sharpe_selected": ev.true_sharpe(sel) if sel else None,
            "true_sharpe_selected_late": ev.true_sharpe(sel, 0.99) if sel else None,
            "true_sharpe_max": task.get("height", 0.0),
            "true_sharpe_mean_trials": float(np.mean([ev.true_sharpe(p) for p in space.grid()])),
            "trial_sharpe_max": float(np.nanmax(_trial_sharpes(study))),
            "mean_rho": _mean_rho(study),
            "runtime_study_s": t_study, "runtime_gates_s": time.perf_counter() - t0 - t_study}


# ---------------------------------------------------------------- worker
def _worker_init() -> None:
    os.environ["POLARS_MAX_THREADS"] = "1"
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    warnings.filterwarnings("ignore")


def run_task(task: dict[str, Any], work_root: str | None = None) -> dict[str, Any]:
    """One calibration study end to end; returns a JSON-able result row (never raises)."""
    warnings.filterwarnings("ignore")
    work = Path(tempfile.mkdtemp(prefix=f"calib_{task['task_id']}_", dir=work_root))
    t0 = time.perf_counter()
    try:
        if task["experiment"] == "synthetic":
            row = _run_synthetic(task, work)
        else:
            if len(evaluators._CACHE) > 2:          # bound per-worker RAM (≤ 2 datasets cached)
                evaluators.clear_cache()
            row, study, ev, rep = _run_real_study(task, work)
            if task["experiment"] == "holdout":
                row.update(_holdout_exam(task, study, rep))
        row["error"] = None
    except Exception as e:  # noqa: BLE001 — recorded, the batch goes on
        row = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=8)}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    row["runtime_total_s"] = time.perf_counter() - t0
    return _json({**task, **row})


# =========================================================================== task lists
# p levels chosen from the pilot (see the v1.1 report) to span true net Sharpe ≈ 0.3 … 3;
# 0.59 / 0.61 added in v1.2 to resolve the predicted 80 % point (≈ SR 1.2).
ORACLE_P = (0.54, 0.555, 0.575, 0.59, 0.60, 0.61, 0.62, 0.65, 0.68, 0.72)
HOLDOUT_P = (0.60, 0.65, 0.72)
DEAD_P = (0.62, 0.68, 0.72)
DEAD_FRACS = (0.45, 0.55)


def build_tasks(experiments: set[str], n_null: int = 200, n_null_xau: int = 100, n_planted: int = 20,
                n_syn: int = 200, n_ho: int = 20, n_ho_null: int = 40, n_corr: int = 100,
                n_dead: int = 20, n_sobol: int = 40) -> list[dict[str, Any]]:
    T: list[dict[str, Any]] = []
    if "null" in experiments:
        T += [{"task_id": f"null-eur-{s:03d}", "experiment": "null", "kind": "null", "setup": "EURUSD_H1",
               "salt": 1000 + s} for s in range(n_null)]
        T += [{"task_id": f"null-xau-{s:03d}", "experiment": "null", "kind": "null", "setup": "XAUUSD_H4",
               "salt": 2000 + s} for s in range(n_null_xau)]
        T += [{"task_id": f"null-eur0-{s:03d}", "experiment": "null", "kind": "null", "setup": "EURUSD_H1",
               "salt": 2500 + s, "cost": "zero"} for s in range(n_null_xau)]
    if "planted" in experiments:
        for p in ORACLE_P:
            T += [{"task_id": f"planted-eur-p{p:.3f}-{s:03d}", "experiment": "planted", "kind": "oracle",
                   "setup": "EURUSD_H1", "p": p, "salt": 3000 + s} for s in range(n_planted)]
    if "holdout" in experiments:
        for p in HOLDOUT_P:
            T += [{"task_id": f"holdout-eur-p{p:.3f}-{s:03d}", "experiment": "holdout", "kind": "oracle",
                   "setup": "EURUSD_H1", "p": p, "salt": 4000 + s, "end": SPLIT_CAL_END} for s in range(n_ho)]
        T += [{"task_id": f"holdout-eur-null-{s:03d}", "experiment": "holdout", "kind": "null",
               "setup": "EURUSD_H1", "salt": 5000 + s, "end": SPLIT_CAL_END} for s in range(n_ho_null)]
    if "synthetic" in experiments:
        T += [{"task_id": f"syn-zero-{s:03d}", "experiment": "synthetic", "scenario": "zero", "salt": 6000 + s}
              for s in range(n_syn)]
        for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
            T += [{"task_id": f"syn-plateau-h{h:.1f}-{s:03d}", "experiment": "synthetic", "scenario": "plateau",
                   "height": h, "salt": 7000 + s} for s in range(max(10, n_syn // 5))]
        T += [{"task_id": f"syn-spike-h3.0-{s:03d}", "experiment": "synthetic", "scenario": "spike",
               "height": 3.0, "salt": 8000 + s} for s in range(max(10, n_syn // 5))]
        T += [{"task_id": f"syn-regime-h3.0-{s:03d}", "experiment": "synthetic", "scenario": "regime",
               "height": 3.0, "salt": 9000 + s} for s in range(max(10, n_syn // 5))]
    if "corr" in experiments:
        base = {"experiment": "corr", "setup": "EURUSD_H1"}
        T += [{**base, "task_id": f"corr-shared-eur0-{s:03d}", "family": "shared", "kind": "null",
               "space": "exits", "cost": "zero", "salt": 11000 + s} for s in range(n_corr)]
        T += [{**base, "task_id": f"corr-shared-eur-{s:03d}", "family": "shared", "kind": "null",
               "space": "exits", "salt": 12000 + s} for s in range(n_corr)]
        T += [{**base, "task_id": f"corr-sma-eur0-{s:03d}", "family": "sma_flip", "kind": "sma",
               "space": "sma", "cost": "zero", "salt": 13000 + s} for s in range(n_corr)]
        T += [{**base, "task_id": f"corr-sma-eur-{s:03d}", "family": "sma_flip", "kind": "sma",
               "space": "sma", "salt": 14000 + s} for s in range(n_corr // 2)]
        for sym in SMA_SYMBOLS:
            for c in (None, "zero"):
                T.append({"task_id": f"sma-real-{sym}-{c or 'base'}", "experiment": "corr", "family": "sma_real",
                          "kind": "sma", "space": "sma", "setup": f"{sym}_H1", "salt": 0, "flip": False,
                          **({"cost": c} if c else {})})
    if "dead" in experiments:
        for p in DEAD_P:
            for fr in DEAD_FRACS:
                T += [{"task_id": f"dead-p{p:.3f}-d{int(fr * 100)}-{s:03d}", "experiment": "dead", "kind": "oracle",
                       "setup": "EURUSD_H1", "p": p, "death_frac": fr, "death": frac_date(fr),
                       "salt": 3000 + s} for s in range(n_dead)]
    if "sobol" in experiments:
        base = {"experiment": "sobol", "setup": "EURUSD_H1", "method": "sobol"}
        T += [{**base, "task_id": f"sobol-null-eur0-{s:03d}", "kind": "null", "space": "sobol4", "cost": "zero",
               "salt": 16000 + s} for s in range(n_sobol)]
        T += [{**base, "task_id": f"sobol-null-eur-{s:03d}", "kind": "null", "space": "sobol4",
               "salt": 17000 + s} for s in range(n_sobol)]
        T += [{**base, "task_id": f"sobol-oracle-p0.650-{s:03d}", "kind": "oracle", "space": "sobol4_oracle",
               "p": 0.65, "salt": 18000 + s} for s in range(n_sobol // 2)]
    return T


FINAL_CHECKPOINT = OUT_DIR / "phase1_final_results.jsonl"
FINAL_ORACLE_P = (0.555, 0.575, 0.59, 0.60, 0.61, 0.62, 0.65, 0.72)


def build_final_tasks(n_null: int = 100, n_chf: int = 60, n_corr: int = 100, n_sobol0: int = 60,
                      n_sobol: int = 40, n_planted: int = 20, n_dead: int = 15, n_syn: int = 100,
                      n_syn_other: int = 20) -> list[dict[str, Any]]:
    """Targeted final confirmation run (2026-09-25, library @ 48e696f): judge-run plateau (min over
    axes), L1 eval-start clamp, L4 hold in rows, manifest holdout horizon.  Task ids and salts
    are those of the v1.2 run wherever the family existed (paired rows); new families:
    ``null-chf`` (USDCHF H1, L1 check) and ``sobol-oracle-p0.620``."""
    T: list[dict[str, Any]] = []
    mh = {"manifest_horizon": True}
    T += [{"task_id": f"null-eur-{s:03d}", "experiment": "null", "kind": "null", "setup": "EURUSD_H1",
           "salt": 1000 + s, **mh} for s in range(n_null)]
    T += [{"task_id": f"null-eur0-{s:03d}", "experiment": "null", "kind": "null", "setup": "EURUSD_H1",
           "salt": 2500 + s, "cost": "zero", **mh} for s in range(n_null)]
    T += [{"task_id": f"null-xau-{s:03d}", "experiment": "null", "kind": "null", "setup": "XAUUSD_H4",
           "salt": 2000 + s, **mh} for s in range(n_null)]
    T += [{"task_id": f"null-chf-{s:03d}", "experiment": "null", "kind": "null", "setup": "USDCHF_H1",
           "salt": 19000 + s, **mh} for s in range(n_chf)]
    base = {"experiment": "corr", "setup": "EURUSD_H1", **mh}
    T += [{**base, "task_id": f"corr-shared-eur0-{s:03d}", "family": "shared", "kind": "null",
           "space": "exits", "cost": "zero", "salt": 11000 + s} for s in range(n_corr)]
    T += [{**base, "task_id": f"corr-sma-eur0-{s:03d}", "family": "sma_flip", "kind": "sma",
           "space": "sma", "cost": "zero", "salt": 13000 + s} for s in range(n_corr)]
    base = {"experiment": "sobol", "setup": "EURUSD_H1", "method": "sobol", **mh}
    T += [{**base, "task_id": f"sobol-null-eur0-{s:03d}", "kind": "null", "space": "sobol4", "cost": "zero",
           "salt": 16000 + s} for s in range(n_sobol0)]
    T += [{**base, "task_id": f"sobol-null-eur-{s:03d}", "kind": "null", "space": "sobol4",
           "salt": 17000 + s} for s in range(n_sobol)]
    T += [{**base, "task_id": f"sobol-oracle-p0.650-{s:03d}", "kind": "oracle", "space": "sobol4_oracle",
           "p": 0.65, "salt": 18000 + s} for s in range(20)]
    T += [{**base, "task_id": f"sobol-oracle-p0.620-{s:03d}", "kind": "oracle", "space": "sobol4_oracle",
           "p": 0.62, "salt": 18500 + s} for s in range(15)]
    for p in FINAL_ORACLE_P:
        T += [{"task_id": f"planted-eur-p{p:.3f}-{s:03d}", "experiment": "planted", "kind": "oracle",
               "setup": "EURUSD_H1", "p": p, "salt": 3000 + s, **mh} for s in range(n_planted)]
    for p in DEAD_P:
        for fr in DEAD_FRACS:
            T += [{"task_id": f"dead-p{p:.3f}-d{int(fr * 100)}-{s:03d}", "experiment": "dead", "kind": "oracle",
                   "setup": "EURUSD_H1", "p": p, "death_frac": fr, "death": frac_date(fr),
                   "salt": 3000 + s, **mh} for s in range(n_dead)]
    T += [{"task_id": f"syn-zero-{s:03d}", "experiment": "synthetic", "scenario": "zero", "salt": 6000 + s}
          for s in range(n_syn)]
    for h in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        T += [{"task_id": f"syn-plateau-h{h:.1f}-{s:03d}", "experiment": "synthetic", "scenario": "plateau",
               "height": h, "salt": 7000 + s} for s in range(n_syn_other)]
    T += [{"task_id": f"syn-spike-h3.0-{s:03d}", "experiment": "synthetic", "scenario": "spike",
           "height": 3.0, "salt": 8000 + s} for s in range(2 * n_syn_other)]
    T += [{"task_id": f"syn-regime-h3.0-{s:03d}", "experiment": "synthetic", "scenario": "regime",
           "height": 3.0, "salt": 9000 + s} for s in range(2 * n_syn_other)]
    return T


def load_results(path: Path = CHECKPOINT) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["task_id"] not in rows or not r.get("error"):
                rows[r["task_id"]] = r          # last successful row per task wins (dedupe/backfill)
    return list(rows.values())


def run_all(tasks: list[dict[str, Any]], workers: int = MAX_WORKERS, checkpoint: Path = CHECKPOINT,
            work_root: str | None = None, log=print, require_key: str | None = None) -> list[dict[str, Any]]:
    """Run ``tasks`` on a process pool, skipping any task_id already in ``checkpoint``
    (``require_key``: also re-run done tasks whose row lacks that key — backfill)."""
    if checkpoint.resolve() == BASELINE.resolve():
        raise ValueError("refusing to append to the archived v1.1 baseline results")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in load_results(checkpoint)
            if not r.get("error") and (require_key is None or r.get(require_key) is not None)}
    todo = [t for t in tasks if t["task_id"] not in done]
    log(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run on {workers} workers")
    # long tasks first (holdout exams, then real data), grouped by setup for the data cache
    todo.sort(key=lambda t: (t["experiment"] == "synthetic", t["experiment"] != "holdout", t.get("setup", ""),
                             t.get("end", ""), t["task_id"]))
    t0 = time.perf_counter()
    out = []
    workers = max(1, min(int(workers), MAX_WORKERS))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futs = {pool.submit(run_task, t, work_root): t for t in todo}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            with checkpoint.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            out.append(r)
            el = time.perf_counter() - t0
            log(f"[{k}/{len(todo)}] {r['task_id']}: {'ERROR ' + r['error'] if r.get('error') else r.get('verdict')}"
                f" stat_pass={r.get('stat_pass')} ({r.get('runtime_total_s', 0):.0f}s; elapsed {el / 60:.1f} min)")
            sys.stdout.flush()
    return out


# =========================================================================== analysis
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _rate_k(k: int, n: int) -> str:
    lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return f"{k}/{n} = {k / n if n else float('nan'):.1%} [{lo:.1%}, {hi:.1%}]"


def _rate(rows, key="stat_pass") -> str:
    return _rate_k(sum(bool(r.get(key) if not callable(key) else key(r)) for r in rows), len(rows))


def pass_without(r: dict[str, Any], drop: tuple[str, ...]) -> bool:
    return all(s == "PASS" for g, s in r["gate_status"].items() if g not in drop)


def wfo_recent(r: dict[str, Any]) -> float:
    w = r.get("wfo_diag") or {}
    v = w.get("sharpe_recent")
    return float("nan") if v is None else float(v)


def pass_wfo_recent_at(r: dict[str, Any], thr: float, strict: bool) -> bool:
    """stat_pass with the wfo_oos recent-third condition replaced by (> thr | ≥ thr)."""
    if not pass_without(r, ("wfo_oos",)):
        return False
    w = r.get("wfo_diag") or {}
    a, rec = w.get("sharpe_all"), w.get("sharpe_recent")
    if a is None or rec is None or r["gate_status"]["wfo_oos"] == "SKIPPED":
        return False
    return a >= gates.GATE_THRESHOLDS["wfo_oos"][1] and (rec > thr if strict else rec >= thr)


def standalone(rows, g: str) -> int:
    return sum(r["gate_status"][g] == "PASS" for r in rows)


def logistic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """MLE logistic P(y=1) = 1/(1+exp(-(a+b x))); returns (a, b, x50, x80)."""
    from scipy.optimize import minimize
    x, y = np.asarray(x, float), np.asarray(y, float)

    def nll(th):
        z = th[0] + th[1] * x
        return float(np.sum(np.logaddexp(0, z) - y * z))
    res = minimize(nll, np.array([-3.0, 2.0]), method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10,
                                                                               "maxiter": 5000})
    a, b = res.x
    return float(a), float(b), float(-a / b), float((math.log(4.0) - a) / b)


def gate_rejection_table(rows: list[dict[str, Any]], gates_: tuple[str, ...] = STAT_GATES) -> str:
    n = len(rows)
    L = ["| Gate | Reject (FAIL or SKIPPED) | of which SKIPPED | Only-rejecting | Standalone pass | Median value |",
         "|---|---|---|---|---|---|"]
    for g in gates_:
        rej = [r for r in rows if r["gate_status"][g] != "PASS"]
        sk = sum(r["gate_status"][g] == "SKIPPED" for r in rows)
        only = [r for r in rej if sum(s != "PASS" for s in r["gate_status"].values()) == 1]
        vals = [r["gate_values"][g] for r in rows if r["gate_values"].get(g) is not None]
        med = f"{np.median(vals):.3g}" if vals else "—"
        L.append(f"| {g} | {len(rej)}/{n} ({len(rej) / n:.0%}) | {sk} | {len(only)} | {n - len(rej)} | {med} |")
    return "\n".join(L)


FAMILIES = (
    ("NULL EURUSD H1 costs (seed×hold)", lambda r: r["experiment"] == "null" and r["setup"] == "EURUSD_H1"
     and r.get("cost") is None),
    ("NULL XAUUSD H4 costs (seed×hold)", lambda r: r["experiment"] == "null" and r["setup"] == "XAUUSD_H4"),
    ("NULL EURUSD H1 frictionless (seed×hold)", lambda r: r["experiment"] == "null" and r.get("cost") == "zero"),
    ("NULL EURUSD H1 split window", lambda r: r["experiment"] == "holdout" and r["kind"] == "null"),
    ("CORR shared schedule, frictionless", lambda r: r["experiment"] == "corr" and r.get("family") == "shared"
     and r.get("cost") == "zero"),
    ("CORR shared schedule, costs", lambda r: r["experiment"] == "corr" and r.get("family") == "shared"
     and r.get("cost") is None),
    ("CORR sign-random SMA, frictionless", lambda r: r["experiment"] == "corr" and r.get("family") == "sma_flip"
     and r.get("cost") == "zero"),
    ("CORR sign-random SMA, costs", lambda r: r["experiment"] == "corr" and r.get("family") == "sma_flip"
     and r.get("cost") is None),
    ("SOBOL null frictionless (4 params)", lambda r: r["experiment"] == "sobol" and r["kind"] == "null"
     and r.get("cost") == "zero"),
    ("SOBOL null costs (4 params)", lambda r: r["experiment"] == "sobol" and r["kind"] == "null"
     and r.get("cost") is None),
    ("NULL USDCHF H1 costs (seed×hold)", lambda r: r["experiment"] == "null" and r["setup"] == "USDCHF_H1"),
    ("SYN zero", lambda r: r.get("scenario") == "zero"),
)


def summarise(results: list[dict[str, Any]]) -> str:
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    L = [f"# Phase 1 calibration v1.2 — summary ({len(ok)} studies ok, {len(err)} errors)", ""]
    if err:
        L += ["Errors: " + ", ".join(f"{r['task_id']} ({r['error'][:80]})" for r in err[:20]), ""]
    # ---- nulls
    L += ["## NULL families", "",
          "| Family | n | stat_pass | mean ρ | N_eff eigen / cluster / Li–Ji (median) | mean trial SR | median best SR |"
          " median SR0 (V0, raw N) | DSR standalone pass | v1.1-DSR (N_eff, V_cross) standalone |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    allnull, realnull = [], []
    for name, sel in FAMILIES:
        rows = [r for r in ok if sel(r)]
        if not rows:
            continue
        allnull += rows
        if not name.startswith("SYN"):
            realnull += rows
        med = lambda k: np.nanmedian([np.nan if r.get(k) is None else r[k] for r in rows])  # noqa: E731
        L.append(f"| {name} | {len(rows)} | {_rate(rows)} | {med('mean_rho'):.2f} | {med('n_eff_eigen'):.1f} / "
                 f"{med('n_eff_cluster'):.1f} / {med('n_eff_liji'):.1f} | "
                 f"{np.mean([r['true_sharpe_mean_trials'] for r in rows]):.2f} | {med('trial_sharpe_max'):.2f} | "
                 f"{med('sr0_annual'):.2f} | {_rate_k(standalone(rows, 'dsr'), len(rows))} | "
                 f"{_rate_k(sum((r.get('dsr_neff_cross') or 0) >= 0.95 for r in rows), len(rows))} |")
    if realnull:
        L += ["", f"**All real-data nulls:** {_rate(realnull)}; **all nulls incl. synthetic:** {_rate(allnull)}", ""]
    for name, sel in FAMILIES:
        rows = [r for r in ok if sel(r)]
        if rows:
            L += [f"### {name} — gate by gate", "", gate_rejection_table(rows), "",
                  "Leave-one-gate-out pass counts: " + ", ".join(
                      f"−{g}: {sum(pass_without(r, (g,)) for r in rows)}" for g in STAT_GATES), ""]
    # ---- descriptive SMA
    rows = [r for r in ok if r.get("family") == "sma_real"]
    if rows:
        L += ["## Plain SMA grid on real H1 data (descriptive, truth unknown)", "",
              "| study | mean ρ | N_eff eigen | best SR | selected SR | verdict | failing gates |", "|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: r["task_id"]):
            fails = [g for g, s in r["gate_status"].items() if s != "PASS"]
            L.append(f"| {r['task_id']} | {r['mean_rho']:.2f} | {r['n_eff_eigen']:.1f} | {r['trial_sharpe_max']:.2f} |"
                     f" {r['sr_selected']:.2f} | {'PASS' if r['stat_pass'] else 'FAIL'} | {', '.join(fails)} |")
        L.append("")
    # ---- planted power
    for name, rows, key in (("PLANTED (real, oracle)", [r for r in ok if r["experiment"] == "planted"], "p"),
                            ("SYN plateau", [r for r in ok if r.get("scenario") == "plateau"], "height")):
        if not rows:
            continue
        L += [f"## {name} — power curve", "",
              f"| {key} | n | true SR | median OOS SR | P(PASS) v1.2 | 95% CP | w/o wfo_oos | recent ≥ 0.5 | "
              f"failing gates (count) |", "|---|---|---|---|---|---|---|---|---|"]
        for v in sorted({r[key] for r in rows}):
            rr = [r for r in rows if r[key] == v]
            k = sum(r["stat_pass"] for r in rr)
            lo, hi = clopper_pearson(k, len(rr))
            fails: dict[str, int] = {}
            for r in rr:
                for g, s in r["gate_status"].items():
                    if s != "PASS":
                        fails[g] = fails.get(g, 0) + 1
            top = ", ".join(f"{g} {c}" for g, c in sorted(fails.items(), key=lambda x: -x[1])) or "—"
            L.append(f"| {v} | {len(rr)} | {np.mean([r['true_sharpe_mean_trials'] for r in rr]):.2f} | "
                     f"{np.nanmedian([r['gate_values']['oos_sharpe'] or np.nan for r in rr]):.2f} | {k / len(rr):.0%} | "
                     f"[{lo:.0%}, {hi:.0%}] | {np.mean([pass_without(r, ('wfo_oos',)) for r in rr]):.0%} | "
                     f"{np.mean([pass_wfo_recent_at(r, 0.5, False) for r in rr]):.0%} | {top} |")
        if key == "p":
            x = np.array([r["true_sharpe_mean_trials"] for r in rows])
            for lab, fn in (("v1.2", lambda r: r["stat_pass"]),
                            ("v1.2 without wfo_oos", lambda r: pass_without(r, ("wfo_oos",))),
                            ("v1.2 with recent-third ≥ 0.5", lambda r: pass_wfo_recent_at(r, 0.5, False))):
                y = np.array([bool(fn(r)) for r in rows], float)
                a, b, x50, x80 = logistic_fit(x, y)
                L.append(f"\nLogistic fit ({lab}): 50 % at SR {x50:.2f}, **80 % at SR {x80:.2f}** (a={a:.2f}, b={b:.2f})")
        L.append("")
    # ---- dead edge
    dead = [r for r in ok if r["experiment"] == "dead"]
    if dead:
        L += ["## DEAD edge", "", "| p | death | n | SR alive | SR dead | full-dev SR | stat_pass v1.2 | without wfo_oos |"
              " recent > 0 | recent ≥ 0.5 | wfo_oos fails | median WFO SR all / recent |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in DEAD_P:
            for fr in DEAD_FRACS:
                rr = [r for r in dead if r["p"] == p and r["death_frac"] == fr]
                if not rr:
                    continue
                L.append(f"| {p} | {fr:.0%} | {len(rr)} | {np.mean([r['true_sharpe_alive'] for r in rr]):.2f} | "
                         f"{np.mean([r['true_sharpe_dead'] for r in rr]):.2f} | "
                         f"{np.mean([r['true_sharpe_mean_trials'] for r in rr]):.2f} | {_rate(rr)} | "
                         f"{_rate(rr, lambda r: pass_without(r, ('wfo_oos',)))} | "
                         f"{_rate(rr, lambda r: pass_wfo_recent_at(r, 0.0, True))} | "
                         f"{_rate(rr, lambda r: pass_wfo_recent_at(r, 0.5, False))} | "
                         f"{sum(r['gate_status']['wfo_oos'] != 'PASS' for r in rr)} | "
                         f"{np.median([(r.get('wfo_diag') or {}).get('sharpe_all', np.nan) for r in rr]):.2f} / "
                         f"{np.median([wfo_recent(r) for r in rr]):.2f} |")
        L.append("")
    # ---- sobol oracle
    so = [r for r in ok if r["experiment"] == "sobol" and r["kind"] == "oracle"]
    if so:
        L += ["## SOBOL oracle p=0.65", "", f"stat_pass {_rate(so)}; true SR {np.mean([r['true_sharpe_mean_trials'] for r in so]):.2f}",
              "", gate_rejection_table(so), ""]
    # ---- holdout
    ho = [r for r in ok if r["experiment"] == "holdout" and r.get("exam_live")]
    if ho:
        L += ["## HOLDOUT (dev split)", "",
              "| group | h | n | band α | P(pass|zero) band | decisive | live PASS | S | R | D | T | dead PASS | mean live HO SR |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for p in sorted({r.get("p") for r in ho}, key=lambda v: -1 if v is None else v):
            rr = [r for r in ho if r.get("p") == p]
            for h in ("1y", "2y"):
                bk = "holdout_band" if h == "1y" else "holdout_band_2y"
                ck = [r["exam_live"][h]["check"] for r in rr if r["exam_live"][h]["check"]]
                c = lambda k: sum(x["checks"][k] for x in ck)  # noqa: E731
                dd = [_crit(r["exam_dead"][h]["check"]) for r in rr if r.get("exam_dead") and r["exam_dead"][h]["check"]]
                L.append(f"| {'null' if p is None else f'p={p}'} | {h} | {len(rr)} | "
                         f"{np.median([r[bk]['tail_level'] for r in rr]):.3f} | "
                         f"{np.median([r[bk]['p_pass_zero_edge'] for r in rr]):.2f} | "
                         f"{sum(bool(r[bk]['decisive']) for r in rr)}/{len(rr)} | {_rate_k(sum(_crit(x) for x in ck), len(ck))} | "
                         f"{c('sharpe')} | {c('ret_at_budget')} | {c('max_dd')} | {c('trades')} | "
                         f"{_rate_k(sum(dd), len(dd)) if dd else '—'} | "
                         f"{np.mean([r['exam_live'][h]['sharpe'] for r in rr]):.2f} |")
        L.append("")
    rt = [r["runtime_total_s"] for r in ok if r["experiment"] != "synthetic"]
    if rt:
        L += [f"Runtime per real study: median {np.median(rt):.0f}s, max {np.max(rt):.0f}s; "
              f"sum {np.sum([r['runtime_total_s'] for r in ok]) / 3600:.2f} core-h", ""]
    return "\n".join(L)


def _pl_status(r: dict[str, Any], kind: str) -> str:
    if kind == "judge":
        return r["gate_status"]["plateau"]
    return (r.get("plateau_matrix") or {}).get("status", "ERROR")


def summarise_final(results: list[dict[str, Any]], v12_path: Path = CHECKPOINT) -> str:
    """Final-run comparisons: v1.2 pairing, judge vs matrix plateau, runtime of the judge plateau."""
    ok = [r for r in results if not r.get("error")]
    v12 = {r["task_id"]: r for r in load_results(v12_path) if not r.get("error")}
    L = ["# Final-run comparisons", ""]
    # ---- power
    pl_ = [r for r in ok if r["experiment"] == "planted"]
    if pl_:
        x = np.array([r["true_sharpe_mean_trials"] for r in pl_])
        pv = [v12[r["task_id"]] for r in pl_ if r["task_id"] in v12]
        xv = np.array([r["true_sharpe_mean_trials"] for r in pv])
        L += ["## Power (logistic on true net SR)", "", "| Gate set | n | 50 % | 80 % |", "|---|---|---|---|"]
        fits = (("final as built", pl_, x, lambda r: r["stat_pass"]),
                ("final without cost_stress", pl_, x, lambda r: pass_without(r, ("cost_stress_sharpe",))),
                ("final without plateau", pl_, x, lambda r: pass_without(r, ("plateau",))),
                ("final with matrix plateau instead of judge", pl_, x,
                 lambda r: pass_without(r, ("plateau",)) and _pl_status(r, "matrix") == "PASS"),
                ("v1.2 same task ids", pv, xv, lambda r: r["stat_pass"]))
        for lab, rows, xx, fn in fits:
            y = np.array([bool(fn(r)) for r in rows], float)
            if len(rows) and 0 < y.sum() < len(rows):
                _a, _b, x50, x80 = logistic_fit(xx, y)
                L.append(f"| {lab} | {len(rows)} | {x50:.2f} | {x80:.2f} |")
        L += ["", "| p | n | true SR | final PASS | v1.2 PASS (paired) | changed (v1.2→final) | judge plateau PASS |"
              " matrix plateau PASS | failing gates |", "|---|---|---|---|---|---|---|---|---|"]
        for p_ in sorted({r["p"] for r in pl_}):
            rr = [r for r in pl_ if r["p"] == p_]
            pr = [(r, v12.get(r["task_id"])) for r in rr]
            ch = [f"{r['task_id'][-3:]}:{int(bool(v['stat_pass']))}→{int(r['stat_pass'])}" for r, v in pr
                  if v is not None and bool(v["stat_pass"]) != bool(r["stat_pass"])]
            fails: dict[str, int] = {}
            for r in rr:
                for g, st_ in r["gate_status"].items():
                    if st_ != "PASS":
                        fails[g] = fails.get(g, 0) + 1
            L.append(f"| {p_} | {len(rr)} | {np.mean([r['true_sharpe_mean_trials'] for r in rr]):.2f} | "
                     f"{_rate(rr)} | {sum(bool(v and v['stat_pass']) for _, v in pr)}/{sum(v is not None for _, v in pr)} | "
                     f"{', '.join(ch) or '—'} | {sum(_pl_status(r, 'judge') == 'PASS' for r in rr)} | "
                     f"{sum(_pl_status(r, 'matrix') == 'PASS' for r in rr)} | "
                     f"{', '.join(f'{g} {c}' for g, c in sorted(fails.items(), key=lambda t: -t[1])) or '—'} |")
        L.append("")
    # ---- plateau: judge vs matrix, by family
    fams = list(FAMILIES) + [
        ("PLANTED grid (all p)", lambda r: r["experiment"] == "planted"),
        ("PLANTED grid p ≥ 0.65", lambda r: r["experiment"] == "planted" and r["p"] >= 0.65),
        ("SOBOL oracle p=0.65", lambda r: r["experiment"] == "sobol" and r["kind"] == "oracle" and r["p"] == 0.65),
        ("SOBOL oracle p=0.62", lambda r: r["experiment"] == "sobol" and r["kind"] == "oracle" and r["p"] == 0.62),
        ("DEAD (all)", lambda r: r["experiment"] == "dead"),
        ("SYN plateau h ≥ 1.5", lambda r: r.get("scenario") == "plateau" and r["height"] >= 1.5),
        ("SYN spike", lambda r: r.get("scenario") == "spike"),
        ("SYN regime", lambda r: r.get("scenario") == "regime")]
    L += ["## Plateau: judge-run (gate) vs matrix box (old), standalone", "",
          "| Family | n | judge PASS | judge SKIPPED | matrix PASS | matrix SKIPPED | agree | median judge evals |"
          " median judge s | weakest axis (mode) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, sel in fams:
        rr = [r for r in ok if sel(r)]
        if not rr:
            continue
        j = [_pl_status(r, "judge") for r in rr]
        m = [_pl_status(r, "matrix") for r in rr]
        pdg = [r.get("plateau_diag") or {} for r in rr]
        ax: dict[str, int] = {}
        for d in pdg:
            if d.get("weakest_axis"):
                ax[d["weakest_axis"]] = ax.get(d["weakest_axis"], 0) + 1
        L.append(f"| {name} | {len(rr)} | {_rate_k(j.count('PASS'), len(rr))} | {j.count('SKIPPED')} | "
                 f"{_rate_k(m.count('PASS'), len(rr))} | {m.count('SKIPPED')} | "
                 f"{sum((a == 'PASS') == (b == 'PASS') for a, b in zip(j, m))}/{len(rr)} | "
                 f"{np.median([d.get('n_evaluations', np.nan) or np.nan for d in pdg]):.0f} | "
                 f"{np.nanmedian([d.get('runtime_s', np.nan) or np.nan for d in pdg]):.2f} | "
                 f"{max(ax, key=ax.get) if ax else '—'} |")
    L.append("")
    # ---- L1 / L4 / L3 checks
    chf = [r for r in results if r.get("setup") == "USDCHF_H1"]
    xau = [r for r in ok if r.get("setup") == "XAUUSD_H4"]
    sob = [r for r in ok if r["experiment"] == "sobol"]
    L += ["## Library checks", "",
          f"- L1 USDCHF: {len(chf)} studies, {sum(bool(r.get('error')) for r in chf)} errors; "
          f"median runtime {np.median([r['runtime_total_s'] for r in chf]) if chf else float('nan'):.0f}s",
          f"- L4 XAUUSD: embargo days median {np.median([r['embargo_days'] for r in xau]) if xau else float('nan')}, "
          f"capped {sum(bool(r['embargo_capped']) for r in xau)}/{len(xau)}; oos_sharpe SKIPPED "
          f"{sum(r['gate_status']['oos_sharpe'] == 'SKIPPED' for r in xau)}, cscv SKIPPED "
          f"{sum(r['gate_status']['cscv_oos_loss'] == 'SKIPPED' for r in xau)}",
          f"- L3 Sobol: judge plateau SKIPPED {sum(r['gate_status']['plateau'] == 'SKIPPED' for r in sob)}/{len(sob)};"
          f" matrix box empty {sum((r.get('plateau_matrix') or {}).get('n_neighbours') == 0 for r in sob)}/{len(sob)}",
          f"- errors overall: {len(results) - len(ok)}", ""]
    # ---- holdout band (manifest horizon)
    hb = [r for r in ok if r.get("manifest_horizon") and r.get("holdout_band")]
    if hb:
        src: dict[str, int] = {}
        for r in hb:
            k = f"{r['holdout_band'].get('horizon_source')}:{r['holdout_band'].get('horizon_days')}"
            src[k] = src.get(k, 0) + 1
        L += [f"Holdout band horizon (real studies): {src}; notes: "
              f"{sum(bool(r.get('holdout_horizon_note')) for r in hb)}", ""]
        pp = [r for r in hb if r["experiment"] == "planted" and r["stat_pass"]]
        if pp:
            L.append(f"Gate-passing planted studies: band decisive {sum(bool(r['holdout_band'].get('decisive')) for r in pp)}"
                     f"/{len(pp)}; median P(pass | zero edge) {np.median([r['holdout_band']['p_pass_zero_edge'] for r in pp]):.2f}")
            for p_ in sorted({r['p'] for r in pp}):
                q = [r for r in pp if r["p"] == p_]
                L.append(f"  p={p_}: decisive {sum(bool(r['holdout_band'].get('decisive')) for r in q)}/{len(q)}, "
                         f"median P0 {np.median([r['holdout_band']['p_pass_zero_edge'] for r in q]):.2f}")
            L.append("")
    # ---- runtime
    real = [r for r in ok if r["experiment"] != "synthetic"]
    if real:
        jr = [((r.get("plateau_diag") or {}).get("runtime_s") or np.nan) for r in real]
        L += ["## Runtime", "",
              f"- real studies {len(real)}: total per study median {np.median([r['runtime_total_s'] for r in real]):.1f}s; "
              f"run_study {np.median([r['runtime_study_s'] for r in real]):.1f}s; gates "
              f"{np.median([r['runtime_gates_s'] for r in real]):.1f}s; judge plateau median {np.nanmedian(jr):.2f}s "
              f"(p95 {np.nanpercentile(jr, 95):.2f}s, max {np.nanmax(jr):.2f}s), "
              f"{np.nanmedian(np.array(jr) / np.array([r['runtime_gates_s'] for r in real])):.0%} of gate time",
              f"- by space: " + "; ".join(
                  f"{sp}: {np.nanmedian([(r.get('plateau_diag') or {}).get('runtime_s') or np.nan for r in real if r.get('space', 'seedhold') == sp]):.2f}s "
                  f"/ {np.nanmedian([(r.get('plateau_diag') or {}).get('n_evaluations') or np.nan for r in real if r.get('space', 'seedhold') == sp]):.0f} evals"
                  for sp in sorted({r.get('space', 'seedhold') for r in real})),
              f"- core-h total {np.sum([r['runtime_total_s'] for r in results]) / 3600:.2f}", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments", default="null,planted,synthetic,holdout,corr,dead,sobol")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--summarise", action="store_true", help="only print the summary of the checkpoint")
    ap.add_argument("--final", action="store_true",
                    help="the targeted final confirmation run (build_final_tasks → output/phase1_final_results.jsonl)")
    ap.add_argument("--require-key", default=None,
                    help="re-run finished tasks whose checkpoint row lacks this key (backfill)")
    ap.add_argument("--only", default=None, help="comma-separated task_id prefixes to run (pilot)")
    ap.add_argument("--n-null", type=int, default=200, help="EURUSD H1 null studies (costs on)")
    ap.add_argument("--n-null-xau", type=int, default=100,
                    help="XAUUSD H4 null studies, and as many frictionless EURUSD null studies")
    ap.add_argument("--n-planted", type=int, default=20, help="studies per oracle accuracy p")
    ap.add_argument("--n-syn", type=int, default=200, help="synthetic zero-edge studies (other scenarios n/5)")
    ap.add_argument("--n-ho", type=int, default=20, help="holdout-split studies per p")
    ap.add_argument("--n-ho-null", type=int, default=40, help="holdout-split null studies")
    ap.add_argument("--n-corr", type=int, default=100, help="correlated-grid null studies per family")
    ap.add_argument("--n-dead", type=int, default=20, help="dead-edge studies per (p, death)")
    ap.add_argument("--n-sobol", type=int, default=40, help="Sobol null studies per cost setting")
    a = ap.parse_args(argv)
    ck = Path(a.checkpoint)
    if a.final and a.checkpoint == str(CHECKPOINT):
        ck = FINAL_CHECKPOINT
    if not a.summarise:
        tasks = build_final_tasks() if a.final else build_tasks(set(a.experiments.split(",")), a.n_null, a.n_null_xau, a.n_planted, a.n_syn,
                            a.n_ho, a.n_ho_null, a.n_corr, a.n_dead, a.n_sobol)
        if a.only:
            pre = tuple(a.only.split(","))
            tasks = [t for t in tasks if t["task_id"].startswith(pre)]
        t0 = time.perf_counter()
        run_all(tasks, a.workers, ck, require_key=a.require_key or None)
        print(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")
    print(summarise(load_results(ck)))
    if a.final:
        print(summarise_final(load_results(ck)))


if __name__ == "__main__":
    main()
