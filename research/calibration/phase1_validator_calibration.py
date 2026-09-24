"""Phase 1 exit test — "validate the validator" (DESIGN §10 Phase 1).

Runs the REAL pipeline (``opt.run_study`` → ``gates.evaluate_gates``) many times on systems
whose truth is known, to measure

1. NULL      the false-pass rate of the §4.2 gates on real data with no edge
             (random entries + ATR stop/target, the best of 144 seeds×holds is data-mined);
2. PLANTED   the power curve: P(PASS) vs true net Sharpe for a *noisy oracle*;
3. SYNTHETIC SyntheticEvaluator surfaces (zero edge, broad plateau, isolated spike, regime shift);
4. HOLDOUT   the coverage of the pre-registered holdout band (§4.4): calibrate on
             2016-05→2023-05-15, run the frozen WFO procedure on 2023-05-15→2024-05-15.
             The real locked holdout (≥ 2025-05-15) is never touched.

!!! ``NoisyOracle`` reads FUTURE prices on purpose.  It is a calibration device that plants an
!!! edge of known size; it is NOT a strategy and must never be used as one.

Every study uses its own temporary ledger + studies directory (never ``research/ledger``).
Studies run in parallel on a process pool (≤ 8 workers, one study per worker, ``n_jobs=1``
inside ``run_study``).  Each finished study appends one JSON line to the checkpoint file, so
an interrupted run resumes where it stopped (``--resume`` is the default behaviour).

Usage::

    PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py \
        --experiments null,planted,synthetic,holdout --workers 8
    PYTHONPATH=. .venv/bin/python research/calibration/phase1_validator_calibration.py --summarise

Headline definitions
--------------------
* ``stat_pass`` = every *statistical* gate (DSR, PBO, OOS Sharpe, cost stress, plateau, positive
  years, max-year share, trade count) is PASS.  The mechanism gate is MANUAL by construction
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
from pathlib import Path
from typing import Any

# Thread caps must be set BEFORE polars/numba are imported: pool workers are forked and inherit
# the parent's already-initialised Polars pool (setting the env var in the pool initializer is
# too late — measured 16 threads per worker, 4x slowdown with 8 workers).
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
CHECKPOINT = OUT_DIR / "phase1_results.jsonl"
DEV_START = "2016-05-02"
DEV_END = "2025-05-15"            # exclusive: holdout start (evaluator default)
SPLIT_CAL_END = "2023-05-15"      # holdout-band experiment: calibration end (exclusive)
SPLIT_HO_END = "2024-05-15"       # pseudo-holdout end (exclusive) — still inside dev
MAX_WORKERS = 8
STAT_GATES = ("dsr", "pbo", "oos_sharpe", "cost_stress_sharpe", "plateau", "positive_years",
              "max_year_share", "trade_count")

# symbol/timeframe setups: entry rate per bar and the hold grid (bars)
SETUPS = {
    "EURUSD_H1": {"symbol": "EURUSD", "timeframe": "H1", "rate": 1 / 24, "holds": (12, 24, 48)},
    "XAUUSD_H4": {"symbol": "XAUUSD", "timeframe": "H4", "rate": 1 / 6, "holds": (3, 6, 12)},
}
N_SEEDS = 48                       # 48 seeds × 3 holds = 144 trials per study


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


# =========================================================================== NULL strategy
@dataclass(frozen=True)
class RandomEntryParams(Params):
    seed: int = 0
    hold: int = 24
    salt: int = 0              # study master seed (fixed per study, not searched)
    rate: float = 1 / 24       # entry probability per bar (fixed)
    stop_mult: float = 2.0     # ATR multiples (fixed)
    target_mult: float = 3.0
    atr_len: int = 14


class RandomEntryATR:
    """Zero-information entries: side and timing are coin flips; ATR stop/target + time exit.

    Causal (row i uses bars ≤ i: ATR and a timestamp hash).  Risk type A (fixed-fraction risk
    on the ATR stop).  Any Sharpe it shows is luck ± costs, so selecting the best seed is pure
    data-mining — the null the gates must reject."""

    name = "calib_random_entry_atr"
    risk_type = RiskType.A
    params_cls = RandomEntryParams

    def __init__(self, params: RandomEntryParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        n = bars.height
        ts = bars["ts"].dt.epoch("ms").to_numpy()
        atr = _atr(bars, p.atr_len)
        key = p.salt * 100_003 + p.seed
        u_entry, u_side = _uniform(ts, key, 1), _uniform(ts, key, 2)
        ok = np.isfinite(atr) & (atr > 0)
        cand = np.flatnonzero(ok & (u_entry < p.rate))
        sig = np.zeros(n, dtype=np.int8)
        valid = np.zeros(n, dtype=bool)
        for i in _schedule(cand, p.hold, n):
            sig[i] = 1 if u_side[i] < 0.5 else -1
            valid[i] = True
            if i + p.hold < n:
                sig[i + p.hold] = 0
                valid[i + p.hold] = True
        atr0 = np.where(ok, atr, np.nan)
        return _frame(sig, valid, atr0 * p.stop_mult, atr0 * p.target_mult)


# =========================================================================== PLANTED edge (look-ahead!)
@dataclass(frozen=True)
class OracleParams(Params):
    seed: int = 0
    hold: int = 24
    salt: int = 0
    p: float = 0.55            # accuracy of the oracle (fixed per study)
    rate: float = 1 / 24
    stop_mult: float = 3.0     # safety stop only; exit is by time
    atr_len: int = 14


class NoisyOracle:
    """CALIBRATION DEVICE — USES FUTURE DATA ON PURPOSE.  NOT A STRATEGY.

    Same random entry schedule as :class:`RandomEntryATR`, but the side is the sign of the
    price change over the trade's own life (open of bar i+1 → open of bar i+hold+1, the
    engine's actual entry/time-exit fills), reported correctly with probability ``p`` and
    flipped otherwise.  Every configuration carries the same planted edge (the seed is an
    irrelevant parameter), so a search still happens but the truth is a flat plateau."""

    name = "calib_noisy_oracle_LOOKAHEAD"
    risk_type = RiskType.A
    params_cls = OracleParams

    def __init__(self, params: OracleParams):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        n = bars.height
        ts = bars["ts"].dt.epoch("ms").to_numpy()
        op = bars["open"].to_numpy()
        atr = _atr(bars, p.atr_len)
        key = p.salt * 100_003 + p.seed
        u_entry, u_flip = _uniform(ts, key, 1), _uniform(ts, key, 3)
        ok = np.isfinite(atr) & (atr > 0)
        ok[max(0, n - p.hold - 1):] = False                   # need the future exit fill
        cand = np.flatnonzero(ok & (u_entry < p.rate))
        sig = np.zeros(n, dtype=np.int8)
        valid = np.zeros(n, dtype=bool)
        for i in _schedule(cand, p.hold, n):
            move = op[i + p.hold + 1] - op[i + 1]              # LOOK-AHEAD (deliberate)
            if move == 0:
                continue
            side = 1 if move > 0 else -1
            if u_flip[i] >= p.p:
                side = -side
            sig[i] = side
            valid[i] = True
            sig[i + p.hold] = 0
            valid[i + p.hold] = True
        atr0 = np.where(ok | valid, atr, np.nan)
        return _frame(sig, valid, atr0 * p.stop_mult, None)


# =========================================================================== study plumbing
def _space(holds) -> opt.SearchSpace:
    return opt.SearchSpace((opt.IntParam("seed", 0, N_SEEDS - 1), opt.CategoricalParam("hold", holds, ordered=True)))


def _cost(name: str | None) -> CostModel:
    """``None``/"base": the book's base cost model.  "zero": no spread, no swap — a frictionless
    run used ONLY for the exact-zero-edge real-data null (the gross Sharpe of random entries is 0;
    with costs on it is negative, which makes the null too easy)."""
    if name in (None, "base"):
        return CostModel()
    if name == "zero":
        return CostModel(version_tag="calib-frictionless", spread_multiplier=0.0, swap_multiplier=0.0)
    raise ValueError(name)


def _rule_evaluator(kind: str, setup: str, salt: int, p: float | None, end: str,
                    cost: str | None = None) -> evaluators.RuleEvaluator:
    s = SETUPS[setup]
    fixed = {"salt": salt, "rate": s["rate"]}
    cls = RandomEntryATR
    if kind == "oracle":
        cls, fixed = NoisyOracle, {**fixed, "p": p}
    return evaluators.RuleEvaluator(cls, symbol=s["symbol"], timeframe=s["timeframe"], book="FBS",
                                    start=DEV_START, end=end, risk_fraction=0.01, fixed_params=fixed,
                                    cost=_cost(cost))


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


def _trial_sharpes(study) -> np.ndarray:
    R = study.returns.drop("date").to_numpy()
    return np.asarray(st.sharpe_per_period(R, axis=0), float) * math.sqrt(260.0)


def _dsr_null_variance(rep: gates.GateReport) -> float | None:
    """DIAGNOSTIC (not the gate): DSR with the expected-max hurdle built from the sampling
    variance of a zero-Sharpe estimator, V0 = 1/(T−1), instead of the cross-sectional variance
    of the trial Sharpes (which also contains genuine differences in true Sharpe across the
    grid).  Same N_eff, same PSR.  Used to quantify the recalibration proposal."""
    et = rep.effective_trials
    try:
        T = int(et["n_obs"])
        sr0 = st.expected_max_sharpe(float(et["n_eff"]), 1.0 / (T - 1))
        return st.psr(float(et["sr"]), sr0, T, float(et["skew"]), float(et["kurt"]))
    except (KeyError, TypeError, ValueError):
        return None


def _summarise_report(rep: gates.GateReport) -> dict[str, Any]:
    stat = {g: rep.row(g).status for g in STAT_GATES}
    dsr_nv = _dsr_null_variance(rep)
    alt_ok = dsr_nv is not None and dsr_nv >= gates.GATE_THRESHOLDS["dsr"][1]
    return {
        "dsr_nullvar": dsr_nv,
        "stat_pass_dsr_nullvar": alt_ok and all(s == "PASS" for g, s in stat.items() if g != "dsr"),
        "var_sr_cross": et_get(rep, "var_sr"),
        "verdict": rep.verdict,
        "stat_pass": all(s == "PASS" for s in stat.values()),
        "gate_status": stat,
        "gate_values": {g: rep.row(g).value for g in STAT_GATES},
        "n_eff": rep.effective_trials.get("n_eff"),
        "n_eff_eigen": rep.effective_trials.get("eigen"),
        "n_eff_cluster": rep.effective_trials.get("cluster"),
        "dsr_raw_n": rep.effective_trials.get("dsr_raw_n"),
        "sr0_annual": rep.effective_trials.get("sr0_annual"),
        "sr_selected": rep.effective_trials.get("sr_annual"),
        "holdout_band": rep.holdout_band,
        "cpcv_path_sharpe": rep.diagnostics.get("cpcv_path_sharpe"),
        "pbo_detail": rep.diagnostics.get("pbo_detail"),
        "cost_stress_by_swap_mult": rep.diagnostics.get("cost_stress_by_swap_mult"),
        "min_trl_days": rep.diagnostics.get("min_trl_days"),
    }


def et_get(rep: gates.GateReport, k: str):
    return rep.effective_trials.get(k)


def _run_real_study(task: dict[str, Any], work: Path) -> tuple[dict[str, Any], Any, Any, Any]:
    end = task.get("end", DEV_END)
    ev = _rule_evaluator(task["kind"], task["setup"], task["salt"], task.get("p"), end, task.get("cost"))
    space = _space(SETUPS[task["setup"]]["holds"])
    t0 = time.perf_counter()
    study = opt.run_study(ev, space, book="FBS", system=f"calib_{task['kind']}", issue=0, attempt=1,
                          method="grid", n_jobs=1, seed=task["salt"], study_id=task["task_id"],
                          cv=opt.CPCVConfig(10, 2), wfo=opt.WFOConfig(),
                          ledger_dir=work / "ledger", studies_dir=work / "studies")
    t_study = time.perf_counter() - t0
    t1 = time.perf_counter()
    rep = gates.evaluate_gates(study, ev, periods_per_year=ev.periods_per_year, n_boot=2000,
                               seed=task["salt"])
    t_gates = time.perf_counter() - t1
    srs = _trial_sharpes(study)
    tids = [int(c[1:]) for c in study.returns.columns if c != "date"]
    hold_of = dict(zip(study.trials["trial_id"].to_list(), study.trials["param_hold"].to_list()))
    by_hold = {str(h): float(np.nanmean([s for t, s in zip(tids, srs) if hold_of.get(t) == h]))
               for h in SETUPS[task["setup"]]["holds"]}
    sel_hold = str(study.selected_params.get("hold")) if study.selected_params else None
    row = {
        "true_sharpe_by_hold": by_hold,
        "true_sharpe_selected_hold": by_hold.get(sel_hold) if sel_hold else None,
        **_summarise_report(rep),
        "selected_params": study.selected_params,
        "true_sharpe_mean_trials": float(np.nanmean(srs)),
        "trial_sharpe_sd": float(np.nanstd(srs, ddof=1)),
        "trial_sharpe_max": float(np.nanmax(srs)),
        "wfo_oos_sharpe": metrics.sharpe(study.wfo_oos.select("ret"), 260.0) if study.wfo_oos.height else None,
        "median_trades": study.meta.get("median_trades_ok"),
        "embargo_days": study.meta["cpcv"]["embargo_days"],
        "runtime_study_s": t_study, "runtime_gates_s": t_gates,
    }
    return row, study, ev, rep


def _holdout_exam(task: dict[str, Any], study, rep) -> dict[str, Any]:
    """Run the frozen WFO procedure over [SPLIT_CAL_END, SPLIT_HO_END) and apply the band.

    The grid is re-evaluated on the extended window (dev data only); refits inside the
    pseudo-holdout use only rows before each refit date — exactly the live procedure."""
    ev2 = _rule_evaluator(task["kind"], task["setup"], task["salt"], task.get("p"), SPLIT_HO_END, task.get("cost"))
    space = _space(SETUPS[task["setup"]]["holds"])
    trials = study.trials
    cols: dict[str, Any] = {}
    ccols: dict[str, Any] = {}
    dates = None
    for row in trials.iter_rows(named=True):
        if row["status"] not in ("ok", "low_trades"):
            continue
        params = json.loads(row["params"])
        out = ev2(params)
        d = out.daily.sort("date")
        if dates is None:
            dates = d["date"]
        cols[f"t{row['trial_id']}"] = d["ret"].to_numpy()
        tr = out.trades.filter(~pl.col("skipped")) if "skipped" in out.trades.columns else out.trades
        cnt = tr.group_by(pl.col("exit_ts").dt.date().alias("date")).agg(pl.len().alias("n"))
        ccols[f"t{row['trial_id']}"] = (pl.DataFrame({"date": dates}).join(cnt, on="date", how="left")
                                        .fill_null(0)["n"].to_numpy().astype(float))
    R = pl.DataFrame({"date": dates, **cols})
    C = pl.DataFrame({"date": dates, **ccols})
    wfo_oos, wfo_params, _ = opt.walk_forward(R, trials, space, opt.WFOConfig(), trade_counts=C,
                                              periods_per_year=260.0, method="grid")
    ho_start = np.datetime64(SPLIT_CAL_END)
    ho = wfo_oos.filter(pl.col("date") >= pl.lit(ho_start).cast(pl.Date)).sort("date")
    # trades closed in the holdout by whichever configuration was live at the time
    n_tr = 0.0
    for r in wfo_params.iter_rows(named=True):
        lo = max(np.datetime64(r["refit_date"]), ho_start)
        hi = np.datetime64(r["test_end"])
        if hi < ho_start or r["selected_trial"] is None:
            continue
        c = C.filter((pl.col("date") >= pl.lit(lo).cast(pl.Date)) & (pl.col("date") <= pl.lit(hi).cast(pl.Date)))
        n_tr += float(c[f"t{r['selected_trial']}"].sum())
    band = rep.holdout_band
    chk = st.holdout_check(band, ho.select("ret"), n_tr, 260.0) if band else None
    # the same exam on the fixed dev-selected parameters (no refit) — diagnostic
    sel_col = f"t{study.selection.get('trial_id')}"
    fixed = R.filter(pl.col("date") >= pl.lit(ho_start).cast(pl.Date)).select(pl.col(sel_col).alias("ret"))
    return {
        "holdout_days": ho.height,
        "holdout_sharpe": metrics.sharpe(ho.select("ret"), 260.0),
        "holdout_sharpe_fixed_params": metrics.sharpe(fixed, 260.0),
        "holdout_trades": n_tr,
        "holdout_check": chk,
        "holdout_true_sharpe_mean_trials": float(np.nanmean(
            np.asarray(st.sharpe_per_period(R.filter(pl.col("date") >= pl.lit(ho_start).cast(pl.Date))
                                            .drop("date").to_numpy(), axis=0)) * math.sqrt(260.0))),
    }


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
                               seed=task["salt"], selected_trades=tr)
    sel = study.selected_params
    return {**_summarise_report(rep), "selected_params": sel,
            "true_sharpe_selected": ev.true_sharpe(sel) if sel else None,
            "true_sharpe_selected_late": ev.true_sharpe(sel, 0.99) if sel else None,
            "true_sharpe_max": task.get("height", 0.0),
            "true_sharpe_mean_trials": float(np.mean([ev.true_sharpe(p) for p in space.grid()])),
            "trial_sharpe_max": float(np.nanmax(_trial_sharpes(study))),
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
# p levels chosen from the pilot (see the report) to span true net Sharpe ≈ 0.3 … 3.
ORACLE_P = (0.54, 0.555, 0.575, 0.60, 0.62, 0.65, 0.68, 0.72)   # true net SR ≈ 0.3 … 2.9 (pilot)
HOLDOUT_P = (0.60, 0.65, 0.72)


def build_tasks(experiments: set[str], n_null: int = 40, n_null_xau: int = 20, n_planted: int = 10,
                n_syn: int = 100, n_ho: int = 10, n_ho_null: int = 20) -> list[dict[str, Any]]:
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
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    done = {r["task_id"] for r in load_results(checkpoint)
            if not r.get("error") and (require_key is None or r.get(require_key) is not None)}
    todo = [t for t in tasks if t["task_id"] not in done]
    log(f"{len(tasks)} tasks, {len(tasks) - len(todo)} already done, {len(todo)} to run on {workers} workers")
    # real-data tasks first (long), grouped by setup so each worker reuses its cached data
    todo.sort(key=lambda t: (t["experiment"] == "synthetic", t.get("setup", ""), t.get("end", ""), t["task_id"]))
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
    return out


# =========================================================================== analysis
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def _rate(rows, key="stat_pass") -> str:
    n = len(rows)
    k = sum(bool(r.get(key)) for r in rows)
    lo, hi = clopper_pearson(k, n) if n else (float("nan"), float("nan"))
    return f"{k}/{n} = {k / n if n else float('nan'):.1%} (95% CP [{lo:.1%}, {hi:.1%}])"


def gate_rejection_table(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    L = ["| Gate | Reject rate | Only-rejecting gate | Median value |", "|---|---|---|---|"]
    for g in STAT_GATES:
        rej = [r for r in rows if r["gate_status"][g] != "PASS"]
        only = [r for r in rej if sum(s != "PASS" for s in r["gate_status"].values()) == 1]
        vals = [r["gate_values"][g] for r in rows if r["gate_values"][g] is not None]
        med = f"{np.median(vals):.3g}" if vals else "—"
        L.append(f"| {g} | {len(rej)}/{n} ({len(rej) / n:.0%}) | {len(only)} | {med} |")
    return "\n".join(L)


def summarise(results: list[dict[str, Any]]) -> str:
    ok = [r for r in results if not r.get("error")]
    err = [r for r in results if r.get("error")]
    L = [f"# Phase 1 calibration — summary ({len(ok)} studies ok, {len(err)} errors)", ""]
    for name, sel in (("NULL EURUSD H1", lambda r: r["experiment"] == "null" and r["setup"] == "EURUSD_H1"
                       and r.get("cost") is None),
                      ("NULL EURUSD H1 frictionless (exact zero edge)", lambda r: r["experiment"] == "null"
                       and r.get("cost") == "zero"),
                      ("NULL XAUUSD H4", lambda r: r["experiment"] == "null" and r["setup"] == "XAUUSD_H4"),
                      ("NULL real, costs on", lambda r: r["experiment"] == "null" and r.get("cost") is None),
                      ("SYN zero", lambda r: r.get("scenario") == "zero"),
                      ("SYN spike h3", lambda r: r.get("scenario") == "spike"),
                      ("SYN regime h3", lambda r: r.get("scenario") == "regime")):
        rows = [r for r in ok if sel(r)]
        if not rows:
            continue
        L += [f"## {name}: stat_pass {_rate(rows)}", "",
              f"mean trial Sharpe {np.mean([r['true_sharpe_mean_trials'] for r in rows]):.2f}; "
              f"median selected Sharpe {np.median([r['sr_selected'] for r in rows]):.2f}; "
              f"median N_eff {np.median([r['n_eff'] for r in rows]):.1f}; "
              f"stat_pass with null-variance DSR (diagnostic) {_rate(rows, 'stat_pass_dsr_nullvar')}", "",
              gate_rejection_table(rows), ""]
    for name, key in (("PLANTED (real, oracle)", "p"), ("SYN plateau", "height")):
        rows = [r for r in ok if (r["experiment"] == "planted" if key == "p" else r.get("scenario") == "plateau")]
        if not rows:
            continue
        L += [f"## {name} — power curve", "",
              f"| {key} | n | true net SR (mean trials) | median OOS SR | P(PASS) | 95% CP | P(PASS) null-var DSR |"
              f" most frequent failing gate |",
              "|---|---|---|---|---|---|---|---|"]
        for v in sorted({r[key] for r in rows}):
            rr = [r for r in rows if r[key] == v]
            k = sum(r["stat_pass"] for r in rr)
            lo, hi = clopper_pearson(k, len(rr))
            fails: dict[str, int] = {}
            for r in rr:
                for g, s in r["gate_status"].items():
                    if s != "PASS":
                        fails[g] = fails.get(g, 0) + 1
            top = ", ".join(f"{g} {c}" for g, c in sorted(fails.items(), key=lambda x: -x[1])[:3]) or "—"
            L.append(f"| {v} | {len(rr)} | {np.mean([r['true_sharpe_mean_trials'] for r in rr]):.2f} | "
                     f"{np.median([r['gate_values']['oos_sharpe'] or np.nan for r in rr]):.2f} | {k / len(rr):.0%} | "
                     f"[{lo:.0%}, {hi:.0%}] | {np.mean([r['stat_pass_dsr_nullvar'] for r in rr]):.0%} | {top} |")
        L.append("")
    ho = [r for r in ok if r["experiment"] == "holdout" and r.get("holdout_check")]
    if ho:
        L += ["## HOLDOUT band coverage", "", "| group | n | band pass | sharpe ≥ p10 | ret@budget ≥ p10 | DD ≤ p95 | "
              "trades in range | mean true holdout SR |", "|---|---|---|---|---|---|---|---|"]
        for p in sorted({r.get("p") for r in ho}, key=lambda v: -1 if v is None else v):
            rr = [r for r in ho if r.get("p") == p]
            c = lambda k: sum(r["holdout_check"]["checks"][k] for r in rr)  # noqa: E731
            L.append(f"| {'null' if p is None else f'p={p}'} | {len(rr)} | {sum(r['holdout_check']['pass'] for r in rr)}/{len(rr)} | "
                     f"{c('sharpe')} | {c('ret_at_budget')} | {c('max_dd')} | {c('trades')} | "
                     f"{np.mean([r['holdout_true_sharpe_mean_trials'] for r in rr]):.2f} |")
        L.append("")
    rt = [r["runtime_total_s"] for r in ok if r["experiment"] != "synthetic"]
    if rt:
        L += [f"Runtime per real study: median {np.median(rt):.0f}s, max {np.max(rt):.0f}s; "
              f"sum {np.sum(rt) / 3600:.2f} core-h", ""]
    return "\n".join(L)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--experiments", default="null,planted,synthetic,holdout")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--summarise", action="store_true", help="only print the summary of the checkpoint")
    ap.add_argument("--require-key", default="pbo_detail",
                    help="re-run finished tasks whose checkpoint row lacks this key (backfill)")
    ap.add_argument("--n-null", type=int, default=200, help="EURUSD H1 null studies (costs on)")
    ap.add_argument("--n-null-xau", type=int, default=100,
                    help="XAUUSD H4 null studies, and as many frictionless EURUSD null studies")
    ap.add_argument("--n-planted", type=int, default=20, help="studies per oracle accuracy p")
    ap.add_argument("--n-syn", type=int, default=200, help="synthetic zero-edge studies (other scenarios n/5)")
    ap.add_argument("--n-ho", type=int, default=20, help="holdout-split studies per p")
    ap.add_argument("--n-ho-null", type=int, default=40, help="holdout-split null studies")
    a = ap.parse_args(argv)
    ck = Path(a.checkpoint)
    if not a.summarise:
        tasks = build_tasks(set(a.experiments.split(",")), a.n_null, a.n_null_xau, a.n_planted, a.n_syn,
                            a.n_ho, a.n_ho_null)
        t0 = time.perf_counter()
        run_all(tasks, a.workers, ck, require_key=a.require_key or None)
        print(f"total wall time {(time.perf_counter() - t0) / 60:.1f} min")
    print(summarise(load_results(ck)))


if __name__ == "__main__":
    main()
