"""End-to-end gate tests on synthetic StudyResults with a fake evaluator (DESIGN §4.2 v1.2, §10 Phase 1).

Regression tests for the Phase 1 fix round (red-team B1–B4, M1, M3–M5, m1, m3, m4, m9;
calibration F4) are marked in their docstrings; each fails on the a741fa2 code.  Fix round 2
(re-verification N1–N4, N7, N10) tests are marked the same way and fail on e64efcf.

Every study is registered in a (temporary) ledger before it is gated: the gates read the
study's identity, prior trials, plateau radius and candidate-set flag from its ledger row (N2).
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import warnings
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantlab import gates as G
from quantlab import ledger, opt
from quantlab import stats as S
from quantlab.contracts import Outcome, StudyResult
from quantlab.costs import CostModel, load_instrument, pip_points
from quantlab.evaluators import SyntheticEvaluator

PPY = 260.0
GRID_SPACE = opt.SearchSpace([opt.IntParam("a", 1, 5, plateau_scale="relative"), opt.IntParam("b", 1, 5, plateau_scale="relative")]).to_json()


def write_store(study: StudyResult, studies_dir: Path) -> Path:
    """Write ``study``'s trials + returns as its trial store (the parquet parts ``opt.run_study``
    writes through ``ledger.TrialRecorder``), unless one exists — the gates load and check the
    in-memory study against it (red-team R2-4)."""
    d = Path(studies_dir) / study.study_id
    if list(d.glob("trials-*.parquet")):
        return Path(studies_dir)
    d.mkdir(parents=True, exist_ok=True)
    t = study.trials
    pcols = [c for c in t.columns if c.startswith("param_")]
    rows = []
    for r in t.iter_rows(named=True):
        row = {"trial_id": int(r["trial_id"]), "status": r.get("status", "ok"),
               "params": json.dumps({c[6:]: r[c] for c in pcols}, sort_keys=True, default=str)}
        row.update({c: r[c] for c in t.columns if c.startswith("m_")})
        if "source" in t.columns:
            row["source"] = r["source"]
        rows.append(row)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(d / "trials-00000.parquet")
    cols = [c for c in study.returns.columns if c != "date"]
    if cols:
        long = pl.concat([study.returns.select(pl.col("date").cast(pl.Date), pl.col(c).cast(pl.Float64).alias("ret"))
                          .drop_nulls("ret").with_columns(pl.lit(c).alias("trial")) for c in cols])
        long.write_parquet(d / "returns-00000.parquet")
    return Path(studies_dir)


FAKE_DESC = {"evaluator": "FakeEvaluator"}


def record_selection(study: StudyResult, ledger_dir: Path, studies_dir: Path) -> None:
    """Write the study's OOS artifacts into its store and log the ``selection`` event with their
    hashes (as ``opt.run_study`` does, R3-2), unless the study already has one."""
    if ledger.study_events(study.study_id, "selection", ledger_dir=ledger_dir):
        return
    m = study.meta or {}
    sha = ledger.write_study_artifacts(study.study_id, {
        "cpcv_paths": study.cpcv_paths, "wfo_oos": study.wfo_oos, "wfo_params": study.wfo_params,
        "trade_counts": m.get("trade_counts"), "entry_counts": m.get("entry_counts")}, studies_dir)
    ledger.log_event(study.study_id, "selection", ledger_dir=ledger_dir, selected_params=study.selected_params,
                     selection={k: v for k, v in (study.selection or {}).items()}, artifact_sha256=sha)


def register(study: StudyResult, ledger_dir: Path | None = None, **over) -> Path:
    """Create the study's ``study_created`` ledger row from its meta (identity, method, space,
    candidate-set flag, radius) in ``ledger_dir`` (default: a fresh temporary ledger), with the
    evaluator description (default: FakeEvaluator's) and a trial store under ``<ledger_dir>/studies``
    (recorded in the row) — as ``opt.run_study`` would (R2-4)."""
    ld = Path(ledger_dir) if ledger_dir is not None else Path(tempfile.mkdtemp(prefix="ql_gates_ledger_"))
    m = study.meta or {}
    idm = re.search(r"-(\d+)-a(\d+)$", study.study_id)
    sd = write_store(study, ld / "studies")
    row = dict(study_id=study.study_id, book=m.get("book", "FBS"), system=m.get("system", "toy"),
               issue=m.get("issue", int(idm.group(1)) if idm else 99),
               attempt=m.get("attempt", int(idm.group(2)) if idm else 1),
               dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-test", cv_scheme="CPCV(n=6,k=2)",
               evaluator=FAKE_DESC, studies_dir=str(sd.resolve()))
    for k_meta, k_led in (("method", "method"), ("space", "search_space"), ("seed", "seed"),
                          ("plateau_radius", "plateau_radius"),
                          ("candidate_set_data_dependent", "candidate_set_data_dependent")):
        if m.get(k_meta) is not None:
            row[k_led] = m[k_meta]
    row.update(over)
    ledger.create_study(ledger_dir=ld, **row)
    record_selection(study, ld, sd)
    return ld
with warnings.catch_warnings():
    warnings.simplefilter("ignore")          # uncalibrated fallback spec is fine for tests
    EURUSD = load_instrument("EURUSD", book="FBS")


def _bdays(n: int, start=date(2016, 5, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _sr_to_mu(sr_annual: float, sd: float) -> float:
    return sr_annual / math.sqrt(PPY) * sd


def make_wfo(dates: list[date], sr: float | np.ndarray, sd: float, rng, sel_id: int, *,
             start_frac: float = 0.3, refit_every: int = 63,
             trades_per_day: float = 0.5) -> tuple[pl.DataFrame, pl.DataFrame]:
    """WFO OOS series from ``start_frac`` of dev onwards, quarterly refits all picking ``sel_id``.
    ``sr`` may be an array (per-day true annual Sharpe).  ``n_trades`` = Poisson entries/day."""
    d = dates[int(start_frac * len(dates)):]
    n = len(d)
    mu = _sr_to_mu(1.0, sd) * (np.asarray(sr, float) if np.ndim(sr) else np.full(n, float(sr)))
    r = mu + sd * rng.normal(0, 1, n)
    rid = np.arange(n) // refit_every
    wfo = pl.DataFrame({"date": d, "ret": r, "refit_id": rid.astype(np.int32),
                        "n_trades": rng.poisson(trades_per_day, n).astype(np.int64)})
    wp = pl.DataFrame({"refit_id": np.unique(rid).astype(np.int32),
                       "refit_date": [d[i * refit_every] for i in np.unique(rid)],
                       "selected_trial": [sel_id] * len(np.unique(rid))})
    return wfo, wp


def make_study(*, seed: int = 0, n_days: int = 2340, peak_sr: float = 2.2, slope: float = 0.2,
               oos_sr: float = 1.6, wfo_sr: float | None = 1.6, plateau_score: float | None = 0.9,
               n_trades: int = 1200, sel_override: np.ndarray | None = None,
               extra_cols: dict[str, np.ndarray] | None = None, cpcv_paths: bool = True,
               meta: dict | None = None):
    """5×5 grid (a, b); true SR(a,b) = peak − slope·(|a−3| + |b−3|), selected = centre (3,3).
    Trials share a common factor (ρ ≈ 0.5) plus idiosyncratic noise, daily sd 0.6 %.
    CPCV paths carry true SR ``oos_sr``; the WFO OOS series true SR ``wfo_sr`` (None → no WFO)."""
    rng = np.random.default_rng(seed)
    dates = _bdays(n_days)
    sd = 0.006
    common = rng.normal(0, 1, n_days)
    cols, trows = {}, []
    tid = 0
    for a in range(1, 6):
        for b in range(1, 6):
            sr = peak_sr - slope * (abs(a - 3) + abs(b - 3))
            z = math.sqrt(0.5) * common + math.sqrt(0.5) * rng.normal(0, 1, n_days)
            r = _sr_to_mu(sr, sd) + sd * z
            if (a, b) == (3, 3) and sel_override is not None:
                r = sel_override
            cols[f"t{tid}"] = r
            trows.append({"trial_id": tid, "status": "ok", "param_a": a, "param_b": b,
                          "m_sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(PPY))})
            if (a, b) == (3, 3):
                sel_id = tid
            tid += 1
    for name, r in (extra_cols or {}).items():
        k = int(name[1:])
        cols[name] = r
        trows.append({"trial_id": k, "status": "ok", "param_a": 100 + k, "param_b": 100 + k,
                      "m_sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(PPY))})
    returns = pl.DataFrame({"date": dates, **cols})
    paths = []
    if cpcv_paths:
        for pid in range(6):
            pr = _sr_to_mu(oos_sr, sd) + sd * rng.normal(0, 1, n_days)
            paths.append(pl.DataFrame({"date": dates, "path_id": [pid] * n_days, "ret": pr}))
    cp = pl.concat(paths) if paths else pl.DataFrame(schema={"date": pl.Date, "path_id": pl.Int64, "ret": pl.Float64})
    if wfo_sr is None:
        wfo, wp = pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64, "refit_id": pl.Int64}), pl.DataFrame()
    else:
        wfo, wp = make_wfo(dates, wfo_sr, sd, rng, sel_id)
    sel = {"method": "plateau", "trial_id": sel_id}
    if plateau_score is not None:
        sel["plateau_score"] = plateau_score
    study = StudyResult(
        study_id="fbs-0099-a1", param_names=("a", "b"), trials=pl.DataFrame(trows), returns=returns,
        selected_params={"a": 3, "b": 3}, selection=sel, cpcv_paths=cp, wfo_oos=wfo, wfo_params=wp,
        meta={"n_trials": len(cols), "method": "grid", "space": GRID_SPACE, **(meta or {})})
    trades = make_trades(returns["date"], returns[f"t{sel_id}"].to_numpy(), n_trades, rng)
    return study, trades


def make_trades(dates: pl.Series, daily: np.ndarray, n_trades: int, rng) -> pl.DataFrame:
    """Trades whose per-trade returns sum to the daily PnL (entries spread over the dev window)."""
    n = len(daily)
    ent = np.sort(rng.choice(n, size=min(n_trades, n), replace=False))
    bounds = np.r_[ent, n]
    pnl = np.array([daily[bounds[i]:bounds[i + 1]].sum() for i in range(len(ent))])
    ts = [datetime.combine(dates[int(i)], datetime.min.time()) + timedelta(hours=10) for i in ent]
    return pl.DataFrame({"entry_ts": ts, "pnl_ccy": pnl * 100_000.0, "equity_before": [100_000.0] * len(ent),
                         "skipped": [False] * len(ent)})


class FakeEvaluator:
    """Returns the recorded trial series of the requested (a, b) minus a cost drag that grows
    with the stressed spread / slippage / swap knobs; params not in the study raise (a failed
    evaluation)."""

    book = "FBS"
    periods_per_year = PPY
    requires_refit = False

    def __init__(self, study: StudyResult, trades: pl.DataFrame, drag_per_unit: float = 0.00002, spec=EURUSD):
        self.cost = CostModel(version_tag="fbs-test")
        self.spec = spec
        self.study, self.trades, self.drag = study, trades, drag_per_unit
        self.calls: list[CostModel] = []

    def __call__(self, params, *, cost=None):
        c = cost or self.cost
        self.calls.append(c)
        t = self.study.trials
        for k, v in params.items():
            if f"param_{k}" in t.columns:
                t = t.filter(pl.col(f"param_{k}") == v)
        if t.height == 0:
            raise KeyError(f"no recorded trial for {params}")
        base = self.study.returns[f"t{int(t['trial_id'][0])}"].to_numpy()
        units = (c.spread_multiplier - 1.0) * 2 + min(c.slippage_points, 1.0) + abs(c.swap_multiplier - 1.0)
        daily = self.study.returns.select("date").with_columns(pl.Series("ret", base - self.drag * units))
        return Outcome(daily=daily, trades=self.trades, metrics={})


def _mech_ok(study, evaluator):
    return True, "Filter beats random filter of equal selectivity (p = 0.01)."


def run(study, trades, evaluator="fake", **kw):
    ev = FakeEvaluator(study, trades) if evaluator == "fake" else evaluator
    kw.setdefault("mechanism_check", _mech_ok)
    if "ledger_dir" not in kw:
        kw["ledger_dir"] = register(study)
    if "studies_dir" not in kw and "studies_dir" not in (ledger.created_row(study.study_id, ledger_dir=kw["ledger_dir"])
                                                           or {}):
        kw["studies_dir"] = write_store(study, Path(kw["ledger_dir"]) / "studies")
    record_selection(study, Path(kw["ledger_dir"]),
                     kw.get("studies_dir") or Path(ledger.created_row(study.study_id, ledger_dir=kw["ledger_dir"])
                                                   ["studies_dir"]))
    return G.evaluate_gates(study, ev, periods_per_year=PPY, selected_trades=trades, **kw), ev


def _as_argmax(study):
    """Re-select the in-sample best trial (a data-mined pick)."""
    best = int(study.trials.sort("m_sharpe", descending=True)["trial_id"][0])
    row = study.trials.filter(pl.col("trial_id") == best)
    return replace(study, selection={"trial_id": best},
                   selected_params={"a": row["param_a"][0], "b": row["param_b"][0]})


@pytest.fixture(scope="module")
def good():
    return make_study(seed=1)


# ============================================================================ basics
def test_thresholds_match_design_and_are_read_only():
    T = G.GATE_THRESHOLDS
    assert T["dsr"] == (">=", 0.95)
    assert T["cscv_oos_loss"] == ("<", 0.10) and "pbo" not in T
    assert T["oos_sharpe"] == (">=", 1.0)
    assert T["wfo_oos"] == (">=", 0.5) and T["wfo_oos_recent"] == (">", 0.0)
    assert T["cost_stress_sharpe"] == (">", 0.5)
    assert T["plateau"] == (">=", 0.60) and G.PLATEAU_PEAK_FRACTION == 0.5 and G.PLATEAU_RADIUS == 0.20
    assert T["positive_years"] == (">=", 0.60)
    assert T["max_year_share"] == ("<=", 0.40)
    assert set(G.GATE_ORDER) <= set(T)
    with pytest.raises(TypeError):
        G.GATE_THRESHOLDS["dsr"] = (">=", 0.5)  # type: ignore[index]


def test_all_pass(good):
    study, trades = good
    rep, ev = run(study, trades)
    assert rep.verdict == "PASS", rep.to_markdown()
    assert [r.gate for r in rep.rows] == list(G.GATE_ORDER)
    expect = ev.cost.stressed(spec=EURUSD)
    assert expect.slippage_points == ev.cost.slippage_points + pip_points(EURUSD) == ev.cost.slippage_points + 10
    stressed = [c for c in ev.calls if c.spread_multiplier == expect.spread_multiplier]
    assert len(stressed) == 3 and all(replace(c, swap_multiplier=expect.swap_multiplier) == expect for c in stressed)
    assert sorted(c.swap_multiplier for c in stressed) == [0.5, 1.0, 1.5]
    hb = rep.holdout_band
    assert hb and hb["sharpe_lo"] < hb["sharpe_median"] and hb["trades_lo"] < hb["trades_hi"]
    assert 0.85 <= hb["joint_coverage"] <= 0.95 and 0 < hb["tail_level"] < 0.10
    md = rep.to_markdown()
    assert "| Deflated Sharpe probability |" in md and "Verdict: **PASS**" in md
    assert "Pre-registered holdout pass band" in md and "N used = raw 25 trials" in md
    cs = rep.row("cost_stress_sharpe").interpretation
    assert "1 stress unit of slippage" in cs and "on all market and stop fills" in cs
    # minor (cost-stress text): the real slippage in points and as a multiple of the median spread
    assert "slippage 10 points per fill for EURUSD" in cs and "× the median dev spread of" in cs
    d = rep.diagnostics["cost_stress_pip_points"]
    assert d["added_slippage_points"] == 10 and 0 < d["slippage_x_median_spread"] < 5
    assert hb["trades_source"] == "wfo_oos.n_trades" and hb["trades_joint"] is True
    assert rep.effective_trials["n_trials_study"] == 25 and rep.effective_trials["n_trials_prior"] == 0
    assert "pbo" in rep.diagnostics["pbo_detail"]


# ============================================================================ DSR (B1 / R1)
def test_dsr_fails_when_system_was_searched_heavily():
    study, trades = make_study(seed=1, peak_sr=1.6, slope=0.2)
    base, _ = run(study, trades)
    assert base.row("dsr").status == "PASS", base.to_markdown()
    rep, _ = run(study, trades, prior_trials=1e6)
    assert rep.row("dsr").status == "FAIL" and rep.verdict == "FAIL"
    et = rep.effective_trials
    assert et["n_trials"] == 1e6 + 25 and et["dsr"] <= et["psr0"]
    others = {r.gate: r.status for r in rep.rows if r.gate != "dsr"}
    assert set(others.values()) == {"PASS"}, others


def test_dsr_hurdle_is_v0_raw_n():
    """R1: SR0 = √(1/(T−1))·E[max of N raw] — independent of the trial correlation structure."""
    study, trades = make_study(seed=3)
    rep, _ = run(study, trades, prior_trials=75)
    et = rep.effective_trials
    T = study.returns.height
    assert et["sr0"] == pytest.approx(S.expected_max_sharpe(100, 1.0 / (T - 1)))
    assert "eigen" in et and "cluster" in et and "liji" in et and "var_sr" in et


def _equicorr_null(rng, n_days=2340, n=100, rho=0.8):
    common = rng.normal(0, 1, (n_days, 1))
    return 0.006 * (math.sqrt(rho) * common + math.sqrt(1 - rho) * rng.normal(0, 1, (n_days, n)))


def test_dsr_size_on_equicorrelated_null_grid():
    """B1 regression: equicorrelated ρ = 0.8 null grid, N = 100, argmax selected.  The v1.1 gate
    (N_eff ≈ 1.5 × cross-sectional V) passed ≈ 20 %; the v1.2 gate must stay near nominal 5 %."""
    rng = np.random.default_rng(801)
    n_worlds, passes, old = 60, 0, 0
    for _ in range(n_worlds):
        m = _equicorr_null(rng)
        j = int(np.argmax(m.mean(0) / m.std(0, ddof=1)))
        frame = pl.DataFrame({f"t{i}": m[:, i] for i in range(m.shape[1])})
        res = S.dsr_from_matrix(frame, f"t{j}")
        passes += res.dsr >= 0.95
        old += res.dsr_neff_cross >= 0.95
    assert passes / n_worlds <= 0.07
    assert old > passes        # the v1.1 construction (kept as a diagnostic) is the lenient one


@pytest.mark.slow
def test_dsr_size_on_equicorrelated_null_grid_large():
    rng = np.random.default_rng(802)
    n_worlds, passes = 300, 0
    for _ in range(n_worlds):
        m = _equicorr_null(rng)
        j = int(np.argmax(m.mean(0) / m.std(0, ddof=1)))
        res = S.dsr_from_matrix(m, j, methods=())
        passes += res.dsr >= 0.95
    assert passes / n_worlds <= 0.06


# ============================================================================ CSCV OOS loss (R2 / M1)
def test_cscv_oos_loss_fails_on_regime_flipping_trials():
    rng = np.random.default_rng(3)
    n = 2340
    blocks = np.arange(n) // (n // 16)
    flip = np.where(blocks % 2 == 0, 1.0, -1.0)
    extra = {}
    for k in range(30):
        s = 1.0 if k < 15 else -1.0
        extra[f"t{100 + k}"] = 0.006 * (rng.normal(0, 1, n) * 0.3 + s * flip * 1.0)
    study, trades = make_study(seed=2, extra_cols=extra)
    rep, _ = run(study, trades)
    assert rep.row("cscv_oos_loss").status == "FAIL" and rep.verdict == "FAIL"


def test_cscv_gate_not_gamed_by_bleeder_padding():
    """M1 regression: 40 zero-edge configs (ρ = 0.5) padded with 160 consistent bleeders
    (−3 bp/day).  PBO collapses to ≈ 0 (the v1.1 gate PASSes); the absolute CSCV OOS-loss
    probability of the IS-best stays high and the gate FAILs."""
    rng = np.random.default_rng(31)
    n = 2340
    common = rng.normal(0, 1, (n, 1))
    zero = 0.006 * (math.sqrt(0.5) * common + math.sqrt(0.5) * rng.normal(0, 1, (n, 40)))
    bleed = -0.0003 + 0.006 * (math.sqrt(0.5) * common + math.sqrt(0.5) * rng.normal(0, 1, (n, 160)))
    m = np.column_stack([zero, bleed])
    res = S.pbo_cscv(m, n_splits=16)
    assert res.pbo < 0.30                  # would have passed the old PBO gate
    assert res.prob_oos_loss >= 0.10       # fails the v1.2 gate
    # through the gates: padded grid, argmax selected
    dates = _bdays(n)
    returns = pl.DataFrame({"date": dates, **{f"t{i}": m[:, i] for i in range(m.shape[1])}})
    j = int(np.argmax(m.mean(0) / m.std(0, ddof=1)))
    trials = pl.DataFrame({"trial_id": list(range(m.shape[1])), "status": ["ok"] * m.shape[1],
                           "param_a": list(range(m.shape[1]))})
    study = StudyResult("fbs-0098-a1", ("a",), trials, returns, {"a": j}, {"trial_id": j},
                        pl.DataFrame(schema={"date": pl.Date, "path_id": pl.Int64, "ret": pl.Float64}),
                        pl.DataFrame(), pl.DataFrame(), {"method": "grid"})
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, ledger_dir=register(study))
    assert rep.row("cscv_oos_loss").status == "FAIL"
    assert rep.diagnostics["pbo_detail"]["pbo"] < 0.30


# ============================================================================ CPCV OOS
def test_oos_sharpe_fails_when_cpcv_paths_are_weak():
    study, trades = make_study(seed=1, oos_sr=0.4)
    rep, _ = run(study, trades)
    assert rep.row("oos_sharpe").status == "FAIL"
    assert rep.verdict == "FAIL"


def test_missing_cpcv_paths_skips_oos_gate():
    study, trades = make_study(seed=1, cpcv_paths=False)
    rep, _ = run(study, trades)
    assert rep.row("oos_sharpe").status == "SKIPPED" and rep.verdict != "PASS"
    assert rep.holdout_band["source"] == "wfo_oos"          # band comes from the WFO procedure
    study, trades = make_study(seed=1, cpcv_paths=False, wfo_sr=None)
    rep, _ = run(study, trades)
    assert rep.holdout_band is None and rep.row("wfo_oos").status == "SKIPPED"


# ============================================================================ WFO gate (B2)
def test_wfo_gate_fails_dead_edge():
    """B2 regression: an edge that is strong in dev-selected, CPCV and early WFO terms but dead
    over the last ~45 % of dev.  Every other gate passes; v1.1 had no gate on this series, the
    v1.2 recency condition fails it."""
    rng = np.random.default_rng(55)
    study, trades = make_study(seed=5)
    dates = study.returns["date"].to_list()
    n0 = int(0.3 * len(dates))
    n = len(dates) - n0
    true = np.where(np.arange(n) < int(0.55 * n), 3.0, 0.0)
    wfo, wp = make_wfo(dates, true, 0.006, rng, study.selection["trial_id"])
    rec = wfo["ret"].to_numpy().copy()
    k = int(math.ceil(n / 3))
    rec[-k:] = rec[-k:] - rec[-k:].mean() - 1e-5          # recent third: exactly dead (slightly negative)
    wfo = wfo.with_columns(pl.Series("ret", rec))
    dead = replace(study, wfo_oos=wfo, wfo_params=wp)
    rep, _ = run(dead, trades)
    row = rep.row("wfo_oos")
    assert row.status == "FAIL", rep.to_markdown()
    assert row.value >= 0.5                             # whole span alone would pass …
    assert rep.diagnostics["wfo_oos"]["sharpe_recent"] < 0   # … the recent third does not
    assert "most recent third" in row.interpretation and str(wfo["date"][-1]) in row.interpretation
    others = {r.gate: r.status for r in rep.rows if r.gate != "wfo_oos"}
    assert set(others.values()) == {"PASS"}, others
    assert rep.verdict == "FAIL"


def test_wfo_gate_fails_weak_procedure():
    study, trades = make_study(seed=6, wfo_sr=0.0)
    rep, _ = run(study, trades)
    assert rep.row("wfo_oos").status == "FAIL" and rep.verdict == "FAIL"


# ============================================================================ B3 / m3 skips
def test_data_dependent_candidate_set_skips_oos_gates():
    """B3: TPE (or any data-dependent candidate set) → oos_sharpe, wfo_oos, cscv_oos_loss SKIPPED."""
    study, trades = make_study(seed=1, meta={"method": "tpe"})
    rep, _ = run(study, trades)
    for g in ("oos_sharpe", "wfo_oos", "cscv_oos_loss"):
        assert rep.row(g).status == "SKIPPED" and "data-dependent" in rep.row(g).interpretation
    assert rep.verdict == "INCOMPLETE"
    study2 = replace(study, meta={**study.meta, "method": "grid"})
    assert run(study2, trades)[0].verdict == "PASS"
    study3 = replace(study, meta={**study.meta, "method": "sobol", "candidate_set_data_dependent": True})
    assert run(study3, trades)[0].row("wfo_oos").status == "SKIPPED"


def test_embargo_cap_skips_cpcv_gates():
    """m3: embargo cap binds → oos_sharpe and cscv_oos_loss SKIPPED with instructions."""
    study, trades = make_study(seed=1, meta={"embargo_capped": True})
    rep, _ = run(study, trades)
    for g in ("oos_sharpe", "cscv_oos_loss"):
        assert rep.row(g).status == "SKIPPED" and "fewer CPCV groups" in rep.row(g).interpretation
    assert rep.row("wfo_oos").status == "PASS" and rep.verdict == "INCOMPLETE"
    nested = replace(study, meta={"n_trials": 25, "method": "grid", "cpcv": {"embargo_capped": True}})
    assert run(nested, trades)[0].row("oos_sharpe").status == "SKIPPED"


# ============================================================================ cost stress
def test_cost_stress_fails_for_cost_fragile_system(good):
    study, trades = good
    ev = FakeEvaluator(study, trades, drag_per_unit=0.0003)
    rep, _ = run(study, trades, evaluator=ev)
    assert rep.row("cost_stress_sharpe").status == "FAIL" and rep.verdict == "FAIL"
    assert {r.status for r in rep.rows if r.gate != "cost_stress_sharpe"} == {"PASS"}


def test_cost_stress_pip_conversion_from_spec(good):
    """M2 wiring: the stress adds 1 pip converted with the instrument spec (explicit spec=,
    evaluator.spec, or load_instrument(evaluator.symbol)); a real evaluator without any spec is
    SKIPPED rather than silently using 1 pip = 1 point."""
    study, trades = good
    ev = FakeEvaluator(study, trades, spec=None)
    rep, _ = run(study, trades, evaluator=ev)
    row = rep.row("cost_stress_sharpe")
    assert row.status == "SKIPPED" and "No instrument spec" in row.interpretation and rep.verdict == "INCOMPLETE"
    assert not any(c.spread_multiplier == 1.5 for c in ev.calls)          # never ran the 1-point fallback
    ev = FakeEvaluator(study, trades, spec=None)
    rep, _ = run(study, trades, evaluator=ev, spec=EURUSD)
    assert rep.row("cost_stress_sharpe").status == "PASS"
    assert {c.slippage_points for c in ev.calls if c.spread_multiplier == 1.5} == {10.0}
    ev = FakeEvaluator(study, trades, spec=None)
    ev.symbol = "XAUUSD"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep, _ = run(study, trades, evaluator=ev)
        gold = pip_points(load_instrument("XAUUSD", book="FBS"))
    assert f"slippage {gold:g} points per fill for XAUUSD" in rep.row("cost_stress_sharpe").interpretation
    assert rep.diagnostics["cost_stress_pip_points"]["pip_points"] == gold
    ev = FakeEvaluator(study, trades, spec=None)
    ev.is_synthetic = True
    rep, _ = run(study, trades, evaluator=ev)
    assert rep.row("cost_stress_sharpe").status == "PASS" and "synthetic" in rep.row("cost_stress_sharpe").interpretation


def test_trade_criterion_unavailable_when_wfo_counts_missing(good):
    """Null WFO n_trades means "unavailable" — never a [0, 0] band, and never the dev-selected
    trial's trade rate (F4): the trade criterion is not applied."""
    study, trades = good
    s2 = replace(study, wfo_oos=study.wfo_oos.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_trades")))
    rep, _ = run(s2, trades)
    hb = rep.holdout_band
    assert not np.isfinite(hb["trades_lo"]) and "not available" in hb["trades_source"]
    assert "criterion not applied" in rep.to_markdown()
    assert S.holdout_check(hb, np.full(260, 0.001) + np.random.default_rng(0).normal(0, 1e-4, 260), 0)["checks"]["trades"]


def test_no_evaluator_skips_cost_stress_and_blocks_pass(good):
    study, trades = good
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades,
                           mechanism_check=_mech_ok, ledger_dir=register(study))
    assert rep.row("cost_stress_sharpe").status == "SKIPPED"
    assert rep.verdict == "INCOMPLETE"


# ============================================================================ plateau (B4 / M5)
def test_plateau_ignores_optimizer_score(good):
    """B4 regression: the optimizer's own plateau_score is a diagnostic; the gate recomputes."""
    study, trades = good
    fake_low = replace(study, selection={**study.selection, "plateau_score": 0.0})
    rep, _ = run(fake_low, trades)
    assert rep.row("plateau").status == "PASS" and rep.diagnostics["plateau_optimizer"]["score"] == 0.0
    spiky, tr2 = make_study(seed=1, slope=1.5, plateau_score=1.0)   # optimizer claims a perfect plateau
    rep, _ = run(spiky, tr2)
    assert rep.row("plateau").status == "FAIL"


def test_plateau_recomputed_on_coarse_grid():
    smooth, _ = make_study(seed=1, plateau_score=None, slope=0.2)
    info = G.plateau_score(smooth, PPY)
    assert info["plateau_score"] == 1.0 and info["n_neighbours"] == 8     # ±20 % of 3 → adjacent levels
    assert "adjacent" in info["rules"]["a"]
    spiky, _ = make_study(seed=1, plateau_score=None, slope=1.5)
    assert G.plateau_score(spiky, PPY)["plateau_score"] < 0.6
    # a wider pre-registered radius reaches the second ring
    wide = G.plateau_score(smooth, PPY, radius={"a": 0.7, "b": 0.7})
    assert wide["n_neighbours"] == 24


def _bump_study(step: int, *, claim: float = 1.0, height: float = 2.0, width: float = 2.0,
                base: float = 0.5, n_days: int = 2340, seed: int = 3) -> StudyResult:
    """Red-team p10b surface: a in [0, 100] step ``step``, narrow Gaussian bump (sd ``width``
    units) of ``height`` annual Sharpe on a ``base`` floor at a = 50; common noise (ρ = 1)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n_days)
    sd = 0.006
    a_vals = list(range(0, 101, step))
    cols = {}
    for i, a in enumerate(a_vals):
        sr = base + height * math.exp(-0.5 * ((a - 50) / width) ** 2)
        cols[f"t{i}"] = _sr_to_mu(sr, sd) + sd * z
    sel = a_vals.index(50)
    trials = pl.DataFrame({"trial_id": list(range(len(a_vals))), "status": ["ok"] * len(a_vals), "param_a": a_vals})
    return StudyResult(f"fbs-0097-a1", ("a",), trials, pl.DataFrame({"date": _bdays(n_days), **cols}),
                       {"a": 50}, {"trial_id": sel, "plateau_score": claim,
                                   "neighbourhood": "PlateauConfig(peak_fraction=0.05)"},
                       pl.DataFrame(), pl.DataFrame(), pl.DataFrame(),
                       {"method": "grid", "space": {"params": [{"name": "a", "kind": "int", "low": 0, "high": 100,
                                                                "step": step}]}})


@pytest.mark.parametrize("step", [1, 2, 5, 10])
def test_plateau_narrow_bump_fails_at_every_grid_resolution(step):
    """M5 + B4 regression (red-team p10): the same narrow bump scored 0.00 at steps 10/5 and
    1.00 at steps 2/1 (grid-step neighbourhoods), and the optimizer's peak_fraction=0.05 score
    was taken by the gate.  The economic ±20 % neighbourhood fails it at every resolution."""
    study = _bump_study(step)
    info = G.plateau_score(study, PPY)
    assert info["plateau_score"] < 0.6, info
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, ledger_dir=register(study))
    assert rep.row("plateau").status == "FAIL" and "Matrix-based fallback" in rep.row("plateau").interpretation
    assert rep.diagnostics["plateau_optimizer"]["score"] == 1.0


def test_plateau_broad_bump_passes_at_every_grid_resolution():
    for step in (1, 2, 5, 10):
        info = G.plateau_score(_bump_study(step, width=25.0), PPY)
        assert info["plateau_score"] >= 0.6, (step, info)


def test_plateau_2d_spike_with_generous_optimizer_config():
    """p10 (2-D): a single sharp spike on a flat-zero surface; the optimizer reports 1.00 under
    peak_fraction = 0.05 — the gate still FAILs."""
    rng = np.random.default_rng(11)
    n = 2340
    z = rng.normal(0, 1, (n, 1))
    cols, rows = {}, []
    tid = 0
    for a in range(11):
        for b in range(11):
            sr = 2.5 if (a, b) == (5, 5) else 0.0
            cols[f"t{tid}"] = _sr_to_mu(sr, 0.006) + 0.006 * (math.sqrt(0.7) * z[:, 0] + math.sqrt(0.3) * rng.normal(0, 1, n))
            rows.append({"trial_id": tid, "status": "ok", "param_a": a, "param_b": b})
            if (a, b) == (5, 5):
                sel = tid
            tid += 1
    study = StudyResult("fbs-0096-a1", ("a", "b"), pl.DataFrame(rows), pl.DataFrame({"date": _bdays(n), **cols}),
                        {"a": 5, "b": 5}, {"trial_id": sel, "plateau_score": 1.0}, pl.DataFrame(), pl.DataFrame(),
                        pl.DataFrame(), {"method": "grid"})
    assert G.plateau_score(study, PPY)["plateau_score"] < 0.6
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, ledger_dir=register(study))
    assert rep.row("plateau").status == "FAIL"


def test_plateau_relative_vs_range_and_categoricals():
    # param x in [-1, 1] (not strictly positive → ±20 % of the range = ±0.4); categorical c must match
    xs = [-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0]
    rows, cols = [], {}
    rng = np.random.default_rng(0)
    n = 600
    z = rng.normal(0, 1, n)
    tid = 0
    for c in ("long", "short"):
        for x in xs:
            rows.append({"trial_id": tid, "status": "ok", "param_x": x, "param_c": c})
            cols[f"t{tid}"] = 0.001 + 0.006 * z + (0.0 if c == "long" else -0.002)
            tid += 1
    trials = pl.DataFrame(rows)
    sel = 3  # x = 0.0, c = long
    study = StudyResult("fbs-0095-a1", ("x", "c"), trials, pl.DataFrame({"date": _bdays(n), **cols}),
                        {"x": 0.0, "c": "long"}, {"trial_id": sel}, pl.DataFrame(), pl.DataFrame(), pl.DataFrame(), {})
    info = G.plateau_score(study, PPY)
    assert sorted(info["neighbour_trials"]) == [2, 4]     # x = ±0.2 only, same category
    assert "range" in info["rules"]["x"] and info["rules"]["c"] == "categorical: equal"


# ============================================================================ time stability / trades / mechanism
def test_time_stability_fails_when_pnl_concentrated():
    n = 2340
    rng = np.random.default_rng(4)
    dates = _bdays(n)
    years = np.array([d.year for d in dates])
    r = 0.004 * rng.normal(0, 1, n) + 0.00005
    r[years == 2020] += 0.004        # one monster year
    study, trades = make_study(seed=1, sel_override=r)
    rep, _ = run(study, trades)
    assert rep.row("max_year_share").status == "FAIL"
    assert rep.verdict == "FAIL"


def test_positive_years_fails_when_most_years_lose():
    n = 2340
    rng = np.random.default_rng(5)
    dates = _bdays(n)
    years = np.array([d.year for d in dates])
    r = 0.004 * rng.normal(0, 1, n) - 0.0004
    r[(years == 2017) | (years == 2021)] += 0.006
    study, trades = make_study(seed=1, sel_override=r)
    rep, _ = run(study, trades)
    assert rep.row("positive_years").status == "FAIL"


def test_trade_count_fails_with_too_few_noisy_trades():
    study, trades = make_study(seed=1)
    rng = np.random.default_rng(6)
    few = trades.head(15).with_columns(pl.Series("pnl_ccy", rng.normal(0.002, 0.02, 15) * 100_000.0))
    rep, _ = run(study, few)
    row = rep.row("trade_count")
    assert row.status == "FAIL" and "15 trades" in row.interpretation
    assert rep.verdict == "FAIL"


def test_trade_count_skipped_without_trades(good):
    study, _ = good
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, mechanism_check=_mech_ok, 
                           ledger_dir=register(study))
    row = rep.row("trade_count")
    assert row.status == "SKIPPED" and "No per-trade data" in row.interpretation
    assert rep.verdict != "PASS"


def test_mechanism_manual_and_failing():
    study, trades = make_study(seed=1)
    rep, _ = run(study, trades, mechanism_check=None)
    assert rep.row("mechanism").status == "MANUAL" and rep.verdict == "INCOMPLETE"
    rep, _ = run(study, trades, mechanism_check=lambda s, e: (False, "Random filter does as well."))
    assert rep.row("mechanism").status == "FAIL" and rep.verdict == "FAIL"


# ============================================================================ holdout band (M3 / F4)
def test_holdout_band_uses_wfo_procedure_trade_counts():
    """F4 regression: the procedure trades 0.6/day (WFO picks trial 7) while the dev-selected
    trial trades 0.2/day.  The trade range must come from the procedure."""
    study, _ = make_study(seed=8)
    sel = study.selection["trial_id"]
    dates = study.returns["date"]
    rng = np.random.default_rng(9)
    tc = pl.DataFrame({"date": dates, **{f"t{i}": rng.poisson(0.6 if i == 7 else 0.2, dates.len()).astype(float)
                                        for i in range(25)}})
    wp = study.wfo_params.with_columns(pl.lit(7).alias("selected_trial"))
    s2 = replace(study, wfo_oos=study.wfo_oos.drop("n_trades"), wfo_params=wp, meta={**study.meta, "trade_counts": tc})
    tpd, src = G.wfo_trades_per_day(s2)
    j = s2.wfo_oos.join(tc, on="date", how="left")
    np.testing.assert_array_equal(tpd, j["t7"].to_numpy())
    assert "WFO procedure" in src
    low_trades = make_trades(dates, study.returns[f"t{sel}"].to_numpy(), int(0.2 * dates.len()), rng)
    rep, _ = run(s2, low_trades)
    hb = rep.holdout_band
    assert hb["trades_lo"] < 0.6 * 260 < hb["trades_hi"] and hb["trades_lo"] > 0.2 * 260
    assert hb["trades_joint"] is True and "WFO procedure" in hb["trades_source"]
    assert "approximate" in hb["trades_source"] and "exit date" in hb["trades_source"]
    # an explicit n_trades column on wfo_oos wins
    s3 = replace(s2, wfo_oos=s2.wfo_oos.with_columns(pl.lit(1.0).alias("n_trades")))
    assert G.wfo_trades_per_day(s3)[1] == "wfo_oos.n_trades"
    # any null n_trades (entry counts unavailable) → rebuild from trade_counts, never zero-fill
    s4 = replace(s2, wfo_oos=s2.wfo_oos.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_trades")))
    assert "WFO procedure" in G.wfo_trades_per_day(s4)[1]


def test_holdout_band_reports_power_and_decisiveness():
    """M3: the band reports P(pass | zero edge); a weak WFO edge is flagged not decisive."""
    weak, tr = make_study(seed=12, wfo_sr=0.6)
    rep, _ = run(weak, tr)
    hb = rep.holdout_band
    assert hb["p_pass_zero_edge"] > 0.30 and hb["decisive"] is False
    assert "Not decisive at this horizon" in rep.to_markdown() and "NOT_DECISIVE" in rep.to_markdown()
    strong, tr = make_study(seed=12, wfo_sr=4.0)
    hb = run(strong, tr)[0].holdout_band
    assert hb["p_pass_zero_edge"] < 0.30 and hb["decisive"] is True


# ============================================================================ ledger: M4, m1, m9, N2
def _create(tmp, sid, attempt, system="toy", issue=99, n_trials=None, **extra):
    extra.setdefault("search_space", GRID_SPACE)
    extra.setdefault("method", "grid")
    extra.setdefault("evaluator", FAKE_DESC)
    ledger.create_study(ledger_dir=tmp, study_id=sid, book="FBS", system=system, issue=issue, attempt=attempt,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-test",
                        cv_scheme="CPCV(n=6,k=2)", **extra)
    if n_trials is not None:
        ledger.log_event(sid, "trials", ledger_dir=tmp, n_trials=n_trials)


def test_prior_trials_read_from_ledger_and_raise_when_missing(tmp_path, good):
    """M4 regression: evaluate_gates reads earlier attempts' raw trials itself; attempt 2 with
    no ledger history raises instead of silently deflating by 0.  N2: an explicit prior below
    the ledger's count raises; the deprecated alias is gone."""
    study, trades = good
    a2 = replace(study, study_id="fbs-0099-a2", meta={**study.meta, "book": "FBS", "system": "toy", "attempt": 2})
    _create(tmp_path, "fbs-0099-a2", 2)
    with pytest.raises(G.GateError, match="attempt 2"):
        run(a2, trades, ledger_dir=tmp_path)
    tmp2 = tmp_path / "l2"
    _create(tmp2, "fbs-0099-a1", 1, n_trials=400)
    _create(tmp2, "fbs-0099-a2", 2)
    _create(tmp2, "fbs-0099-a3", 3, n_trials=9999)              # created later: counts too (R2 minor N2b)
    rep, _ = run(a2, trades, ledger_dir=tmp2)
    et = rep.effective_trials
    assert et["n_trials_prior"] == 400 + 9999 and et["n_trials"] == 25 + 400 + 9999
    assert "fbs-0099-a1" in et["prior_source"] and "fbs-0099-a3" in et["prior_source"]
    # N2: explicit prior may only raise N
    assert run(a2, trades, ledger_dir=tmp2, prior_trials=20000)[0].effective_trials["n_trials_prior"] == 20000
    with pytest.raises(G.GateError, match="below the ledger"):
        run(a2, trades, ledger_dir=tmp2, prior_trials=1000)
    with pytest.raises(TypeError, match="prior_trials"):
        run(a2, trades, ledger_dir=tmp2, prior_effective_trials=7.0)
    # identity comes from the ledger row when meta lacks it
    bare = replace(study, study_id="fbs-0099-a2", meta={"n_trials": 25, "method": "grid", "space": GRID_SPACE})
    assert run(bare, trades, ledger_dir=tmp2)[0].effective_trials["n_trials_prior"] == 400 + 9999


def test_gates_require_the_ledger_row_and_matching_meta(tmp_path, good):
    """N2 regression: a study that is not in the ledger cannot be gated; a meta that disagrees
    with the ledger (the r1_p11 (f) relabel: ledger sma/3, meta other/1) raises."""
    study, trades = good
    with pytest.raises(G.GateError, match="not in the ledger"):
        G.evaluate_gates(study, FakeEvaluator(study, trades), periods_per_year=PPY, ledger_dir=tmp_path / "empty")
    s3 = replace(study, study_id="fbs-0007-a3", meta={**study.meta, "system": "sma", "attempt": 3})
    _create(tmp_path, "fbs-0007-a1", 1, system="sma", issue=7, n_trials=49)
    _create(tmp_path, "fbs-0007-a2", 2, system="sma", issue=7, n_trials=49)
    _create(tmp_path, "fbs-0007-a3", 3, system="sma", issue=7)
    rep, _ = run(s3, trades, ledger_dir=tmp_path)
    assert rep.effective_trials["n_trials_prior"] == 98
    relabel = replace(s3, meta={**s3.meta, "system": "other", "attempt": 1})
    with pytest.raises(G.GateError, match="disagrees with the ledger"):
        run(relabel, trades, ledger_dir=tmp_path)
    for bad in ({"issue": 8}, {"book": "B3"}, {"method": "sobol"}, {"plateau_radius": 0.3},
                {"space": opt.SearchSpace([opt.IntParam("a", 1, 9, plateau_scale="relative"), opt.IntParam("b", 1, 5, plateau_scale="relative")]).to_json()}):
        with pytest.raises(G.GateError, match="disagrees"):
            run(replace(s3, meta={**s3.meta, **bad}), trades, ledger_dir=tmp_path)


def test_prior_trials_follow_system_name_or_issue_not_labels(tmp_path, good):
    """N2 regression (r1_p11 b/c): a renamed system (same normalised name, or same issue) and a
    new issue on the same system both count every earlier related study, whatever its attempt
    label; an unrelated study (other name, other issue) does not."""
    study, trades = good
    _create(tmp_path, "fbs-0007-a1", 1, system="SMA-Cross", issue=7, n_trials=49)
    _create(tmp_path, "fbs-0007-a2", 2, system="sma_cross", issue=7, n_trials=49)
    _create(tmp_path, "fbs-0007-a3", 3, system="sma cross", issue=7, n_trials=49)
    _create(tmp_path, "fbs-0050-a1", 1, system="rsi", issue=50, n_trials=1000)          # unrelated
    # rename to a new label under the same issue, attempt reset to 1
    _create(tmp_path, "fbs-0007-a1x", 1, system="sma_v2", issue=7)
    s = replace(study, study_id="fbs-0007-a1x", meta={**study.meta, "system": "sma_v2", "attempt": 1})
    et = run(s, trades, ledger_dir=tmp_path)[0].effective_trials
    assert et["n_trials_prior"] == 147 and set(et["prior_studies"]) == {"fbs-0007-a1", "fbs-0007-a2", "fbs-0007-a3"}
    # same system (normalised) under a NEW issue, attempt 1
    _create(tmp_path, "fbs-0008-a1", 1, system="SMACROSS", issue=8)
    s = replace(study, study_id="fbs-0008-a1", meta={**study.meta, "system": "SMACROSS", "attempt": 1})
    et = run(s, trades, ledger_dir=tmp_path)[0].effective_trials
    assert et["n_trials_prior"] == 147 and "fbs-0050-a1" not in et["prior_studies"]
    assert ledger.normalise_system("SMA-Cross") == ledger.normalise_system("sma cross") == "smacross"


def test_plateau_radius_comes_from_the_ledger(tmp_path, good):
    """N3 regression: the gate-time plateau_radius kwarg is gone (TypeError); the radius is the
    one pre-registered in the study's ledger row, and a radius below the floor is refused."""
    study, trades = good
    with pytest.raises(TypeError, match="plateau_radius"):
        run(study, trades, plateau_radius=1e-6)
    ld = register(study, plateau_radius={"a": 0.5, "b": 0.5})
    rep, _ = run(study, trades, ledger_dir=ld)
    assert rep.ledger_context["plateau_radius"] == {"a": 0.5, "b": 0.5}
    assert rep.plateau_detail["radius"] == {"a": 0.5, "b": 0.5}
    offs = [q["offset"] for q in rep.plateau_detail["points"] if q["param"] == "a"]
    assert offs == ["x0.5", "x0.75", "x1.25", "x1.5"] and "ledger study_created row" in rep.row("plateau").interpretation
    rep0, _ = run(study, trades)                                   # no radius in the row → DESIGN default
    assert rep0.ledger_context["plateau_radius"] == 0.20 and "default" in rep0.plateau_detail["radius_source"]
    with pytest.raises(G.GateError, match="plateau_radius"):
        run(study, trades, ledger_dir=register(study, plateau_radius=1e-6))


def test_data_dependent_flag_is_read_from_the_ledger(tmp_path, good):
    """N1 regression (gate side): any ledger event that set candidate_set_data_dependent=True
    (or logged TPE-sourced trials) makes the study data-dependent, and a meta that says False
    raises instead of clearing it."""
    study, trades = good
    sid = study.study_id
    _create(tmp_path, sid, 1, system="toy", issue=99, candidate_set_data_dependent=False)
    ledger.log_event(sid, "trials", ledger_dir=tmp_path, n_trials=25, n_by_source={"tpe": 5, "sobol": 20})
    ledger.log_event(sid, "note", ledger_dir=tmp_path, candidate_set_data_dependent=False)
    assert ledger.candidate_set_data_dependent(sid, ledger_dir=tmp_path) is True
    with pytest.raises(G.GateError, match="candidate_set_data_dependent"):
        run(replace(study, meta={**study.meta, "candidate_set_data_dependent": False}), trades, ledger_dir=tmp_path)
    rep, _ = run(study, trades, ledger_dir=tmp_path)             # meta silent → ledger decides
    assert rep.row("wfo_oos").status == "SKIPPED" and rep.verdict == "INCOMPLETE"


def test_log_gates_counts_own_trials_and_flags_reruns(tmp_path, good):
    """m1 + m9 regression: the ledger gets the study's own trial count (no cumulative double
    count across attempts) and a gate-run index; a re-run is flagged in the report."""
    study, trades = good
    s1 = replace(study, meta={**study.meta, "book": "FBS", "system": "toy", "attempt": 1})
    _create(tmp_path, s1.study_id, 1)
    rep, _ = run(s1, trades, ledger_dir=tmp_path, log=True)
    assert rep.prior_gate_runs == [] and rep.ledger_row["event"] == "gates"
    with pytest.raises(TypeError, match="R3-1"):
        G.log_gates(rep, ledger_dir=tmp_path)          # an externally supplied report never reaches the ledger
    state = ledger.studies(tmp_path)[s1.study_id]
    assert state["events"] == ["study_created", "selection", "gates"] and state["verdict"] == "PASS"
    assert state["gates"]["cscv_oos_loss"] == pytest.approx(rep.row("cscv_oos_loss").value)
    assert state["gate_run"] == 1 and state["n_trials_study"] == 25 and state["n_trials_dsr"] == 25
    assert state["holdout_band"]["sharpe_lo"] == pytest.approx(rep.holdout_band["sharpe_lo"])
    # N4: the judge-run plateau evaluations are logged (count + params + Sharpe) and not in N
    pl_log = state["plateau"]
    assert pl_log["kind"] == "judge-run" and pl_log["n_evaluations"] == 17 and len(pl_log["points"]) == 16
    assert all({"params", "sharpe", "status", "pass"} <= set(q) for q in pl_log["points"])
    assert state["n_trials_dsr"] == 25 and state["ledger_context"]["system"] == "toy"
    # attempt 2: prior = 25 (a1's own); after logging, attempt 3's prior = 25 + 25, not 25 + 50
    s2 = replace(study, study_id="fbs-0099-a2", meta={**s1.meta, "attempt": 2})
    _create(tmp_path, "fbs-0099-a2", 2)
    r2, _ = run(s2, trades, ledger_dir=tmp_path, log=True)
    assert r2.effective_trials["n_trials_prior"] == 25
    assert ledger.system_prior_trials("FBS", "toy", exclude_study="fbs-0099-a3", ledger_dir=tmp_path)[0] == 50
    # re-run on the same study → flagged
    again, _ = run(s1, trades, ledger_dir=tmp_path, log=True)
    assert len(again.prior_gate_runs) == 1 and again.prior_gate_runs[0]["verdict"] == "PASS"
    assert "gated 1 time(s) before" in again.to_markdown()
    assert ledger.studies(tmp_path)[s1.study_id]["gate_run"] == 2
    ledger.verify_chain(tmp_path)


# ============================================================================ Phase 1 exit test
def test_zero_edge_system_fails():
    """Phase 1 exit test (fast version): a zero-edge grid, selected as its in-sample best."""
    fails = 0
    for seed in range(3):
        study, trades = make_study(seed=10 + seed, peak_sr=0.0, slope=0.0, oos_sr=0.0, wfo_sr=0.0,
                                   plateau_score=None)
        rep, _ = run(_as_argmax(study), trades)
        fails += rep.verdict == "FAIL"
    assert fails == 3


@pytest.mark.slow
def test_zero_edge_fails_95pct_and_planted_edge_passes():
    """Phase 1 exit test: zero-edge system fails ≥ 95 % of the time; planted edge passes."""
    n = 40
    fails = 0
    for seed in range(n):
        study, trades = make_study(seed=1000 + seed, peak_sr=0.0, slope=0.0, oos_sr=0.0, wfo_sr=0.0)
        rep, _ = run(_as_argmax(study), trades)
        fails += rep.verdict == "FAIL"
    assert fails / n >= 0.95
    passes = sum(run(*make_study(seed=2000 + s))[0].verdict == "PASS" for s in range(10))
    assert passes >= 8


# ============================================================================ fix round 2: N4 judge-run plateau, N7
def _study_run(ev, space, tmp, sid, **kw):
    kw.setdefault("wfo", None)
    kw.setdefault("cv", opt.CPCVConfig(6, 2))
    return opt.run_study(ev, space, book="FBS", system="n4", issue=4, attempt=1, study_id=sid, n_jobs=1,
                         ledger_dir=tmp / "ledger", studies_dir=tmp / "studies", **kw)


def _gate(study, ev, tmp, **kw):
    return G.evaluate_gates(study, ev, periods_per_year=PPY, ledger_dir=tmp / "ledger",
                            mechanism_check=_mech_ok, **kw)


@pytest.mark.parametrize("step", [1, pytest.param(2, marks=pytest.mark.slow),
                                  pytest.param(5, marks=pytest.mark.slow), 10])
def test_judge_plateau_spike_fails_broad_passes_at_every_grid_step(tmp_path, step):
    """N4 regression (red-team p10 surface, SyntheticEvaluator, real run_study): with an evaluator
    the gate perturbs the selected config itself, so the verdict no longer depends on the grid
    step: the 2 %-wide spike FAILs and a broad bump PASSes at steps 1 / 2 / 5 / 10."""
    sp = opt.SearchSpace([opt.IntParam("a", 0, 100, step, plateau_step=20)])
    out = {}
    for width in (0.02, 0.30):
        ev = SyntheticEvaluator(bounds={"a": (0, 100)}, bumps=({"center": {"a": 50}, "height": 2.0, "width": width},),
                                base_sharpe=0.5, rho=1.0, seed=3)
        res = _study_run(ev, sp, tmp_path, f"spike-{step}-{width}")
        assert res.selected_params["a"] == 50
        rep = _gate(res, ev, tmp_path)
        row = rep.row("plateau")
        assert "judge-run" in row.interpretation and rep.plateau_detail["kind"] == "judge-run"
        assert [q["value"] for q in rep.plateau_detail["points"]] == [30, 40, 60, 70]      # ± r/2, ± r of range
        assert rep.plateau_detail["n_evaluations"] == 5
        out[width] = row.status
    assert out == {0.02: "FAIL", 0.30: "PASS"}, out


def _sobol5():
    return opt.SearchSpace([opt.IntParam("look", 10, 200, plateau_scale="relative"), opt.FloatParam("mult", 1.0, 4.0, plateau_scale="relative"),
                            opt.IntParam("hold", 1, 20, plateau_scale="relative"), opt.FloatParam("thr", 0.0, 2.0, plateau_step=0.4),
                            opt.IntParam("slow", 20, 300, plateau_scale="relative")])


def test_judge_plateau_on_d5_sobol_study(tmp_path):
    """N4 regression (r1_p10e: a d = 5 Sobol study had 0 neighbours in the ±20 % box in 100 % of
    studies → plateau SKIPPED, verdict never PASS).  The judge-run perturbations always exist:
    a spike planted on a Sobol candidate FAILs, a broad optimum there PASSes; the matrix fallback
    on the same study cannot judge it."""
    sp = _sobol5()
    cands = opt.candidate_set(sp, 64, method="sobol", seed=1)
    margin = [min(min(p.unit(c[p.name]), 1 - p.unit(c[p.name])) for p in sp.params) for c in cands]
    k = int(np.argmax(margin))
    centre = cands[k]
    bounds = {p.name: (p.low, p.high) for p in sp.params}
    got = {}
    for width in (0.02, 0.35):
        ev = SyntheticEvaluator(bounds=bounds, bumps=({"center": centre, "height": 2.5, "width": width},),
                                base_sharpe=0.0, rho=0.8, seed=7)
        res = _study_run(ev, sp, tmp_path, f"sobol5-{width}", method="sobol", n_trials=64, seed=1)
        assert res.trials["source"].to_list() == ["sobol"] * 64
        # judged at the planted centre (not the study's own pick): the judge is called directly, since
        # evaluate_gates refuses a selection that differs from the ledger's (R3-2)
        judged = replace(res, selected_params=dict(centre), selection={**res.selection, "trial_id": k})
        if res.selection["trial_id"] != k:
            with pytest.raises(G.GateError, match="R3-2"):
                _gate(judged, ev, tmp_path)
        pdl = G.judge_plateau(judged, ev, space=sp, radius=0.2, periods_per_year=PPY)
        assert len(pdl["points"]) == 28 and pdl["n_evaluations"] <= 29   # 5×4 + 8 joint
        assert sum(q["param"] == "joint" for q in pdl["points"]) == 8
        assert pdl["peak_sharpe"] > 1.5
        got[width] = "PASS" if G._cmp(">=", pdl["plateau_score"], 0.6) else "FAIL"
        fb = G.plateau_score(judged, PPY, radius=0.2, space=sp.to_json())
        assert not (fb["plateau_score"] >= 0.6) or fb["n_neighbours"] == 0      # the matrix fallback cannot judge it
    assert got == {0.02: "FAIL", 0.35: "PASS"}, got


def test_judge_plateau_validity_and_search_bounds(tmp_path):
    """N4 + round 2b: points rejected by space.is_valid or by natural validity (lookback int
    ≥ 1, positive params > 0) fail; points outside the SEARCH bounds are evaluated and reported;
    ints that round back to x move to the next distinct int; score = min over axes."""
    sp = opt.SearchSpace([opt.IntParam("a", 1, 10, plateau_scale="relative"), opt.FloatParam("x", 0.0, 1.0, plateau_step=0.2)],
                         constraint=lambda p: p["x"] < 0.75)
    ev = SyntheticEvaluator(bounds={"a": (1, 10), "x": (0.0, 1.0)}, base_sharpe=1.5, rho=1.0, seed=2)
    pts = G.plateau_perturbations(sp, {"a": 2, "x": 0.6}, 0.2)
    assert [q["value"] for q in pts if q["param"] == "a"] == [0, 1, 3, 4]      # 1.6→2→1, 1.8→2 … distinct
    assert [q["natural_valid"] for q in pts if q["param"] == "a"] == [False, True, True, True]
    assert [q["value"] for q in pts if q["param"] == "x"] == [0.4, 0.5, 0.7, 0.8]
    res = _study_run(ev, sp, tmp_path, "oob", method="sobol", n_trials=16, seed=0)
    judged = replace(res, selected_params={"a": 2, "x": 0.6},
                     selection={**res.selection, "trial_id": int(res.trials["trial_id"][0])})
    pj = G.judge_plateau(judged, ev, space=sp, radius=0.2, periods_per_year=PPY)
    st = {(q["param"], q["value"]): q["status"] for q in pj["points"] if q["param"] != "joint"}
    assert st[("a", 0)] == "invalid_natural" and st[("x", 0.8)] == "invalid"
    # joint axis (R2-1): (a, x) = (0, .4) and (0, .8) naturally invalid, (4, .8) rejected by the constraint
    js = {q["offset"]: (q["value"]["a"], q["value"]["x"], q["status"]) for q in pj["points"] if q["param"] == "joint"}
    assert js["all -r"] == (0, 0.4, "invalid_natural") and js["all +r"] == (4, 0.8, "invalid")
    assert js["alt- r"] == (0, 0.8, "invalid_natural") and js["alt+ r"] == (4, 0.4, "ok")
    assert pj["pass_share_by_param"] == {"a": 0.75, "x": 0.75, "joint": 0.625} and pj["plateau_score"] == 0.625
    assert pj["n_evaluations"] == 12
    assert [p for p in pj["outside_search_bounds_points"] if p[0] != "joint"] == [("a", 0)]   # invalid anyway
    # outside the search bounds but naturally valid → evaluated, reported, and the edge is flagged
    pj = G.judge_plateau(replace(judged, selected_params={"a": 10, "x": 0.0}), ev, space=sp, radius=0.2,
                         periods_per_year=PPY)
    out = {(q["param"], q["value"]) for q in pj["points"] if q["outside_search_bounds"] and q["param"] != "joint"}
    assert out == {("a", 11), ("a", 12), ("x", -0.2), ("x", -0.1)}
    assert all(q["status"] == "ok" for q in pj["points"]) and pj["plateau_score"] == 1.0
    assert pj["selected_at_edge"] == ["a", "x"]


def _n8_study(tmp, nd, sid):
    sp = opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)] + [opt.IntParam(f"z{i}", 0, 20, plateau_step=4) for i in range(nd)])
    ev0 = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=1.0, seed=5)
    res = _study_run(ev0, sp, tmp, sid, method="sobol", n_trials=8, seed=0, cv=opt.CPCVConfig(4, 1))
    sel = {"a": 5, **{f"z{i}": 10 for i in range(nd)}}
    return sp, replace(res, selected_params=sel, selection={**res.selection, "trial_id": 0})


def test_plateau_min_over_axes_is_not_lifted_by_irrelevant_params(tmp_path):
    """N8 regression (round 2b): pooled over axes, a spike with 2 irrelevant dummies scored
    0.67 (PASS) and a half-plateau with 1 dummy 0.75 (PASS).  With the min over axes, dummies
    (share 1.0 each) cannot lift the relevant axis: spike and partial plateau FAIL with any
    number of dummies; a broad plateau PASSes."""
    for nd in (0, 1, 2, 3, 4):
        sp, judged = _n8_study(tmp_path, nd, f"n8-{nd}")
        got = {}
        for name, width in (("spike", 0.02), ("partial", 0.09), ("broad", 0.30)):
            ev = SyntheticEvaluator(bounds={"a": (0, 10)}, bumps=({"center": {"a": 5}, "height": 2.0, "width": width},),
                                    rho=1.0, seed=5)
            pj = G.judge_plateau(judged, ev, space=sp, radius=0.2, periods_per_year=PPY)
            assert all(pj["pass_share_by_param"][f"z{i}"] == 1.0 for i in range(nd))
            axis_pts = [q["pass"] for q in pj["points"] if q["param"] != "joint"]
            got[name] = (pj["plateau_score"], float(np.mean(axis_pts)))
        assert got["spike"][0] == 0.0 and got["partial"][0] == 0.5 and got["broad"][0] == 1.0, (nd, got)
        assert [G._cmp(">=", got[k][0], 0.6) for k in ("spike", "partial", "broad")] == [False, False, True]
        if nd >= 2:
            assert got["spike"][1] >= 0.6            # the pooled per-axis share would have passed


def test_broad_plateau_selected_at_search_space_edge_passes(tmp_path):
    """Round 2b: bounds are search limits.  A broad optimum sitting on the edge of the searched
    space is judged on its real neighbourhood (points beyond the bound are evaluated), in both
    relative (low > 0) and range mode, and the edge is reported."""
    for lo, hi, x in ((10, 50, 50), (0, 10, 10)):
        sp = opt.SearchSpace([opt.IntParam("look", lo, hi, **({"plateau_scale": "relative"} if lo > 0
                                                             else {"plateau_step": 0.2 * (hi - lo)}))])
        ev = SyntheticEvaluator(bounds={"look": (lo, hi)},
                                bumps=({"center": {"look": x}, "height": 2.0, "width": 0.4},), rho=1.0, seed=4)
        res = _study_run(ev, sp, tmp_path, f"edge-{lo}")
        judged = replace(res, selected_params={"look": x},
                         selection={**res.selection, "trial_id": int(res.trials.filter(pl.col("param_look") == x)
                                                                      ["trial_id"][0])})
        rep = _gate(judged, ev, tmp_path)
        row = rep.row("plateau")
        assert row.status == "PASS", row.interpretation
        assert "selected at search-space edge (look)" in row.interpretation
        assert rep.diagnostics["plateau"]["n_outside_search_bounds"] == 2
        assert rep.diagnostics["plateau"]["selected_at_search_space_edge"] == ["look"]
        assert "Weakest parameter axis: look keeps 4/4" in row.interpretation


def test_explicit_spec_must_match_evaluator_symbol(good):
    """N7 regression: spec= for another instrument than evaluator.symbol raises."""
    study, trades = good
    ev = FakeEvaluator(study, trades, spec=None)
    ev.symbol = "EURUSD"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xau = load_instrument("XAUUSD", book="FBS")
    with pytest.raises(G.GateError, match="N7"):
        run(study, trades, evaluator=ev, spec=xau)
    assert run(study, trades, evaluator=ev, spec=EURUSD)[0].row("cost_stress_sharpe").status == "PASS"
