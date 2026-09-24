"""End-to-end gate tests on synthetic StudyResults with a fake evaluator (DESIGN §4.2 v1.2, §10 Phase 1).

Regression tests for the Phase 1 fix round (red-team B1–B4, M1, M3–M5, m1, m3, m4, m9;
calibration F4) are marked in their docstrings; each fails on the a741fa2 code.
"""

from __future__ import annotations

import math
import tempfile
import warnings
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantlab import gates as G
from quantlab import ledger
from quantlab import stats as S
from quantlab.contracts import Outcome, StudyResult
from quantlab.costs import CostModel, load_instrument, pip_points

PPY = 260.0
EMPTY_LEDGER = Path(tempfile.mkdtemp(prefix="ql_gates_empty_ledger_"))
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
        meta={"n_trials": len(cols), "method": "grid", **(meta or {})})
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
    """Returns the selected trial's daily series minus a cost drag that grows with the
    stressed spread / slippage / swap knobs."""

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
        tid = self.study.selection["trial_id"]
        base = self.study.returns[f"t{tid}"].to_numpy()
        units = (c.spread_multiplier - 1.0) * 2 + min(c.slippage_points, 1.0) + abs(c.swap_multiplier - 1.0)
        daily = self.study.returns.select("date").with_columns(pl.Series("ret", base - self.drag * units))
        return Outcome(daily=daily, trades=self.trades, metrics={})


def _mech_ok(study, evaluator):
    return True, "Filter beats random filter of equal selectivity (p = 0.01)."


def run(study, trades, evaluator="fake", **kw):
    ev = FakeEvaluator(study, trades) if evaluator == "fake" else evaluator
    kw.setdefault("mechanism_check", _mech_ok)
    kw.setdefault("n_boot", 400)
    kw.setdefault("ledger_dir", EMPTY_LEDGER)
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
    assert "1 pip slippage on all market and stop fills" in rep.row("cost_stress_sharpe").interpretation
    assert "1 pip = 10 points for EURUSD" in rep.row("cost_stress_sharpe").interpretation
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
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, n_boot=200, ledger_dir=EMPTY_LEDGER)
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
    study2 = replace(study, meta={**study.meta, "candidate_set_data_dependent": False})
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
    assert f"1 pip = {gold:g} points for XAUUSD" in rep.row("cost_stress_sharpe").interpretation
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
                           mechanism_check=_mech_ok, n_boot=300, ledger_dir=EMPTY_LEDGER)
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
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, n_boot=200, ledger_dir=EMPTY_LEDGER)
    assert rep.row("plateau").status == "FAIL"
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
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, n_boot=200, ledger_dir=EMPTY_LEDGER)
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
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, mechanism_check=_mech_ok, n_boot=300,
                           ledger_dir=EMPTY_LEDGER)
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
    assert "Not decisive on its own" in rep.to_markdown()
    strong, tr = make_study(seed=12, wfo_sr=4.0)
    hb = run(strong, tr)[0].holdout_band
    assert hb["p_pass_zero_edge"] < 0.30 and hb["decisive"] is True


# ============================================================================ ledger: M4, m1, m9
def _create(tmp, sid, attempt, system="toy"):
    ledger.create_study(ledger_dir=tmp, study_id=sid, book="FBS", system=system, issue=99, attempt=attempt,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-test",
                        cv_scheme="CPCV(n=6,k=2)")


def test_prior_trials_read_from_ledger_and_raise_when_missing(tmp_path, good):
    """M4 regression: evaluate_gates reads earlier attempts' raw trials itself; attempt 2 with
    no ledger history raises instead of silently deflating by 0."""
    study, trades = good
    a2 = replace(study, study_id="fbs-0099-a2", meta={**study.meta, "book": "FBS", "system": "toy", "attempt": 2})
    _create(tmp_path, "fbs-0099-a2", 2)
    with pytest.raises(ValueError, match="attempt 2"):
        run(a2, trades, ledger_dir=tmp_path)
    _create(tmp_path, "fbs-0099-a1", 1)
    ledger.log_event("fbs-0099-a1", "trials", ledger_dir=tmp_path, n_trials=400)
    _create(tmp_path, "fbs-0099-a3", 3)                      # a later attempt must not count
    ledger.log_event("fbs-0099-a3", "trials", ledger_dir=tmp_path, n_trials=9999)
    rep, _ = run(a2, trades, ledger_dir=tmp_path)
    et = rep.effective_trials
    assert et["n_trials_prior"] == 400 and et["n_trials"] == 425 and "fbs-0099-a1" in et["prior_source"]
    # explicit prior wins; deprecated alias still works with a warning
    assert run(a2, trades, ledger_dir=tmp_path, prior_trials=10)[0].effective_trials["n_trials_prior"] == 10
    with pytest.warns(DeprecationWarning):
        rep, _ = run(a2, trades, ledger_dir=tmp_path, prior_effective_trials=7.0)
    assert rep.effective_trials["n_trials_prior"] == 7.0
    # attempt taken from the ledger row / the -aN suffix when meta lacks it
    bare = replace(study, study_id="fbs-0099-a2", meta={"n_trials": 25, "method": "grid"})
    assert run(bare, trades, ledger_dir=tmp_path)[0].effective_trials["n_trials_prior"] == 400


def test_log_gates_counts_own_trials_and_flags_reruns(tmp_path, good):
    """m1 + m9 regression: the ledger gets the study's own trial count (no cumulative double
    count across attempts) and a gate-run index; a re-run is flagged in the report."""
    study, trades = good
    s1 = replace(study, meta={**study.meta, "book": "FBS", "system": "toy", "attempt": 1})
    _create(tmp_path, s1.study_id, 1)
    rep, _ = run(s1, trades, ledger_dir=tmp_path)
    assert rep.prior_gate_runs == []
    G.log_gates(rep, ledger_dir=tmp_path)
    state = ledger.studies(tmp_path)[s1.study_id]
    assert state["events"] == ["study_created", "gates"] and state["verdict"] == "PASS"
    assert state["gates"]["cscv_oos_loss"] == pytest.approx(rep.row("cscv_oos_loss").value)
    assert state["gate_run"] == 1 and state["n_trials_study"] == 25 and state["n_trials_dsr"] == 25
    assert state["holdout_band"]["sharpe_lo"] == pytest.approx(rep.holdout_band["sharpe_lo"])
    # attempt 2: prior = 25 (a1's own); after logging, attempt 3's prior = 25 + 25, not 25 + 50
    s2 = replace(study, study_id="fbs-0099-a2", meta={**s1.meta, "attempt": 2})
    _create(tmp_path, "fbs-0099-a2", 2)
    r2, _ = run(s2, trades, ledger_dir=tmp_path)
    assert r2.effective_trials["n_trials_prior"] == 25
    G.log_gates(r2, ledger_dir=tmp_path)
    assert ledger.system_prior_trials("FBS", "toy", exclude_study="fbs-0099-a3", ledger_dir=tmp_path)[0] == 50
    # re-run on the same study → flagged
    again, _ = run(s1, trades, ledger_dir=tmp_path)
    assert len(again.prior_gate_runs) == 1 and again.prior_gate_runs[0]["verdict"] == "PASS"
    assert "gated 1 time(s) before" in again.to_markdown()
    G.log_gates(again, ledger_dir=tmp_path)
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
        rep, _ = run(_as_argmax(study), trades, n_boot=200)
        fails += rep.verdict == "FAIL"
    assert fails / n >= 0.95
    passes = sum(run(*make_study(seed=2000 + s), n_boot=200)[0].verdict == "PASS" for s in range(10))
    assert passes >= 8
