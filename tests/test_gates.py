"""End-to-end gate tests on synthetic StudyResults with a fake evaluator (DESIGN §4.2, §10 Phase 1)."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab import gates as G
from quantlab import ledger
from quantlab.contracts import Outcome, StudyResult
from quantlab.costs import CostModel

PPY = 260.0


def _bdays(n: int, start=date(2016, 5, 2)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _sr_to_mu(sr_annual: float, sd: float) -> float:
    return sr_annual / math.sqrt(PPY) * sd


def make_study(*, seed: int = 0, n_days: int = 2340, peak_sr: float = 2.2, slope: float = 0.2,
               oos_sr: float = 1.6, plateau_score: float | None = 0.9, n_trades: int = 1200,
               sel_override: np.ndarray | None = None, extra_cols: dict[str, np.ndarray] | None = None,
               cpcv_paths: bool = True):
    """5×5 grid (a, b); true SR(a,b) = peak − slope·(|a−3| + |b−3|), selected = centre (3,3).
    Trials share a common factor (ρ ≈ 0.5) plus idiosyncratic noise, daily sd 0.6 %."""
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
    # CPCV paths: 6 OOS paths of the selection procedure with true SR = oos_sr
    paths = []
    if cpcv_paths:
        for pid in range(6):
            pr = _sr_to_mu(oos_sr, sd) + sd * rng.normal(0, 1, n_days)
            paths.append(pl.DataFrame({"date": dates, "path_id": [pid] * n_days, "ret": pr}))
    cp = pl.concat(paths) if paths else pl.DataFrame(schema={"date": pl.Date, "path_id": pl.Int64, "ret": pl.Float64})
    sel = {"method": "plateau", "trial_id": sel_id}
    if plateau_score is not None:
        sel["plateau_score"] = plateau_score
    study = StudyResult(
        study_id="fbs-0099-a1", param_names=("a", "b"), trials=pl.DataFrame(trows), returns=returns,
        selected_params={"a": 3, "b": 3}, selection=sel, cpcv_paths=cp,
        wfo_oos=pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64, "refit_id": pl.Int64}),
        wfo_params=pl.DataFrame(), meta={"n_trials": len(cols)})
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

    def __init__(self, study: StudyResult, trades: pl.DataFrame, drag_per_unit: float = 0.00002):
        self.cost = CostModel(version_tag="fbs-test")
        self.study, self.trades, self.drag = study, trades, drag_per_unit
        self.calls: list[CostModel] = []

    def __call__(self, params, *, cost=None):
        c = cost or self.cost
        self.calls.append(c)
        tid = self.study.selection["trial_id"]
        base = self.study.returns[f"t{tid}"].to_numpy()
        units = (c.spread_multiplier - 1.0) * 2 + c.slippage_points + abs(c.swap_multiplier - 1.0)
        daily = self.study.returns.select("date").with_columns(pl.Series("ret", base - self.drag * units))
        return Outcome(daily=daily, trades=self.trades, metrics={})


def _mech_ok(study, evaluator):
    return True, "Filter beats random filter of equal selectivity (p = 0.01)."


def run(study, trades, evaluator="fake", **kw):
    ev = FakeEvaluator(study, trades) if evaluator == "fake" else evaluator
    kw.setdefault("mechanism_check", _mech_ok)
    kw.setdefault("n_boot", 400)
    return G.evaluate_gates(study, ev, periods_per_year=PPY, selected_trades=trades, **kw), ev


@pytest.fixture(scope="module")
def good():
    return make_study(seed=1)


def test_thresholds_match_design_and_are_read_only():
    assert G.GATE_THRESHOLDS["dsr"] == (">=", 0.95)
    assert G.GATE_THRESHOLDS["pbo"] == ("<", 0.30)
    assert G.GATE_THRESHOLDS["oos_sharpe"] == (">=", 1.0)
    assert G.GATE_THRESHOLDS["cost_stress_sharpe"] == (">", 0.5)
    assert G.GATE_THRESHOLDS["plateau"] == (">=", 0.60) and G.PLATEAU_PEAK_FRACTION == 0.5
    assert G.GATE_THRESHOLDS["positive_years"] == (">=", 0.60)
    assert G.GATE_THRESHOLDS["max_year_share"] == ("<=", 0.40)
    with pytest.raises(TypeError):
        G.GATE_THRESHOLDS["dsr"] = (">=", 0.5)  # type: ignore[index]


def test_all_pass(good):
    study, trades = good
    rep, ev = run(study, trades)
    assert rep.verdict == "PASS", rep.to_markdown()
    assert [r.gate for r in rep.rows] == list(G.GATE_ORDER)
    # cost stress used CostModel.stressed(): 1.5× spread, +1 pt slippage, bar_extreme fills
    stressed = [c for c in ev.calls if c.spread_multiplier == 1.5]
    assert len(stressed) == 3 and all(c.slippage_points == 1.0 and c.stop_fill == "bar_extreme" for c in stressed)
    assert sorted(c.swap_multiplier for c in stressed) == [0.5, 1.0, 1.5]
    hb = rep.holdout_band
    assert hb and hb["sharpe_p10"] < hb["sharpe_median"] and hb["trades_lo"] < hb["trades_hi"]
    md = rep.to_markdown()
    assert "| Deflated Sharpe probability |" in md and "Verdict: **PASS**" in md
    assert "Pre-registered holdout pass band" in md and "N_eff used" in md


def test_dsr_fails_when_system_was_searched_heavily():
    # DSR's hurdle scales with the cross-trial Sharpe dispersion: a wider grid (SR 0.1–1.3)
    # searched ~10^4 effective times cannot distinguish its best from the expected maximum.
    study, trades = make_study(seed=1, peak_sr=1.3, slope=0.3)
    base, _ = run(study, trades)
    assert base.row("dsr").status == "PASS"
    rep, _ = run(study, trades, prior_effective_trials=1e4)
    assert rep.row("dsr").status == "FAIL" and rep.verdict == "FAIL"
    assert rep.effective_trials["dsr_raw_n"] <= rep.effective_trials["psr0"]
    others = {r.gate: r.status for r in rep.rows if r.gate != "dsr"}
    assert set(others.values()) == {"PASS"}


def test_pbo_fails_on_regime_flipping_trials():
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
    assert rep.row("pbo").status == "FAIL" and rep.verdict == "FAIL"


def test_oos_sharpe_fails_when_cpcv_paths_are_weak():
    study, trades = make_study(seed=1, oos_sr=0.4)
    rep, _ = run(study, trades)
    assert rep.row("oos_sharpe").status == "FAIL"
    assert rep.verdict == "FAIL"


def test_cost_stress_fails_for_cost_fragile_system(good):
    study, trades = good
    ev = FakeEvaluator(study, trades, drag_per_unit=0.0003)
    rep, _ = run(study, trades, evaluator=ev)
    assert rep.row("cost_stress_sharpe").status == "FAIL" and rep.verdict == "FAIL"
    assert {r.status for r in rep.rows if r.gate != "cost_stress_sharpe"} == {"PASS"}


def test_plateau_fails_on_sharp_optimum(good):
    study, trades = good
    sharp = replace(study, selection={**study.selection, "plateau_score": 0.3})
    rep, _ = run(sharp, trades)
    assert rep.row("plateau").status == "FAIL" and rep.verdict == "FAIL"


def test_plateau_recomputed_when_missing():
    smooth, trades = make_study(seed=1, plateau_score=None, slope=0.2)
    assert G.plateau_score(smooth, PPY)["plateau_score"] == 1.0
    assert G.plateau_score(smooth, PPY)["n_neighbours"] == 8
    spiky, _ = make_study(seed=1, plateau_score=None, slope=1.5)
    assert G.plateau_score(spiky, PPY)["plateau_score"] < 0.6
    rep, _ = run(spiky, trades)
    assert rep.row("plateau").status == "FAIL" and "recomputed" in rep.row("plateau").interpretation


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
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, mechanism_check=_mech_ok, n_boot=300)
    row = rep.row("trade_count")
    assert row.status == "SKIPPED" and "No per-trade data" in row.interpretation
    assert rep.verdict != "PASS"


def test_mechanism_manual_and_failing():
    study, trades = make_study(seed=1)
    rep, _ = run(study, trades, mechanism_check=None)
    assert rep.row("mechanism").status == "MANUAL" and rep.verdict == "INCOMPLETE"
    rep, _ = run(study, trades, mechanism_check=lambda s, e: (False, "Random filter does as well."))
    assert rep.row("mechanism").status == "FAIL" and rep.verdict == "FAIL"


def test_no_evaluator_skips_cost_stress_and_blocks_pass(good):
    study, trades = good
    rep = G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades,
                           mechanism_check=_mech_ok, n_boot=300)
    assert rep.row("cost_stress_sharpe").status == "SKIPPED"
    assert rep.verdict == "INCOMPLETE"


def test_missing_cpcv_paths_skips_oos_gate():
    study, trades = make_study(seed=1, cpcv_paths=False)
    rep, _ = run(study, trades)
    assert rep.row("oos_sharpe").status == "SKIPPED" and rep.verdict != "PASS"
    assert rep.holdout_band is None


def test_zero_edge_system_fails():
    """Phase 1 exit test (fast version): a zero-edge grid, selected as its in-sample best."""
    fails = 0
    for seed in range(3):
        study, trades = make_study(seed=10 + seed, peak_sr=0.0, slope=0.0, oos_sr=0.0)
        best = int(study.trials.sort("m_sharpe", descending=True)["trial_id"][0])
        row = study.trials.filter(pl.col("trial_id") == best)
        study = replace(study, selection={"trial_id": best, "plateau_score": None},
                        selected_params={"a": row["param_a"][0], "b": row["param_b"][0]})
        study.selection.pop("plateau_score")
        rep, _ = run(study, trades)
        fails += rep.verdict == "FAIL"
    assert fails == 3


@pytest.mark.slow
def test_zero_edge_fails_95pct_and_planted_edge_passes():
    """Phase 1 exit test: zero-edge system fails ≥ 95 % of the time; planted edge passes."""
    n = 40
    fails = 0
    for seed in range(n):
        study, trades = make_study(seed=1000 + seed, peak_sr=0.0, slope=0.0, oos_sr=0.0)
        best = int(study.trials.sort("m_sharpe", descending=True)["trial_id"][0])
        row = study.trials.filter(pl.col("trial_id") == best)
        study = replace(study, selection={"trial_id": best},
                        selected_params={"a": row["param_a"][0], "b": row["param_b"][0]})
        rep, _ = run(study, trades, n_boot=200)
        fails += rep.verdict == "FAIL"
    assert fails / n >= 0.95
    passes = sum(run(*make_study(seed=2000 + s), n_boot=200)[0].verdict == "PASS" for s in range(10))
    assert passes >= 8


def test_log_gates_appends_to_ledger(tmp_path, good):
    study, trades = good
    ledger.create_study(ledger_dir=tmp_path, study_id=study.study_id, book="FBS", system="toy", issue=99,
                        attempt=1, dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-test",
                        cv_scheme="CPCV(n=6,k=2)")
    rep, _ = run(study, trades)
    G.log_gates(rep, ledger_dir=tmp_path)
    state = ledger.studies(tmp_path)[study.study_id]
    assert state["events"] == ["study_created", "gates"]
    assert state["verdict"] == "PASS"
    assert state["gates"]["pbo"] == pytest.approx(rep.row("pbo").value)
    assert state["effective_trials"] == pytest.approx(rep.effective_trials["n_eff"])
    assert state["holdout_band"]["sharpe_p10"] == pytest.approx(rep.holdout_band["sharpe_p10"])
    ledger.verify_chain(tmp_path)
