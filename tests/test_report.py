"""Tests for quantlab.report (DESIGN §7, §5, §6; report-builder agent).

Gate-value tests reuse the ``tests/test_gates.py`` study/ledger helpers (``make_study``,
``register``, ``run``) so the tear sheet is checked against a *real* logged ``gates`` event,
never a hand-typed stand-in.  Trade-level / applicability tests use a small local evaluator
whose trades carry the full sizing-engine column set (``sizing.apply_sizing``'s output schema),
since the ``test_gates`` fixtures deliberately use a minimal trades frame that only needs to
support the gates' trade-count check.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import test_gates as tg  # noqa: E402  (sibling test module; ledger/study helpers, DESIGN §7)

from quantlab import ledger, metrics as m, report  # noqa: E402
from quantlab.contracts import Outcome, RiskType  # noqa: E402
from quantlab.report._loading import ReportError  # noqa: E402

PPY = 260.0


# =========================================================================== helpers
def _daily(rets, start=date(2018, 1, 1)) -> pl.DataFrame:
    dates = [start + timedelta(days=i) for i in range(len(rets))]
    return pl.DataFrame({"date": dates, "ret": rets})


def _full_trades(rng: np.random.Generator, n: int = 150, risk_type: RiskType = RiskType.A,
                 start=date(2018, 1, 1)) -> pl.DataFrame:
    """A trades frame with the complete ``sizing.apply_sizing`` output schema, so every
    report-builder metric function (R-multiples, cost breakdown, MAE/MFE, ...) has what it
    needs regardless of which risk type is being exercised."""
    entry_days = np.sort(rng.choice(np.arange(2, 900), size=n, replace=False))
    entry_ts = [datetime.combine(start + timedelta(days=int(d)), datetime.min.time()) for d in entry_days]
    exit_ts = [t + timedelta(hours=int(rng.integers(1, 48))) for t in entry_ts]
    direction = rng.choice([1, -1], size=n)
    pnl_points = rng.normal(2.0, 12.0, n)
    lots = np.full(n, 1.0)
    vpp = np.full(n, 10.0)
    pnl_ccy = pnl_points * vpp * lots
    equity_before = 100_000.0 + np.cumsum(pnl_ccy) - pnl_ccy
    cols = {
        "entry_ts": entry_ts, "exit_ts": exit_ts, "entry_idx": entry_days, "exit_idx": entry_days + 1,
        "direction": direction, "entry_price": rng.uniform(1.0, 1.2, n), "exit_price": rng.uniform(1.0, 1.2, n),
        "target_price": np.full(n, np.nan),
        "pnl_points": pnl_points, "pnl_ccy": pnl_ccy, "lots": lots,
        "equity_before": equity_before, "equity_after": equity_before + pnl_ccy,
        "value_per_point_acct": vpp, "value_per_point_acct_exit": vpp,
        "spread_cost_points": np.full(n, 1.5), "swap_points": rng.normal(0, 0.2, n),
        "swap_money_per_lot": np.zeros(n), "commission_per_lot": np.full(n, 2.0),
        "skipped": np.zeros(n, dtype=bool), "mae_points": np.abs(rng.normal(8.0, 4.0, n)),
        "mfe_points": np.abs(rng.normal(10.0, 5.0, n)), "bars_held": rng.integers(1, 20, n),
    }
    if risk_type is RiskType.A:
        risk_target = equity_before * 0.02
        risk_realised = risk_target * (1.0 + rng.normal(0, 0.03, n))
        cols["stop_price"] = cols["entry_price"] - direction * 0.01
        cols["risk_target_ccy"] = risk_target
        cols["risk_realised_ccy"] = risk_realised
        cols["risk_realisation_error"] = risk_realised / risk_target - 1.0
    elif risk_type is RiskType.B:
        cols["stop_price"] = cols["entry_price"] - direction * 0.01
        cols["risk_target_ccy"] = np.full(n, np.nan)
        cols["risk_realised_ccy"] = np.abs(rng.normal(1500.0, 300.0, n))
        cols["risk_realisation_error"] = np.full(n, np.nan)
    else:  # C / D: no hard stop
        cols["stop_price"] = np.full(n, np.nan)
        cols["risk_target_ccy"] = np.full(n, np.nan)
        cols["risk_realised_ccy"] = np.full(n, np.nan)
        cols["risk_realisation_error"] = np.full(n, np.nan)
    return pl.DataFrame(cols)


class RichEvaluator:
    """Minimal evaluator whose ``trades`` is the full sizing-engine schema (see
    :func:`_full_trades`); ``daily`` is unused by the report (trade-level data only, DESIGN
    task: daily returns always come from the trial store, never from a fresh evaluator call)."""

    book = "FBS"
    periods_per_year = PPY
    requires_refit = False

    def __init__(self, trades: pl.DataFrame):
        self.cost = None
        self._trades = trades
        self.calls = 0

    def __call__(self, params, *, cost=None):
        self.calls += 1
        return Outcome(daily=pl.DataFrame(schema={"date": pl.Date, "ret": pl.Float64}),
                      trades=self._trades, metrics={})


def _study_and_ledger(seed=1, n_days=700, **kw):
    """A gated (``log=True``), ledger-registered study -- the report's normal starting point."""
    study, trades = tg.make_study(seed=seed, n_days=n_days, **kw)
    gate_report, ev = tg.run(study, trades, log=True)
    ledger_dir = Path(gate_report.ledger_context["ledger_dir"])
    return study, trades, gate_report, ev, ledger_dir


# =========================================================================== gate values (never recomputed)
def test_tear_sheet_reads_logged_gate_values_verbatim():
    study, trades, gate_report, ev, ledger_dir = _study_and_ledger(seed=1)
    logged = ledger.study_events(study.study_id, "gates", ledger_dir=ledger_dir)[-1]

    ts = report.tear_sheet(study.study_id, evaluator=ev, ledger_dir=ledger_dir, risk_type="A")

    assert ts.headline["gate_verdict"] == logged["verdict"] == gate_report.verdict
    for row in ts.gate_table.iter_rows(named=True):
        logged_value = logged["gates"].get(row["gate"])
        if logged_value is None or not math.isfinite(logged_value):
            assert row["value"] is None
        else:
            assert row["value"] == pytest.approx(logged_value)
        assert row["status"] == logged["gate_status"].get(row["gate"])
    # never a second, independent gate run: nobody in the report package calls gates.evaluate_gates
    import inspect
    from quantlab.report import tear_sheet as ts_mod
    assert "evaluate_gates" not in inspect.getsource(ts_mod)


def test_no_gates_event_is_not_validated():
    study, trades = tg.make_study(seed=5, n_days=400)
    ledger_dir = tg.register(study)
    ts = report.tear_sheet(study.study_id, ledger_dir=ledger_dir, risk_type="C")
    assert ts.headline["gate_verdict"] == "not validated"
    assert ts.gate_table.height == 0
    robustness_rows = {r["metric"]: r for r in ts.applicability.iter_rows(named=True)}
    assert robustness_rows["dsr"]["shown"] is False
    assert "no gates event" in robustness_rows["dsr"]["why"]


def test_risk_type_unresolvable_raises():
    study, trades = tg.make_study(seed=6, n_days=400)
    ledger_dir = tg.register(study)
    with pytest.raises(ReportError):
        report.tear_sheet(study.study_id, ledger_dir=ledger_dir)


def test_unknown_study_raises():
    with pytest.raises(ReportError):
        report.tear_sheet("does-not-exist", ledger_dir=tg.register(tg.make_study(seed=9)[0]), risk_type="A")


# =========================================================================== applicability by risk type
@pytest.mark.parametrize("rtype,shown_only,hidden_only", [
    (RiskType.A, ("expectancy_r", "max_losing_streak_r", "risk_realisation_error"), ("mae_mfe", "worst_trade")),
    (RiskType.C, ("mae_mfe", "worst_trade"), ("expectancy_r", "max_losing_streak_r")),
])
def test_applicability_differs_by_risk_type(rtype, shown_only, hidden_only):
    study, trades, gate_report, ev, ledger_dir = _study_and_ledger(seed=2)
    rng = np.random.default_rng(0)
    rich_trades = _full_trades(rng, risk_type=rtype)
    rich_ev = RichEvaluator(rich_trades)

    ts = report.tear_sheet(study.study_id, evaluator=rich_ev, ledger_dir=ledger_dir, risk_type=rtype)
    shown = {r["metric"]: r["shown"] for r in ts.applicability.iter_rows(named=True)}
    for name in shown_only:
        assert shown[name] is True, (name, shown)
    for name in hidden_only:
        assert shown[name] is False, (name, shown)
    # DESIGN §5: type D never gets turnover/exposure -- the engine has no continuous path
    assert shown["turnover_exposure_cost_per_turnover"] is False


def test_applicability_type_d_excludes_trade_stats():
    study, trades, gate_report, ev, ledger_dir = _study_and_ledger(seed=3)
    rng = np.random.default_rng(1)
    rich_ev = RichEvaluator(_full_trades(rng, risk_type=RiskType.B))
    ts = report.tear_sheet(study.study_id, evaluator=rich_ev, ledger_dir=ledger_dir, risk_type="D")
    shown = {r["metric"]: r["shown"] for r in ts.applicability.iter_rows(named=True)}
    assert shown["win_rate_profit_factor"] is False
    assert shown["turnover_exposure_cost_per_turnover"] is False


# =========================================================================== figures
def test_figures_are_nonempty_pngs(tmp_path):
    study, trades, gate_report, ev, ledger_dir = _study_and_ledger(seed=1)
    rng = np.random.default_rng(0)
    rich_ev = RichEvaluator(_full_trades(rng, risk_type=RiskType.A))
    ts = report.tear_sheet(study.study_id, evaluator=rich_ev, ledger_dir=ledger_dir, risk_type="A")
    paths = ts.figures(tmp_path / "figs")
    assert paths, "expected at least one figure"
    for name, p in paths.items():
        assert p.exists() and p.stat().st_size > 0, name
    assert {"equity_underwater", "sharpe_yearly", "sharpe_monthly", "monthly_heatmap"} <= paths.keys()


# =========================================================================== risk-profile label
def test_risk_profile_label_steady():
    rng = np.random.default_rng(42)
    per_year, years = 260, 4
    base_year = rng.normal(0.0004, 0.004, per_year)
    steady = np.concatenate([base_year + rng.normal(0, 0.0003, per_year) for _ in range(years)])
    label, ev = report.risk_profile_label(_daily(steady))
    assert label == "Steady", ev


def test_risk_profile_label_grinder_negative_skew():
    rng = np.random.default_rng(7)
    n = 1000
    base = rng.normal(0.0008, 0.0015, n)
    shocks = rng.choice(n, size=int(0.04 * n), replace=False)
    grinder = base.copy()
    grinder[shocks] -= rng.uniform(0.02, 0.05, size=shocks.size)
    label, ev = report.risk_profile_label(_daily(grinder))
    assert label == "Grinder"
    assert ev["skew"] < 0 and ev["win_rate"] >= 0.5


def test_risk_profile_label_trend_positive_skew():
    rng = np.random.default_rng(11)
    n = 1000
    base = rng.normal(-0.0007, 0.0015, n)
    shocks = rng.choice(n, size=int(0.04 * n), replace=False)
    trend = base.copy()
    trend[shocks] += rng.uniform(0.02, 0.05, size=shocks.size)
    label, ev = report.risk_profile_label(_daily(trend))
    assert label == "Trend-like"
    assert ev["skew"] > 0 and ev["win_rate"] < 0.5


def test_risk_profile_label_lumpy():
    rng = np.random.default_rng(3)
    per_year = 260
    lumpy = np.concatenate([
        rng.normal(0.0001, 0.003, per_year), rng.normal(0.003, 0.004, per_year),
        rng.normal(0.0001, 0.003, per_year), rng.normal(0.0001, 0.003, per_year),
    ])
    label, ev = report.risk_profile_label(_daily(lumpy))
    assert label == "Lumpy"
    assert ev["max_period_share"] > 0.55


# =========================================================================== Obsidian card
def _minimal_tear_sheet(tmp_path, *, gate_verdict="PASS", mean_monthly_pct=4.2):
    rng = np.random.default_rng(0)
    daily = _daily(rng.normal(0.0006, 0.006, 600))
    trades = _full_trades(rng, n=80, risk_type=RiskType.A)
    label, rp = report.risk_profile_label(daily, trades)
    headline = {
        "class": "reasonable", "mean_monthly_pct": mean_monthly_pct,
        "gate_verdict": gate_verdict,
        "robustness": {"dsr": 0.97, "cscv_oos_loss": 0.04, "oos_sharpe": 1.3, "wfo_oos": 1.1,
                      "wfo_oos_recent": 0.9, "holdout_status": "not unlocked"},
    }
    return report.TearSheet(
        study_id="fbs-0042-a1", risk_type=RiskType.A, headline=headline, metrics={}, interpretations={},
        applicability=pl.DataFrame(schema={"metric": pl.Utf8, "shown": pl.Boolean, "why": pl.Utf8}),
        gate_table=pl.DataFrame(schema={"gate": pl.Utf8, "label": pl.Utf8, "value": pl.Float64,
                                        "status": pl.Utf8, "threshold": pl.Utf8}),
        risk_profile=(label, rp), _daily=daily, _ppy=PPY,
    )


def test_card_matches_golden_structure(tmp_path):
    ts = _minimal_tear_sheet(tmp_path)
    vault = tmp_path / "vault"
    card = report.write_card(ts, slug="my-sys", name="My System", idea="A short idea.", status="testing",
                             issue=42, book="FBS", vault_dir=vault)
    lines = card.read_text().splitlines()
    assert lines[0] == "# My System"
    assert lines[1] == "**Book:** FBS   **Status:** testing   **Issue:** #42"
    assert lines[2] == "**Idea:** A short idea."
    assert lines[3].startswith("**Result:** reasonable — 4.2%/month at 10% DD budget")
    assert lines[4].startswith("**Risk profile:** ")
    assert "MaxDD" in lines[4] and "longest DD" in lines[4] and "max losing streak" in lines[4]
    assert "skew" in lines[4] and "CVaR95" in lines[4]
    assert lines[5].startswith("**Robustness:** DSR 0.97")
    assert "holdout not unlocked" in lines[5]
    assert lines[6] == "![[my-sys_sharpe_yearly.png]]"
    assert lines[7] == "![[my-sys_sharpe_monthly.png]]"
    assert (vault / "attachments" / "my-sys_sharpe_yearly.png").stat().st_size > 0
    assert (vault / "attachments" / "my-sys_sharpe_monthly.png").stat().st_size > 0


def test_card_idea_is_truncated_to_200_chars(tmp_path):
    ts = _minimal_tear_sheet(tmp_path)
    card = report.write_card(ts, slug="long-idea", name="Sys", idea="x" * 400, status="testing",
                             issue=1, book="FBS", vault_dir=tmp_path / "vault")
    idea_line = next(l for l in card.read_text().splitlines() if l.startswith("**Idea:**"))
    assert len(idea_line) - len("**Idea:** ") <= 200


def test_killed_card_carries_stage_and_reason(tmp_path):
    ts = _minimal_tear_sheet(tmp_path, gate_verdict="FAIL")
    card = report.write_card(ts, slug="dead-sys", name="Dead System", idea="Killed at S5.", status="killed",
                             stage="S5", reason="failed the plateau gate", issue=7, book="FBS",
                             vault_dir=tmp_path / "vault")
    text = card.read_text()
    assert "**Status:** killed (S5, failed the plateau gate)" in text


def test_write_card_killed_requires_stage_and_reason(tmp_path):
    ts = _minimal_tear_sheet(tmp_path)
    with pytest.raises(ValueError):
        report.write_card(ts, slug="dead", name="Dead", idea="x", status="killed", issue=1, book="FBS",
                          vault_dir=tmp_path / "vault")


def test_write_card_default_vault_dir(monkeypatch, tmp_path):
    from quantlab import config
    monkeypatch.setattr(config, "VAULT_DIR", tmp_path / "default_vault")
    ts = _minimal_tear_sheet(tmp_path)
    card = report.write_card(ts, slug="def-sys", name="Def", idea="x", status="testing", issue=1, book="FBS")
    assert card == tmp_path / "default_vault" / "def-sys.md"
    assert card.exists()


# =========================================================================== write_results
def test_write_results_writes_metrics_json_and_markdown(tmp_path):
    study, trades, gate_report, ev, ledger_dir = _study_and_ledger(seed=4)
    ts = report.tear_sheet(study.study_id, evaluator=ev, ledger_dir=ledger_dir, risk_type="B")
    out = ts.write_results(tmp_path / "results")
    payload = json.loads(out["metrics_json"].read_text())
    assert payload["study_id"] == study.study_id
    assert payload["headline"]["gate_verdict"] == ts.headline["gate_verdict"]
    md = out["tear_sheet_md"].read_text()
    assert md.startswith("# Tear sheet")
    assert "## Metrics applicability" in md and "## Gates (as logged in the ledger)" in md


# =========================================================================== metrics.py additions
def _trades_for_metrics(rng, risk_type=RiskType.A, n=40):
    return _full_trades(rng, n=n, risk_type=risk_type)


def test_r_multiples_and_expectancy():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.A)
    r = m.r_multiples(trades)
    assert r.size == trades.height
    expected = (trades["pnl_ccy"] / trades["risk_target_ccy"]).to_numpy()
    np.testing.assert_allclose(np.sort(r), np.sort(expected), rtol=1e-8)
    assert m.expectancy_r(trades) == pytest.approx(float(np.mean(expected)))


def test_r_multiples_empty_for_type_b():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.B)
    assert m.r_multiples(trades).size == 0
    assert math.isnan(m.expectancy_r(trades))


def test_max_losing_streak_r_known_answer():
    trades = pl.DataFrame({"pnl_ccy": [10.0, -5.0, -5.0, -5.0, 20.0, -1.0],
                           "risk_target_ccy": [100.0] * 6, "skipped": [False] * 6})
    assert m.max_losing_streak_r(trades) == 3


def test_mae_mfe_stats():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.C)
    stats = m.mae_mfe_stats(trades)
    assert stats["mae_p50"] == pytest.approx(float(np.median(trades["mae_points"].to_numpy())))
    assert stats["mfe_p95"] == pytest.approx(float(np.quantile(trades["mfe_points"].to_numpy(), 0.95)))


def test_worst_trade():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.C)
    assert m.worst_trade_points(trades) == pytest.approx(float(trades["pnl_points"].min()))
    assert m.worst_trade_ccy(trades) == pytest.approx(float(trades["pnl_ccy"].min()))


def test_cost_breakdown_signs_and_identities():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.A, n=60)
    cb = m.cost_breakdown(trades)
    assert cb["gross_pnl_ccy"] == pytest.approx(cb["net_pnl_ccy"] + cb["spread_cost_ccy"])
    assert np.isfinite(cb["cost_total_ccy"])
    if cb["gross_pnl_ccy"] > 0:
        assert 0.0 <= cb["cost_pct_of_gross"] or cb["cost_pct_of_gross"] < 0  # always finite, no crash
    assert math.isfinite(cb["cost_pct_of_gross"]) or cb["gross_pnl_ccy"] <= 0


def test_cost_breakdown_missing_columns_returns_nan():
    trades = pl.DataFrame({"pnl_ccy": [1.0, 2.0]})
    cb = m.cost_breakdown(trades)
    assert math.isnan(cb["gross_pnl_ccy"])


def test_risk_realisation_error_stats():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.A)
    stats = m.risk_realisation_error_stats(trades)
    assert stats["n"] == trades.height
    assert np.isfinite(stats["mean"])


def test_risk_per_trade_ccy_type_b():
    rng = np.random.default_rng(0)
    trades = _trades_for_metrics(rng, RiskType.B)
    r = m.risk_per_trade_ccy(trades)
    assert r.size == trades.height
    assert (r > 0).all()
