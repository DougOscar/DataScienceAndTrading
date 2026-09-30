"""Regression tests for the Phase 1 re-verification round 3 findings (red-team audit
2026-09-24, section "Re-verification round 3 (30b23b6)"): R3-1 … R3-5, R3-7.

Each test replays the scenario of the matching probe (``research/audits/probes/phase1/r3_*.py``)
and fails on 30b23b6.  Synthetic data and hand-written manifests only.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantlab import config, contracts, data, ledger, opt
from quantlab import gates as G
from quantlab import stats as S
from quantlab.evaluators import SyntheticEvaluator
from quantlab.ledger import LedgerError

from conftest import catalog_row, write_m1_parquet                               # noqa: E402
from test_gates import FakeEvaluator, _as_argmax, _mech_ok, make_study, register  # noqa: E402
from test_redteam_r2 import (MapEval, _new, _reg_band, _spike, _study, _unlock,   # noqa: E402,F401
                             clean_tree, manifest)

PPY = 260.0


def _gate(study, ev, ld, trades=None, **kw):
    return G.evaluate_gates(study, ev, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                            mechanism_check=_mech_ok, **kw)


# =========================================================================== R3-1 logged content is recomputed
def test_r3_1_only_evaluate_gates_logs_and_the_ledger_register_unlock_are_private(tmp_path):
    study, trades = make_study(seed=1)
    ld = register(study, tmp_path / "led")
    rep = _gate(study, FakeEvaluator(study, trades), ld, trades)
    assert rep.ledger_row is None and not ledger.study_events(study.study_id, "gates", ledger_dir=ld)
    rep.verdict = "PASS"
    with pytest.raises(TypeError, match="R3-1"):
        G.log_gates(rep, ledger_dir=ld)
    with pytest.raises(LedgerError, match="R3-1"):
        ledger.register_holdout_band(study_id=study.study_id, band={}, reason="x", ledger_dir=ld)
    with pytest.raises(LedgerError, match="R3-1"):
        ledger.record_holdout_unlock(book="FBS", system="toy", study_id=study.study_id, pass_band={},
                                     user_confirmation="x", ledger_dir=ld)
    rep = _gate(study, FakeEvaluator(study, trades), ld, trades, log=True)
    row = rep.ledger_row
    assert row["event"] == "gates" and row["verdict"] == rep.verdict and row["holdout_band"] == rep.holdout_band


def _s5(tmp_path, study, trades, ev):
    ld = register(study, tmp_path / "led")
    rep = G.evaluate_gates(study, ev, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                           mechanism_check=_mech_ok, holdout_symbols="EURUSD", log=True)
    return ld, rep


def _unlock_study(study, ev, ld, trades, **kw):
    kw.setdefault("mechanism_check", _mech_ok)
    return G.unlock_holdout(study, ev, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                            user_confirmation=ledger.unlock_phrase("FBS", "toy"), **kw)


def test_r3_1_doctored_band_is_never_unlocked(manifest, clean_tree, tmp_path):
    """r3_p03 H2f / r3_p01 m5: a band with the right seed / n_boot / symbols but edited thresholds
    was registered and unlocked, and a zero-edge holdout PASSed it.  Now the unlock recomputes the
    band from the verified study and refuses any difference."""
    study, trades = make_study(seed=1)
    ev = FakeEvaluator(study, trades)
    ld, rep = _s5(tmp_path, study, trades, ev)
    assert rep.verdict == "PASS"
    reg = ledger.registered_holdout_band(study.study_id, ledger_dir=ld)
    doctored = {**reg, "sharpe_lo": -9.0, "max_dd_mag_hi": 1.0, "trades_lo": 0.0, "trades_hi": 1e9,
                "p_pass_zero_edge": 0.05}
    ledger.check_band_construction(study.study_id, doctored, ledger_dir=ld)          # construction looks fine …
    ledger.log_event(study.study_id, "holdout_band_registered", ledger_dir=ld, band=doctored, band_version=1,
                     reason="hand-written", horizon_days=doctored["horizon_days"], horizon_end=doctored["horizon_end"])
    with pytest.raises(G.GateError, match="differs from the band recomputed"):
        _unlock_study(study, ev, ld, trades)                                          # … but it is refused
    assert not ledger.holdout_unlocks(ld)


def test_r3_1_unlock_recomputes_the_verdict(manifest, clean_tree, tmp_path):
    """r3_p01 m5: a zero-edge study whose honest verdict is FAIL, logged as PASS, reached the unlock.
    Now the unlock recomputes every gate; a MANUAL mechanism needs a recorded review."""
    z, tz = make_study(seed=10, peak_sr=0.0, slope=0.0, oos_sr=0.0, wfo_sr=0.0, plateau_score=None)
    z = _as_argmax(z)
    ev = FakeEvaluator(z, tz)
    ld, rep = _s5(tmp_path, z, tz, ev)
    assert rep.verdict == "FAIL"
    ledger.log_event(z.study_id, "gates", ledger_dir=ld, verdict="PASS", gate_run=99)   # a forged verdict row
    with pytest.raises(G.GateError, match="not all PASS"):
        _unlock_study(z, ev, ld, tz)
    # a strong study unlocks; without a mechanism callable it needs a passing human review
    study, trades = make_study(seed=1)
    ev = FakeEvaluator(study, trades)
    ld2, rep = _s5(tmp_path / "b", study, trades, ev)
    with pytest.raises(G.GateError, match="mechanism"):
        _unlock_study(study, ev, ld2, trades, mechanism_check=None)
    ledger.record_mechanism_review(study.study_id, passed=True, note="ablation matches card §3", ledger_dir=ld2)
    row = _unlock_study(study, ev, ld2, trades, mechanism_check=None)
    assert row["exam"] == 1 and row["verification"]["verdict_recomputed"] == "PASS"
    assert row["verification"]["band_recomputed_equal"] is True and row["pass_band"] == rep.holdout_band


# =========================================================================== R3-2 OOS series / selection
def test_r3_2_run_study_stores_and_hashes_the_oos_artifacts(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.8, seed=2)
    res = opt.run_study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), book="FBS", system="art",
                        issue=5, attempt=1, study_id="art-a1", n_jobs=1, cv=opt.CPCVConfig(6, 2),
                        wfo=opt.WFOConfig(min_train="3y"), ledger_dir=tmp_path / "ledger", studies_dir=tmp_path / "st")
    sel = ledger.study_events("art-a1", "selection", ledger_dir=tmp_path / "ledger")[-1]
    sha = sel["artifact_sha256"]
    assert set(sha) == set(ledger.STUDY_ARTIFACTS)
    for name, frame in (("cpcv_paths", res.cpcv_paths), ("wfo_oos", res.wfo_oos), ("wfo_params", res.wfo_params),
                        ("trade_counts", res.meta["trade_counts"])):
        assert sha[name] == ledger.frame_sha256(frame) != ledger.EMPTY_FRAME_SHA
        assert ledger.frame_sha256(ledger.load_study_artifact("art-a1", name, tmp_path / "st")) == sha[name]
    G.evaluate_gates(res, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")      # the untouched study gates


def test_r3_2_forged_oos_series_and_selection_raise(tmp_path):
    """r3_p01 m1–m4 / r3_p04: CPCV paths or the WFO series replaced by the IS-best column (or
    recomputed with another scheme), or the selection moved to the IS-best trial, were gated
    without a flag.  Each now raises."""
    z, tz = make_study(seed=10, peak_sr=0.0, slope=0.0, oos_sr=0.0, wfo_sr=0.0, plateau_score=None)
    ld = register(z, tmp_path / "led")
    ev = FakeEvaluator(z, tz)
    best = int(z.trials.sort("m_sharpe", descending=True)["trial_id"][0])
    col = z.returns[f"t{best}"]
    forged_paths = z.cpcv_paths.join(z.returns.select("date", pl.col(f"t{best}").alias("best")), on="date") \
        .with_columns(pl.col("best").alias("ret")).drop("best")
    forged_wfo = z.wfo_oos.join(z.returns.select("date", pl.col(f"t{best}").alias("best")), on="date") \
        .with_columns(pl.col("best").alias("ret")).drop("best")
    assert col.len() > 0
    for forged in (replace(z, cpcv_paths=forged_paths), replace(z, wfo_oos=forged_wfo),
                   replace(z, wfo_oos=z.wfo_oos.head(z.wfo_oos.height // 2))):
        with pytest.raises(G.GateError, match="R3-2"):
            _gate(forged, ev, ld, tz)
    moved = _as_argmax(z)
    if moved.selection["trial_id"] != z.selection["trial_id"]:
        with pytest.raises(G.GateError, match="R3-2"):
            _gate(moved, ev, ld, tz)
    moved_params = replace(z, selected_params={"a": 1, "b": 1})
    with pytest.raises(G.GateError, match="R3-2"):
        _gate(moved_params, ev, ld, tz)
    _gate(z, ev, ld, tz)                                                            # the honest one gates


def test_r3_2_trade_counts_come_from_the_store(tmp_path):
    study, trades = make_study(seed=1)
    tc = study.returns.select("date", *[pl.lit(0.2).alias(c) for c in study.returns.columns if c != "date"])
    study = replace(study, meta={**study.meta, "trade_counts": tc},
                    wfo_oos=study.wfo_oos.with_columns(pl.lit(None, dtype=pl.Int64).alias("n_trades")))
    ld = register(study, tmp_path / "led")
    no_meta = replace(study, meta={k: v for k, v in study.meta.items() if k != "trade_counts"})
    rep = _gate(no_meta, FakeEvaluator(study, trades), ld, trades)
    assert "WFO procedure" in rep.holdout_band["trades_source"]                     # read from the store
    other = replace(study, meta={**study.meta, "trade_counts": tc.with_columns(pl.lit(9.0).alias("t0"))})
    with pytest.raises(G.GateError, match="trade_counts"):
        _gate(other, FakeEvaluator(study, trades), ld, trades)


# =========================================================================== R3-3 plateau floor + categoricals
def _score(ev, space, tmp, sid, sel):
    res = _study(ev, space, tmp, sid)
    t = res.trials
    for k, v in sel.items():
        t = t.filter(pl.col(f"param_{k}") == v)
    judged = replace(res, selected_params=dict(sel), selection={**res.selection, "trial_id": int(t["trial_id"][0])})
    return G.judge_plateau(judged, ev, space=space, radius=0.2, periods_per_year=PPY)["plateau_score"]


@pytest.mark.parametrize("case", ["G1-relative", "G1-step", "G1c", "G1b"])
def test_r3_3_floor_catches_the_author_chosen_scales(tmp_path, case):
    """r3_p02: every reproduced tiny-scale case PASSed 1.0; with the 5 %-of-range floor all FAIL."""
    th = SyntheticEvaluator(bounds={"a": (0.01, 2.0)}, bumps=({"center": {"a": 0.05}, "height": 2.0, "width": 0.02},),
                            base_sharpe=0.5, rho=1.0, seed=3)
    cases = {
        "G1-relative": (MapEval(_spike(0.02), lambda p: 49.9 + p["d"], name="g1r"),
                        opt.FloatParam("d", 0.05, 5.0, 0.05, plateau_scale="relative"), {"d": 0.05}),
        "G1-step": (MapEval(_spike(0.02), lambda p: 49.9 + p["d"], name="g1s"),
                    opt.FloatParam("d", 0.05, 5.0, 0.05, plateau_step=0.01), {"d": 0.05}),
        "G1c": (MapEval(_spike(0.02), lambda p: p["a"], name="g1c"),
                opt.IntParam("a", 1, 101, 1, plateau_step=1), {"a": 50}),
        "G1b": (MapEval(th, lambda p: p["thr"], name="g1b"),
                opt.FloatParam("thr", 0.01, 2.0, 0.01, plateau_scale="relative"), {"thr": 0.05}),
    }
    ev, param, sel = cases[case]
    assert _score(ev, opt.SearchSpace([param]), tmp_path, case.lower(), sel) < 0.6
    assert G.PLATEAU_MIN_RANGE_FRACTION == 0.05


@pytest.mark.parametrize("centre", [10, 20, 50, 120])
def test_r3_3_floor_keeps_honest_broad_lookback_edges(tmp_path, centre):
    """r3_p02 false-positive check: a broad edge in log(lookback) on [5, 200] (Sharpe halves at ×2 /
    ×½ of the optimum), relative 0.20, still PASSes at every optimum with the floor."""
    lb = SyntheticEvaluator(bounds={"a": (math.log(5), math.log(200))},
                            bumps=({"center": {"a": math.log(centre)}, "height": 2.0, "width": 0.16},),
                            base_sharpe=0.0, rho=1.0, seed=3)
    ev = MapEval(lb, lambda p: math.log(p["L"]), name=f"lb{centre}")
    sp = opt.SearchSpace([opt.IntParam("L", 5, 200, 1, plateau_scale="relative")])
    assert _score(ev, sp, tmp_path, f"lb-{centre}", {"L": centre}) >= 0.6


def test_r3_3_perturbation_ladder_is_floored():
    sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 1, plateau_step=1)])
    assert [q["value"] for q in G.plateau_perturbations(sp, {"a": 50}, 0.2)] == [45, 48, 53, 55]  # ±5, ±2.5
    sp = opt.SearchSpace([opt.FloatParam("x", 0.0, 10.0, plateau_step=3.0)])       # declared > floor: unchanged
    assert [q["value"] for q in G.plateau_perturbations(sp, {"x": 5.0}, 0.2)] == [2.0, 3.5, 6.5, 8.0]


def test_r3_3_numeric_categoricals_refused_and_unordered_listed_as_not_judged(tmp_path):
    with pytest.raises(ValueError, match="numeric choices"):
        opt.CategoricalParam("a", (48, 49, 50, 51, 52), ordered=True)               # G4
    with pytest.raises(ValueError, match="numeric choices"):
        opt.CategoricalParam("a", tuple(str(v) for v in range(10, 101, 10)))        # numeric strings
    with pytest.raises(ValueError, match="numeric choices"):
        opt.CategoricalParam("a", ("1.5", "2.5"), ordered=True)
    labels = tuple(f"L{v}" for v in range(10, 101, 10))                            # labels cannot be caught …
    sp = opt.SearchSpace([opt.CategoricalParam("a", labels), opt.IntParam("b", 5, 15, plateau_scale="relative")])
    ev = MapEval(_spike(0.02), lambda p: float(p["a"][1:]), name="g3s")
    res = _study(ev, sp, tmp_path, "g3s", issue=7)
    rep = G.evaluate_gates(res, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    row = rep.row("plateau")
    assert "unordered categorical(s) a: not judged — review at S2" in row.interpretation   # … so they are flagged
    assert rep.diagnostics["plateau"]["unordered_not_judged"] == ["a"]
    # uneven numeric grids use levels=
    p = opt.IntParam("hold", 12, 48, levels=(12, 24, 48), plateau_scale="relative")
    assert p.grid_values() == [12, 24, 48] and opt.space_from_json(opt.SearchSpace([p]).to_json()).params[0] == p


# =========================================================================== R3-4 family
def test_r3_4_family_is_per_book_symmetric_and_issue_is_required(manifest, clean_tree, tmp_path):
    ld = tmp_path / "led"
    with pytest.raises(LedgerError, match="issue"):
        ledger.create_study(ledger_dir=ld, study_id="x-a1", book="FBS", system="x", issue=None, attempt=1,
                            dev_window=["a", "b"], cost_model_version="t", cv_scheme="t")
    with pytest.raises(ValueError, match="issue"):
        opt.run_study(SyntheticEvaluator(bounds={"a": (0, 1)}), opt.SearchSpace([opt.IntParam("a", 0, 1, plateau_step=1)]),
                      book="FBS", system="x", issue=None, attempt=1, ledger_dir=ld, studies_dir=tmp_path / "s")
    b = _new(ld, "fbs-0042-a1", "probe", 42)
    _new(ld, "fbs-0042-c1", "carry_basket", 42)               # carry_basket shares issue 42 …
    _new(ld, "fbs-0200-c2", "carry_basket", 200)              # … and has a study on its own issue 200,
    _new(ld, "fbs-0200-m1", "mr_gold", 200)                   # which mr_gold shares
    _unlock(ld, "fbs-0042-a1", "probe", b)
    G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1", holdout_daily=np.full(b["horizon_days"], -0.001),
                       n_trades=100, periods_per_year=PPY, ledger_dir=ld)
    for sysname, sid in (("carry_basket", "fbs-0200-c2"), ("mr_gold", "fbs-0200-m1"), ("probe", None)):
        assert ledger.holdout_state("FBS", sysname, ld, study_id=sid)["killed"], sysname   # symmetric closure
    # the issue clause is per book: a B3 system on issue 42 keeps its own (independent) holdout
    ledger.create_study(ledger_dir=ld, study_id="b3-0042-a1", book="B3", system="win_probe", issue=42, attempt=1,
                        dev_window=["a", "b"], cost_model_version="t", cv_scheme="t")
    assert not ledger.holdout_state("B3", "win_probe", ld, study_id="b3-0042-a1")["killed"]
    assert ledger.holdout_family("B3", "probe", ld)["book"] == "B3"
    assert not ledger.holdout_state("B3", "probe", ld)["killed"]


# =========================================================================== R3-5 read access + cost
def test_r3_5_load_bars_requires_the_unlocked_systems_name(monkeypatch, tmp_path, clean_tree, install_catalog):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "led")
    rr, t = [], datetime(2025, 4, 1)
    while t < datetime(2026, 5, 16):
        if t.weekday() < 5:
            rr.append((t, 1.1, 1.1001, 1.0999, 1.1, 1, 0, 10))
        t += timedelta(hours=1)
    f = write_m1_parquet(tmp_path / "EURUSD_M1.parquet", rr)
    install_catalog([catalog_row(file=f, symbol="EURUSD", start=datetime(2025, 4, 1), end=datetime(2026, 5, 15, 23),
                                 rows=len(rr))])
    b = _new(None, "fbs-0060-a1", "passed_sys", 60, symbols=("EURUSD",))
    _new(None, "fbs-0060-b1", "sibling", 60, symbols=("EURUSD",))        # same issue, never unlocked
    _unlock(None, "fbs-0060-a1", "passed_sys", b)
    ok = data.load_bars("EURUSD", "M1", end="2025-07-01", include_holdout=True, system="PASSED-SYS")  # normalised
    assert ok.height
    for alias in ("sibling", "unknown_alias"):
        with pytest.raises(contracts.HoldoutLocked):
            data.load_bars("EURUSD", "M1", end="2025-07-01", include_holdout=True, system=alias)
    reads = ledger.holdout_reads()
    assert len(reads) == 1 and reads[0]["system"] == "PASSED-SYS" and reads[0]["unlocked_system"] == "passed_sys"
    assert not any(r.get("event") == "holdout_read" for r in ledger.holdout_unlocks())   # own file
    ledger.verify_chain()


def test_r3_5_appends_do_not_reread_the_file_and_reads_are_cached(tmp_path, monkeypatch):
    path = tmp_path / ledger.READS_FILE
    for i in range(300):
        ledger._append(path, {"event": "holdout_read", "i": i, "pad": "x" * 200})
    calls = []
    real = ledger._raw_lines
    monkeypatch.setattr(ledger, "_raw_lines", lambda p: calls.append(p) or real(p))
    t0 = time.perf_counter()
    for i in range(200):
        ledger._append(path, {"event": "holdout_read", "i": 300 + i})
    assert not calls and time.perf_counter() - t0 < 2.0                 # tail read only
    ledger.verify_chain(tmp_path)
    rows = ledger._read(path)
    assert [r["seq"] for r in rows] == list(range(500))
    n = len(calls)
    ledger._read(path)
    ledger._read(path)
    assert len(calls) == n                                              # cached per (size, mtime)


# =========================================================================== R3-7 strategy source hash
def test_r3_7_strategy_source_change_is_detected(tmp_path):
    mod_path = tmp_path / "my_strategy_mod.py"
    mod_path.write_text("from quantlab.evaluators import SyntheticEvaluator\n\n\n"
                        "class MyEval(SyntheticEvaluator):\n    pass\n")
    spec = importlib.util.spec_from_file_location("my_strategy_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["my_strategy_mod"] = mod
    spec.loader.exec_module(mod)
    ev = mod.MyEval(bounds={"a": (0, 10)}, rho=0.9, seed=1)
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "src-a1", issue=3)
    row = ledger.created_row("src-a1", ledger_dir=tmp_path / "ledger")
    assert row["evaluator_code"]["sha256"] == opt.evaluator_code_fingerprint(ev)["sha256"]
    G.evaluate_gates(res, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    mod_path.write_text(mod_path.read_text() + "\n# drag = 0  (behaviour change describe() does not show)\n")
    with pytest.raises(G.GateError, match="R3-7"):
        G.evaluate_gates(res, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    sys.modules.pop("my_strategy_mod", None)
