"""Regression tests for the Phase 1 re-verification round 2 findings (red-team audit
2026-09-24, section "Re-verification round 2 (48e696f)"): R2-1 … R2-4 and the minors.

Each test replays the scenario of the matching probe (``research/audits/probes/phase1/r2_*.py``)
and fails on 48e696f.  Synthetic data only, except where a manifest-shaped fixture is written;
no real holdout bar is read (the load_bars scope test uses hand-built M1 files).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quantlab import config, contracts, data, ledger, opt
from quantlab import gates as G
from quantlab import stats as S
from quantlab.costs import CostModel
from quantlab.evaluators import SyntheticEvaluator
from quantlab.ledger import LedgerError

from conftest import catalog_row, write_m1_parquet                     # noqa: E402
from test_gates import FakeEvaluator, make_study, register, write_store  # noqa: E402

PPY = 260.0
LOCKED_END = datetime(2026, 5, 15, 10, 36)


# =========================================================================== helpers
@dataclass
class MapEval:
    """The author's parametrisation: params → the underlying economic parameter ``a`` of a
    SyntheticEvaluator (r2_p02's MapEval)."""
    inner: SyntheticEvaluator
    fn: object
    name: str = "map"
    periods_per_year: float = 260.0
    book: str = "FBS"
    requires_refit = False
    cost_version = "synthetic"
    is_synthetic = True

    @property
    def dev_window(self):
        return self.inner.dev_window

    def describe(self):
        return {"evaluator": "MapEval", "map": self.name, **self.inner.describe()}

    def prepare(self):
        pass

    def __call__(self, params, *, cost=None):
        return self.inner({"a": float(self.fn(params))}, cost=cost)


def _spike(width: float, lo: float = 1, hi: float = 101, centre: float = 50) -> SyntheticEvaluator:
    return SyntheticEvaluator(bounds={"a": (lo, hi)}, bumps=({"center": {"a": centre}, "height": 2.0,
                                                              "width": width},), base_sharpe=0.5, rho=1.0, seed=3)


def _study(ev, space, tmp, sid, **kw):
    kw.setdefault("wfo", None)
    kw.setdefault("cv", opt.CPCVConfig(4, 1))
    return opt.run_study(ev, space, book="FBS", system=kw.pop("system", sid), issue=kw.pop("issue", 9999), attempt=1,
                         study_id=sid, n_jobs=1, ledger_dir=tmp / "ledger", studies_dir=tmp / "studies",
                         allow_large_grid=True, **kw)


def _judge_at(res, ev, space, sel):
    t = res.trials
    for k, v in sel.items():
        t = t.filter(pl.col(f"param_{k}") == v)
    judged = replace(res, selected_params=dict(sel), selection={**res.selection, "trial_id": int(t["trial_id"][0])})
    return G.judge_plateau(judged, ev, space=space, radius=0.2, periods_per_year=PPY)


# =========================================================================== R2-1 plateau scale
def test_r2_1a_numeric_params_must_declare_a_plateau_scale():
    with pytest.raises(ValueError, match="exactly one plateau scale"):
        opt.IntParam("a", 1, 10)
    with pytest.raises(ValueError, match="exactly one plateau scale"):
        opt.FloatParam("x", 1.0, 2.0, plateau_scale="relative", plateau_step=0.1)
    with pytest.raises(ValueError, match="strictly positive"):
        opt.FloatParam("thr", 0.0, 2.0, plateau_scale="relative")      # the G1 offset / threshold case
    with pytest.raises(ValueError, match="> 0"):
        opt.FloatParam("thr", 0.0, 2.0, plateau_step=0.0)
    with pytest.raises(ValueError, match="'relative'"):
        opt.FloatParam("x", 1.0, 2.0, plateau_scale="range")
    p = opt.FloatParam("thr", 0.01, 2.0, 0.01, plateau_step=0.4)
    js = opt.SearchSpace([p]).to_json()
    assert js["params"][0]["plateau_step"] == 0.4                     # pre-registered in the space JSON
    legacy = {"params": [{"name": "a", "kind": "int", "low": 1, "high": 10, "step": 1}], "constraint": None}
    with pytest.raises(ValueError, match="plateau scale"):
        opt.space_from_json(legacy)
    assert "S2 red-team reviews" in opt.Param.__doc__


def test_r2_1a_small_positive_param_is_judged_on_its_declared_step(tmp_path):
    """G1b: thr on [0.01, 2] selected at 0.05 was judged at 0.04–0.06 (relative mode by default)
    and PASSed a spike.  Declared with an economic step it FAILs; relative mode is still
    available for it, and then the author's scale is on the record for the S2 red team."""
    th = SyntheticEvaluator(bounds={"a": (0.01, 2.0)}, bumps=({"center": {"a": 0.05}, "height": 2.0, "width": 0.02},),
                            base_sharpe=0.5, rho=1.0, seed=3)
    ev = MapEval(th, lambda p: p["thr"], name="thr")
    sp = opt.SearchSpace([opt.FloatParam("thr", 0.01, 2.0, 0.01, plateau_step=0.2)])
    res = _study(ev, sp, tmp_path, "g1b")
    pj = _judge_at(res, ev, sp, {"thr": 0.05})
    assert pj["plateau_score"] < 0.6
    row = ledger.created_row("g1b", ledger_dir=tmp_path / "ledger")
    assert row["search_space"]["params"][0]["plateau_step"] == 0.2


def test_r2_1b_numeric_unordered_categorical_refused_and_ordered_levels_pm2():
    with pytest.raises(ValueError, match="use an ordered/numeric param"):
        opt.CategoricalParam("a", tuple(range(10, 101, 10)), ordered=False)            # G3
    opt.CategoricalParam("mode", ("fast", "slow"))                                    # non-numeric: fine
    with pytest.raises(ValueError, match="numeric parameters only"):
        opt.Param("m", "categorical", choices=("x", "y"), ordered=True, plateau_step=1.0)
    sp = opt.SearchSpace([opt.CategoricalParam("a", ("vl", "l", "m", "h", "vh"), ordered=True)])
    pts = G.plateau_perturbations(sp, {"a": "m"}, 0.2)
    assert [(q["offset"], q["value"]) for q in pts] == [("level -2", "vl"), ("level -1", "l"), ("level +1", "h"),
                                                         ("level +2", "vh")]
    edge = G.plateau_perturbations(sp, {"a": "vh"}, 0.2)              # 2 real levels + 2 missing (listed)
    assert len(edge) == 4 and [q.get("missing_level", False) for q in edge] == [False, False, True, True]


def test_r2_1b_ordered_categorical_judged_on_two_levels_each_side(tmp_path):
    """G4-type edge case (red-team minor): at an edge an ordered categorical had ONE point (share
    0 or 1).  Now ±1 and ±2 levels are used: at the edge both inner levels must pass, and a
    spike that survives only the adjacent level FAILs; missing levels are listed, not scored."""
    lv = {"low": 30, "mid": 40, "high": 50}                            # labels (numeric levels are refused, R3-3)
    sp = opt.SearchSpace([opt.CategoricalParam("a", tuple(lv), ordered=True)])
    ev = MapEval(_spike(0.30), lambda p: lv[p["a"]], name="ord-broad")
    pj = _judge_at(_study(ev, sp, tmp_path, "ord-broad"), ev, sp, {"a": "high"})
    assert [q["status"] for q in pj["points"]].count("missing_level") == 2 and pj["n_missing_levels"] == 2
    assert pj["pass_count_by_param"]["a"] == (2, 2) and pj["plateau_score"] == 1.0
    narrow = MapEval(_spike(0.08), lambda p: lv[p["a"]], name="ord-narrow")   # mid passes, low does not
    pj = _judge_at(_study(narrow, sp, tmp_path, "ord-narrow"), narrow, sp, {"a": "high"})
    assert pj["pass_count_by_param"]["a"] == (1, 2) and pj["plateau_score"] == 0.5


@pytest.mark.parametrize("k", [4, 5])
def test_r2_1c_mean_of_duplicates_fails_with_the_joint_axis(tmp_path, k):
    """G2: a = mean(a1..ak).  Moving one duplicate at a time moves a by r·a/k, so every axis
    passed (score 1.0 at k = 4, 5).  The joint axis moves all of them: FAIL."""
    ev = MapEval(_spike(0.02), lambda p, k=k: np.mean([p[f"a{i}"] for i in range(k)]), name=f"mean{k}")
    sp = opt.SearchSpace([opt.IntParam(f"a{i}", 48, 52, 2, plateau_scale="relative") for i in range(k)])
    res = _study(ev, sp, tmp_path, f"g2-{k}", method="sobol", n_trials=16, seed=0)
    sel = {f"a{i}": 50 for i in range(k)}
    judged = replace(res, selected_params=sel, selection={**res.selection, "trial_id": 0})
    pj = G.judge_plateau(judged, ev, space=sp, radius=0.2, periods_per_year=PPY)
    per_axis = {n: s for n, s in pj["pass_share_by_param"].items() if n != "joint"}
    assert set(per_axis.values()) == {1.0}                            # the old min-over-axes score
    assert pj["n_joint_points"] == 8 and pj["pass_share_by_param"]["joint"] <= 0.5
    assert pj["plateau_score"] < 0.6 and pj["weakest_axis"] == "joint"


def test_r2_1c_p10_spike_fails_and_broad_plateau_passes_with_the_joint_axis(tmp_path):
    """The red-team p10 surface (2 %-wide spike) with a harmless second param FAILs; a broad
    bump PASSes on every axis including the joint one."""
    sp = opt.SearchSpace([opt.IntParam("a", 0, 100, 5, plateau_step=20), opt.IntParam("b", 5, 15, plateau_scale="relative")])
    got = {}
    for width in (0.02, 0.30):
        ev = MapEval(_spike(width, 0, 100), lambda p: p["a"], name=f"p10-{width}")
        res = _study(ev, sp, tmp_path, f"p10-{width}", method="sobol", n_trials=16, seed=0)
        judged = replace(res, selected_params={"a": 50, "b": 10}, selection={**res.selection, "trial_id": 0})
        pj = G.judge_plateau(judged, ev, space=sp, radius=0.2, periods_per_year=PPY)
        assert pj["n_axes"] == 3 and pj["n_joint_points"] == 8
        got[width] = (pj["plateau_score"], pj["pass_share_by_param"]["joint"])
    assert got[0.02][0] < 0.6 and got[0.02][1] == 0.0
    assert got[0.30] == (1.0, 1.0)


# =========================================================================== R2-2 holdout identity / access
@pytest.fixture
def clean_tree(monkeypatch):
    monkeypatch.setattr(ledger, "git_commit", lambda: "abc1234")


@pytest.fixture
def manifest(monkeypatch, tmp_path):
    """Hand-written manifest (metadata only; files never opened).  Returns a setter {symbol: end}."""
    path = tmp_path / "manifest.json"

    def _set(ends: dict[str, datetime]):
        entries = {}
        for sym, end in ends.items():
            market = "crypto" if sym.startswith("BTC") else "forex"
            rel = f"data/{market}/{sym}_M1_DOES_NOT_EXIST.parquet"
            entries[rel] = {"file": rel, "market": market, "broker": "x", "symbol": sym, "timeframe": "M1",
                            "kind": "bars", "rows": 1, "start": "2016-05-02 00:00:00", "end": str(end)}
        path.write_text(json.dumps(entries))
        monkeypatch.setattr(config, "MANIFEST_PATH", path)
        return path

    _set({"EURUSD": LOCKED_END, "GBPUSD": LOCKED_END})
    return _set


class _WfoStudy:
    def __init__(self, sr: float, n: int = 1500, seed: int = 3, sd: float = 0.006):
        z = np.random.default_rng(seed).standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        self.mu = sr / math.sqrt(260) * sd
        self.wfo_oos = pl.DataFrame({"date": [date(2018, 1, 1) + timedelta(days=i) for i in range(n)],
                                     "ret": self.mu + sd * z})
        self.cpcv_paths = None


def _reg_band(sid: str, sr: float = 2.8, symbols=("EURUSD",)) -> dict:
    return S.holdout_band(_WfoStudy(sr), data.holdout_horizon("FBS", list(symbols)), 260.0,
                          n_boot=S.HOLDOUT_BAND_N_BOOT, n_power=S.HOLDOUT_BAND_N_POWER, trades_per_day=0.4,
                          seed=S.holdout_band_seed(sid)).as_dict()


def _new(ld, sid, system, issue, *, sr=2.8, symbols=None):
    extra = {"symbols": list(symbols), "conversion_legs": []} if symbols else {}
    ledger.create_study(ledger_dir=ld, study_id=sid, book="FBS", system=system, issue=issue, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t", **extra)
    b = _reg_band(sid, sr, symbols or ("EURUSD",))
    ledger._register_holdout_band(study_id=sid, band=b, reason="S5", ledger_dir=ld)
    return b


def _unlock(ld, sid, system, band):
    return ledger._record_holdout_unlock(book="FBS", system=system, study_id=sid, pass_band=band,
                                        user_confirmation=ledger.unlock_phrase("FBS", system), ledger_dir=ld)


def test_r2_2a_fail_kills_the_renamed_system_and_the_issue(manifest, clean_tree, tmp_path):
    """H3: after 'probe' (issue 42) FAILs, 'Probe' (same normalised name) and 'probe_v2' (same
    issue) no longer get a fresh exam-1 unlock; an unrelated system still can."""
    ld = tmp_path / "led"
    b = _new(ld, "fbs-0042-a1", "probe", 42)
    _unlock(ld, "fbs-0042-a1", "probe", b)
    r = G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1",
                           holdout_daily=np.full(b["horizon_days"], -0.001), n_trades=100, periods_per_year=PPY,
                           ledger_dir=ld)
    assert r["status"] == "FAIL"
    for sid, name, iss in (("fbs-0042-a2", "Probe", 42), ("fbs-0043-a1", "probe_v2", 42), ("fbs-0077-a1", "PROBE", 77)):
        with pytest.raises(LedgerError, match="killed"):
            _new(ld, sid, name, iss)                                   # no band either
        st = ledger.holdout_state("FBS", name, ld, study_id=sid)
        assert st["killed"] and "probe" in st["systems"]
    b2 = _new(ld, "fbs-0050-a1", "other", 50)
    assert _unlock(ld, "fbs-0050-a1", "other", b2)["exam"] == 1


def test_r2_2a_pending_member_blocks_the_family(manifest, clean_tree, tmp_path):
    ld = tmp_path / "led"
    b = _new(ld, "fbs-0042-a1", "probe", 42)
    b2 = _new(ld, "fbs-0042-a2", "probe_v2", 42)
    _unlock(ld, "fbs-0042-a1", "probe", b)
    with pytest.raises(LedgerError, match="pending"):
        _unlock(ld, "fbs-0042-a2", "probe_v2", b2)
    with pytest.raises(LedgerError, match="study fbs-0042-a1 belongs to"):
        _unlock(ld, "fbs-0042-a1", "Probe", b)                        # the unlock names the study's own system


def test_r2_2b_load_bars_scoped_to_registered_symbols_and_logged(monkeypatch, tmp_path, clean_tree, install_catalog):
    """H5: an unlock opened ANY symbol for its system.  Now only the study's registered symbols
    (traded + conversion legs) are served, and every holdout read is logged."""
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path / "led")
    rows = []
    for sym in ("EURUSD", "GBPUSD"):
        rr, t = [], datetime(2025, 4, 1)
        while t < datetime(2026, 5, 16):
            if t.weekday() < 5:
                rr.append((t, 1.1, 1.1001, 1.0999, 1.1, 1, 0, 10))
            t += timedelta(hours=1)
        f = write_m1_parquet(tmp_path / f"{sym}_M1.parquet", rr)
        rows.append(catalog_row(file=f, symbol=sym, start=datetime(2025, 4, 1), end=datetime(2026, 5, 15, 23),
                                rows=len(rr)))
    install_catalog(rows)
    b = _new(None, "fbs-0060-a1", "scoped", 60, symbols=("EURUSD",))
    row = _unlock(None, "fbs-0060-a1", "scoped", b)
    assert row["symbols"] == ["EURUSD"] and row["conversion_legs"] == []
    bars = data.load_bars("EURUSD", "M1", end="2025-07-01", include_holdout=True, system="scoped")
    assert bars["ts"].max() >= datetime(2025, 6, 30)
    with pytest.raises(contracts.HoldoutLocked, match="covers only"):
        data.load_bars("GBPUSD", "M1", end="2025-07-01", include_holdout=True, system="scoped")
    with pytest.raises(contracts.HoldoutLocked):
        data.load_bars("GBPUSD", "M1", end="2025-07-01", include_holdout=True, system="never_unlocked")
    reads = ledger.holdout_reads()
    assert len(reads) == 1 and reads[0]["symbol"] == "EURUSD" and reads[0]["system"] == "scoped"
    assert reads[0]["study_id"] == "fbs-0060-a1" and reads[0]["end"].startswith("2025-07-01")
    assert reads[0]["start"].startswith("2025-05-15")
    ledger.verify_chain()


class _SymEval:
    """A non-synthetic evaluator stand-in that trades ``symbol`` (returns from a SyntheticEvaluator)."""
    book, periods_per_year, requires_refit, cost_version = "FBS", 260.0, False, "fbs-test"

    def __init__(self, symbol: str):
        self.symbol, self._ev = symbol, SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.9)
        self.dev_window = self._ev.dev_window

    def describe(self):
        return {"evaluator": "_SymEval", "symbol": self.symbol}

    def __call__(self, params, *, cost=None):
        return self._ev(params)


def test_r2_2_symbols_and_legs_recorded_at_study_creation(tmp_path):
    """run_study records a real evaluator's traded symbol + conversion legs in study_created;
    they scope holdout access and feed the horizon (synthetic evaluators record none)."""
    sp = opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)])
    _study(_SymEval("EURJPY"), sp, tmp_path, "legs")
    row = ledger.created_row("legs", ledger_dir=tmp_path / "ledger")
    assert row["symbols"] == ["EURJPY"] and "USDJPY" in row["conversion_legs"]
    assert row["studies_dir"] == str((tmp_path / "studies").resolve())
    assert ledger.study_symbols("legs", ledger_dir=tmp_path / "ledger") == (["EURJPY"], row["conversion_legs"])
    _study(SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.9), sp, tmp_path, "nosym")
    assert "symbols" not in ledger.created_row("nosym", ledger_dir=tmp_path / "ledger")


# =========================================================================== R2-3 band freezing
def test_r2_3_band_is_frozen_at_the_first_gates_run_and_construction_is_fixed(manifest, clean_tree, tmp_path):
    """H2: seven gate runs with seeds 1–7 and n_boot down to 60 re-rolled the band, and the last
    one was unlockable.  Now seed / n_boot are read-only, the first run's band is the registered
    one, later runs are diagnostics, and a band built otherwise is refused."""
    study, trades = make_study(seed=4)
    ld = register(study, tmp_path / "led")
    for bad in ({"seed": 1}, {"n_boot": 60}):
        with pytest.raises(G.GateError, match="R2-3"):
            G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                             holdout_symbols="EURUSD", **bad)
    r1 = G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                          holdout_symbols="EURUSD")
    assert r1.holdout_band["seed"] == S.holdout_band_seed(study.study_id)
    assert r1.holdout_band["n_boot_requested"] == S.HOLDOUT_BAND_N_BOOT
    assert r1.ledger_row is None                                  # not logged unless log=True
    r1 = G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                          holdout_symbols="EURUSD", log=True)
    reg = ledger.registered_holdout_band(study.study_id, ledger_dir=ld)
    assert reg == r1.holdout_band
    # a later gate run never replaces it: its band is logged as a diagnostic
    G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                     holdout_symbols="EURUSD", log=True)
    ev2 = ledger.study_events(study.study_id, "gates", ledger_dir=ld)[-1]
    assert "holdout_band" not in ev2 and ev2["holdout_band_diagnostic"]["sharpe_lo"] == reg["sharpe_lo"]
    assert ledger.registered_holdout_band(study.study_id, ledger_dir=ld) == reg
    # a thinned / re-seeded band can neither be registered nor unlocked
    for forged in ({**reg, "seed": 7}, {**reg, "n_boot_requested": 60}, {**reg, "n_power_requested": 50}):
        with pytest.raises(LedgerError, match="fixed construction"):
            ledger.check_band_construction(study.study_id, forged, ledger_dir=ld)
    # holdout_symbols may not differ from the study's registered symbols once they exist
    with pytest.raises(G.GateError, match="registered in the ledger"):
        G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                         holdout_symbols="GBPUSD")


def test_r2_3_rebuild_refuses_overrides_and_takes_symbols_from_the_ledger(manifest, clean_tree, tmp_path):
    study, trades = make_study(seed=4)
    ld = register(study, tmp_path / "led")
    G.evaluate_gates(study, None, periods_per_year=PPY, selected_trades=trades, ledger_dir=ld,
                     holdout_symbols="EURUSD", log=True)
    manifest({"EURUSD": datetime(2026, 9, 30, 23, 59), "GBPUSD": datetime(2027, 3, 1, 23, 59)})
    for kw in ({"seed": 999}, {"n_boot": 40}, {"symbols": ["GBPUSD"]}):
        with pytest.raises(TypeError, match="refused"):
            G.rebuild_holdout_band(study, periods_per_year=PPY, reason="export", ledger_dir=ld, **kw)
    nb = G.rebuild_holdout_band(study, periods_per_year=PPY, reason="export", ledger_dir=ld)
    assert nb["horizon_end"] == "2026-09-30 23:59:00" and nb["symbols"] == ["EURUSD"]   # not GBPUSD's 2027
    assert nb["seed"] == S.holdout_band_seed(study.study_id) and nb["n_boot_requested"] == S.HOLDOUT_BAND_N_BOOT


# =========================================================================== R2-4 in-memory study trust
def test_r2_4a_truncated_study_raises_and_n_never_below_the_ledger(tmp_path):
    study, trades = make_study(seed=1)
    ld = register(study, tmp_path / "led")
    keep = [f"t{i}" for i in range(10)]
    trunc = replace(study, trials=study.trials.filter(pl.col("trial_id") < 10),
                    returns=study.returns.select(["date", *keep]))
    trunc = replace(trunc, selection={"method": "plateau", "trial_id": 0}, selected_params={"a": 1, "b": 1})
    with pytest.raises(G.GateError, match="does not match the trial store"):
        G.evaluate_gates(trunc, FakeEvaluator(study, trades), periods_per_year=PPY, selected_trades=trades,
                         ledger_dir=ld)
    edited = replace(study, returns=study.returns.with_columns(pl.col("t12") * 2.0))
    with pytest.raises(G.GateError, match="content check"):
        G.evaluate_gates(edited, FakeEvaluator(study, trades), periods_per_year=PPY, ledger_dir=ld)
    # the optimizer logged more trials than the store holds (e.g. a partially lost store): N uses the max
    ledger.log_event(study.study_id, "trials", ledger_dir=ld, n_trials=441, n_invalid=0)
    rep = G.evaluate_gates(study, FakeEvaluator(study, trades), periods_per_year=PPY, selected_trades=trades,
                           ledger_dir=ld, log=True)
    et = rep.effective_trials
    assert et["n_trials_study"] == 441 and et["n_trials_in_memory"] == 25 and et["n_trials_store"] == 25
    state = ledger.studies(ld)[study.study_id]
    assert state["n_trials_study"] == 441 and ledger.study_trial_count(state) == 441
    # b) a gates event can never lower the optimizer's count seen by later attempts
    assert ledger.study_trial_count({"n_trials_study": 10, "n_trials": 441}) == 441


def test_r2_4c_tpe_study_relabelled_as_a_clean_study_fails(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)}, rho=0.8, seed=1)
    sp = opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2), opt.IntParam("b", 0, 10, plateau_step=2)])
    tpe = _study(ev, sp, tmp_path, "tpe-a1", method="tpe", n_trials=20, seed=0, system="relabel")
    clean = _study(ev, sp, tmp_path, "clean-a1", method="sobol", n_trials=20, seed=0, system="relabel")
    forged = replace(tpe, study_id="clean-a1", trials=tpe.trials.drop("source"),
                     meta={k: v for k, v in tpe.meta.items()
                           if k not in ("candidate_set", "candidate_set_data_dependent", "method", "seed", "study_id")})
    with pytest.raises(G.GateError, match="R2-4"):
        G.evaluate_gates(forged, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    rep = G.evaluate_gates(clean, ev, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    assert rep.ledger_context["trial_store"].endswith("studies")


def test_r2_4d_evaluator_and_cost_model_must_be_the_studys(tmp_path):
    """r2_p03: a broad, zero-drag evaluator swapped in for the study's spiky one turned the
    plateau FAIL into PASS.  Now the ledger's evaluator description / cost model are compared."""
    sp = opt.SearchSpace([opt.IntParam("a", 1, 101, 5, plateau_scale="relative")])
    spiky = _spike(0.02, centre=51)
    broad = _spike(0.5, centre=51)
    res = _study(spiky, sp, tmp_path, "sw-a1")
    with pytest.raises(G.GateError, match="not the study's"):
        G.evaluate_gates(res, broad, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    with pytest.raises(G.GateError, match="cost model version"):
        G.evaluate_gates(res, spiky, periods_per_year=PPY, ledger_dir=tmp_path / "ledger",
                         base_cost=CostModel(version_tag="other"))
    rep = G.evaluate_gates(res, spiky, periods_per_year=PPY, ledger_dir=tmp_path / "ledger")
    assert rep.row("plateau").status == "FAIL"


# =========================================================================== minors
def test_minor_prior_trials_count_later_created_attempts(tmp_path):
    """N2b: re-gating attempt 1 after attempts 2–5 exist counts them (was 0)."""
    study, trades = make_study(seed=1)
    ld = register(study, tmp_path / "led")
    for i, n in ((2, 49), (3, 49), (4, 49)):
        ledger.create_study(ledger_dir=ld, study_id=f"fbs-0099-a{i}x", book="FBS", system="toy", issue=99, attempt=i,
                            dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-test", cv_scheme="t")
        ledger.log_event(f"fbs-0099-a{i}x", "trials", ledger_dir=ld, n_trials=n)
    rep = G.evaluate_gates(study, FakeEvaluator(study, trades), periods_per_year=PPY, selected_trades=trades,
                           ledger_dir=ld)
    assert rep.effective_trials["n_trials_prior"] == 147


def test_minor_horizon_includes_conversion_legs(manifest):
    """H7: EURJPY alone ended 2027-05-14; its USDJPY conversion leg ends 2026-05-15."""
    manifest({"EURJPY": datetime(2027, 5, 14, 23, 59), "USDJPY": LOCKED_END, "EURUSD": datetime(2027, 5, 14, 23, 59)})
    h = data.holdout_horizon("FBS", "EURJPY")
    assert h["conversion_legs"] == ["USDJPY"] and h["horizon_end"] == str(LOCKED_END)
    assert data.holdout_horizon("FBS", "EURJPY", include_legs=False)["horizon_end"] == "2027-05-14 23:59:00"
    assert data.conversion_legs("EURUSD", book="FBS") == []


def test_minor_manifest_rechecked_at_unlock(manifest, clean_tree, tmp_path):
    """H6: a manifest whose end moved under the band (e.g. edited earlier) is refused at unlock;
    a manifest change elsewhere is accepted and recorded."""
    ld = tmp_path / "led"
    b = _new(ld, "fbs-0070-a1", "mani", 70)
    manifest({"EURUSD": datetime(2026, 5, 1), "GBPUSD": LOCKED_END})           # end edited earlier
    with pytest.raises(LedgerError, match="no longer supports"):
        _unlock(ld, "fbs-0070-a1", "mani", b)
    manifest({"EURUSD": LOCKED_END, "GBPUSD": LOCKED_END, "BTCUSD": datetime(2026, 5, 14)})   # unrelated change
    row = _unlock(ld, "fbs-0070-a1", "mani", b)
    assert row["manifest_changed"] is True and row["manifest_sha_now"] != row["manifest_sha_band"]


def test_minor_span_tolerance_tightened():
    band = S.holdout_band(_WfoStudy(2.8), 262, 260.0, n_boot=300, n_power=0, trades_per_day=0.4, seed=1)
    ok = np.full(262, 0.001)
    S.holdout_check(band, ok[:257], 100)                                        # −5 days: accepted
    with pytest.raises(ValueError, match="built for"):
        S.holdout_check(band, ok[:256], 100)                                    # −6 days (was ±10)
    band2 = S.holdout_band(_WfoStudy(2.8), 652, 260.0, n_boot=300, n_power=0, trades_per_day=0.4, seed=1)
    long = np.full(652, 0.001)
    S.holdout_check(band2, long[:639], 250)                                     # 2 % of 652 = 13 days
    with pytest.raises(ValueError, match="built for"):
        S.holdout_check(band2, long[:638], 250)                                 # was accepted up to 32


def test_minor_unlock_without_exam_reported_overdue(manifest, clean_tree, tmp_path, monkeypatch):
    """H4: an unlock never followed by a recorded exam is reported (not killed, report only)."""
    ld = tmp_path / "led"
    b = _new(ld, "fbs-0080-a1", "late", 80)
    monkeypatch.setattr(ledger, "_now", lambda: "2026-01-01T00:00:00Z")
    _unlock(ld, "fbs-0080-a1", "late", b)
    st = ledger.holdout_state("FBS", "late", ld)
    assert st["pending"] and st["exam_overdue"] is True and st["pending_days"] > ledger.HOLDOUT_EXAM_OVERDUE_DAYS
    assert not st["killed"]
