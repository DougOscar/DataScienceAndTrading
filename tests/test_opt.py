"""Tests for quantlab.opt: logging of every trial, determinism, CPCV, plateau, walk-forward."""

from __future__ import annotations

import math
import os
import sys
import time
import warnings
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from quantlab import ledger, opt
from quantlab.contracts import Outcome
from quantlab.evaluators import SyntheticEvaluator

sys.path.insert(0, os.path.dirname(__file__))


def _study(ev, space, tmp_path, sid, **kw):
    kw.setdefault("n_jobs", 1)
    return opt.run_study(ev, space, book="FBS", system="synthetic", issue=1, attempt=1, study_id=sid,
                         ledger_dir=tmp_path / "ledger", studies_dir=tmp_path / "studies", **kw)


def _space2(constraint=None):
    return opt.SearchSpace([opt.IntParam("a", 0, 10), opt.IntParam("b", 0, 10)], constraint=constraint)


def _no_five(p):
    return p["a"] != 5


PLATEAU_EV = SyntheticEvaluator(
    bounds={"a": (0, 10), "b": (0, 10)},
    bumps=({"center": {"a": 3, "b": 3}, "height": 1.5, "width": 0.2, "kind": "box"},     # a,b in 1..5
           {"center": {"a": 8, "b": 8}, "height": 3.0, "width": 0.01, "kind": "box"}),   # single point
    rho=1.0, seed=3)


# --------------------------------------------------------------------------- search space
def test_search_space_grid_and_units():
    sp = opt.SearchSpace([opt.IntParam("n", 5, 20, 5), opt.FloatParam("x", 0.5, 1.5, 0.25),
                          opt.CategoricalParam("mode", ["a", "b"])])
    g = sp.grid()
    assert len(g) == sp.grid_size() == 4 * 5 * 2
    assert g[0] == {"n": 5, "x": 0.5, "mode": "a"}
    assert sorted({p["x"] for p in g}) == [0.5, 0.75, 1.0, 1.25, 1.5]
    U = sp.unit_coords([{"n": 20, "x": 1.0, "mode": "b"}])
    np.testing.assert_allclose(U, [[1.0, 0.5, 1.0]])
    assert sp.is_discrete
    assert not opt.SearchSpace([opt.FloatParam("x", 0, 1)]).is_discrete
    with pytest.raises(ValueError):
        opt.SearchSpace([opt.FloatParam("x", 0, 1)]).grid()
    js = sp.to_json()
    assert js["params"][0] == {"name": "n", "kind": "int", "low": 5, "high": 20, "step": 5}


# --------------------------------------------------------------------------- logging
def test_every_trial_logged_including_invalid_and_error(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)}, error_when=({"a": 7},), rho=0.9)
    res = _study(ev, _space2(_no_five), tmp_path, "fbs-0001-a1", wfo=opt.WFOConfig())
    st = res.trials["status"].value_counts().sort("status")
    counts = dict(zip(st["status"], st["count"]))
    assert counts == {"ok": 99, "invalid": 11, "error": 11}
    assert res.meta["n_trials"] == 121 and res.meta["n_invalid"] == 11 and res.meta["n_error"] == 11
    # ledger: study created, trial count = raw count incl. invalid/error, selection logged
    s = ledger.studies(tmp_path / "ledger")["fbs-0001-a1"]
    assert s["n_trials"] == 121 and s["status"] == "complete"
    assert s["events"] == ["study_created", "trials", "selection"]
    assert s["selected_params"] == res.selected_params
    assert ledger.system_trial_count("FBS", "synthetic", tmp_path / "ledger") == 121
    ledger.verify_chain(tmp_path / "ledger")
    # TrialRecorder files
    d = tmp_path / "studies" / "fbs-0001-a1"
    assert list(d.glob("trials-*.parquet")) and list(d.glob("returns-*.parquet"))
    assert list(d.glob("opt_aux-errors-*.parquet"))
    lt = ledger.load_trials("fbs-0001-a1", tmp_path / "studies")
    assert lt.height == 121 and lt["trial_id"].to_list() == list(range(121))
    assert set(lt["status"]) == {"ok", "invalid", "error"}
    wide = ledger.load_trial_returns("fbs-0001-a1", tmp_path / "studies")
    assert wide.width - 1 == 99                                   # only evaluated trials carry returns
    assert res.returns.width - 1 == 99 and res.returns.height == ev.n_days
    assert all(e["error"].startswith("RuntimeError") for e in res.meta["errors"])
    # invalid / error never selected
    assert res.selected_params["a"] not in (5, 7)


def test_low_trades_status_and_never_selected(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, trades_per_year=5,
                            bumps=({"center": {"a": 5}, "height": 2.0, "width": 0.2},), rho=0.9)
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10)]), tmp_path, "s-low", min_trades=1000, wfo=None)
    assert set(res.trials["status"]) == {"low_trades"}
    assert res.trials["m_objective"].is_infinite().all()
    assert res.selected_params == {} and res.selection["status"] == "no_eligible"
    s = ledger.studies(tmp_path / "ledger")["s-low"]
    assert s["n_trials"] == 11 and s["n_low_trades"] == 11


class _Abort(BaseException):
    pass


class _BoomEvaluator:
    book, periods_per_year, requires_refit, cost_version = "FBS", 260.0, False, "x"
    dev_window = ("2016-05-02", "2025-05-14")

    def __init__(self):
        self.inner = SyntheticEvaluator(bounds={"a": (0, 10)})
        self.n = 0

    def __call__(self, params, *, cost=None) -> Outcome:
        self.n += 1
        if self.n == 4:
            raise _Abort()
        return self.inner(params, cost=cost)


def test_aborted_study_still_logs_trials(tmp_path):
    with pytest.raises(_Abort):
        _study(_BoomEvaluator(), opt.SearchSpace([opt.IntParam("a", 0, 10)]), tmp_path, "s-abort",
               checkpoint_every=2)
    s = ledger.studies(tmp_path / "ledger")["s-abort"]
    assert s["status"] == "aborted" and s["n_trials"] == 3
    assert ledger.load_trials("s-abort", tmp_path / "studies").height == 3


def test_requires_refit_and_grid_budget(tmp_path):
    class ML(SyntheticEvaluator):
        requires_refit = True

    with pytest.raises(NotImplementedError):
        _study(ML(bounds={"a": (0, 1)}), opt.SearchSpace([opt.IntParam("a", 0, 1)]), tmp_path, "s-ml")
    big = opt.SearchSpace([opt.IntParam(n, 0, 1) for n in "abcd"])
    with pytest.raises(ValueError, match="<= 3"):
        _study(SyntheticEvaluator(bounds={"a": (0, 1)}), big, tmp_path, "s-big")
    with pytest.raises(ValueError, match="book"):
        _study(SyntheticEvaluator(bounds={"a": (0, 1)}, book="B3"), opt.SearchSpace([opt.IntParam("a", 0, 1)]),
               tmp_path, "s-book")


# --------------------------------------------------------------------------- determinism
def _assert_same(r1, r2):
    assert r1.trials.equals(r2.trials)
    assert r1.returns.equals(r2.returns)
    assert r1.selected_params == r2.selected_params
    assert r1.cpcv_paths.equals(r2.cpcv_paths)
    assert r1.wfo_oos.equals(r2.wfo_oos) and r1.wfo_params.equals(r2.wfo_params)


def test_grid_determinism_across_n_jobs(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)}, rho=0.7,
                            bumps=({"center": {"a": 4, "b": 6}, "height": 1.2, "width": 0.25},))
    r1 = _study(ev, _space2(_no_five), tmp_path, "g1", n_jobs=1)
    r4 = _study(ev, _space2(_no_five), tmp_path, "g4", n_jobs=4)
    _assert_same(r1, r4)
    assert r4.meta["n_jobs"] == 4


def test_tpe_determinism_across_n_jobs_and_resume(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "x": (0.0, 1.0)}, rho=0.8,
                            bumps=({"center": {"a": 3, "x": 0.3}, "height": 1.5, "width": 0.2},))
    sp = opt.SearchSpace([opt.IntParam("a", 0, 10), opt.FloatParam("x", 0.0, 1.0)],
                         constraint=lambda p: not (p["a"] == 9 and p["x"] > 0.5))
    r1 = _study(ev, sp, tmp_path, "t1", method="tpe", n_trials=40, seed=7, n_jobs=1)
    r3 = _study(ev, sp, tmp_path, "t3", method="tpe", n_trials=40, seed=7, n_jobs=3)
    _assert_same(r1, r3)
    assert r1.meta["n_trials"] == 40 == r1.trials.height
    assert r1.selection["neighbourhood"]["neighbourhood"] == "knn"
    # resume: continue t1 to 55 trials; the first 40 are untouched
    r1b = _study(ev, sp, tmp_path, "t1", method="tpe", n_trials=55, seed=7, resume=True)
    assert r1b.trials.height == 55 and r1b.trials["trial_id"].to_list() == list(range(55))
    assert r1b.trials.head(40).select("trial_id", "status", "params").equals(r1.trials.select("trial_id", "status", "params"))
    s = ledger.studies(tmp_path / "ledger")["t1"]
    assert s["n_trials"] == 55 and "resumed" in s["events"]


def test_tpe_rdb_storage(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.8)
    url = f"sqlite:///{tmp_path / 'optuna.db'}"
    r = _study(ev, opt.SearchSpace([opt.FloatParam("a", 0, 10)]), tmp_path, "t-rdb", method="tpe",
               n_trials=12, storage=url)
    import optuna
    st = optuna.load_study(study_name="t-rdb", storage=url)
    assert len(st.trials) >= 12 and r.trials.height == 12


# --------------------------------------------------------------------------- CPCV
def test_cpcv_split_counts_and_paths():
    T, N, k = 1000, 10, 2
    splits, groups = opt.cpcv_splits(T, N, k, embargo_days=7, purge_days=5)
    assert len(splits) == math.comb(N, k) == 45
    appear = np.zeros(T, dtype=int)
    for sp in splits:
        appear[sp.test_idx] += 1
        assert np.intersect1d(sp.train_idx, sp.test_idx).size == 0
    phi = math.comb(N - 1, k - 1)
    assert (appear == phi).all()                     # every day in exactly φ test sets
    pmap = opt.cpcv_path_map(N, k)
    assert pmap.shape == (phi, N)
    # every split supplies exactly k (path, group) cells; each path covers every group once
    used = np.bincount(pmap.ravel(), minlength=len(splits))
    assert (used == k).all()
    for p in range(phi):
        for g in range(N):
            assert g in splits[pmap[p, g]].test_groups


def test_cpcv_purge_and_embargo_respected():
    T = 1000
    splits, groups = opt.cpcv_splits(T, 10, 2, embargo_days=7, purge_days=5)
    for sp in splits:
        test = np.zeros(T, bool)
        test[sp.test_idx] = True
        train = np.zeros(T, bool)
        train[sp.train_idx] = True
        edges = np.flatnonzero(np.diff(np.concatenate([[0], test.astype(int), [0]])))
        expected_drop = test.copy()
        for a, b in zip(edges[::2], edges[1::2]):
            assert not train[max(0, a - 5):a].any()          # purge before each test block
            assert not train[b:b + 7].any()                  # embargo after each test block
            expected_drop[max(0, a - 5):a] = True
            expected_drop[b:b + 7] = True
        assert (train == ~expected_drop).all()               # nothing else removed
    # adjacent test groups form one block: no purge/embargo between them
    sp = next(s for s in splits if s.test_groups == (3, 4))
    assert np.array_equal(sp.test_idx, np.concatenate([groups[3], groups[4]]))


def test_cpcv_paths_in_study(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.8, hold_days=4,
                            bumps=({"center": {"a": 6}, "height": 1.5, "width": 0.2},))
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10)]), tmp_path, "c1",
                 cv=opt.CPCVConfig(n_groups=6, k_test=2))
    p = res.cpcv_paths
    phi = math.comb(5, 1)
    assert p["path_id"].n_unique() == phi == res.meta["cpcv"]["n_paths"]
    per = p.group_by("path_id").agg(pl.len().alias("n"), pl.col("date").n_unique().alias("u"))
    assert (per["n"] == ev.n_days).all() and (per["u"] == ev.n_days).all()
    assert res.meta["cpcv"]["embargo_days"] == 4 == res.meta["cpcv"]["purge_days"]   # auto = max hold
    assert res.meta["cpcv_splits"].height == 15
    assert "CPCV(n=6,k=2,purge=4d,embargo=4d)" in res.meta["cv_scheme"]
    # each path's return on a day equals the return of the trial selected on the split that supplied it
    sp = res.meta["cpcv_splits"]
    assert sp["selected_trial"].null_count() == 0


# --------------------------------------------------------------------------- plateau
def test_plateau_select_prefers_broad_region_over_spike():
    # 1-D: a spike of 3.0 at index 2, a plateau of 1.5 at 6..10
    s = np.array([0, 0, 3.0, 0, 0, 0, 1.5, 1.5, 1.5, 1.5, 1.5, 0, 0])
    sp = opt.SearchSpace([opt.IntParam("a", 0, 12)])
    nb = opt.neighbourhoods(sp, [{"a": i} for i in range(13)], opt.PlateauConfig(neighbourhood="grid"))
    sel = opt.plateau_select(s, nb)
    assert sel["raw_index"] == 2
    assert sel["index"] == 7                    # first interior plateau point (ties → lowest)
    assert sel["plateau_score"] == 1.0
    assert sel["raw_plateau_score"] == 0.0


def test_plateau_rejects_hole_surrounded_by_good_neighbours():
    s = np.array([0.0, 1.0, 1.0, -2.0, 1.0, 1.0, 0.0])
    sp = opt.SearchSpace([opt.IntParam("a", 0, 6)])
    nb = opt.neighbourhoods(sp, [{"a": i} for i in range(7)], opt.PlateauConfig(neighbourhood="grid", radius=2))
    assert opt.plateau_select(s, nb)["index"] != 3


def test_study_selects_plateau_not_spike(tmp_path):
    res = _study(PLATEAU_EV, _space2(), tmp_path, "p1")
    assert res.selection["raw_argmax_params"] == {"a": 8, "b": 8}
    sel = res.selected_params
    assert 2 <= sel["a"] <= 4 and 2 <= sel["b"] <= 4       # interior of the planted plateau
    assert res.selection["plateau_score"] == 1.0
    assert res.selection["raw_argmax_plateau_score"] == 0.0
    # rho=1: every column shares the same noise, so Sharpe differences are the planted ones
    base = res.trials.filter((pl.col("param_a") == 0) & (pl.col("param_b") == 10))["m_sharpe"][0]
    assert res.selection["sharpe"] - base == pytest.approx(1.5, abs=0.1)


def test_knn_neighbourhood():
    sp = opt.SearchSpace([opt.FloatParam("x", 0, 1), opt.CategoricalParam("m", ["p", "q"])])
    pts = [{"x": v, "m": m} for m in "pq" for v in (0.0, 0.1, 0.2, 0.9)]
    nb = opt.neighbourhoods(sp, pts, opt.PlateauConfig(neighbourhood="knn", knn=2))
    assert nb.sum(axis=1).tolist() == [2] * 8
    assert not nb[:4, 4:].any() and not nb[4:, :4].any()    # unordered categoricals never mix
    assert nb[0, 1] and nb[0, 2] and not nb[0, 3]


# --------------------------------------------------------------------------- walk-forward
def _wfo_inputs(M=9, T=1560, seed=1):
    rng = np.random.default_rng(seed)
    d0 = np.datetime64("2016-05-02")
    dates = np.busday_offset(d0, np.arange(T), roll="forward")
    mu = np.linspace(-0.0002, 0.0004, M)
    R = 0.004 * rng.standard_normal((T, M)) + mu
    returns = pl.DataFrame({"date": dates, **{f"t{j}": R[:, j] for j in range(M)}}).with_columns(pl.col("date").cast(pl.Date))
    trials = pl.DataFrame({"trial_id": np.arange(M), "status": ["ok"] * M, "param_a": np.arange(M)})
    return returns, trials, opt.SearchSpace([opt.IntParam("a", 0, M - 1)])


def test_walk_forward_causality_poisoning_future():
    returns, trials, sp = _wfo_inputs()
    sched = opt.WFOConfig(refit_every="3mo", window="anchored", min_train="2y")
    oos, par, meta = opt.walk_forward(returns, trials, sp, sched)
    assert meta["n_refits"] == par.height > 5
    k = 3
    cut = par["refit_date"][k]
    # poison trial 0 (the worst) with huge returns strictly on/after refit k's date
    pois = returns.with_columns(pl.when(pl.col("date") >= cut).then(pl.col("t0") + 0.02)
                                .otherwise(pl.col("t0")).alias("t0"))
    oos2, par2, _ = opt.walk_forward(pois, trials, sp, sched)
    assert par2["selected_trial"][: k + 1].to_list() == par["selected_trial"][: k + 1].to_list()
    assert 0 in par2["selected_trial"][k + 1:].to_list()   # sensitive: later refits do see the poison
    assert 0 not in par["selected_trial"].to_list()
    # every refit trains strictly before its refit date and trades [refit_date, next refit)
    assert (par["train_end"] < par["refit_date"]).all()
    j = oos.join(par.select("refit_id", "refit_date"), on="refit_id")
    assert (j["date"] >= j["refit_date"]).all()
    assert oos["date"].is_sorted() and oos["date"].n_unique() == oos.height
    first_refit = par["refit_date"][0]
    assert first_refit == date(2018, 5, 2)
    assert oos["date"].min() >= first_refit


def test_walk_forward_anchored_vs_rolling():
    returns, trials, sp = _wfo_inputs()
    first = returns["date"][0]
    _, pa, _ = opt.walk_forward(returns, trials, sp, opt.WFOConfig(window="anchored", min_train="2y"))
    assert (pa["train_start"] == first).all()
    _, pr, meta = opt.walk_forward(returns, trials, sp,
                                   opt.WFOConfig(window="rolling", window_length="1y", min_train="1y"))
    for row in pr.iter_rows(named=True):
        lo = row["refit_date"] - timedelta(days=366)
        assert row["train_start"] >= lo and row["train_start"] >= first
        assert row["train_end"] < row["refit_date"]
    later = pr.filter(pl.col("refit_date") > first + timedelta(days=500))
    assert (later["train_start"] > first).all()             # the window really rolls
    assert {"drift_mean", "drift_max", "drift_distances", "n_distinct_selections"} <= set(meta)


def test_wfo_parameter_drift_on_regime_change(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.9,
                            bumps=({"center": {"a": 2}, "height": 2.0, "width": 0.1},),
                            regimes=({"from": 0.6, "bumps": ({"center": {"a": 8}, "height": 2.0, "width": 0.1},)},))
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10)]), tmp_path, "w1",
                 wfo=opt.WFOConfig(window="rolling", window_length="1y", min_train="1y"))
    p = res.wfo_params
    assert p["param_a"][0] in (1, 2, 3) and p["param_a"][-1] in (7, 8, 9)
    assert res.meta["wfo"]["drift_max"] >= 0.4


# --------------------------------------------------------------------------- real-data smoke + benchmark
@pytest.mark.slow
def test_benchmark_200_trial_grid_eurusd_h1(tmp_path):
    """DESIGN §7 budget check: 200-trial grid on EURUSD H1 (dev window) with M1 + CPCV(10,2) + WFO."""
    from test_evaluators import SmaCross
    from quantlab.evaluators import RuleEvaluator

    warnings.filterwarnings("ignore")
    ev = RuleEvaluator(SmaCross, symbol="EURUSD", timeframe="H1", start="2016-05-02", end="2025-05-14")
    sp = opt.SearchSpace([opt.IntParam("fast", 5, 50, 5), opt.IntParam("slow", 60, 250, 10)])
    t0 = time.perf_counter()
    res = opt.run_study(ev, sp, book="FBS", system="sma_bench", issue=999, attempt=1, study_id="bench-sma",
                        n_jobs="auto", cv=opt.CPCVConfig(10, 2), wfo=opt.WFOConfig(), min_trades=100,
                        ledger_dir=tmp_path / "ledger", studies_dir=tmp_path / "studies")
    wall = time.perf_counter() - t0
    print(f"\n200-trial EURUSD H1 grid study: {wall:.1f}s wall, n_jobs={res.meta['n_jobs']}, "
          f"selected={res.selected_params}, plateau={res.selection.get('plateau_score')}, "
          f"cv={res.meta['cv_scheme']}, refits={res.meta['wfo']['n_refits']}, "
          f"drift_mean={res.meta['wfo']['drift_mean']:.3f}")
    assert res.meta["n_trials"] == 200 and res.meta["n_error"] == 0
    assert res.cpcv_paths["path_id"].n_unique() == 9
    assert wall < 600
