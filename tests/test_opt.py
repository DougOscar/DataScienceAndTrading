"""Tests for quantlab.opt: logging of every trial, determinism, CPCV, plateau, walk-forward."""

from __future__ import annotations

import json
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
    return opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2), opt.IntParam("b", 0, 10, plateau_step=2)], constraint=constraint)


def _no_five(p):
    return p["a"] != 5


PLATEAU_EV = SyntheticEvaluator(
    bounds={"a": (0, 10), "b": (0, 10)},
    bumps=({"center": {"a": 3, "b": 3}, "height": 1.5, "width": 0.2, "kind": "box"},     # a,b in 1..5
           {"center": {"a": 8, "b": 8}, "height": 3.0, "width": 0.01, "kind": "box"}),   # single point
    rho=1.0, seed=3)


# --------------------------------------------------------------------------- search space
def test_search_space_grid_and_units():
    sp = opt.SearchSpace([opt.IntParam("n", 5, 20, 5, plateau_scale="relative"), opt.FloatParam("x", 0.5, 1.5, 0.25, plateau_scale="relative"),
                          opt.CategoricalParam("mode", ["a", "b"])])
    g = sp.grid()
    assert len(g) == sp.grid_size() == 4 * 5 * 2
    assert g[0] == {"n": 5, "x": 0.5, "mode": "a"}
    assert sorted({p["x"] for p in g}) == [0.5, 0.75, 1.0, 1.25, 1.5]
    U = sp.unit_coords([{"n": 20, "x": 1.0, "mode": "b"}])
    np.testing.assert_allclose(U, [[1.0, 0.5, 1.0]])
    assert sp.is_discrete
    assert not opt.SearchSpace([opt.FloatParam("x", 0, 1, plateau_step=0.2)]).is_discrete
    with pytest.raises(ValueError):
        opt.SearchSpace([opt.FloatParam("x", 0, 1, plateau_step=0.2)]).grid()
    js = sp.to_json()
    assert js["params"][0] == {"name": "n", "kind": "int", "low": 5, "high": 20, "step": 5, "plateau_scale": "relative"}
    assert js["params"][1]["plateau_scale"] == "relative" and "plateau_scale" not in js["params"][2]
    assert opt.space_from_json(js).to_json() == js


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
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "s-low", min_trades=1000, wfo=None)
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
        _study(_BoomEvaluator(), opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "s-abort",
               checkpoint_every=2)
    s = ledger.studies(tmp_path / "ledger")["s-abort"]
    assert s["status"] == "aborted" and s["n_trials"] == 3
    assert ledger.load_trials("s-abort", tmp_path / "studies").height == 3


def test_requires_refit_and_grid_budget(tmp_path):
    class ML(SyntheticEvaluator):
        requires_refit = True

    with pytest.raises(NotImplementedError):
        _study(ML(bounds={"a": (0, 1)}), opt.SearchSpace([opt.IntParam("a", 0, 1, plateau_step=0.2)]), tmp_path, "s-ml")
    big = opt.SearchSpace([opt.IntParam(n, 0, 1, plateau_step=0.2) for n in "abcd"])
    with pytest.raises(ValueError, match="<= 3"):
        _study(SyntheticEvaluator(bounds={"a": (0, 1)}), big, tmp_path, "s-big", method="grid")
    # B3: the default ("auto") for > 3 params is Sobol, which needs a trial budget
    with pytest.raises(ValueError, match="'sobol' needs n_trials"):
        _study(SyntheticEvaluator(bounds={"a": (0, 1)}), big, tmp_path, "s-big2")
    with pytest.raises(ValueError, match="book"):
        _study(SyntheticEvaluator(bounds={"a": (0, 1)}, book="B3"), opt.SearchSpace([opt.IntParam("a", 0, 1, plateau_step=0.2)]),
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
    sp = opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2), opt.FloatParam("x", 0.0, 1.0, plateau_step=0.2)],
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
    r = _study(ev, opt.SearchSpace([opt.FloatParam("a", 0, 10, plateau_step=2)]), tmp_path, "t-rdb", method="tpe",
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
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "c1",
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
    sp = opt.SearchSpace([opt.IntParam("a", 0, 12, plateau_step=2.4)])
    nb = opt.neighbourhoods(sp, [{"a": i} for i in range(13)], opt.PlateauConfig(neighbourhood="grid"))
    sel = opt.plateau_select(s, nb)
    assert sel["raw_index"] == 2
    assert sel["index"] == 7                    # first interior plateau point (ties → lowest)
    assert sel["plateau_score"] == 1.0
    assert sel["raw_plateau_score"] == 0.0


def test_plateau_rejects_hole_surrounded_by_good_neighbours():
    s = np.array([0.0, 1.0, 1.0, -2.0, 1.0, 1.0, 0.0])
    sp = opt.SearchSpace([opt.IntParam("a", 0, 6, plateau_step=1.2)])
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
    sp = opt.SearchSpace([opt.FloatParam("x", 0, 1, plateau_step=0.2), opt.CategoricalParam("m", ["p", "q"])])
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
    return returns, trials, opt.SearchSpace([opt.IntParam("a", 0, M - 1, plateau_step=1)])


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
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "w1",
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
    sp = opt.SearchSpace([opt.IntParam("fast", 5, 50, 5, plateau_scale="relative"), opt.IntParam("slow", 60, 250, 10, plateau_scale="relative")])
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


# --------------------------------------------------------------------------- B3: candidate sets
class _SmoothNull:
    """Zero-edge world with parameter-smooth noise (red-team probe p07): returns are a smooth
    random field over (x, y) in [0, 1]^2 plus a common factor.  Every configuration has true
    Sharpe 0, but neighbouring configurations share their luck, so a candidate set chosen with
    the full-sample objective (TPE) concentrates on the region that was lucky out-of-sample too."""

    book, periods_per_year, requires_refit, cost_version = "FBS", 260.0, False, "probe"
    T = 2340

    def __init__(self, seed, L=0.15, rho=0.3, vol=0.005):
        rng = np.random.default_rng(seed)
        g = np.linspace(0, 1, 8)
        self.C = np.array([(x, y) for x in g for y in g])
        self.Z = rng.standard_normal((self.T, len(self.C)))
        self.zc = rng.standard_normal(self.T)
        self.L, self.rho, self.vol = L, rho, vol
        self.dates = np.busday_offset(np.datetime64("2016-05-02"), np.arange(self.T), roll="forward")
        self.dev_window = (str(self.dates[0]), str(self.dates[-1]))

    def __call__(self, params, *, cost=None):
        p = np.array([params["x"], params["y"]])
        w = np.exp(-((self.C - p) ** 2).sum(1) / (2 * self.L ** 2))
        r = self.vol * (math.sqrt(self.rho) * self.zc + math.sqrt(1 - self.rho) * (self.Z @ w / np.linalg.norm(w)))
        daily = pl.DataFrame({"date": self.dates.astype("datetime64[D]"), "ret": r}).with_columns(pl.col("date").cast(pl.Date))
        tr = pl.DataFrame({"entry_ts": [], "exit_ts": []}, schema={"entry_ts": pl.Datetime("ms"), "exit_ts": pl.Datetime("ms")})
        return Outcome(daily=daily, trades=tr, metrics={"n_trades": 900.0, "hold_days_max": 3.0})


def _cpcv_median_sharpe(res):
    s = [g.sort("date")["ret"].to_numpy() for _, g in res.cpcv_paths.group_by("path_id")]
    return float(np.median([x.mean() / x.std(ddof=1) * math.sqrt(260) for x in s]))


_B3_BIAS = 0.08    # paired mean (candidate set − full grid) CPCV path-median OOS Sharpe, 12 seeds


def _null_bias(tmp_path, methods, n_seeds, step=0.1, n_sobol=40):
    space = opt.SearchSpace([opt.FloatParam("x", 0.0, 1.0, step, plateau_step=0.2), opt.FloatParam("y", 0.0, 1.0, step, plateau_step=0.2)])
    out = {m: [] for m in methods}
    for s in range(n_seeds):
        ev = _SmoothNull(seed=s)
        for m in methods:
            res = _study(ev, space, tmp_path, f"b3-{m}-{s}", method=m, seed=s, wfo=None,
                         n_trials=None if m == "grid" else n_sobol, cv=opt.CPCVConfig(6, 2))
            out[m].append(_cpcv_median_sharpe(res))
    return {m: np.asarray(v) for m, v in out.items()}


def test_sobol_candidate_set_has_no_oos_bias_under_smooth_null(tmp_path):
    """B3 fix: a Sobol candidate set is fixed before any evaluation, so its CPCV OOS Sharpe
    under a zero-edge smooth field is centred like the full grid's (TPE: +0.17, probe p07)."""
    r = _null_bias(tmp_path, ("grid", "sobol"), n_seeds=12)
    d = r["sobol"] - r["grid"]
    assert d.mean() < _B3_BIAS, d.mean()            # measured +0.03 (se 0.06); TPE: +0.18
    assert abs(r["sobol"].mean()) < 0.15


@pytest.mark.slow
def test_tpe_candidate_set_bias_is_detected_by_the_same_check(tmp_path):
    """Power of the check above: the same null with TPE shows the p07 leak."""
    r = _null_bias(tmp_path, ("grid", "tpe"), n_seeds=12)
    d = r["tpe"] - r["grid"]
    assert d.mean() > _B3_BIAS, d.mean()


def test_candidate_set_sobol_mapping_distinct_and_prefix_stable():
    sp = opt.SearchSpace([opt.IntParam("n", 5, 50, 5, plateau_scale="relative"), opt.IntParam("k", 2, 200, log=True, plateau_scale="relative"),
                          opt.FloatParam("x", 0.1, 10.0, log=True, plateau_scale="relative"), opt.FloatParam("z", -1.0, 1.0, plateau_step=0.4),
                          opt.FloatParam("s", 0.0, 1.0, 0.25, plateau_step=0.2), opt.CategoricalParam("m", ["a", "b", "c"])])
    a = opt.candidate_set(sp, 200, method="sobol", seed=3)
    assert len(a) == 200 and len({opt._pjson(p) for p in a}) == 200
    assert a == opt.candidate_set(sp, 200, method="sobol", seed=3)              # seeded
    assert a != opt.candidate_set(sp, 200, method="sobol", seed=4)
    assert opt.candidate_set(sp, 64, method="sobol", seed=3) == a[:64]         # prefix-stable (resume)
    for p in a:
        assert p["n"] in range(5, 51, 5) and isinstance(p["n"], int)
        assert 2 <= p["k"] <= 200 and isinstance(p["k"], int)
        assert 0.1 <= p["x"] <= 10.0 and -1.0 <= p["z"] <= 1.0
        assert p["s"] in (0.0, 0.25, 0.5, 0.75, 1.0) and p["m"] in ("a", "b", "c")
    # every level of the discrete params is visited; log params are log-uniform (median ~ geometric mean)
    assert {p["n"] for p in a} == set(range(5, 51, 5)) and {p["m"] for p in a} == {"a", "b", "c"}
    assert 0.5 < float(np.median([p["x"] for p in a])) < 2.0
    # a discrete space smaller than n: every configuration once, with a warning
    small = opt.SearchSpace([opt.IntParam("a", 0, 3, plateau_step=0.6), opt.CategoricalParam("b", ["u", "v"])])
    with pytest.warns(UserWarning, match="only 8 distinct"):
        c = opt.candidate_set(small, 20, method="sobol", seed=0, max_draws=4096)
    assert len(c) == 8
    r = opt.candidate_set(sp, 50, method="random", seed=1)
    assert len(r) == 50 and r == opt.candidate_set(sp, 50, method="random", seed=1)


def _space4(constraint=None):
    return opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2), opt.IntParam("b", 0, 10, plateau_step=2), opt.FloatParam("x", 0.0, 1.0, plateau_step=0.2),
                            opt.CategoricalParam("m", ["p", "q"])], constraint=constraint)


def _bad_a9(p):
    return not (p["a"] == 9 and p["m"] == "q")


def test_auto_method_is_sobol_for_4_params_and_meta_for_gates(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10), "x": (0.0, 1.0)}, rho=0.8,
                            bumps=({"center": {"a": 3, "b": 6, "x": 0.4}, "height": 1.5, "width": 0.3},))
    res = _study(ev, _space4(_bad_a9), tmp_path, "s4", n_trials=64, seed=5)
    m = res.meta
    assert m["method"] == m["candidate_set"] == "sobol" and m["method_requested"] == "auto"
    assert m["candidate_set_data_dependent"] is False
    assert (m["book"], m["system"], m["issue"], m["attempt"]) == ("FBS", "synthetic", 1, 1)
    assert m["space"] == _space4(_bad_a9).to_json()
    assert m["embargo_capped"] is False and m["embargo_days_uncapped"] == 3
    # every candidate is a logged trial (invalid ones too), in candidate-set order
    assert res.trials.height == 64 == m["n_trials"]
    cands = opt.candidate_set(_space4(), 64, method="sobol", seed=5)
    assert [json.loads(p) for p in res.trials["params"]] == [opt._py(c) for c in cands]
    n_bad = sum(not _bad_a9(c) for c in cands)
    assert n_bad > 0 and m["n_invalid"] == n_bad
    assert res.selection["neighbourhood"]["neighbourhood"] == "knn"
    s = ledger.studies(tmp_path / "ledger")["s4"]
    assert s["candidate_set"] == "sobol" and s["candidate_set_data_dependent"] is False
    assert s["embargo_capped"] is False and s["n_trials"] == 64
    # grid: data-independent too; TPE: flagged data-dependent (B3)
    g = _study(ev, _space2(), tmp_path, "s2g")
    assert g.meta["candidate_set"] == "grid" and g.meta["candidate_set_data_dependent"] is False
    t = _study(ev, _space4(), tmp_path, "s4t", method="tpe", n_trials=20, wfo=None)
    assert t.meta["candidate_set"] == "tpe" and t.meta["candidate_set_data_dependent"] is True
    assert ledger.studies(tmp_path / "ledger")["s4t"]["candidate_set_data_dependent"] is True


def test_sobol_resume_extends_the_same_candidate_set(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10), "x": (0.0, 1.0)}, rho=0.8)
    r1 = _study(ev, _space4(), tmp_path, "sr", n_trials=30, seed=2, wfo=None)
    r2 = _study(ev, _space4(), tmp_path, "sr", n_trials=50, seed=2, wfo=None, resume=True)
    full = _study(ev, _space4(), tmp_path, "sr-full", n_trials=50, seed=2, wfo=None)
    assert r2.trials.head(30).select("trial_id", "params").equals(r1.trials.select("trial_id", "params"))
    assert r2.trials.select("trial_id", "params").equals(full.trials.select("trial_id", "params"))
    assert r2.selected_params == full.selected_params


def test_embargo_cap_is_recorded(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.8, hold_days=80, trades_per_year=20)
    with pytest.warns(UserWarning, match="capped"):
        res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "cap", wfo=None)
    cap = (ev.n_days // 10) // 4
    assert res.meta["embargo_capped"] is True and res.meta["embargo_days_uncapped"] == 80
    assert res.meta["cpcv"]["embargo_days"] == cap and res.meta["cpcv"]["embargo_cap"] == cap
    assert ledger.studies(tmp_path / "ledger")["cap"]["embargo_capped"] is True
    res2 = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "nocap", wfo=None,
                  cv=opt.CPCVConfig(n_groups=4, k_test=1))
    assert res2.meta["embargo_capped"] is False and res2.meta["cpcv"]["embargo_days"] == 80


class _NoEntryTs(SyntheticEvaluator):
    def __call__(self, params, *, cost=None):
        o = super().__call__(params, cost=cost)
        return Outcome(daily=o.daily, trades=o.trades.select(pl.col("exit_ts")), metrics=o.metrics)


def test_wfo_oos_n_trades_are_the_active_configs_entries(tmp_path):
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.9, hold_days=4, trades_per_year=60,
                            bumps=({"center": {"a": 2}, "height": 2.0, "width": 0.1},),
                            regimes=({"from": 0.6, "bumps": ({"center": {"a": 8}, "height": 2.0, "width": 0.1},)},))
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "wn",
                 wfo=opt.WFOConfig(window="rolling", window_length="1y", min_train="1y"))
    w = res.wfo_oos
    assert w["n_trades"].dtype == pl.Int64 and w["n_trades"].null_count() == 0
    expected = []
    for row in res.wfo_params.iter_rows(named=True):
        tr = ev({"a": row["param_a"]}).trades
        d = tr["entry_ts"].dt.date()
        expected.append(int(((d >= row["refit_date"]) & (d <= row["test_end"])).sum()))
    got = w.group_by("refit_id").agg(pl.col("n_trades").sum()).sort("refit_id")["n_trades"].to_list()
    assert got == expected and sum(got) > 0
    assert res.wfo_params["param_a"].n_unique() > 1                 # the active config really switches
    # no entry timestamps in the evaluator's trades -> null, not a made-up zero
    res2 = _study(_NoEntryTs(bounds={"a": (0, 10)}, rho=0.9), opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]),
                  tmp_path, "wn2")
    assert res2.wfo_oos.height and res2.wfo_oos["n_trades"].null_count() == res2.wfo_oos.height
    # resume restores the entry counts from the opt_aux sidecar
    res3 = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "wn", resume=True,
                  wfo=opt.WFOConfig(window="rolling", window_length="1y", min_train="1y"))
    assert res3.wfo_oos.equals(res.wfo_oos)


def test_worker_processes_start_with_thread_caps(tmp_path):
    """F1: the Polars pool inside a worker really has 1 thread (the old initializer-time
    setdefault had no effect: the child had already initialised Polars), and the parent's
    environment is left untouched."""
    pl.DataFrame({"a": [1, 2]}).sum()                               # parent pool initialised
    before = {k: os.environ.get(k) for k in opt.WORKER_THREAD_ENV}
    r = opt._Runner(SyntheticEvaluator(bounds={"a": (0, 1)}), None, 2)
    try:
        assert r.start_method == "spawn"
        infos = [r.pool.submit(opt._worker_threads).result() for _ in range(4)]
    finally:
        r.close()
    assert {i["polars_threads"] for i in infos} == {1}
    assert all(i["POLARS_MAX_THREADS"] == "1" and i["NUMBA_NUM_THREADS"] == "1" for i in infos)
    assert {k: os.environ.get(k) for k in opt.WORKER_THREAD_ENV} == before
    res = _study(SyntheticEvaluator(bounds={"a": (0, 10)}), opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]),
                 tmp_path, "thr", n_jobs=2, wfo=None)
    assert res.meta["mp_start_method"] == "spawn" and res.meta["worker_threads"]["polars_threads"] == 1


# --------------------------------------------------------------------------- fix round 2: N1 / N3 / N10
def _resume_kw(**over):
    kw = dict(book="FBS", system="synthetic", issue=1, attempt=1, method="tpe", n_trials=16, seed=3, wfo=None,
              n_jobs=1, resume=True)
    kw.update(over)
    return kw


def test_resume_cannot_change_method_seed_space_or_identity(tmp_path):
    """N1 regression (r1_p07): TPE 12 → resume as Sobol cleared the data-dependence flag.  A resume
    must match the ledger's study_created row (method / seed / space / radius / identity); a
    refused resume leaves no event.  Each trial records its source; once TPE, always flagged."""
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10), "x": (0.0, 1.0)}, rho=0.8)
    sp = _space4()
    t = _study(ev, sp, tmp_path, "n1", method="tpe", n_trials=12, seed=3, wfo=None)
    assert t.trials["source"].to_list() == ["tpe"] * 12 and t.meta["candidate_set_data_dependent"] is True
    led, sd = tmp_path / "ledger", tmp_path / "studies"
    n_events = len(ledger.study_events("n1", ledger_dir=led))
    bad = [dict(method="sobol", n_trials=24), dict(seed=4), dict(system="other"), dict(attempt=2),
           dict(issue=2), dict(plateau_radius=0.3)]
    for over in bad:
        with pytest.raises(opt.StudyError, match="pre-registered settings"):
            opt.run_study(ev, sp, study_id="n1", ledger_dir=led, studies_dir=sd, **_resume_kw(**over))
    with pytest.raises(opt.StudyError, match="search_space"):
        opt.run_study(ev, opt.SearchSpace(_space4().params[:3]), study_id="n1", ledger_dir=led,
                      studies_dir=sd, **_resume_kw())
    assert len(ledger.study_events("n1", ledger_dir=led)) == n_events           # nothing logged
    t2 = opt.run_study(ev, sp, study_id="n1", ledger_dir=led, studies_dir=sd, **_resume_kw())
    assert t2.trials.height == 16 and t2.trials["source"].to_list() == ["tpe"] * 16
    assert t2.meta["candidate_set_data_dependent"] is True and t2.meta["plateau_radius"] == 0.20
    st = ledger.studies(led)["n1"]
    assert st["n_by_source"] == {"tpe": 16} and st["candidate_set_data_dependent"] is True
    assert ledger.candidate_set_data_dependent("n1", ledger_dir=led) is True
    # N10: the change of planned trials is an auditable event
    ch = ledger.study_events("n1", "n_trials_changed", ledger_dir=led)
    assert len(ch) == 1 and (ch[0]["n_trials_planned_old"], ch[0]["n_trials_planned_new"]) == (12, 16)
    assert ch[0]["n_trials_recorded"] == 12 and ch[0]["gate_runs_before"] == 0


def test_legacy_trials_without_source_take_the_ledger_method(tmp_path):
    """N1: trials recorded before sources existed get the method of the ledger's created row."""
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, rho=0.8)
    sp = opt.SearchSpace([opt.FloatParam("a", 0.0, 10.0, plateau_step=2)])
    _study(ev, sp, tmp_path, "leg", method="tpe", n_trials=8, wfo=None)
    d = tmp_path / "studies" / "leg"
    for f in d.glob("trials-*.parquet"):
        pl.read_parquet(f).drop("source").write_parquet(f)
    assert opt._load_existing("leg", tmp_path / "studies", default_source="tpe")[0].source == "tpe"
    r = _study(ev, sp, tmp_path, "leg", method="tpe", n_trials=10, wfo=None, resume=True)
    assert r.trials["source"].to_list() == ["tpe"] * 10 and r.meta["candidate_set_data_dependent"] is True


def test_sobol_resume_logs_n_trials_change_and_keeps_flag_false(tmp_path):
    """N10: look → resume with more trials is logged with old / new values and prior gate runs."""
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10), "x": (0.0, 1.0)}, rho=0.8)
    r1 = _study(ev, _space4(), tmp_path, "n10", n_trials=20, seed=2, wfo=None)
    assert r1.trials["source"].to_list() == ["sobol"] * 20
    ledger.log_event("n10", "gates", ledger_dir=tmp_path / "ledger", verdict="FAIL", gate_run=1)
    r2 = _study(ev, _space4(), tmp_path, "n10", n_trials=30, seed=2, wfo=None, resume=True)
    assert r2.meta["candidate_set_data_dependent"] is False
    ch = ledger.study_events("n10", "n_trials_changed", ledger_dir=tmp_path / "ledger")
    assert [(c["n_trials_planned_old"], c["n_trials_planned_new"], c["gate_runs_before"]) for c in ch] == [(20, 30, 1)]
    assert ledger.studies(tmp_path / "ledger")["n10"]["n_trials_planned"] == 30
    _study(ev, _space4(), tmp_path, "n10", n_trials=30, seed=2, wfo=None, resume=True)   # same n: no new event
    assert len(ledger.study_events("n10", "n_trials_changed", ledger_dir=tmp_path / "ledger")) == 1


def test_plateau_radius_is_pre_registered_in_the_ledger(tmp_path):
    """N3: run_study(plateau_radius=) is stored in study_created and meta; floor 0.10; names checked."""
    ev = SyntheticEvaluator(bounds={"a": (0, 10), "b": (0, 10)}, rho=0.8)
    sp = opt.SearchSpace([opt.IntParam("a", 1, 4, plateau_scale="relative"),
                          opt.IntParam("b", 1, 4, plateau_scale="relative")])
    r = _study(ev, sp, tmp_path, "rad", wfo=None, plateau_radius={"b": 0.3, "a": 0.25})
    row = ledger.created_row("rad", ledger_dir=tmp_path / "ledger")
    assert row["plateau_radius"] == {"a": 0.25, "b": 0.3} == r.meta["plateau_radius"]
    assert ledger.created_row("rad", ledger_dir=tmp_path / "ledger")["seq"] == 0
    _study(ev, sp, tmp_path, "rad0", wfo=None)
    assert ledger.created_row("rad0", ledger_dir=tmp_path / "ledger")["plateau_radius"] == 0.20
    with pytest.raises(ValueError, match="floor"):
        _study(ev, sp, tmp_path, "rad1", wfo=None, plateau_radius=1e-6)
    with pytest.raises(ValueError, match="unknown parameter"):
        _study(ev, sp, tmp_path, "rad2", wfo=None, plateau_radius={"zz": 0.2})
    # R2-1: a per-param radius cannot name a param with an absolute plateau_step (it would not apply)
    sp_abs = opt.SearchSpace([opt.IntParam("a", 0, 3, plateau_step=1), opt.IntParam("b", 1, 4, plateau_scale="relative")])
    with pytest.raises(ValueError, match="absolute plateau_step"):
        _study(ev, sp_abs, tmp_path, "rad3", wfo=None, plateau_radius={"a": 0.3})
    # resume without a radius takes the ledger's; the same radius is accepted
    _study(ev, sp, tmp_path, "rad", wfo=None, resume=True)
    _study(ev, sp, tmp_path, "rad", wfo=None, resume=True, plateau_radius={"a": 0.25, "b": 0.3})


# --------------------------------------------------------------------------- L1: uniform first-batch errors
class _DataBroken(SyntheticEvaluator):
    """Every call fails with the same data error (like USDCHF before the L1 fix)."""

    def __call__(self, params, *, cost=None):
        raise ValueError("USDCHF: no M1 bar had opened yet at/before some requested timestamps")


def test_uniform_first_batch_errors_abort_the_study(tmp_path):
    ev = _DataBroken(bounds={"a": (0, 10)})
    with pytest.raises(opt.StudyError, match="same error"):
        _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "s-uni", wfo=None)
    s = ledger.studies(tmp_path / "ledger")["s-uni"]
    assert s["status"] == "aborted" and s["n_trials"] == 2 and s["n_error"] == 2   # max(n_jobs, 2)
    assert ledger.load_trials("s-uni", tmp_path / "studies").height == 2
    # opt-out grinds through the whole grid (then fails: nothing evaluated)
    with pytest.raises(opt.StudyError, match="no configuration could be evaluated"):
        _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "s-uni2", wfo=None,
               abort_on_uniform_errors=False)
    assert ledger.studies(tmp_path / "ledger")["s-uni2"]["n_trials"] == 11


def test_non_uniform_first_batch_errors_do_not_abort(tmp_path):
    # first two trials fail with *different* messages (parameter-specific) → no abort
    ev = SyntheticEvaluator(bounds={"a": (0, 10)}, error_when=({"a": 0}, {"a": 1}), rho=0.9)
    res = _study(ev, opt.SearchSpace([opt.IntParam("a", 0, 10, plateau_step=2)]), tmp_path, "s-mixed", wfo=None)
    assert res.meta["n_error"] == 2 and res.meta["n_ok"] == 9
    assert res.meta["eval_start_effective"] is None and res.meta["data_gaps"] == []
