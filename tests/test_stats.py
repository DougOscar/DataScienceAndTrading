"""Known-answer and small Monte-Carlo tests for quantlab.stats (Phase 1 'validate the validator')."""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats as sps

from quantlab import metrics
from quantlab import stats as S


def _daily(ret: np.ndarray, start=date(2016, 5, 2)) -> pl.DataFrame:
    """Business-day dated frame (Mon–Fri) for a return array."""
    days, d = [], start
    while len(days) < len(ret):
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return pl.DataFrame({"date": days, "ret": np.asarray(ret, float)})


# ------------------------------------------------------------------ PSR / DSR / MinTRL
def test_psr_hand_computation():
    sr, n, sk, ku = 0.1, 101, -0.5, 5.0
    z = sr * math.sqrt(n - 1) / math.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2)
    assert S.psr(sr, 0.0, n, sk, ku) == pytest.approx(sps.norm.cdf(z), abs=1e-12)
    # normal case, benchmark = observed → 0.5
    assert S.psr(0.08, 0.08, 500) == pytest.approx(0.5)


def test_dsr_reproduces_bailey_lopez_de_prado_2014_example():
    """BLdP (2014) 'The Deflated Sharpe Ratio', numerical example: annualised SR 2.5 on 1,250
    daily obs (250/yr), N = 100 trials, V[SR_annual] = 0.5, skew −3, kurtosis 10 →
    E[max SR] ≈ 0.1132 (per-period) and DSR ≈ 0.9004."""
    var_pp = 0.5 / 250.0
    sr_pp = 2.5 / math.sqrt(250.0)
    assert S.expected_max_sharpe(100, var_pp) == pytest.approx(0.1132, abs=1e-4)
    d = S.dsr(sr=sr_pp, n=1250, skew=-3.0, kurt=10.0, var_sr=var_pp, n_eff=100)
    assert d == pytest.approx(0.9004, abs=5e-4)


def test_dsr_decreases_with_n_eff_and_equals_psr0_at_one_trial():
    rng = np.random.default_rng(1)
    r = rng.normal(0.0008, 0.01, 2000)
    vals = [S.dsr(r, var_sr=1e-3, n_eff=n) for n in (1, 2, 10, 100, 1000, 1e6)]
    assert all(a > b for a, b in zip(vals, vals[1:]))
    sk, ku = metrics.skew_kurt(r)
    assert vals[0] == pytest.approx(S.psr(S.sharpe_per_period(r), 0.0, r.size, sk, ku))


def test_expected_max_sharpe_matches_monte_carlo():
    rng = np.random.default_rng(2)
    n, sd = 50, 0.03
    mc = rng.normal(0, sd, (20000, n)).max(axis=1).mean()
    assert S.expected_max_sharpe(n, sd ** 2) == pytest.approx(mc, rel=0.03)


@pytest.mark.parametrize("sr,skew,kurt", [(0.1, 0.0, 3.0), (0.05, -1.0, 8.0), (0.2, 0.5, 4.0)])
def test_min_trl_inverse_of_psr(sr, skew, kurt):
    mt = S.min_trl(sr, 0.0, skew, kurt, prob=0.95)
    assert S.psr(sr, 0.0, mt["periods"], skew, kurt) == pytest.approx(0.95, abs=1e-9)
    assert mt["years"] == pytest.approx(mt["periods"] / 260.0)
    assert S.min_trl(0.0, 0.0)["periods"] == math.inf


def test_power_check_flags_infeasible_low_sharpe():
    ok = S.power_check(2.0, 50, 9.0)
    bad = S.power_check(0.4, 50, 9.0)
    assert ok["feasible"] and not bad["feasible"]
    assert bad["min_trl_years"] > 9.0


# ------------------------------------------------------------------ effective N
def test_effective_n_limits():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.01, (3000, 1))
    ident = np.repeat(base, 40, axis=1) * rng.uniform(0.5, 2, 40)  # scaled copies: ρ = 1
    indep = rng.normal(0, 0.01, (3000, 40))
    for m in ("eigen", "cluster", "liji"):
        assert S.effective_n_trials(ident, m) == pytest.approx(1.0, abs=1e-6)
    assert S.effective_n_trials(indep, "cluster") == 40
    assert S.effective_n_trials(indep, "eigen") > 0.95 * 40 / (1 + 40 / 3000) - 1
    assert S.effective_n_trials(indep, "liji") > 35
    # 5 blocks of 8 near-duplicates
    f = rng.normal(0, 0.01, (3000, 5))
    blocks = np.repeat(f, 8, axis=1) + rng.normal(0, 0.002, (3000, 40))
    assert S.effective_n_trials(blocks, "cluster") == 5
    assert 4.0 < S.effective_n_trials(blocks, "eigen") < 6.0
    # flat (no-trade) columns are ignored
    with_flat = np.column_stack([indep[:, :10], np.zeros((3000, 5))])
    assert S.effective_n_trials(with_flat, "cluster") == 10


def test_dsr_from_matrix_reports_both():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 0.01, (2000, 30))
    m[:, 7] += 0.001
    frame = pl.DataFrame({f"t{i}": m[:, i] for i in range(30)})
    res = S.dsr_from_matrix(frame, "t7")
    # R1 gate: V0 = 1/(T−1), N = raw column count
    assert res.n_trials == 30 and res.v0 == pytest.approx(1 / 1999)
    assert res.sr0 == pytest.approx(S.expected_max_sharpe(30, 1 / 1999))
    assert res.dsr == res.dsr_raw_n and res.dsr <= res.psr0
    # diagnostics
    assert res.n_eff == max(res.n_eff_by_method["eigen"], res.n_eff_by_method["cluster"])
    assert set(res.n_eff_by_method) == {"eigen", "cluster", "liji"}
    assert res.sr_annual == pytest.approx(metrics.sharpe(m[:, 7]))
    more = S.dsr_from_matrix(frame, "t7", extra_trials=500)
    assert more.dsr < res.dsr and more.n_trials == 530 and more.n_trials_prior == 500
    explicit = S.dsr_from_matrix(frame, "t7", n_trials=45)
    assert explicit.n_trials_study == 45 and explicit.dsr < res.dsr


def test_dsr_hurdle_ignores_correlation_structure_and_grid_heterogeneity():
    """R1 / F2: the hurdle depends only on T and raw N — not on how correlated the trials are,
    nor on genuine Sharpe differences across the grid (which inflated V_cross)."""
    rng = np.random.default_rng(41)
    T = 2340
    common = rng.normal(0, 0.006, (T, 1))
    corr = common + rng.normal(0, 0.003, (T, 40))
    hetero = rng.normal(0, 0.006, (T, 40)) + np.linspace(0, 0.0006, 40)
    a = S.dsr_from_matrix(corr, 0)
    b = S.dsr_from_matrix(hetero, 39)
    assert a.sr0 == pytest.approx(b.sr0) == pytest.approx(S.expected_max_sharpe(40, 1 / (T - 1)))
    assert b.var_sr > 1 / (T - 1)            # V_cross inflated by the planted gradient …
    assert b.dsr > b.dsr_raw_cross           # … which used to deflate a real edge (F2)


def test_dsr_rejects_annualised_sharpe():
    """m4 regression: an annualised Sharpe passed as per-period must raise."""
    with pytest.raises(ValueError, match="annualised"):
        S.dsr(sr=2.0, n=2340, var_sr=1e-3, n_eff=10)
    with pytest.raises(ValueError):
        S.dsr(sr=-1.5, n=2340, var_sr=1e-3, n_eff=10)
    assert 0.0 <= S.dsr(sr=0.1, n=2340, var_sr=1e-3, n_eff=10) <= 1.0


# ------------------------------------------------------------------ PBO
def test_pbo_noise_is_about_half():
    vals = []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        vals.append(S.pbo_cscv(rng.normal(0, 0.01, (1600, 20)), n_splits=10).pbo)
    assert 0.35 < np.mean(vals) < 0.65


def test_pbo_dominant_strategy_is_zero():
    rng = np.random.default_rng(5)
    m = rng.normal(0, 0.01, (1600, 20))
    m[:, 3] += 0.004
    r = S.pbo_cscv(m, n_splits=16)
    assert r.pbo == 0.0
    assert r.n_combos == math.comb(16, 8)
    assert r.prob_oos_loss == 0.0


def test_pbo_regime_flip_is_high():
    rng = np.random.default_rng(6)
    t, n = 1600, 20
    m = rng.normal(0, 0.01, (t, n))
    blocks = np.arange(t) // (t // 16)
    sign = np.where(blocks % 2 == 0, 1.0, -1.0)
    m[:, :10] += 0.002 * sign[:, None]
    m[:, 10:] -= 0.002 * sign[:, None]
    assert S.pbo_cscv(m, n_splits=16).pbo > 0.9


def test_pbo_callable_metric_matches_vectorised():
    rng = np.random.default_rng(7)
    m = rng.normal(0, 0.01, (400, 6))
    a = S.pbo_cscv(m, n_splits=8, metric="mean")
    b = S.pbo_cscv(m, n_splits=8, metric=lambda x: x.mean(axis=0))
    assert a.pbo == b.pbo
    np.testing.assert_allclose(a.logits, b.logits)


# ------------------------------------------------------------------ bootstrap
def test_stationary_bootstrap_indices_structure():
    rng = np.random.default_rng(8)
    idx = S.stationary_bootstrap_indices(500, 10.0, 400, rng)
    assert idx.shape == (400, 500) and idx.min() >= 0 and idx.max() < 500
    cont = (np.diff(idx, axis=1) == 1) | ((idx[:, :-1] == 499) & (idx[:, 1:] == 0))
    assert 1 - cont.mean() == pytest.approx(0.1, abs=0.01)   # new-block rate ≈ 1/mean_block


def test_stationary_bootstrap_preserves_mean_and_short_lag_acf():
    rng = np.random.default_rng(9)
    n, phi = 3000, 0.5
    e = rng.normal(0, 1, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    x += 0.3
    idx = S.stationary_bootstrap_indices(n, 40.0, 300, rng)
    xb = x[idx]
    assert xb.mean() == pytest.approx(x.mean(), abs=0.02)
    xc = xb - xb.mean(axis=1, keepdims=True)
    lag1 = (xc[:, 1:] * xc[:, :-1]).sum(1) / (xc ** 2).sum(1)
    emp = np.corrcoef(x[1:], x[:-1])[0, 1]
    assert lag1.mean() == pytest.approx(emp * (1 - 1 / 40), abs=0.05)


def test_optimal_block_length_grows_with_dependence():
    rng = np.random.default_rng(10)
    iid = rng.normal(0, 1, 3000)
    ar = np.empty(3000)
    ar[0] = 0
    e = rng.normal(0, 1, 3000)
    for t in range(1, 3000):
        ar[t] = 0.7 * ar[t - 1] + e[t]
    assert S.optimal_block_length(iid) < 3.0
    assert S.optimal_block_length(ar) > 5.0
    # volatility clustering (dependence only in r²) also lengthens the block
    h = np.empty(3000)
    h[0] = 1e-4
    r = np.empty(3000)
    z = rng.normal(0, 1, 3000)
    for t in range(3000):
        if t:
            h[t] = 1e-6 + 0.1 * r[t - 1] ** 2 + 0.88 * h[t - 1]
        r[t] = math.sqrt(h[t]) * z[t]
    assert S.optimal_block_length(r) > S._pw_single(r)


def test_bootstrap_stats_shapes_and_sharpe_centre():
    rng = np.random.default_rng(11)
    r = rng.normal(0.0005, 0.01, 1500)
    out = S.bootstrap_stats(r, n_boot=600, rng=rng, chunk=250)
    assert out["sharpe_pp"].shape == (600,) and (out["max_dd"] <= 0).all()
    assert np.median(out["sharpe_pp"]) == pytest.approx(S.sharpe_per_period(r), abs=0.01)


# ------------------------------------------------------------------ return at DD budget
def test_return_at_dd_budget_scale_consistency():
    rng = np.random.default_rng(12)
    r = rng.standard_t(4, 1500) * 0.006 + 0.0004
    a = S.return_at_dd_budget(r, n_boot=500, rng=np.random.default_rng(0))
    b = S.return_at_dd_budget(r / 2, n_boot=500, rng=np.random.default_rng(0))
    assert b.leverage == pytest.approx(2 * a.leverage, rel=1e-5)
    assert b.mean_monthly == pytest.approx(a.mean_monthly, rel=1e-5)
    # at k the bootstrapped q95 |maxDD| equals the budget
    idx = S.stationary_bootstrap_indices(r.size, a.mean_block, 500, np.random.default_rng(0))
    q = np.quantile(-S._rows_maxdd(a.leverage * r[idx]), 0.95)
    assert q == pytest.approx(0.10, rel=1e-4)


def test_return_at_dd_budget_with_dates_and_label():
    rng = np.random.default_rng(13)
    d = _daily(rng.normal(0.001, 0.005, 1300))
    res = S.return_at_dd_budget(d, n_boot=400)
    assert res.mean_monthly_band[0] <= res.mean_monthly_band[1] <= res.mean_monthly_band[2]
    assert res.label == S.classify_monthly_return(res.mean_monthly)
    assert S.classify_monthly_return(0.12) == "profitable"
    assert S.classify_monthly_return(0.08) == "reasonable"
    assert S.classify_monthly_return(0.03) == "unprofitable"


# ------------------------------------------------------------------ SPA / RC / BH
def _spa_mc(mu_best: float, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    rej_spa = rej_rc = 0
    for _ in range(reps):
        d = rng.normal(0, 1, (300, 5))
        d[:, 0] += mu_best
        p1 = S.spa_test(d, n_boot=300, mean_block=1.0, rng=rng)["p_consistent"]
        p2 = S.white_reality_check(d, n_boot=300, mean_block=1.0, rng=rng)["p_value"]
        rej_spa += p1 <= 0.10
        rej_rc += p2 <= 0.10
    return rej_spa / reps, rej_rc / reps


def test_spa_and_rc_size_under_null():
    spa, rc = _spa_mc(0.0, 120, 14)
    assert 0.02 <= spa <= 0.20
    assert 0.02 <= rc <= 0.20


def test_spa_and_rc_power_under_alternative():
    spa, rc = _spa_mc(0.25, 40, 15)
    assert spa >= 0.85 and rc >= 0.8


def test_spa_benchmark_and_ordering_of_pvalues():
    rng = np.random.default_rng(16)
    perf = rng.normal(0.0, 1, (400, 4))
    bm = rng.normal(0.0, 1, 400)
    out = S.spa_test(perf, bm, n_boot=300, rng=rng)
    assert out["p_lower"] <= out["p_consistent"] <= out["p_upper"]


def test_bh_known_answer():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    res = S.bh_fdr(p, q=0.05)
    assert res["reject"].tolist() == [True, True, True, True, False]
    np.testing.assert_allclose(res["p_adj"], [0.025, 0.05, 0.05, 0.025, 0.2])
    res = S.bh_fdr([0.02, 0.5, 0.9], q=0.05)
    assert res["reject"].tolist() == [False, False, False]


# ------------------------------------------------------------------ stability / regimes
def test_time_stability_known_answer():
    rows = []
    for y, pnl in zip(range(2016, 2021), [0.10, -0.05, 0.20, 0.05, 0.20]):
        days = [date(y, 1, 1) + timedelta(days=i) for i in range(100)]
        rows += [(d, pnl / 100) for d in days]
    df = pl.DataFrame(rows, schema=["date", "ret"], orient="row")
    ts = S.time_stability(df)
    assert ts["pos_year_share"] == pytest.approx(0.8)
    assert ts["max_year_share"] == pytest.approx(0.2 / 0.5)
    assert ts["n_years"] == 5
    neg = df.with_columns(pl.col("ret") * -1)
    assert S.time_stability(neg)["max_year_share"] == math.inf


def test_volatility_regimes_are_causal():
    rng = np.random.default_rng(17)
    r = rng.normal(0, 0.01, 800) * np.repeat(rng.uniform(0.5, 2, 8), 100)
    full = S.volatility_regimes(r, window=21, min_periods=100)
    part = S.volatility_regimes(r[:500], window=21, min_periods=100)
    assert (full[:500] == part).all()
    assert set(v for v in full if v is not None) == {"low_vol", "mid_vol", "high_vol"}
    close = np.cumsum(rng.normal(0, 1, 800)) + 100
    tf = S.trend_regimes(close, window=21, min_periods=100)
    tp = S.trend_regimes(close[:500], window=21, min_periods=100)
    assert (tf[:500] == tp).all()


def test_regime_split_partitions_pnl():
    rng = np.random.default_rng(18)
    d = _daily(rng.normal(0.0005, 0.01, 600))
    labels = np.where(np.arange(600) < 300, "a", "b")
    out = S.regime_split(d, labels)
    assert out["n_days"].sum() == 600
    assert out["pnl"].sum() == pytest.approx(d["ret"].sum())
    assert out["share_of_pnl"].sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ decay
def test_cusum_threshold_arl_monte_carlo():
    k, arl0 = 0.25, 500
    h = S.cusum_threshold(k, arl0)
    rng = np.random.default_rng(19)
    runs = []
    for _ in range(150):
        cur, t = 0.0, 0
        z = rng.normal(0, 1, 20000)
        while t < 20000:
            cur = max(0.0, cur - z[t] - k)
            t += 1
            if cur > h:
                break
        runs.append(t)
    assert 350 < np.mean(runs) < 700


def test_cusum_detects_decay_and_rarely_false_alarms():
    mu, sd = 0.0008, 0.01          # ≈ SR 1.3 annualised
    rng = np.random.default_rng(20)
    false_alarms = sum(S.cusum_decay(rng.normal(mu, sd, 260), mu, sd)["alarm"] for _ in range(200))
    assert false_alarms / 200 < 0.2
    detected = sum(S.cusum_decay(rng.normal(-mu, sd, 2600), mu, sd)["alarm"] for _ in range(50))
    assert detected / 50 > 0.9
    big = S.cusum_decay(rng.normal(-0.01, sd, 500), mu, sd)
    assert big["alarm"] and big["alarm_index"] < 100


def test_sequential_sharpe_test():
    rng = np.random.default_rng(21)
    sd = 0.01
    mu = 2.0 / math.sqrt(260) * sd
    dead = [S.sequential_sharpe_test(rng.normal(0, sd, 5000), 2.0, sd)["decision"] for _ in range(40)]
    alive = [S.sequential_sharpe_test(rng.normal(mu, sd, 5000), 2.0, sd)["decision"] for _ in range(40)]
    assert dead.count("decayed") / 40 >= 0.7
    assert alive.count("decayed") / 40 <= 0.1


# ------------------------------------------------------------------ holdout band
class _MiniStudy:
    def __init__(self, paths):
        rows = []
        for pid, p in enumerate(paths):
            d = _daily(p)
            rows.append(d.with_columns(pl.lit(pid).alias("path_id")))
        self.cpcv_paths = pl.concat(rows).select("date", "path_id", "ret")
        self.wfo_oos = pl.DataFrame()


def test_holdout_band_is_jointly_calibrated():
    """M3 / R4: one common tail level so that ≈ 90 % of holdouts from the same process pass
    all four criteria at once (the v1.1 marginal p10/p10/p95/95 % band passed ≈ 78 %)."""
    rng = np.random.default_rng(22)
    mu, sd, rate = 0.0008, 0.008, 0.4
    study = _MiniStudy([rng.normal(mu, sd, 2000) for _ in range(5)])
    band = S.holdout_band(study, 260, 260.0, n_boot=2000, trades_per_day=rate, n_power=300)
    assert band.sharpe_lo < band.sharpe_median
    assert band.trades_lo < rate * 260 < band.trades_hi
    assert band.max_dd_mag_hi > 0 and band.leverage > 0
    assert 0.0 < band.tail_level < 0.10 and band.joint_coverage == pytest.approx(0.90, abs=0.01)
    passes = [S.holdout_check(band, rng.normal(mu, sd, 260), rng.poisson(rate * 260))["pass"] for _ in range(400)]
    assert 0.84 < np.mean(passes) < 0.96
    assert 0.0 <= band.p_pass_zero_edge <= 1.0 and band.n_power == 300


def test_joint_tail_level_is_stricter_than_marginals():
    rng = np.random.default_rng(5)
    sh, rab, dd = rng.normal(size=(3, 5000))
    t = rng.normal(100, 10, 5000)
    a, cov = S.joint_tail_level(sh, rab, dd, t, 0.90)
    assert a < 0.10 and cov == pytest.approx(0.90, abs=0.005)
    a1, _ = S.joint_tail_level(sh, rab, dd, None, 0.90)
    assert a1 > a                                 # fewer criteria → wider tail per criterion


def test_holdout_band_power_against_zero_edge():
    """M3: power is computed by running de-meaned bootstrap draws through holdout_check."""
    rng = np.random.default_rng(24)
    strong = _MiniStudy([rng.normal(0.0025, 0.008, 2000)])      # SR ≈ 5
    weak = _MiniStudy([rng.normal(0.0002, 0.008, 2000)])        # SR ≈ 0.4
    bs = S.holdout_band(strong, 260, n_boot=1000, n_power=400)
    bw = S.holdout_band(weak, 260, n_boot=1000, n_power=400)
    assert bs.p_pass_zero_edge < 0.05 and bs.decisive is True
    assert bw.p_pass_zero_edge > 0.30 and bw.decisive is False


def test_holdout_band_trade_series():
    rng = np.random.default_rng(23)
    study = _MiniStudy([rng.normal(0.0005, 0.01, 1000)])
    tpd = rng.poisson(0.5, 1000).astype(float)
    band = S.holdout_band(study, 260, n_boot=500, trades_per_day=tpd)
    assert band.trades_lo < 130 < band.trades_hi
    assert band.trades_joint is True              # one path, same length → same bootstrap indices
    two = _MiniStudy([rng.normal(0.0005, 0.01, 1000), rng.normal(0.0005, 0.01, 1000)])
    assert S.holdout_band(two, 260, n_boot=500, trades_per_day=tpd, n_power=0).trades_joint is False


def test_holdout_check_accepts_v11_band_keys():
    old = {"sharpe_p10": 0.0, "ret_at_budget_p10": -1.0, "max_dd_mag_p95": 0.5, "trades_lo": 10,
           "trades_hi": 20, "leverage": 1.0}
    rng = np.random.default_rng(1)
    out = S.holdout_check(old, rng.normal(0.001, 0.005, 260), 15)
    assert out["pass"] is True
# ------------------------------------------------------------------ component nulls
def test_random_selectivity_null_draws_exact_selectivity():
    rng = np.random.default_rng(24)
    signal_rets = rng.normal(0.001, 0.01, 400)
    seen = []

    def evaluate(mask):
        seen.append(mask.sum())
        return float(signal_rets[mask].mean() / signal_rets[mask].std() * math.sqrt(260))

    null = S.random_selectivity_null(evaluate, 0.25, 400, 50, rng)
    assert null.shape == (50,) and set(seen) == {100}
    top = np.isin(np.arange(400), np.argsort(-signal_rets)[:100])   # a filter with foresight
    assert S.empirical_pvalue(evaluate(top), null) < 0.05


def test_random_entry_null_non_overlapping_with_same_holds():
    rng = np.random.default_rng(25)
    holds = rng.integers(1, 20, 30)
    calls = []

    def evaluate(e, h, sides):
        calls.append((e, h, sides))
        return 0.0

    S.random_entry_null(evaluate, 1000, holds, 20, rng, sides=np.r_[np.ones(15), -np.ones(15)])
    for e, h, sides in calls:
        assert (np.diff(e) >= h[:-1]).all() and e[0] >= 0 and e[-1] + h[-1] <= 1000
        assert sorted(h) == sorted(holds)
        assert sides.sum() == 0


def test_ablation_compare_rows():
    rng = np.random.default_rng(26)
    on = rng.normal(0.0005, 0.006, 800)
    off = rng.normal(0.0005, 0.012, 800)
    out = S.ablation_compare(on, off, n_boot=300)
    assert out["rule"].to_list() == ["on", "off"]
    assert out["leverage_at_budget"][0] > out["leverage_at_budget"][1]
