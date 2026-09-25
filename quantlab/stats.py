"""Statistical machinery for validation (DESIGN §4, §5): the judge's toolbox.

Units — read this before calling anything
-----------------------------------------
* **Per-period Sharpe** (``sr``) = mean / std of the *daily* simple returns (ddof=1), NOT
  annualised.  Every formula from Bailey & López de Prado (PSR, DSR, MinTRL, expected max
  Sharpe) is stated in per-period units, and all functions here take/return per-period
  Sharpe unless the argument name ends in ``_annual``.
* **Annualised Sharpe** = per-period Sharpe × √periods_per_year (FX 260, crypto 365,
  B3 252 — ``metrics.PERIODS_PER_YEAR``).  Never computed from per-trade statistics.
* **Kurtosis is raw** (normal = 3), as returned by ``metrics.skew_kurt``.
* Daily returns follow the contracts convention: simple returns of equity on the book's
  nominal account (``date``/``ret`` frames, or wide ``date`` + one column per trial).
* Drawdowns are negative fractions (−0.12 = 12 % below the peak), except where a name ends
  in ``_mag`` (positive magnitude).

References
----------
[BLdP12]  Bailey, López de Prado (2012) "The Sharpe Ratio Efficient Frontier", J. Risk 15(2).
[BLdP14]  Bailey, López de Prado (2014) "The Deflated Sharpe Ratio", J. Portfolio Mgmt 40(5).
[BBLdPZ17] Bailey, Borwein, López de Prado, Zhu (2017) "The Probability of Backtest
          Overfitting", J. Computational Finance 20(4).
[LdP19]   López de Prado (2019) "A Data Science Solution to the Multiple-Testing Crisis in
          Financial Research", J. Financial Data Science 1(1)  (ONC clustering, K trials).
[LJ05]    Li, Ji (2005) "Adjusting multiple testing in multilocus analyses using the
          eigenvalues of a correlation matrix", Heredity 95.
[PR94]    Politis, Romano (1994) "The Stationary Bootstrap", JASA 89.
[PW04]    Politis, White (2004) "Automatic Block-Length Selection for the Dependent
          Bootstrap", Econometric Reviews 23;  corrected by Patton, Politis, White (2009).
[W00]     White (2000) "A Reality Check for Data Snooping", Econometrica 68.
[H05]     Hansen (2005) "A Test for Superior Predictive Ability", JBES 23.
[BH95]    Benjamini, Hochberg (1995) "Controlling the False Discovery Rate", JRSS-B 57.
[P54]     Page (1954) "Continuous Inspection Schemes", Biometrika 41;  Siegmund (1985)
          ARL approximation.
[W45]     Wald (1945) "Sequential Tests of Statistical Hypotheses", Ann. Math. Stat. 16.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import polars as pl
from scipy import stats as sps
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform

from . import metrics

EULER_GAMMA = 0.5772156649015329

# DESIGN §4.4 (decided 2026-09-24): a band whose zero-edge pass probability exceeds this cannot
# tell a real edge from a dead one; an all-criteria pass under it is NOT_DECISIVE.  Read-only.
DECISIVE_MAX_ZERO_EDGE_PASS = 0.30
HOLDOUT_STATUSES = ("PASS", "FAIL", "NOT_DECISIVE")

# Red-team R2-3: the registered holdout band has ONE construction — these bootstrap sizes and a
# seed derived from the study id (holdout_band_seed).  Read-only: the ledger refuses a band built
# otherwise (ledger.check_band_construction), so a band cannot be re-rolled or thinned.
HOLDOUT_BAND_N_BOOT = 2000
HOLDOUT_BAND_N_POWER = 1000
# The realised holdout series must span the band's horizon within ±max(5 days, 2 %) (H6).
HOLDOUT_SPAN_TOL_DAYS = 5.0
HOLDOUT_SPAN_TOL_FRAC = 0.02


def holdout_band_seed(study_id: str) -> int:
    """The fixed bootstrap seed of a study's holdout band: a stable hash of its id (R2-3)."""
    import hashlib
    return int(hashlib.sha256(f"holdout-band|{study_id}".encode()).hexdigest()[:8], 16)

__all__ = [
    "sharpe_per_period", "psr", "expected_max_sharpe", "dsr", "dsr_from_matrix", "DSRResult",
    "effective_n_trials", "min_trl", "power_check", "pbo_cscv", "PBOResult",
    "optimal_block_length", "stationary_bootstrap_indices", "bootstrap_stats", "VEC_STATS",
    "return_at_dd_budget", "BudgetResult", "classify_monthly_return",
    "spa_test", "white_reality_check", "bh_fdr", "time_stability", "regime_split",
    "volatility_regimes", "trend_regimes", "cusum_threshold", "cusum_decay",
    "sequential_sharpe_test", "holdout_band", "HoldoutBand", "holdout_check", "joint_tail_level",
    "holdout_status", "DECISIVE_MAX_ZERO_EDGE_PASS", "HOLDOUT_STATUSES",
    "HOLDOUT_BAND_N_BOOT", "HOLDOUT_BAND_N_POWER", "holdout_band_seed",
    "random_selectivity_null", "random_entry_null", "empirical_pvalue", "ablation_compare",
]


# =========================================================================== helpers
def _ret_array(r) -> np.ndarray:
    """1-D float array of finite returns from a Series / frame(date, ret) / array."""
    return metrics.as_array(r)


def _matrix(returns) -> tuple[np.ndarray, list[str]]:
    """(T×N float matrix, column names) from a wide frame (``date`` + trial columns) or array.

    Nulls/NaNs are filled with 0 (a trial with no row that day was flat)."""
    if isinstance(returns, pl.DataFrame):
        cols = [c for c in returns.columns if c != "date"]
        m = returns.select(cols).to_numpy().astype(float) if cols else np.empty((returns.height, 0))
    else:
        m = np.asarray(returns, dtype=float)
        if m.ndim == 1:
            m = m[:, None]
        cols = [f"c{i}" for i in range(m.shape[1])]
    return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0), cols


def sharpe_per_period(x, axis: int = 0) -> np.ndarray | float:
    """Per-period (non-annualised) Sharpe = mean/std(ddof=1) along ``axis``; NaN if std == 0."""
    a = np.asarray(x, dtype=float)
    mu = a.mean(axis=axis)
    sd = a.std(axis=axis, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / np.where(sd > 0, sd, 1.0), np.nan)
    return float(out) if np.ndim(out) == 0 else out


def _rows_sharpe(x: np.ndarray) -> np.ndarray:
    return sharpe_per_period(x, axis=1)


def _rows_maxdd(x: np.ndarray) -> np.ndarray:
    """Max drawdown (≤ 0) of each row's compounded equity (start at 1, peak includes start).
    Rows with any return ≤ −1 are ruined → −1."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lg = np.log1p(np.maximum(x, -1.0 + 1e-15))
    cum = np.cumsum(lg, axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros((x.shape[0], 1)), cum], axis=1), axis=1)[:, 1:]
    dd = np.expm1(np.min(cum - peak, axis=1))
    dd = np.minimum(dd, 0.0)
    ruined = (x <= -1.0).any(axis=1)
    dd[ruined] = -1.0
    return dd


def _rows_total_return(x: np.ndarray) -> np.ndarray:
    return np.expm1(np.log1p(np.maximum(x, -1.0 + 1e-15)).sum(axis=1))


def _rows_mean_monthly(x: np.ndarray, days_per_month: int) -> np.ndarray:
    """Mean simple monthly return of each row, months = consecutive blocks of ``days_per_month``
    days (the tail remainder is dropped unless the row is shorter than one month, in which case
    the whole row is one month rescaled geometrically)."""
    n = x.shape[1]
    m = n // days_per_month
    lg = np.log1p(np.maximum(x, -1.0 + 1e-15))
    if m == 0:
        return np.expm1(lg.sum(axis=1) * days_per_month / max(n, 1))
    blocks = lg[:, : m * days_per_month].reshape(x.shape[0], m, days_per_month).sum(axis=2)
    return np.expm1(blocks).mean(axis=1)


VEC_STATS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sharpe_pp": _rows_sharpe,          # per-period Sharpe of each bootstrap row
    "max_dd": _rows_maxdd,              # ≤ 0
    "total_return": _rows_total_return,
    "mean": lambda x: x.mean(axis=1),
}


# =========================================================================== PSR / DSR / MinTRL
def _sr_denominator(sr, skew, kurt):
    """√(1 − γ3·SR + (γ4 − 1)/4·SR²) — the non-normal SR standard-error factor [BLdP12 eq. 3],
    with γ4 the RAW kurtosis.  Floored to avoid a negative radicand for extreme inputs."""
    v = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    return np.sqrt(np.maximum(v, 1e-12))


def psr(sr: float, sr_benchmark: float, n: float, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio [BLdP12]: P(true SR > sr_benchmark | observed SR over n periods).

    ``sr``/``sr_benchmark`` are **per-period** Sharpe ratios; ``kurt`` is raw (normal = 3).

        PSR = Φ( (SR − SR*)·√(n − 1) / √(1 − γ3·SR + (γ4 − 1)/4·SR²) )
    """
    if n <= 1 or not np.isfinite(sr):
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1.0) / float(_sr_denominator(sr, skew, kurt))
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_eff: float, var_sr: float) -> float:
    """False-strategy theorem [BLdP14 eq. 5 / Bailey et al. 2014]: expected maximum of n_eff
    independent Sharpe estimates with true SR 0 and cross-sectional variance ``var_sr``:

        E[max SR] ≈ √V · ( (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ),  γ = Euler–Mascheroni.

    Units follow ``var_sr`` (per-period variance → per-period SR).  Returns 0 for N ≤ 1."""
    if not np.isfinite(n_eff) or n_eff <= 1.0 or var_sr <= 0 or not np.isfinite(var_sr):
        return 0.0
    q1 = sps.norm.ppf(1.0 - 1.0 / n_eff)
    q2 = sps.norm.ppf(1.0 - 1.0 / (n_eff * math.e))
    return float(max(0.0, math.sqrt(var_sr) * ((1.0 - EULER_GAMMA) * q1 + EULER_GAMMA * q2)))


def dsr(selected_returns=None, *, trial_sharpes: Sequence[float] | None = None,
        var_sr: float | None = None, n_eff: float, sr: float | None = None, n: int | None = None,
        skew: float | None = None, kurt: float | None = None) -> float:
    """Deflated Sharpe Ratio [BLdP14]: PSR of the selected strategy against the expected
    maximum Sharpe of ``n_eff`` independent trials.  Returns a **probability**.

    Either give ``selected_returns`` (daily returns; SR/n/skew/kurt are estimated from it), or
    the moments directly (``sr`` per-period, ``n``, ``skew``, raw ``kurt``).  The Sharpe
    variance comes from ``var_sr`` (per-period units) or is computed from ``trial_sharpes``
    (per-period, ddof=1).  Raises ``ValueError`` if |sr| > 1 (an annualised Sharpe passed by
    mistake; a daily Sharpe of 1 is ≈ 16 annualised)."""
    if selected_returns is not None:
        a = _ret_array(selected_returns)
        sr = sharpe_per_period(a) if sr is None else sr
        n = a.size if n is None else n
        sk, ku = metrics.skew_kurt(a)
        skew = sk if skew is None else skew
        kurt = ku if kurt is None else kurt
    if sr is None or n is None:
        raise ValueError("dsr needs selected_returns or (sr, n)")
    if np.isfinite(sr) and abs(sr) > 1.0:
        raise ValueError(f"dsr: |per-period Sharpe| = {abs(sr):.3g} > 1 — looks annualised; pass the "
                         "per-period (daily) Sharpe (annual / sqrt(periods_per_year))")
    if var_sr is None:
        if trial_sharpes is None:
            raise ValueError("dsr needs var_sr or trial_sharpes")
        ts = np.asarray(trial_sharpes, dtype=float)
        ts = ts[np.isfinite(ts)]
        var_sr = float(ts.var(ddof=1)) if ts.size > 1 else 0.0
    sr0 = expected_max_sharpe(n_eff, var_sr)
    return psr(sr, sr0, n, 0.0 if skew is None or not np.isfinite(skew) else skew,
               3.0 if kurt is None or not np.isfinite(kurt) else kurt)


# --------------------------------------------------------------------------- effective N
def _usable(m: np.ndarray) -> np.ndarray:
    """Columns with non-zero variance (flat / failed trials carry no information)."""
    return m.std(axis=0) > 0


def effective_n_trials(returns_matrix, method: str = "eigen", *, corr_threshold: float = 0.5,
                       return_labels: bool = False):
    """Effective number of independent trials from the trial return correlation structure.

    methods
    -------
    ``"eigen"``   Participation ratio of the correlation-matrix eigenvalues,
                  N_eff = (Σλ)² / Σλ² = N² / ‖C‖_F².  Identical trials → 1; independent → N
                  (× 1/(1 + N/T) finite-sample shrinkage from the Marchenko–Pastur spread).
                  Dominated by the leading eigenvalue: with equicorrelation ρ it gives
                  ≈ 1/ρ², i.e. it can be *lenient* for large grids with a common factor.
    ``"liji"``    Li & Ji (2005) [LJ05]: Σ_i [ 1{|λ_i| ≥ 1} + (|λ_i| − ⌊|λ_i|⌋) ].
                  Identical → 1; independent → ≈ N; equicorrelation ρ → ≈ 1 + (1 − ρ)(N − 1).
    ``"cluster"`` ONC-style count [LdP19], simplified to be deterministic: average-linkage
                  hierarchical clustering on the correlation distance d = √(½(1 − ρ)), cut
                  where the average within-cluster correlation drops below ``corr_threshold``
                  (default 0.5 → "two trials are the same bet if ρ ≥ 0.5 on average").
                  The number of clusters is K.  LdP19 picks K by silhouette t-stat with
                  k-means; the fixed threshold is used here because the silhouette optimum
                  is undefined in the two limits we must get right (all identical, all
                  independent).

    Flat columns (zero variance) are dropped first.  Returns a float (and cluster labels for
    the usable columns if ``return_labels`` and method == "cluster")."""
    m, _ = _matrix(returns_matrix)
    m = m[:, _usable(m)]
    n = m.shape[1]
    if n <= 1:
        return (float(n), np.zeros(n, dtype=int)) if return_labels else float(n)
    c = np.corrcoef(m, rowvar=False)
    c = np.nan_to_num((c + c.T) / 2.0, nan=0.0)
    np.fill_diagonal(c, 1.0)
    if method == "eigen":
        out = float(n ** 2 / np.sum(c ** 2))
    elif method == "liji":
        lam = np.round(np.abs(np.linalg.eigvalsh(c)), 6)   # kill float noise near integers
        out = float(np.sum((lam >= 1.0).astype(float) + (lam - np.floor(lam))))
        out = min(max(out, 1.0), float(n))
    elif method == "cluster":
        d = np.sqrt(np.clip(0.5 * (1.0 - c), 0.0, None))
        np.fill_diagonal(d, 0.0)
        z = sch.linkage(squareform(d, checks=False), method="average")
        t = math.sqrt(0.5 * (1.0 - corr_threshold))
        labels = sch.fcluster(z, t=t, criterion="distance")
        out = float(len(np.unique(labels)))
        if return_labels:
            return out, labels
    else:
        raise ValueError(f"unknown method {method!r}")
    return out


@dataclass
class DSRResult:
    """DSR gate calculation (DESIGN §4.2 v1.2, R1) plus the effective-N diagnostics.

    Gate: SR0 = √V0 · E[max of N iid N(0,1)], V0 = 1/(T − 1) (null sampling variance of a
    per-period Sharpe estimate, normal-moment approximation), N = raw trials of this study +
    raw trials of earlier attempts.  DSR = PSR(SR | SR0) with the selected series' skew/kurt.

    Why raw N with V0 (red-team B1): under a common factor ρ the max of N correlated null
    Sharpes is √ρ·Z0 + √(1−ρ)·max_N(iid); V0·E[max_N] upper-bounds its mean, whereas an
    effective-N estimate (≈1/ρ² or 1 cluster) *and* the cross-sectional variance both shrink
    with ρ and double-count the correlation.  Using the cross-sectional variance also lets
    genuine Sharpe differences across the grid inflate the hurdle (calibration F2).  The
    eigen / cluster / Li–Ji N_eff and V_cross are therefore reported as diagnostics only."""

    dsr: float                 # gate value (probability): hurdle from V0 and raw N
    sr0: float                 # per-period hurdle √V0 · E[max of N]
    sr0_annual: float
    v0: float                  # 1/(T − 1)
    n_trials: float            # N used = n_trials_study + n_trials_prior
    n_trials_study: float
    n_trials_prior: float
    psr0: float                # PSR against SR* = 0 (no deflation)
    sr: float                  # per-period Sharpe of the selected trial
    sr_annual: float
    n_obs: int
    skew: float
    kurt: float
    # ---- diagnostics (not the gate)
    var_sr: float              # cross-sectional per-period Sharpe variance of the usable trials
    n_eff: float               # max of eigen/cluster N_eff of this study's matrix (no prior)
    n_eff_by_method: dict[str, float]
    dsr_neff_cross: float      # old v1.1 gate: N_eff + prior raw trials, with V_cross
    dsr_raw_cross: float       # raw N with V_cross
    sr0_neff_cross_annual: float
    n_columns: int             # trial return columns in the matrix

    @property
    def dsr_raw_n(self) -> float:   # backward-compatible alias: the gate now uses raw N
        return self.dsr

    @property
    def n_trials_raw(self) -> float:
        return self.n_trials

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dsr_raw_n"] = self.dsr
        d["n_trials_raw"] = self.n_trials
        return d


def dsr_from_matrix(returns_matrix, selected: str | int, *, periods_per_year: float = 260.0,
                    n_trials: float | None = None, extra_trials: float = 0.0,
                    methods: Sequence[str] = ("eigen", "cluster", "liji"),
                    corr_threshold: float = 0.5) -> DSRResult:
    """Everything-in-one DSR from a wide trial return matrix (DESIGN §4.2 v1.2, R1).

    * ``selected``: column name (e.g. ``"t17"``) or column index of the selected trial.
    * ``n_trials``: raw trials of *this* study (default: number of trial columns); pass the
      study's evaluated-trial count when some trials have no return column (errors).
    * ``extra_trials``: raw trials of earlier attempts of the same system (from the ledger).
    * Gate hurdle SR0 = √(1/(T−1)) · E[max of ``n_trials + extra_trials``] (see
      :class:`DSRResult`).  The eigen/cluster/Li–Ji effective N and the cross-sectional Sharpe
      variance are computed for the diagnostics only."""
    m, cols = _matrix(returns_matrix)
    j = cols.index(selected) if isinstance(selected, str) else int(selected)
    sel = m[:, j]
    use = _usable(m)
    srs = sharpe_per_period(m[:, use], axis=0)
    srs = np.atleast_1d(srs)
    var_sr = float(np.var(srs, ddof=1)) if srs.size > 1 else 0.0
    by = {meth: effective_n_trials(m, meth, corr_threshold=corr_threshold) for meth in methods}
    n_study = float(m.shape[1] if n_trials is None else n_trials)
    n_prior = float(extra_trials)
    n_tot = n_study + n_prior
    sr = float(sharpe_per_period(sel))
    sk, ku = metrics.skew_kurt(sel)
    sk = 0.0 if not np.isfinite(sk) else float(sk)
    ku = 3.0 if not np.isfinite(ku) else float(ku)
    t = int(sel.size)
    v0 = 1.0 / (t - 1) if t > 1 else float("nan")
    sr0 = expected_max_sharpe(n_tot, v0)
    ann = math.sqrt(periods_per_year)
    lenient = [by[k] for k in ("eigen", "cluster") if k in by]
    ne = max(lenient) if lenient else float(use.sum())
    sr0_nc = expected_max_sharpe(ne + n_prior, var_sr)
    return DSRResult(
        dsr=psr(sr, sr0, t, sk, ku), sr0=sr0, sr0_annual=sr0 * ann, v0=v0, n_trials=n_tot,
        n_trials_study=n_study, n_trials_prior=n_prior, psr0=psr(sr, 0.0, t, sk, ku), sr=sr,
        sr_annual=sr * ann, n_obs=t, skew=sk, kurt=ku, var_sr=var_sr, n_eff=ne, n_eff_by_method=by,
        dsr_neff_cross=psr(sr, sr0_nc, t, sk, ku),
        dsr_raw_cross=psr(sr, expected_max_sharpe(n_tot, var_sr), t, sk, ku),
        sr0_neff_cross_annual=sr0_nc * ann, n_columns=int(m.shape[1]))


def min_trl(sr: float, target_sr: float = 0.0, skew: float = 0.0, kurt: float = 3.0,
            prob: float = 0.95, periods_per_year: float = 260.0) -> dict[str, float]:
    """Minimum Track Record Length [BLdP12 eq. 11]: number of periods needed for
    PSR(target_sr) ≥ ``prob``.  ``sr``/``target_sr`` per-period; raw ``kurt``.

        MinTRL = 1 + (1 − γ3·SR + (γ4 − 1)/4·SR²) · (Z_prob / (SR − SR*))²

    Returns {"periods", "years"}; inf when SR ≤ SR*."""
    if not np.isfinite(sr) or sr <= target_sr:
        return {"periods": float("inf"), "years": float("inf")}
    v = float(_sr_denominator(sr, skew, kurt)) ** 2
    periods = 1.0 + v * (sps.norm.ppf(prob) / (sr - target_sr)) ** 2
    return {"periods": float(periods), "years": float(periods / periods_per_year)}


def power_check(expected_sr_annual: float, trades_per_year: float, years_available: float, *,
                periods_per_year: float = 260.0, skew: float = 0.0, kurt: float = 3.0,
                prob: float = 0.95) -> dict[str, Any]:
    """S1 power check: can the dev window reach MinTRL for the card's expected Sharpe?

    MinTRL is computed on **daily** returns (per-period SR = annual/√ppy).  Also reports the
    number of trades the dev window will contain and the trades MinTRL implies at the card's
    trade rate.  ``feasible`` False → recommend early kill (or a redesign raising trade count)."""
    sr = expected_sr_annual / math.sqrt(periods_per_year)
    mt = min_trl(sr, 0.0, skew, kurt, prob, periods_per_year)
    return {"min_trl_years": mt["years"], "min_trl_days": mt["periods"],
            "years_available": years_available, "feasible": bool(mt["years"] <= years_available),
            "expected_trades": trades_per_year * years_available,
            "trades_needed": trades_per_year * mt["years"] if np.isfinite(mt["years"]) else float("inf")}


# =========================================================================== PBO (CSCV)
@dataclass
class PBOResult:
    pbo: float                     # P(selected-IS config ranks at/below the OOS median)
    logits: np.ndarray             # λ_c per combination
    is_best_perf: np.ndarray       # IS metric of the IS-best config, per combination
    oos_perf: np.ndarray           # its OOS metric, per combination
    oos_rel_rank: np.ndarray       # ω_c ∈ (0,1)
    slope: float                   # OLS slope of OOS on IS perf (degradation)
    intercept: float
    r2: float
    prob_oos_loss: float           # P(OOS metric of the IS-best < 0)
    n_splits: int
    n_combos: int
    n_strategies: int

    def summary(self) -> dict[str, float]:
        return {"pbo": self.pbo, "slope": self.slope, "r2": self.r2,
                "prob_oos_loss": self.prob_oos_loss, "median_logit": float(np.median(self.logits)),
                "n_combos": self.n_combos, "n_strategies": self.n_strategies}


def pbo_cscv(returns_matrix, n_splits: int = 16, metric: str | Callable = "sharpe",
             max_combos: int | None = None, rng: np.random.Generator | None = None) -> PBOResult:
    """Probability of Backtest Overfitting via Combinatorially-Symmetric CV [BBLdPZ17].

    Rows (time) are split into ``n_splits`` contiguous blocks (the first T mod S rows are
    dropped so blocks are equal).  For every choice of S/2 blocks as IS (the rest OOS), the
    config with the best IS metric is found and its relative OOS rank ω = rank/(N+1) (rank 1 =
    worst) turned into a logit λ = ln(ω/(1−ω)).  PBO = P(λ ≤ 0).

    ``metric``: "sharpe" (per-period, vectorised over all C(S, S/2) combos via block sums),
    "mean", or a callable (T'×N array → N values; slow path).  ``max_combos`` randomly
    subsamples combinations (seeded ``rng``)."""
    m, _ = _matrix(returns_matrix)
    m = m[:, _usable(m)]
    t, n = m.shape
    if n < 2:
        raise ValueError("PBO needs at least 2 non-flat strategies")
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be even and ≥ 2")
    bl = t // n_splits
    if bl < 2:
        raise ValueError("too few rows for n_splits")
    m = m[t - bl * n_splits:]
    blocks = m.reshape(n_splits, bl, n)
    combos = np.array(list(combinations(range(n_splits), n_splits // 2)), dtype=int)
    if max_combos is not None and len(combos) > max_combos:
        rng = rng or np.random.default_rng(0)
        combos = combos[rng.choice(len(combos), max_combos, replace=False)]
    sel = np.zeros((len(combos), n_splits))
    sel[np.arange(len(combos))[:, None], combos] = 1.0
    if metric in ("sharpe", "mean"):
        s1 = blocks.sum(axis=1)                    # S × N
        s2 = (blocks ** 2).sum(axis=1)
        cnt = bl * n_splits / 2

        def perf(w):
            mu = (w @ s1) / cnt
            if metric == "mean":
                return mu
            var = ((w @ s2) - cnt * mu ** 2) / (cnt - 1)
            sd = np.sqrt(np.maximum(var, 0))
            return np.where(sd > 0, mu / np.where(sd > 0, sd, 1), -np.inf)

        is_perf, oos_perf = perf(sel), perf(1.0 - sel)
    else:
        is_perf = np.empty((len(combos), n))
        oos_perf = np.empty((len(combos), n))
        for i, c in enumerate(combos):
            mask = np.zeros(n_splits, bool)
            mask[c] = True
            is_perf[i] = metric(blocks[mask].reshape(-1, n))
            oos_perf[i] = metric(blocks[~mask].reshape(-1, n))
    best = np.argmax(is_perf, axis=1)
    rows = np.arange(len(combos))
    oos_best = oos_perf[rows, best]
    # rank 1 = worst OOS; ties get the average rank
    ranks = sps.rankdata(oos_perf, axis=1, method="average")[rows, best]
    omega = ranks / (n + 1.0)
    logits = np.log(omega / (1.0 - omega))
    is_best = is_perf[rows, best]
    fin = np.isfinite(is_best) & np.isfinite(oos_best)
    if fin.sum() > 2 and np.std(is_best[fin]) > 0:
        slope, intercept, r, _, _ = sps.linregress(is_best[fin], oos_best[fin])
        r2 = r ** 2
    else:
        slope = intercept = r2 = float("nan")
    return PBOResult(pbo=float(np.mean(logits <= 0)), logits=logits, is_best_perf=is_best,
                     oos_perf=oos_best, oos_rel_rank=omega, slope=float(slope),
                     intercept=float(intercept), r2=float(r2),
                     prob_oos_loss=float(np.mean(oos_best < 0)), n_splits=n_splits,
                     n_combos=len(combos), n_strategies=n)


# =========================================================================== bootstrap
def _acf(x: np.ndarray, maxlag: int) -> np.ndarray:
    x = x - x.mean()
    d = float(np.dot(x, x))
    if d == 0:
        return np.zeros(maxlag + 1)
    return np.array([1.0] + [float(np.dot(x[k:], x[:-k])) / d for k in range(1, maxlag + 1)])


def _pw_single(x: np.ndarray) -> float:
    """Politis–White (2004) / Patton–Politis–White (2009) optimal mean block for the stationary
    bootstrap: b = (2·G² / D_SB)^{1/3} · n^{1/3}, flat-top lag window, m̂ from the first run of
    K_N = max(5, √log10 n) insignificant autocorrelations at c = 2."""
    n = x.size
    if n < 10 or x.std() == 0:
        return 1.0
    kn = max(5, int(math.ceil(math.sqrt(math.log10(n)))))
    mmax = int(math.ceil(math.sqrt(n))) + kn
    mmax = min(mmax, n - 1)
    rho = _acf(x, mmax)
    thr = 2.0 * math.sqrt(math.log10(n) / n)
    insig = np.abs(rho[1:]) < thr
    mhat = None
    for m in range(0, mmax - kn + 1):
        if insig[m: m + kn].all():
            mhat = m
            break
    if mhat is None:
        sig = np.nonzero(~insig)[0]
        mhat = int(sig[-1] + 1) if sig.size else 0
    M = min(2 * max(mhat, 1), mmax)
    xc = x - x.mean()
    gamma = np.array([np.dot(xc[k:], xc[: n - k]) / n for k in range(M + 1)])
    lag = np.arange(M + 1)
    t = lag / M
    lam = np.where(t <= 0.5, 1.0, 2.0 * (1.0 - t))
    g = gamma[0] + 2.0 * np.sum(lam[1:] * gamma[1:])
    G = 2.0 * np.sum(lam[1:] * lag[1:] * gamma[1:])
    d_sb = 2.0 * g ** 2
    if d_sb <= 0:
        return 1.0
    b = (2.0 * G ** 2 / d_sb) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(np.clip(b, 1.0, max(1.0, 3.0 * math.sqrt(n))))


def optimal_block_length(x) -> float:
    """Mean block length for the stationary bootstrap: the larger of the Politis–White
    estimates [PW04 + 2009 correction] on the returns **and on their squares**.  Drawdowns
    depend on volatility clustering, which lives in r², not r; using only r would pick b ≈ 1
    for most strategies and understate drawdown tails.  Result is ≥ 1."""
    a = _ret_array(x)
    return max(1.0, _pw_single(a), _pw_single(a ** 2))


def stationary_bootstrap_indices(n: int, mean_block: float, n_boot: int,
                                 rng: np.random.Generator, length: int | None = None) -> np.ndarray:
    """Politis–Romano stationary bootstrap indices [PR94], fully vectorised.

    Returns an int array (n_boot × length) of indices into a series of length ``n`` (circular
    wrap).  Each position starts a new block with probability p = 1/mean_block (block lengths
    are geometric with mean ``mean_block``); otherwise it continues the previous block."""
    length = n if length is None else int(length)
    p = 1.0 / max(float(mean_block), 1.0)
    new = rng.random((n_boot, length)) < p
    new[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, length))
    pos = np.arange(length)
    last = np.maximum.accumulate(np.where(new, pos, 0), axis=1)
    start_val = np.take_along_axis(starts, last, axis=1)
    return (start_val + (pos - last)) % n


def bootstrap_stats(daily, fns: Mapping[str, Callable[[np.ndarray], np.ndarray]] | None = None, *,
                    n_boot: int = 2000, mean_block: float | None = None, horizon: int | None = None,
                    rng: np.random.Generator | None = None, chunk: int = 500) -> dict[str, np.ndarray]:
    """Stationary-bootstrap distributions of statistics of a daily return series.

    ``fns`` maps name → vectorised function taking a 2-D array (rows = bootstrap samples of
    length ``horizon``, default = len(daily)) and returning one value per row; defaults to
    ``VEC_STATS`` (per-period Sharpe, max DD, total return, mean).  Works in chunks to bound
    RAM.  The output also carries ``"_mean_block"``."""
    a = _ret_array(daily)
    fns = dict(VEC_STATS if fns is None else fns)
    rng = rng or np.random.default_rng(0)
    b = optimal_block_length(a) if mean_block is None else float(mean_block)
    h = a.size if horizon is None else int(horizon)
    out: dict[str, list] = {k: [] for k in fns}
    for s in range(0, n_boot, chunk):
        idx = stationary_bootstrap_indices(a.size, b, min(chunk, n_boot - s), rng, h)
        x = a[idx]
        for k, f in fns.items():
            out[k].append(np.asarray(f(x), dtype=float))
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["_mean_block"] = np.array([b])
    return res


# =========================================================================== return at DD budget
def classify_monthly_return(r_month: float) -> str:
    """DESIGN §4.5 label: > 10 %/mo profitable, 6–10 % reasonable, < 6 % unprofitable."""
    if not np.isfinite(r_month) or r_month < 0.06:
        return "unprofitable"
    return "profitable" if r_month > 0.10 else "reasonable"


@dataclass
class BudgetResult:
    leverage: float                 # k: q-quantile bootstrapped max DD of k·r equals the budget
    mean_monthly: float             # mean calendar-month return of k·r (actual series)
    mean_monthly_band: tuple[float, float, float]   # bootstrap p5 / p50 / p95
    label: str                      # DESIGN §4.5 class
    dd_budget: float
    quantile: float
    horizon_days: int
    mean_block: float
    n_boot: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dd_quantile_mag(samples: np.ndarray, k: float, q: float) -> float:
    return float(np.quantile(-_rows_maxdd(k * samples), q))


def _solve_leverage(samples: np.ndarray, budget: float, q: float) -> float:
    """Root of q-quantile(|maxDD(k·r)|) = budget in k (monotone in k; common random numbers —
    the same bootstrap samples at every k), by bracketing + Brent's method (rtol 1e-9)."""
    from scipy.optimize import brentq

    f = lambda k: _dd_quantile_mag(samples, k, q) - budget  # noqa: E731
    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        lo, hi = hi, hi * 4.0
        if hi > 1e6:
            return float("inf")
    return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-9))


def return_at_dd_budget(daily, dd_budget: float = 0.10, quantile: float = 0.95,
                        horizon_days: int | None = None, n_boot: int = 2000, *,
                        periods_per_year: float = 260.0, mean_block: float | None = None,
                        rng: np.random.Generator | None = None) -> BudgetResult:
    """DESIGN §4.5: scale returns by k so that the bootstrapped ``quantile`` of max drawdown
    (over ``horizon_days``, default the full series) equals ``dd_budget``; report the mean
    monthly return at k.

    Scaling is on simple daily returns (k·r, compounded), i.e. leverage on the %-equity curve.
    Point estimate = mean calendar-month return of k·r when ``daily`` has dates, else of
    consecutive round(ppy/12)-day blocks; band = p5/p50/p95 over bootstrap samples (blocks of
    round(ppy/12) days).  Exact scale property: halving returns doubles k."""
    a = _ret_array(daily)
    rng = rng or np.random.default_rng(0)
    b = optimal_block_length(a) if mean_block is None else float(mean_block)
    h = a.size if horizon_days is None else int(horizon_days)
    samples = a[stationary_bootstrap_indices(a.size, b, n_boot, rng, h)]
    k = _solve_leverage(samples, dd_budget, quantile)
    dpm = max(1, int(round(periods_per_year / 12.0)))
    if isinstance(daily, pl.DataFrame) and "date" in daily.columns and np.isfinite(k):
        mm = metrics.period_returns(daily.select("date", (pl.col("ret") * k).alias("ret")), "1mo")["ret"]
        point = float(mm.mean())
    else:
        point = float(_rows_mean_monthly((k * a)[None, :], dpm)[0]) if np.isfinite(k) else float("nan")
    boot = _rows_mean_monthly(k * samples, dpm) if np.isfinite(k) else np.full(n_boot, np.nan)
    band = tuple(float(v) for v in np.quantile(boot, [0.05, 0.5, 0.95]))
    return BudgetResult(leverage=float(k), mean_monthly=point, mean_monthly_band=band,
                        label=classify_monthly_return(point), dd_budget=dd_budget, quantile=quantile,
                        horizon_days=h, mean_block=b, n_boot=n_boot)


# =========================================================================== SPA / RC / FDR
def _boot_means(d: np.ndarray, mean_block: float, n_boot: int, rng, chunk: int = 250) -> np.ndarray:
    """Bootstrap means (n_boot × K) of the columns of d (T × K) via index-count matrices."""
    t = d.shape[0]
    out = []
    for s in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - s)
        idx = stationary_bootstrap_indices(t, mean_block, nb, rng)
        counts = np.zeros((nb, t))
        np.add.at(counts, (np.repeat(np.arange(nb), t), idx.ravel()), 1.0)
        out.append(counts @ d / t)
    return np.vstack(out)


def _prep_diff(perf_matrix, benchmark) -> np.ndarray:
    p, _ = _matrix(perf_matrix)
    if benchmark is None:
        bm = np.zeros(p.shape[0])
    elif np.ndim(benchmark) == 0:
        bm = np.full(p.shape[0], float(benchmark))
    else:
        bm = _ret_array(benchmark) if not isinstance(benchmark, np.ndarray) else benchmark.astype(float)
    return p - bm[:, None]


def spa_test(perf_matrix, benchmark=None, *, n_boot: int = 1000, mean_block: float | None = None,
             rng: np.random.Generator | None = None) -> dict[str, float]:
    """Hansen's Superior Predictive Ability test [H05]; H0: no strategy beats the benchmark.

    ``perf_matrix``: T × K per-period performance (e.g. daily returns) of the K candidates,
    higher = better; ``benchmark``: length-T series, scalar, or None (= 0).  Relative
    performance d_k,t = perf − benchmark.  Studentised statistic
    T = max(0, max_k √n·d̄_k/ω̂_k), ω̂_k² = bootstrap variance of √n·d̄*_k.  Stationary
    bootstrap, mean block = ``mean_block`` or the median Politis–White length over columns.

    Bootstrap statistic T* = max(0, max_k √n·(d̄*_k − d̄_k + μ̂_k)/ω̂_k).  Returns p-values
    for the three null means: ``p_consistent`` (Hansen's recommended SPA_c:
    μ̂_k = d̄_k·1{√n·d̄_k/ω̂_k ≤ −√(2 log log n)}), ``p_lower`` (μ̂ = min(d̄, 0); liberal),
    ``p_upper`` (μ̂ = 0, least favourable, RC-style; conservative), plus the statistic.
    p_lower ≤ p_consistent ≤ p_upper."""
    d = _prep_diff(perf_matrix, benchmark)
    n, _ = d.shape
    rng = rng or np.random.default_rng(0)
    b = mean_block if mean_block is not None else float(np.median([_pw_single(c) for c in d.T]))
    dbar = d.mean(axis=0)
    bm = _boot_means(d, b, n_boot, rng)
    omega = np.sqrt(n) * bm.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.inf)
    stat = max(0.0, float(np.max(np.sqrt(n) * dbar / omega)))
    thr = -math.sqrt(2.0 * math.log(math.log(max(n, 16))))
    # bootstrap null means (Hansen 2005 §2.3): poor models (t ≤ −√(2 log log n)) are
    # recentred at their own (negative) mean so they cannot inflate the null distribution.
    mu_c = np.where(np.sqrt(n) * dbar / omega <= thr, dbar, 0.0)
    mu_l = np.minimum(dbar, 0.0)
    mu_u = np.zeros_like(dbar)
    out = {"stat": stat, "mean_block": b}
    for name, mu in (("p_consistent", mu_c), ("p_lower", mu_l), ("p_upper", mu_u)):
        z = np.sqrt(n) * (bm - dbar + mu) / omega
        tb = np.maximum(0.0, z.max(axis=1))
        out[name] = float(np.mean(tb >= stat))
    return out


def white_reality_check(perf_matrix, benchmark=None, *, n_boot: int = 1000,
                        mean_block: float | None = None,
                        rng: np.random.Generator | None = None) -> dict[str, float]:
    """White's Reality Check [W00]: non-studentised V = max_k √n·d̄_k; bootstrap
    V* = max_k √n·(d̄*_k − d̄_k); p = P(V* ≥ V)."""
    d = _prep_diff(perf_matrix, benchmark)
    n = d.shape[0]
    rng = rng or np.random.default_rng(0)
    b = mean_block if mean_block is not None else float(np.median([_pw_single(c) for c in d.T]))
    dbar = d.mean(axis=0)
    bm = _boot_means(d, b, n_boot, rng)
    v = float(np.max(np.sqrt(n) * dbar))
    vb = np.max(np.sqrt(n) * (bm - dbar), axis=1)
    return {"stat": v, "p_value": float(np.mean(vb >= v)), "mean_block": b}


def bh_fdr(pvalues, q: float = 0.10) -> dict[str, np.ndarray]:
    """Benjamini–Hochberg step-up [BH95] at FDR ``q``.  Returns ``reject`` (bool mask, input
    order) and ``p_adj`` (BH-adjusted p-values, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return {"reject": np.zeros(0, bool), "p_adj": np.zeros(0)}
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.minimum(adj_sorted, 1.0)
    below = p[order] <= q * np.arange(1, m + 1) / m
    kmax = np.nonzero(below)[0].max() + 1 if below.any() else 0
    reject = np.zeros(m, bool)
    reject[order[:kmax]] = True
    return {"reject": reject, "p_adj": adj}


# =========================================================================== stability / regimes
def time_stability(daily: pl.DataFrame, min_days: int = 20) -> dict[str, Any]:
    """Calendar-year stability (DESIGN §4.2).  Per-year PnL = sum of daily simple returns
    (additive on the nominal account, so years add up to the total).  Years with fewer than
    ``min_days`` observations are excluded (stubs).  Returns ``pos_year_share``,
    ``max_year_share`` (largest single-year PnL / total PnL; inf if total ≤ 0) and ``table``."""
    tbl = (daily.sort("date").with_columns(pl.col("date").dt.year().alias("year"))
           .group_by("year", maintain_order=True)
           .agg(pl.col("ret").sum().alias("pnl"), pl.len().alias("n_days"),
                ((pl.col("ret") + 1).product() - 1).alias("ret_compounded"))
           .filter(pl.col("n_days") >= min_days))
    pnl = tbl["pnl"].to_numpy()
    total = float(pnl.sum())
    share = pnl / total if total > 0 else np.full(pnl.size, np.nan)
    tbl = tbl.with_columns(pl.Series("share_of_total", share))
    return {"pos_year_share": float(np.mean(pnl > 0)) if pnl.size else float("nan"),
            "max_year_share": float(np.max(share)) if total > 0 and pnl.size else float("inf"),
            "total_pnl": total, "n_years": int(pnl.size), "table": tbl}


def regime_split(daily: pl.DataFrame, labels, periods_per_year: float = 260.0) -> pl.DataFrame:
    """Metrics per regime label.  ``labels``: array/Series aligned with ``daily`` rows, or a
    frame (date, regime) joined on date.  Null labels (warm-up) are dropped.  Per regime:
    days, annualised Sharpe, mean daily return, PnL (sum), share of total PnL, max DD of the
    concatenated regime days."""
    if isinstance(labels, pl.DataFrame):
        df = daily.join(labels.rename({labels.columns[1]: "regime"}), on="date", how="left")
    else:
        df = daily.with_columns(pl.Series("regime", list(labels)))
    df = df.filter(pl.col("regime").is_not_null())
    total = float(df["ret"].sum())
    rows = []
    for (reg,), g in df.group_by("regime", maintain_order=True):
        r = g["ret"]
        rows.append({"regime": str(reg), "n_days": g.height, "sharpe": metrics.sharpe(r, periods_per_year),
                     "mean": float(r.mean()), "pnl": float(r.sum()),
                     "share_of_pnl": float(r.sum()) / total if total != 0 else float("nan"),
                     "max_dd": metrics.max_drawdown(r)})
    return pl.DataFrame(rows).sort("regime")


def _expanding_terciles(x: np.ndarray, min_periods: int) -> np.ndarray:
    """Causal tercile label (0/1/2) of x[t] against the empirical distribution of x[:t+1]
    (only finite values); −1 before ``min_periods`` finite observations."""
    out = np.full(x.size, -1, dtype=int)
    hist: list[float] = []
    import bisect
    for t, v in enumerate(x):
        if not np.isfinite(v):
            continue
        bisect.insort(hist, v)
        if len(hist) < min_periods:
            continue
        rank = bisect.bisect_right(hist, v) / len(hist)
        out[t] = 0 if rank <= 1 / 3 else (1 if rank <= 2 / 3 else 2)
    return out


def volatility_regimes(returns, window: int = 63, min_periods: int = 252) -> np.ndarray:
    """Causal volatility terciles: rolling std of returns over the ``window`` days *ending
    at t−1* (no same-day information), classified against the expanding distribution of that
    rolling vol.  ``returns``: market (or strategy) daily returns.  Returns an object array of
    "low_vol" / "mid_vol" / "high_vol" / None (warm-up)."""
    a = np.asarray(returns.to_numpy() if isinstance(returns, pl.Series) else returns, dtype=float)
    s = pl.Series(a).rolling_std(window_size=window).shift(1).to_numpy()
    lab = _expanding_terciles(np.asarray(s, dtype=float), min_periods)
    names = np.array(["low_vol", "mid_vol", "high_vol"], dtype=object)
    return np.where(lab >= 0, names[np.clip(lab, 0, 2)], None)


def trend_regimes(close, window: int = 63, min_periods: int = 252) -> np.ndarray:
    """Causal trend/range label from Kaufman's efficiency ratio over the ``window`` closes
    ending at t−1: ER = |c[t−1] − c[t−1−w]| / Σ|Δc|.  "trend" if ER is above its expanding
    median, else "range"; None during warm-up."""
    c = np.asarray(close.to_numpy() if isinstance(close, pl.Series) else close, dtype=float)
    n = c.size
    er = np.full(n, np.nan)
    if n > window + 1:
        absd = np.abs(np.diff(c, prepend=c[0]))
        cs = np.cumsum(absd)
        for t in range(window + 1, n):
            e = t - 1
            path = cs[e] - cs[e - window]
            er[t] = abs(c[e] - c[e - window]) / path if path > 0 else 0.0
    out = np.full(n, None, dtype=object)
    hist: list[float] = []
    import bisect
    for t in range(n):
        if not np.isfinite(er[t]):
            continue
        bisect.insort(hist, er[t])
        if len(hist) >= min_periods:
            med = hist[len(hist) // 2]
            out[t] = "trend" if er[t] > med else "range"
    return out


# =========================================================================== decay (§4.6)
def cusum_threshold(k: float, arl0: float) -> float:
    """h such that the one-sided CUSUM with reference k has in-control ARL ≈ ``arl0`` under
    N(0,1) increments (Siegmund's approximation, h' = h + 1.166):
    ARL0 ≈ (e^{2kh'} − 2kh' − 1) / (2k²)."""
    def arl(h):
        hp = h + 1.166
        return (math.exp(2 * k * hp) - 2 * k * hp - 1) / (2 * k * k)
    lo, hi = 0.0, 1.0
    while arl(hi) < arl0:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if arl(mid) < arl0 else (lo, mid)
    return (lo + hi) / 2


def cusum_decay(daily_new, expected_mean: float, expected_sd: float, k: float | None = None,
                h: float | None = None, arl0: float = 2600.0) -> dict[str, Any]:
    """Lower one-sided CUSUM [P54] on standardised returns z = (r − μ₀)/σ₀ for a *downward*
    shift of the mean (alpha decay).  S_t = max(0, S_{t−1} − z_t − k); alarm when S_t > h.

    Defaults: k = ½·(μ₀/σ₀) — tuned to detect complete loss of the edge (mean → 0);
    h from :func:`cusum_threshold` for an in-control ARL of ``arl0`` days (default 2600 ≈ 10
    years of FX days: ~10 % false-alarm chance per year).  Returns ``alarm`` (bool),
    ``alarm_index`` (first alarm, or None), ``path`` (S_t), ``k``, ``h``."""
    a = _ret_array(daily_new)
    if k is None:
        k = max(0.5 * expected_mean / expected_sd, 1e-3)
    if h is None:
        h = cusum_threshold(k, arl0)
    z = (a - expected_mean) / expected_sd
    s = np.empty(a.size)
    cur = 0.0
    for i, zi in enumerate(z):
        cur = max(0.0, cur - zi - k)
        s[i] = cur
    hit = np.nonzero(s > h)[0]
    return {"alarm": bool(hit.size), "alarm_index": int(hit[0]) if hit.size else None,
            "path": s, "k": float(k), "h": float(h)}


def sequential_sharpe_test(daily_new, expected_sr_annual: float, expected_sd: float, *,
                           periods_per_year: float = 260.0, sr_dead_annual: float = 0.0,
                           alpha: float = 0.05, beta: float = 0.20) -> dict[str, Any]:
    """Wald SPRT [W45] on the per-period Sharpe with the dev-period volatility ``expected_sd``
    treated as known.  H0: SR = expected (edge alive); H1: SR = ``sr_dead_annual`` (dead).
    Log-likelihood ratio of H1 vs H0 accumulates; crossing ln((1−β)/α) → "decayed",
    crossing ln(β/(1−α)) → "alive"; otherwise "continue".  Returns decision, index, llr path."""
    a = _ret_array(daily_new)
    mu0 = expected_sr_annual / math.sqrt(periods_per_year) * expected_sd
    mu1 = sr_dead_annual / math.sqrt(periods_per_year) * expected_sd
    inc = ((mu1 - mu0) * a - 0.5 * (mu1 ** 2 - mu0 ** 2)) / expected_sd ** 2
    llr = np.cumsum(inc)
    up, lo = math.log((1 - beta) / alpha), math.log(beta / (1 - alpha))
    for i, v in enumerate(llr):
        if v >= up:
            return {"decision": "decayed", "index": i, "llr": llr, "bounds": (lo, up)}
        if v <= lo:
            return {"decision": "alive", "index": i, "llr": llr, "bounds": (lo, up)}
    return {"decision": "continue", "index": None, "llr": llr, "bounds": (lo, up)}


# =========================================================================== holdout band (§4.4)
@dataclass
class HoldoutBand:
    """Pre-registered holdout pass band (DESIGN §4.4 v1.2), frozen before checkpoint B.

    Holdout passes iff sharpe_annual ≥ sharpe_lo, mean monthly return at the fixed leverage ≥
    ret_at_budget_lo, |max DD| ≤ max_dd_mag_hi, trades within [trades_lo, trades_hi].

    The four limits share **one** tail level α (``tail_level``): the one-sided limits are the
    α / (1 − α) quantiles and the trade range the α/2 … 1 − α/2 quantiles of the bootstrap
    draws; α is the largest level at which ``joint_coverage`` ≥ ``target_coverage`` (≈ 90 %)
    of the *joint* draws pass all four at once (red-team M3 / calibration R4).

    ``p_pass_zero_edge``: probability that a zero-edge holdout (the same series de-meaned,
    bootstrapped, run through :func:`holdout_check`'s criteria) passes — the band's false-pass
    rate.  ``decisive`` is False when it exceeds :data:`DECISIVE_MAX_ZERO_EDGE_PASS` (0.30); an
    all-criteria pass is then NOT_DECISIVE (:func:`holdout_status`).

    Horizon (DESIGN §4.4, 2026-09-24): ``horizon_days`` runs from the holdout start to the end of
    the data available when the band is built (locked year + newer exports), read from the data
    manifest by ``data.holdout_horizon`` (``horizon_source="manifest"``; ``"explicit"`` for an
    integer override, which the ledger refuses at unlock).  ``horizon_end``, ``holdout_start``,
    ``locked_end``, ``newer_data_included``, ``symbols`` and ``manifest_sha`` record where it
    came from."""

    sharpe_lo: float               # annualised
    ret_at_budget_lo: float        # mean monthly return of leverage·r
    max_dd_mag_hi: float           # positive magnitude, UNlevered returns
    trades_lo: float
    trades_hi: float
    leverage: float                # k fixed from the source series (DESIGN §4.5 budget)
    horizon_days: int
    n_boot: int
    mean_block: float
    source: str
    sharpe_median: float
    seed: int
    tail_level: float = float("nan")
    target_coverage: float = 0.90
    joint_coverage: float = float("nan")
    trades_source: str = "none"
    trades_joint: bool = False     # trade sums drawn with the same bootstrap indices as returns
    p_pass_zero_edge: float = float("nan")
    n_power: int = 0
    max_zero_edge_pass: float = DECISIVE_MAX_ZERO_EDGE_PASS
    decisive: bool | None = None
    horizon_source: str = "explicit"
    horizon_end: str | None = None
    holdout_start: str | None = None
    locked_end: str | None = None
    newer_data_included: bool | None = None
    symbols: Any = None
    periods_per_year: float | None = None
    manifest_sha: str | None = None
    conversion_legs: Any = None
    n_boot_requested: int | None = None    # the n_boot / n_power asked for (R2-3: must be the fixed
    n_power_requested: int | None = None   # HOLDOUT_BAND_N_BOOT / _N_POWER for a registered band)

    def as_dict(self) -> dict[str, Any]:
        out = {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in asdict(self).items()}
        return {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in out.items()}


def _selection_series(study) -> tuple[list[np.ndarray], str]:
    """OOS return paths of the selection procedure: CPCV paths → WFO OOS → selected trial."""
    cp = study.cpcv_paths
    if cp is not None and cp.height:
        paths = [g.sort("date")["ret"].to_numpy().astype(float)
                 for _, g in cp.group_by("path_id", maintain_order=True)]
        return paths, "cpcv_paths"
    if study.wfo_oos is not None and study.wfo_oos.height:
        return [study.wfo_oos.sort("date")["ret"].to_numpy().astype(float)], "wfo_oos"
    return [], "none"


def _band_pass(sh, rab, ddm, tsum, lim) -> np.ndarray:
    """Vectorised holdout_check over bootstrap draws (NaN statistics fail, like holdout_check)."""
    ok = (sh >= lim["sharpe_lo"]) & (ddm <= lim["max_dd_mag_hi"])
    if np.isfinite(lim["ret_at_budget_lo"]):
        ok &= rab >= lim["ret_at_budget_lo"]
    if tsum is not None:
        ok &= (tsum >= lim["trades_lo"]) & (tsum <= lim["trades_hi"])
    return ok


def _band_limits(sh, rab, ddm, tsum, a: float) -> dict[str, float]:
    return {"sharpe_lo": float(np.nanquantile(sh, a)) if np.isfinite(sh).any() else float("nan"),
            "ret_at_budget_lo": float(np.nanquantile(rab, a)) if np.isfinite(rab).any() else float("nan"),
            "max_dd_mag_hi": float(np.quantile(ddm, 1.0 - a)),
            "trades_lo": float(np.quantile(tsum, a / 2.0)) if tsum is not None else float("nan"),
            "trades_hi": float(np.quantile(tsum, 1.0 - a / 2.0)) if tsum is not None else float("nan")}


def joint_tail_level(sh, rab, ddm, tsum=None, target: float = 0.90, iters: int = 40) -> tuple[float, float]:
    """Largest common tail level α ∈ [0, 0.5] whose limits (see :class:`HoldoutBand`) let at
    least ``target`` of the joint draws pass every criterion.  Returns (α, joint coverage).
    Coverage is non-increasing in α, so bisection applies."""
    def cov(a):
        return float(np.mean(_band_pass(sh, rab, ddm, tsum, _band_limits(sh, rab, ddm, tsum, a))))
    lo, hi = 0.0, 0.5
    if cov(hi) >= target:
        return hi, cov(hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cov(mid) >= target:
            lo = mid
        else:
            hi = mid
    return lo, cov(lo)


_HORIZON_META = ("horizon_source", "horizon_end", "holdout_start", "locked_end", "newer_data_included",
                 "symbols", "manifest_sha", "conversion_legs")


def holdout_band(study, horizon_days: int | Mapping[str, Any], periods_per_year: float = 260.0,
                 n_boot: int = 4000, *,
                 trades_per_day=None, dd_budget: float = 0.10, seed: int = 12345,
                 extra_series: np.ndarray | None = None, source: str = "wfo",
                 target_coverage: float = 0.90, trades_source: str | None = None,
                 n_power: int = 1000, max_zero_edge_pass: float | None = None) -> HoldoutBand:
    """Predictive distribution for a holdout of ``horizon_days`` (DESIGN §4.4 v1.2).

    ``horizon_days``: an int (explicit override, tests) or the mapping returned by
    ``data.holdout_horizon`` (the exam horizon: holdout start → end of available data per the
    manifest, whose metadata is copied onto the band).  ``max_zero_edge_pass`` is read-only:
    anything other than :data:`DECISIVE_MAX_ZERO_EDGE_PASS` raises.

    Each source path contributes an equal share of ``n_boot`` stationary-bootstrap samples of
    length ``horizon_days``.  The budget leverage k is solved on full-length bootstrap samples
    of the pooled paths (§4.5) and then held fixed.  The four limits share one tail level set
    on the joint draws (:func:`joint_tail_level`, ``target_coverage`` ≈ 0.90).

    ``trades_per_day``: daily trade counts.  When it is aligned with a single source path (same
    length — e.g. the WFO procedure's own daily counts, F4), the trade sums use the **same**
    bootstrap indices as the returns (joint draws); otherwise they are bootstrapped separately
    (a scalar gives a Poisson rate).  ``extra_series`` overrides the path source.

    ``source="wfo"`` (default): the holdout exam runs the frozen re-optimisation *procedure*
    (DESIGN §4.4 + §4.6.2), so the band is built from the walk-forward OOS series; CPCV paths
    are the fallback when no WFO series exists.  ``source="cpcv"`` inverts the preference.

    Power: ``n_power`` de-meaned (zero-edge) bootstrap draws of the same paths are run through
    :func:`holdout_check`; ``p_pass_zero_edge`` is their pass rate."""
    if max_zero_edge_pass is not None and max_zero_edge_pass != DECISIVE_MAX_ZERO_EDGE_PASS:
        raise ValueError(f"max_zero_edge_pass is fixed at {DECISIVE_MAX_ZERO_EDGE_PASS} (DESIGN §4.4); "
                         f"got {max_zero_edge_pass}")
    hmeta: dict[str, Any] = {"horizon_source": "explicit"}
    if isinstance(horizon_days, Mapping):
        hmeta = {k: horizon_days.get(k) for k in _HORIZON_META}
        horizon_days = int(horizon_days["horizon_days"])
    horizon_days = int(horizon_days)
    if horizon_days < 2:
        raise ValueError(f"holdout horizon must be ≥ 2 days, got {horizon_days}")
    rng = np.random.default_rng(seed)
    if extra_series is not None:
        paths, src = [np.asarray(extra_series, float)], "extra_series"
    else:
        wfo = study.wfo_oos if study.wfo_oos is not None and study.wfo_oos.height else None
        if source == "wfo" and wfo is not None:
            paths, src = [wfo.sort("date")["ret"].to_numpy().astype(float)], "wfo_oos"
        elif source in ("wfo", "cpcv"):
            paths, src = _selection_series(study)
        else:
            raise ValueError(f"source must be 'wfo' or 'cpcv', got {source!r}")
    if not paths:
        raise ValueError("study has no CPCV paths or WFO OOS returns")
    pooled = np.concatenate(paths)
    b = optimal_block_length(pooled)
    tp = None if trades_per_day is None or np.ndim(trades_per_day) == 0 else np.asarray(trades_per_day, float)
    aligned = tp is not None and len(paths) == 1 and tp.size == paths[0].size
    per = max(1, n_boot // len(paths))
    hs, idxs, full = [], [], []
    for p in paths:
        ix = stationary_bootstrap_indices(p.size, b, per, rng, horizon_days)
        idxs.append(ix)
        hs.append(p[ix])
        full.append(p[stationary_bootstrap_indices(p.size, b, max(1, min(n_boot, 2000) // len(paths)), rng)])
    hs_m = np.vstack(hs)
    full_len = min(len(f[0]) for f in full)
    k = _solve_leverage(np.vstack([f[:, :full_len] for f in full]), dd_budget, 0.95)
    ann = math.sqrt(periods_per_year)
    sh = _rows_sharpe(hs_m) * ann
    dpm = max(1, int(round(periods_per_year / 12.0)))
    rab = _rows_mean_monthly(k * hs_m, dpm) if np.isfinite(k) else np.full(hs_m.shape[0], np.nan)
    ddm = -_rows_maxdd(hs_m)
    if trades_per_day is None:
        tsum, tsrc = None, "none"
    elif np.ndim(trades_per_day) == 0:
        lam = float(trades_per_day) * horizon_days
        tsum, tsrc = rng.poisson(lam, hs_m.shape[0]).astype(float), "poisson rate"
    elif aligned:
        tsum, tsrc = tp[idxs[0]].sum(axis=1), "aligned daily counts (joint)"
    else:
        bt = optimal_block_length(tp) if tp.std() > 0 else 1.0
        tsum = tp[stationary_bootstrap_indices(tp.size, bt, hs_m.shape[0], rng, horizon_days)].sum(axis=1)
        tsrc = "daily counts (independent draws)"
    alpha, cov = joint_tail_level(sh, rab, ddm, tsum, target_coverage)
    lim = _band_limits(sh, rab, ddm, tsum, alpha)
    band = HoldoutBand(
        sharpe_lo=lim["sharpe_lo"], ret_at_budget_lo=lim["ret_at_budget_lo"], max_dd_mag_hi=lim["max_dd_mag_hi"],
        trades_lo=lim["trades_lo"], trades_hi=lim["trades_hi"], leverage=float(k),
        horizon_days=int(horizon_days), n_boot=int(hs_m.shape[0]), mean_block=float(b), source=src,
        sharpe_median=float(np.nanmedian(sh)), seed=seed, tail_level=float(alpha),
        target_coverage=float(target_coverage), joint_coverage=float(cov),
        trades_source=trades_source or tsrc, trades_joint=bool(aligned),
        max_zero_edge_pass=DECISIVE_MAX_ZERO_EDGE_PASS, periods_per_year=float(periods_per_year),
        n_boot_requested=int(n_boot), n_power_requested=int(n_power),
        **{k: v for k, v in hmeta.items() if v is not None})
    # ---- power against a zero-edge holdout (de-meaned series through holdout_check)
    if n_power > 0:
        npw = max(1, n_power // len(paths))
        passes, tot = 0, 0
        for pi, p in enumerate(paths):
            z = p - p.mean()
            ix = stationary_bootstrap_indices(z.size, b, npw, rng, horizon_days)
            ts_pw = (tp[ix].sum(axis=1) if aligned else
                     (tsum[rng.integers(0, tsum.size, npw)] if tsum is not None else np.full(npw, np.nan)))
            for r in range(npw):
                passes += holdout_check(band, z[ix[r]], float(ts_pw[r]), periods_per_year,
                                        check_horizon=False)["criteria_pass"]
                tot += 1
        band.p_pass_zero_edge = passes / tot
        band.n_power = tot
        band.decisive = bool(band.p_pass_zero_edge <= DECISIVE_MAX_ZERO_EDGE_PASS)
    return band


_BAND_KEYS = {"sharpe_lo": "sharpe_p10", "ret_at_budget_lo": "ret_at_budget_p10", "max_dd_mag_hi": "max_dd_mag_p95"}


def holdout_status(criteria_pass: bool, p_pass_zero_edge: float | None) -> str:
    """DESIGN §4.4 verdict: FAIL if any criterion fails; NOT_DECISIVE if all pass but the band's
    zero-edge pass probability is above :data:`DECISIVE_MAX_ZERO_EDGE_PASS` (or unknown); PASS
    otherwise.  NOT_DECISIVE is neither a pass nor a fail: the system waits for more data."""
    if not criteria_pass:
        return "FAIL"
    p = float("nan") if p_pass_zero_edge is None else float(p_pass_zero_edge)
    if not np.isfinite(p) or p > DECISIVE_MAX_ZERO_EDGE_PASS:
        return "NOT_DECISIVE"
    return "PASS"


def holdout_check(band: HoldoutBand | Mapping[str, Any], holdout_daily, n_trades: float,
                  periods_per_year: float = 260.0, *, check_horizon: bool = True) -> dict[str, Any]:
    """Apply a pre-registered band to realised holdout returns (only after checkpoint B).
    Accepts v1.2 bands and v1.1 dicts (``sharpe_p10`` / ``ret_at_budget_p10`` / ``max_dd_mag_p95``).

    Returns ``status`` ∈ {PASS, FAIL, NOT_DECISIVE} (:func:`holdout_status`), ``criteria_pass``
    (all four criteria met), ``pass`` (= status == "PASS"; never True when NOT_DECISIVE),
    ``decisive``, ``p_pass_zero_edge``, ``checks``, ``values``, the band's horizon
    (``horizon_days``, ``horizon_end``, ``newer_data_included``), ``n_days`` and the ``band`` used.

    ``check_horizon``: the realised series must span the band's horizon (±max(5 days, 2 %) —
    :data:`HOLDOUT_SPAN_TOL_DAYS` / :data:`HOLDOUT_SPAN_TOL_FRAC`, red-team H6), else ValueError —
    a band built for one horizon says nothing about another."""
    bd = band.as_dict() if isinstance(band, HoldoutBand) else dict(band)
    orig = dict(bd)
    for new, old in _BAND_KEYS.items():
        if new not in bd and old in bd:
            bd[new] = bd[old]
    a = _ret_array(holdout_daily)
    hd = bd.get("horizon_days")
    if check_horizon and hd:
        tol = max(HOLDOUT_SPAN_TOL_DAYS, HOLDOUT_SPAN_TOL_FRAC * float(hd))
        if abs(a.size - float(hd)) > tol:
            raise ValueError(f"holdout series has {a.size} daily returns but the band was built for "
                             f"{hd} (horizon end {bd.get('horizon_end')}); judge the span the band was "
                             f"registered for")
    sh = metrics.sharpe(a, periods_per_year)
    dpm = max(1, int(round(periods_per_year / 12.0)))
    rab = float(_rows_mean_monthly((bd["leverage"] * a)[None, :], dpm)[0])
    dd = -metrics.max_drawdown(a)
    tl, th = bd.get("trades_lo", float("nan")), bd.get("trades_hi", float("nan"))
    tl = float("nan") if tl is None else tl
    checks = {"sharpe": bool(sh >= bd["sharpe_lo"]),
              "ret_at_budget": bool((not np.isfinite(bd["ret_at_budget_lo"])) or rab >= bd["ret_at_budget_lo"]),
              "max_dd": bool(dd <= bd["max_dd_mag_hi"]),
              "trades": bool((not np.isfinite(tl)) or bool(tl <= n_trades <= th))}
    crit = bool(all(checks.values()))
    pz = bd.get("p_pass_zero_edge")
    status = holdout_status(crit, pz)
    return {"status": status, "pass": status == "PASS", "criteria_pass": crit,
            "decisive": holdout_status(True, pz) == "PASS",
            "p_pass_zero_edge": pz, "checks": checks,
            "values": {"sharpe": sh, "ret_at_budget": rab, "max_dd_mag": dd, "trades": n_trades},
            "horizon_days": hd, "horizon_end": bd.get("horizon_end"),
            "newer_data_included": bd.get("newer_data_included"), "n_days": int(a.size), "band": orig}


# =========================================================================== component tests
def _to_sharpe(v, periods_per_year: float) -> float:
    if hasattr(v, "daily"):
        return metrics.sharpe(v.daily, periods_per_year)
    if isinstance(v, (pl.DataFrame, pl.Series, np.ndarray)):
        return metrics.sharpe(v, periods_per_year)
    return float(v)


def random_selectivity_null(evaluate_fn: Callable[[np.ndarray], Any], selectivity: float,
                            n_candidates: int, n_draws: int, rng: np.random.Generator, *,
                            periods_per_year: float = 260.0) -> np.ndarray:
    """Null distribution for a **filter** (DESIGN §4.2 component tests).

    ``evaluate_fn(mask)`` runs the base system with a boolean mask over its ``n_candidates``
    candidate signals (True = allowed) and returns an annualised Sharpe, an Outcome, or daily
    returns.  Each draw keeps exactly round(selectivity·n) random candidates.  Compare the
    real filter's Sharpe with :func:`empirical_pvalue`."""
    keep = int(round(selectivity * n_candidates))
    out = np.empty(n_draws)
    for i in range(n_draws):
        mask = np.zeros(n_candidates, bool)
        mask[rng.choice(n_candidates, keep, replace=False)] = True
        out[i] = _to_sharpe(evaluate_fn(mask), periods_per_year)
    return out


def _random_entries(n_bars: int, holds: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Uniformly random non-overlapping placement of intervals with lengths ``holds`` in
    ``n_bars`` slots (stars and bars: choose the m gap positions among free + m slots).
    Returns (starts, lengths) in increasing start order."""
    holds = rng.permutation(holds)
    free = n_bars - int(holds.sum())
    if free < 0:
        raise ValueError("holding times exceed the number of bars")
    m = holds.size
    pos = np.sort(rng.choice(free + m, m, replace=False))
    starts = pos - np.arange(m) + np.concatenate([[0], np.cumsum(holds)[:-1]])
    return starts, holds


def random_entry_null(evaluate_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Any], n_bars: int,
                      holding_bars, n_draws: int, rng: np.random.Generator, *, sides=None,
                      periods_per_year: float = 260.0) -> np.ndarray:
    """Null distribution for an **entry signal**: random entries with the system's own
    holding-time distribution (the empirical ``holding_bars``, permuted) and side mix.

    ``evaluate_fn(entry_idx, hold_bars, sides)`` must run the system's exits on those entries
    (entry at bar entry_idx, exit hold_bars later) and return Sharpe / Outcome / daily.
    Entries never overlap (one position at a time, like the real system)."""
    holds = np.asarray(holding_bars, dtype=int)
    sd = np.ones(holds.size, dtype=int) if sides is None else np.asarray(sides, dtype=int)
    out = np.empty(n_draws)
    for i in range(n_draws):
        e, h = _random_entries(n_bars, holds, rng)
        out[i] = _to_sharpe(evaluate_fn(e, h, rng.permutation(sd)), periods_per_year)
    return out


def empirical_pvalue(observed: float, null: np.ndarray) -> float:
    """One-sided (greater) Monte-Carlo p-value with the +1 correction: (1 + #{null ≥ obs})/(1 + n)."""
    nl = np.asarray(null, float)
    nl = nl[np.isfinite(nl)]
    return float((1 + np.sum(nl >= observed)) / (1 + nl.size))


def ablation_compare(daily_on, daily_off, *, periods_per_year: float = 260.0, dd_budget: float = 0.10,
                     n_boot: int = 1000, seed: int = 0) -> pl.DataFrame:
    """Stop/sizing-rule component test: the same system with the rule on vs off, judged on
    max DD, daily CVaR95, Sharpe and monthly return at the DD budget (DESIGN §4.2)."""
    rows = []
    for name, d in (("on", daily_on), ("off", daily_off)):
        a = _ret_array(d)
        br = return_at_dd_budget(a, dd_budget, n_boot=n_boot, periods_per_year=periods_per_year,
                                 rng=np.random.default_rng(seed))
        rows.append({"rule": name, "sharpe": metrics.sharpe(a, periods_per_year),
                     "max_dd": metrics.max_drawdown(a), "cvar95": metrics.cvar(a, 0.95),
                     "leverage_at_budget": br.leverage, "monthly_at_budget": br.mean_monthly})
    return pl.DataFrame(rows)
