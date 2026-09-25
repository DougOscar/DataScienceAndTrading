"""Known-answer tests for quantlab.metrics."""

from datetime import date, timedelta
import math

import numpy as np
import polars as pl
import pytest

from quantlab import metrics as m


def _daily(rets, start=date(2020, 1, 1)):
    return pl.DataFrame({"date": [start + timedelta(days=i) for i in range(len(rets))], "ret": rets})


def test_sharpe_known_answer():
    r = np.array([0.01, -0.005, 0.02, 0.0])
    expected = r.mean() / r.std(ddof=1) * math.sqrt(260)
    assert m.sharpe(r) == pytest.approx(expected)
    assert math.isnan(m.sharpe([0.01, 0.01]))            # zero variance


def test_drawdown_and_duration():
    r = [0.10, -0.50, 0.20, 0.0, 1.0]                     # eq 1.1, .55, .66, .66, 1.32
    assert m.max_drawdown(r) == pytest.approx(-0.5)
    assert m.longest_drawdown_periods(r) == 3
    assert m.max_drawdown([0.01, 0.02]) == 0.0
    assert m.max_drawdown([-0.1]) == pytest.approx(-0.1)  # starting peak counts


def test_cagr_and_calmar():
    r = [0.0] * 259 + [0.21]                              # one year, +21%
    assert m.cagr(r) == pytest.approx(0.21)
    assert math.isnan(m.calmar(r))                        # no drawdown


def test_skew_kurt_is_raw_kurtosis():
    rng = np.random.default_rng(0)
    _, k = m.skew_kurt(rng.normal(size=200_000))
    assert k == pytest.approx(3.0, abs=0.05)


def test_cvar_is_mean_of_tail():
    r = np.arange(-50, 50) / 1000.0                       # 100 values
    assert m.cvar(r, 0.95) == pytest.approx(np.mean(r[:5]), abs=1e-4)


def test_period_returns_compound_and_period_sharpe():
    d = _daily([0.01] * 31 + [-0.01] * 29)                # Jan up, Feb down
    mo = m.period_returns(d, "1mo")
    assert mo["ret"][0] == pytest.approx(1.01 ** 31 - 1)
    assert mo["ret"][1] == pytest.approx(0.99 ** 29 - 1)
    ps = m.period_sharpe(_daily(list(np.random.default_rng(1).normal(0, 0.01, 800))), "1y")
    assert ps.height == 3 and ps["n_days"].sum() == 800


def test_summary_keys():
    s = m.summary(_daily(list(np.random.default_rng(2).normal(0.0005, 0.01, 520))))
    for k in ("sharpe", "max_dd", "calmar", "cvar95", "kurtosis", "mean_monthly", "n_days"):
        assert k in s
