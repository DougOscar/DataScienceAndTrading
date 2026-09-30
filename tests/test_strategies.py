"""Tests for quantlab.strategies (the per-system strategy registry + shared indicator helpers)."""

from __future__ import annotations

import importlib
import sys
import textwrap

import numpy as np
import polars as pl
import pytest

import quantlab.strategies as strategies_pkg
from quantlab import strategies
from quantlab.strategies import base, get_strategy
from quantlab.testing import synthetic_bars


# --------------------------------------------------------------------------- base helpers
def test_true_range_matches_hand_computation():
    bars = synthetic_bars(50, seed=1, timeframe="H1")
    got = bars.select(base.true_range().alias("tr"))["tr"].to_numpy()

    high, low, close = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    prev_close = np.empty_like(close)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]
    # pl.max_horizontal ignores nulls (row 0's prev_close-derived terms are null from shift(1)):
    # np.nanmax mirrors that "ignore, don't propagate" semantics, so TR is defined at every row,
    # including row 0 (== high[0] - low[0] there, with no previous close to compare against).
    diffs = np.stack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    expected = np.nanmax(diffs, axis=0)

    assert not np.isnan(got).any()
    np.testing.assert_allclose(got, expected, rtol=1e-9)
    assert got[0] == pytest.approx(high[0] - low[0], rel=1e-9)


def test_atr_is_a_rolling_mean_of_true_range_with_no_partial_windows():
    bars = synthetic_bars(60, seed=2, timeframe="H1")
    period = 14
    got = bars.select(base.atr(period).alias("atr"))["atr"]

    # True range itself is never null (see test above), so "no partial windows" means exactly
    # `period - 1` leading nulls: the window first covers `period` rows (0..period-1) at row
    # index `period - 1`.
    first_valid = period - 1
    assert got.null_count() == first_valid
    assert got[first_valid] is not None

    tr = bars.select(base.true_range().alias("tr"))["tr"].to_numpy()
    expected_at_first_valid = np.mean(tr[:period])
    assert got[first_valid] == pytest.approx(expected_at_first_valid, rel=1e-9)


def test_atr_rejects_non_positive_period():
    with pytest.raises(ValueError, match="period"):
        base.atr(0)


def test_atr_and_true_range_are_causal_via_lookahead_audit():
    """The two helpers are expressions used inside a strategy's signals(); prove a strategy
    built purely from them passes the same audit a real system module must pass."""
    from dataclasses import dataclass

    from quantlab.contracts import Params, RiskType
    from quantlab.testing import assert_no_lookahead

    @dataclass(frozen=True)
    class P(Params):
        period: int = 10

    class AtrProbe:
        name = "atr_probe_test"
        risk_type = RiskType.C
        params_cls = P

        def __init__(self, params: P):
            self.params = params

        def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
            a = base.atr(self.params.period)
            return bars.select(
                pl.when(a.is_null()).then(None).otherwise(1).cast(pl.Int8).alias("signal"),
                pl.lit(None, dtype=pl.Float64).alias("stop_dist"),
                pl.lit(None, dtype=pl.Float64).alias("target_dist"),
            )

    bars = synthetic_bars(400, seed=3, timeframe="H1")
    assert_no_lookahead(lambda: AtrProbe(P()), bars, min_history=60)


# --------------------------------------------------------------------------- registry: get_strategy
_VALID_MODULE = textwrap.dedent(
    """
    from dataclasses import dataclass

    import polars as pl

    from quantlab.contracts import Params, RiskType


    @dataclass(frozen=True)
    class Params(Params):
        fast: int = 5


    class Strategy:
        name = "probe_valid"
        risk_type = RiskType.C

        def __init__(self, params=None):
            self.params = params or Params()

        def signals(self, bars):
            return bars.select(
                pl.lit(0).cast(pl.Int8).alias("signal"),
                pl.lit(None, dtype=pl.Float64).alias("stop_dist"),
                pl.lit(None, dtype=pl.Float64).alias("target_dist"),
            )
    """
)

_STRATEGY_ALIAS_MODULE = textwrap.dedent(
    """
    import polars as pl

    from quantlab.contracts import RiskType


    class _Impl:
        name = "probe_alias"
        risk_type = RiskType.C

        def signals(self, bars):
            return bars.select(
                pl.lit(0).cast(pl.Int8).alias("signal"),
                pl.lit(None, dtype=pl.Float64).alias("stop_dist"),
                pl.lit(None, dtype=pl.Float64).alias("target_dist"),
            )


    STRATEGY = _Impl
    """
)

_AMBIGUOUS_MODULE = textwrap.dedent(
    """
    from quantlab.contracts import RiskType


    class First:
        name = "first"
        risk_type = RiskType.C

        def signals(self, bars):
            return bars

    class Second:
        name = "second"
        risk_type = RiskType.C

        def signals(self, bars):
            return bars
    """
)

_NOT_A_STRATEGY_MODULE = "X = 1\n"


@pytest.fixture
def temp_strategy_package(tmp_path, monkeypatch):
    """Extend quantlab.strategies.__path__ with a tmp directory for the duration of one test,
    so `get_strategy` finds modules written there without touching the real package on disk."""
    extra_dir = tmp_path / "extra_strategies"
    extra_dir.mkdir()
    original_path = list(strategies_pkg.__path__)
    monkeypatch.setattr(strategies_pkg, "__path__", original_path + [str(extra_dir)])

    written: list[str] = []

    def _write(slug: str, source: str) -> None:
        (extra_dir / f"{slug}.py").write_text(source)
        written.append(f"{strategies_pkg.__name__}.{slug}")

    yield _write

    for mod_name in written:
        sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()


def test_get_strategy_resolves_via_named_strategy_attribute(temp_strategy_package):
    temp_strategy_package("probe_valid", _VALID_MODULE)
    cls = get_strategy("probe_valid")
    assert cls.name == "probe_valid"
    inst = cls()
    assert inst.params.fast == 5


def test_get_strategy_resolves_via_explicit_strategy_constant(temp_strategy_package):
    temp_strategy_package("probe_alias", _STRATEGY_ALIAS_MODULE)
    cls = get_strategy("probe_alias")
    assert cls.name == "probe_alias"


def test_get_strategy_missing_module_raises_clear_error(temp_strategy_package):
    with pytest.raises(ValueError, match="no module"):
        get_strategy("does_not_exist_anywhere")


def test_get_strategy_ambiguous_module_raises(temp_strategy_package):
    temp_strategy_package("probe_ambiguous", _AMBIGUOUS_MODULE)
    with pytest.raises(ValueError, match="candidate"):
        get_strategy("probe_ambiguous")


def test_get_strategy_no_candidate_raises(temp_strategy_package):
    temp_strategy_package("probe_empty", _NOT_A_STRATEGY_MODULE)
    with pytest.raises(ValueError, match="no Strategy class"):
        get_strategy("probe_empty")


def test_get_strategy_is_reachable_from_the_quantlab_strategies_namespace():
    """`strategies.get_strategy` (the module-level import used above) and
    `quantlab.strategies.get_strategy` are the same function."""
    assert strategies.get_strategy is get_strategy
