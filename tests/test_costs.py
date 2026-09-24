"""quantlab.costs: InstrumentSpec, CostModel, broker-TSV loading, fallback defaults.

Column names in the TSV fixtures below mirror tools/mql5/ExportBrokerSpecs.mq5
exactly (symbols.tsv / commissions.tsv headers), so these tests double as a
schema-compatibility check against that exporter.
"""

from __future__ import annotations

import sys
import types
import warnings

import numpy as np
import polars as pl
import pytest

from quantlab import config, costs


@pytest.fixture(autouse=True)
def _reset_warn_cache():
    """Fallback-spec warnings are cached per (book, symbol); isolate tests."""
    costs._WARNED_SYMBOLS.clear()
    yield
    costs._WARNED_SYMBOLS.clear()


def _spec(**over):
    base = dict(
        symbol="EURUSD", digits=5, point=0.00001, contract_size=100_000.0, tick_size=0.00001,
        volume_min=0.01, volume_step=0.01, volume_max=500.0, base_ccy="EUR", quote_ccy="USD",
        swap_mode="points", swap_long=-2.5, swap_short=0.3, swap_3day=2,
        commission_per_lot_rt=7.0, stops_level=0.0, calibrated=True,
    )
    base.update(over)
    return costs.InstrumentSpec(**base)


# --------------------------------------------------------------------------- InstrumentSpec

def test_value_per_point_per_lot_fx():
    # 100,000 units/lot * 0.00001 price/point = $1.00 quote-ccy per point per lot
    # (=$10/pip on a 5-digit broker, the standard FX convention).
    spec = _spec(contract_size=100_000.0, point=0.00001)
    assert spec.value_per_point_per_lot == pytest.approx(1.0)


def test_value_per_point_per_lot_gold():
    # 100 oz/lot * 0.01 $/point = $1.00 per point per lot.
    spec = _spec(symbol="XAUUSD", contract_size=100.0, point=0.01, digits=2, tick_size=0.01)
    assert spec.value_per_point_per_lot == pytest.approx(1.0)


def test_instrument_spec_rejects_bad_swap_mode():
    with pytest.raises(ValueError):
        _spec(swap_mode="bogus")


def test_instrument_spec_rejects_bad_swap_3day():
    with pytest.raises(ValueError):
        _spec(swap_3day=9)


def test_instrument_spec_swap_ccy_and_margin_ccy_default(monkeypatch):
    """Regression (red-team N1): with no real broker export, margin_ccy defaults to
    base_ccy and swap_ccy defaults to quote_ccy (the pre-N1 best-effort guess) --
    documented, not silently different from before for every spec built directly."""
    spec = _spec(base_ccy="EUR", quote_ccy="USD")
    assert spec.margin_ccy == "EUR"
    assert spec.swap_ccy == "USD"


def test_instrument_spec_swap_ccy_and_margin_ccy_explicit():
    spec = _spec(base_ccy="EUR", quote_ccy="USD", margin_ccy="GBP", swap_ccy="ACCOUNT")
    assert spec.margin_ccy == "GBP"
    assert spec.swap_ccy == "ACCOUNT"


# --------------------------------------------------------------------------- swap_currency_for (red-team N1)

def test_swap_currency_for_maps_each_mt5_money_mode():
    assert costs.swap_currency_for(
        "SYMBOL_SWAP_MODE_CURRENCY_SYMBOL", base_ccy="USD", margin_ccy="EUR", quote_ccy="JPY"
    ) == "USD"
    assert costs.swap_currency_for(
        "SYMBOL_SWAP_MODE_CURRENCY_MARGIN", base_ccy="USD", margin_ccy="EUR", quote_ccy="JPY"
    ) == "EUR"
    assert costs.swap_currency_for(
        "SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT", base_ccy="USD", margin_ccy="EUR", quote_ccy="JPY"
    ) == "ACCOUNT"


def test_swap_currency_for_non_money_modes_fall_back_to_quote_ccy():
    for raw in (
        "SYMBOL_SWAP_MODE_POINTS", "SYMBOL_SWAP_MODE_INTEREST_CURRENT", "SYMBOL_SWAP_MODE_INTEREST_OPEN",
        "SYMBOL_SWAP_MODE_REOPEN_CURRENT", "SYMBOL_SWAP_MODE_REOPEN_BID", "SYMBOL_SWAP_MODE_DISABLED",
    ):
        assert costs.swap_currency_for(raw, base_ccy="USD", margin_ccy="EUR", quote_ccy="JPY") == "JPY"


# --------------------------------------------------------------------------- CostModel

def test_effective_spread_multiplier_and_floor():
    model = costs.CostModel(spread_multiplier=1.2, extra_spread_points=0.5)
    # 10*1.2+0.5=12.5 ; 20*1.2+0.5=24.5
    out = model.effective_spread(np.array([10.0, 20.0]))
    assert out[0] == pytest.approx(12.5)
    assert out[1] == pytest.approx(24.5)


def test_stressed_default_matches_design_4_2():
    base = costs.CostModel(version_tag="fbs-v1", spread_multiplier=1.0, slippage_points=0.0)
    stressed = base.stressed()  # DESIGN §4.2: 1.5x spread + 1pt slippage
    assert stressed.spread_multiplier == pytest.approx(1.5)
    assert stressed.slippage_points == pytest.approx(1.0)
    assert stressed.stop_fill == "bar_extreme"  # red-team N5: on by default for the stress gate
    assert stressed.version == "fbs-v1+spread_mult1.5+slippage1+stop_fill=bar_extreme"
    # base is frozen/untouched
    assert base.spread_multiplier == 1.0
    assert base.slippage_points == 0.0
    assert base.stop_fill == "level"
    assert base.version == "fbs-v1"


def test_stressed_compounds_on_existing_multiplier():
    base = costs.CostModel(version_tag="fbs-v1", spread_multiplier=1.1, slippage_points=0.2)
    stressed = base.stressed(spread_mult=1.5, extra_slippage=1.0)
    assert stressed.spread_multiplier == pytest.approx(1.1 * 1.5)
    assert stressed.slippage_points == pytest.approx(0.2 + 1.0)


def test_stressed_keeps_the_base_stop_fill_when_told_to():
    base = costs.CostModel(version_tag="fbs-v1")
    stressed = base.stressed(stop_fill="level")
    assert stressed.stop_fill == "level"
    assert stressed.version == "fbs-v1+spread_mult1.5+slippage1"


# --------------------------------------------------------------------------- pip_points (red-team M2)

def test_pip_points_fx_5_digit_and_3_digit_jpy_price_a_pip_at_10_points():
    eurusd = _spec(symbol="EURUSD", digits=5, point=0.00001, base_ccy="EUR", quote_ccy="USD")
    usdjpy = _spec(symbol="USDJPY", digits=3, point=0.001, base_ccy="USD", quote_ccy="JPY")
    assert costs.pip_points(eurusd) == pytest.approx(10.0)
    assert costs.pip_points(usdjpy) == pytest.approx(10.0)


def test_pip_points_fx_legacy_2_or_4_digit_whole_pip_quote_is_1_point():
    eurusd4 = _spec(symbol="EURUSD", digits=4, point=0.0001, base_ccy="EUR", quote_ccy="USD")
    usdjpy2 = _spec(symbol="USDJPY", digits=2, point=0.01, base_ccy="USD", quote_ccy="JPY")
    assert costs.pip_points(eurusd4) == pytest.approx(1.0)
    assert costs.pip_points(usdjpy2) == pytest.approx(1.0)


def test_pip_points_metals_use_the_documented_per_metal_usd_convention():
    """Red-team N6: silver is now its own 0.01 USD convention, not gold's 0.1 -- the old
    shared 0.1 USD number priced a silver "pip" at 2.6x the median FBS D1 spread."""
    xau = _spec(symbol="XAUUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XAU", quote_ccy="USD")
    xag = _spec(symbol="XAGUSD", digits=3, point=0.001, contract_size=5_000.0, tick_size=0.001,
                base_ccy="XAG", quote_ccy="USD")
    xpt = _spec(symbol="XPTUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XPT", quote_ccy="USD")
    xpd = _spec(symbol="XPDUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XPD", quote_ccy="USD")
    assert costs.pip_points(xau) == pytest.approx(10.0)    # 0.1 / 0.01
    assert costs.pip_points(xag) == pytest.approx(10.0)    # 0.01 / 0.001 (N6 fix, was 100.0)
    assert costs.pip_points(xpt) == pytest.approx(10.0)    # 0.1 / 0.01 (gold's convention)
    assert costs.pip_points(xpd) == pytest.approx(10.0)    # 0.1 / 0.01 (gold's convention)


def test_pip_points_crypto_and_b3_futures_default_to_1_tick_1_point():
    """``pip_points`` itself is unchanged for these classes (still a flat 1-point
    fallback, kept for backward-compatible FX/metals-only display code, e.g.
    gates.py's "1 pip = N points" text) -- see ``test_stress_slippage_points_*`` below
    for what ``CostModel.stressed()`` actually uses for these classes since N6."""
    btc = _spec(symbol="BTCUSD", digits=2, point=0.01, contract_size=1.0, tick_size=0.01,
                base_ccy="BTC", quote_ccy="USD")
    win = _spec(symbol="WINZ26", digits=0, point=5.0, contract_size=0.20, tick_size=5.0,
                base_ccy="IBOV", quote_ccy="BRL", volume_min=1.0, volume_step=1.0)
    assert costs.pip_points(btc) == pytest.approx(1.0)
    assert costs.pip_points(win) == pytest.approx(1.0)


def test_stressed_extra_slippage_pips_converts_via_pip_points_when_spec_given():
    eurusd = _spec(symbol="EURUSD", digits=5, point=0.00001, base_ccy="EUR", quote_ccy="USD")
    usdjpy = _spec(symbol="USDJPY", digits=3, point=0.001, base_ccy="USD", quote_ccy="JPY")
    xau = _spec(symbol="XAUUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XAU", quote_ccy="USD")
    base = costs.CostModel(version_tag="fbs-v1")

    # 1 pip default -> pip_points(spec) points of slippage_points, per DESIGN M2's own examples.
    assert base.stressed(spec=eurusd).slippage_points == pytest.approx(10.0)
    assert base.stressed(spec=usdjpy).slippage_points == pytest.approx(10.0)
    assert base.stressed(spec=xau).slippage_points == pytest.approx(10.0)
    # scales with extra_slippage_pips
    assert base.stressed(extra_slippage_pips=2.0, spec=eurusd).slippage_points == pytest.approx(20.0)
    # compounds on an already non-zero base slippage_points, like the old extra_slippage kwarg did
    with_prior = costs.CostModel(version_tag="fbs-v1", slippage_points=0.5)
    assert with_prior.stressed(spec=eurusd).slippage_points == pytest.approx(10.5)


def test_stressed_without_spec_keeps_the_legacy_1_point_fallback():
    """No spec -> the pip size is unknown -> degrades to the pre-M2 numeric default (1 point),
    so a caller that has not yet been updated to pass spec= (e.g. today's gates.py call site)
    sees no change in slippage_points from this fix alone -- only the engine's wider fill-type
    scope (which needs no spec to take effect)."""
    base = costs.CostModel(version_tag="fbs-v1")
    assert base.stressed().slippage_points == pytest.approx(1.0)


def test_stressed_old_extra_slippage_kwarg_still_bypasses_pip_conversion():
    eurusd = _spec(symbol="EURUSD", digits=5, point=0.00001, base_ccy="EUR", quote_ccy="USD")
    base = costs.CostModel(version_tag="fbs-v1")
    stressed = base.stressed(extra_slippage=3.0, spec=eurusd)  # old kwarg wins over pips+spec
    assert stressed.slippage_points == pytest.approx(3.0)


# --------------------------------------------------------------------------- stress_slippage_points (red-team N6)
#
# Reference median D1 bar spread (points) over the FBS/B3 dev window 2024-01..06, taken from
# the phase-1 red-team's own probe (research/audits/probes/phase1/r1_p09_cost_stress.py/.out,
# "Re-verification round 1 (e64efcf)", finding N6) -- used here only to hand-compute each
# class's stress slippage as a documented multiple of what the market actually quotes. This
# worktree has no bar data mounted (data/ is gitignored bulk data, worktree-local), so the
# table below is evidence, not a live recomputation; the crypto/indices lookup path itself is
# exercised separately with a monkeypatched quantlab.data below.
_REF_MEDIAN_SPREAD_POINTS = {
    "EURUSD": 30.0, "USDJPY": 73.0, "XAUUSD": 28.0, "XAGUSD": 39.0,
    "BTCUSD": 1942.0, "ETHUSD": 203.0, "WINZ26": 1.0, "WDOZ26": 1.0,
}


@pytest.mark.parametrize("symbol,spec_kwargs,expected_points", [
    ("EURUSD", dict(digits=5, point=0.00001, tick_size=0.00001, base_ccy="EUR", quote_ccy="USD"),
     10.0),
    ("USDJPY", dict(digits=3, point=0.001, tick_size=0.001, base_ccy="USD", quote_ccy="JPY"),
     10.0),
    ("XAUUSD", dict(digits=2, point=0.01, tick_size=0.01, contract_size=100.0,
                     base_ccy="XAU", quote_ccy="USD"), 10.0),
    ("XAGUSD", dict(digits=3, point=0.001, tick_size=0.001, contract_size=5_000.0,
                     base_ccy="XAG", quote_ccy="USD"), 10.0),  # N6 fix: was 100.0 (2.6x spread)
    ("BTCUSD", dict(digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                     base_ccy="BTC", quote_ccy="USD"), 1942.0),
    ("ETHUSD", dict(digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                     base_ccy="ETH", quote_ccy="USD"), 203.0),
    ("WINZ26", dict(digits=0, point=1.0, tick_size=5.0, contract_size=0.20, base_ccy="IBOV",
                     quote_ccy="BRL", volume_min=1.0, volume_step=1.0), 5.0),
    ("WDOZ26", dict(digits=3, point=0.001, tick_size=0.5, contract_size=10.0, base_ccy="USD",
                     quote_ccy="BRL", volume_min=1.0, volume_step=1.0), 500.0),
], ids=["EURUSD_fx", "USDJPY_fx", "XAUUSD_metal", "XAGUSD_metal", "BTCUSD_crypto",
        "ETHUSD_crypto", "WIN_b3", "WDO_b3"])
def test_stress_slippage_points_table(symbol, spec_kwargs, expected_points):
    """Table-driven (red-team N6): each asset class's stress slippage, in points and as a
    multiple of the reference median dev-window D1 spread. Crypto has no spec-derived
    convention, so it needs median_spread_points= (or a data lookup, tested separately);
    every other class is pure spec arithmetic."""
    spec = _spec(symbol=symbol, **spec_kwargs)
    ref_spread = _REF_MEDIAN_SPREAD_POINTS[symbol]
    if symbol in ("BTCUSD", "ETHUSD"):
        points = costs.stress_slippage_points(spec, median_spread_points=ref_spread)
    else:
        points = costs.stress_slippage_points(spec)
    assert points == pytest.approx(expected_points)
    ratio = points / ref_spread
    print(f"{symbol}: {points:g} pts = {ratio:.3f}x median spread ({ref_spread:g} pts)")


def test_stress_slippage_points_fx_matches_pip_points():
    eurusd = _spec(symbol="EURUSD", digits=5, point=0.00001, base_ccy="EUR", quote_ccy="USD")
    usdjpy = _spec(symbol="USDJPY", digits=3, point=0.001, base_ccy="USD", quote_ccy="JPY")
    assert costs.stress_slippage_points(eurusd) == costs.pip_points(eurusd) == pytest.approx(10.0)
    assert costs.stress_slippage_points(usdjpy) == costs.pip_points(usdjpy) == pytest.approx(10.0)


def test_stress_slippage_points_silver_is_one_tenth_of_gold():
    xau = _spec(symbol="XAUUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XAU", quote_ccy="USD")
    xag = _spec(symbol="XAGUSD", digits=3, point=0.001, contract_size=5_000.0, tick_size=0.001,
                base_ccy="XAG", quote_ccy="USD")
    assert costs.stress_slippage_points(xau) == pytest.approx(10.0)   # 0.1 USD / 0.01 point
    assert costs.stress_slippage_points(xag) == pytest.approx(10.0)   # 0.01 USD / 0.001 point


def test_stress_slippage_points_platinum_and_palladium_documented_convention():
    xpt = _spec(symbol="XPTUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XPT", quote_ccy="USD")
    xpd = _spec(symbol="XPDUSD", digits=2, point=0.01, contract_size=100.0, tick_size=0.01,
                base_ccy="XPD", quote_ccy="USD")
    assert costs.stress_slippage_points(xpt) == pytest.approx(10.0)   # 0.1 USD, gold's convention
    assert costs.stress_slippage_points(xpd) == pytest.approx(10.0)


def test_stress_slippage_points_b3_futures_use_one_tick_converted_to_engine_points():
    """Red-team N6: tick_size is a *price* increment (WIN's 5 index points, WDO's 0.5), not
    already in the engine's points unit -- dividing by spec.point is required (DESIGN §4.3).
    Skipping that conversion is exactly how the pre-fix code priced these at 0.2/0.002 "pip"."""
    win = _spec(symbol="WINZ26", digits=0, point=1.0, tick_size=5.0, contract_size=0.20,
                base_ccy="IBOV", quote_ccy="BRL", volume_min=1.0, volume_step=1.0)
    wdo = _spec(symbol="WDOZ26", digits=3, point=0.001, tick_size=0.5, contract_size=10.0,
                base_ccy="USD", quote_ccy="BRL", volume_min=1.0, volume_step=1.0)
    assert costs.stress_slippage_points(win) == pytest.approx(5.0)     # 5.0 / 1.0
    assert costs.stress_slippage_points(wdo) == pytest.approx(500.0)   # 0.5 / 0.001


def test_stress_slippage_points_crypto_uses_given_median_override():
    btc = _spec(symbol="BTCUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="BTC", quote_ccy="USD")
    assert costs.stress_slippage_points(btc, median_spread_points=1942.0) == pytest.approx(1942.0)


def test_stress_slippage_points_crypto_computes_median_spread_via_data_load_bars(monkeypatch):
    """No override given -> reads the symbol's own median dev-window D1 spread via
    quantlab.data.load_bars (spread column is already in engine points)."""
    calls = []

    def fake_load_bars(symbol, timeframe, *, book=None, **kw):
        calls.append((symbol, timeframe, book))
        return pl.DataFrame({"spread": [1900.0, 1942.0, 2000.0]})

    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = fake_load_bars
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    btc = _spec(symbol="BTCUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="BTC", quote_ccy="USD")
    assert costs.stress_slippage_points(btc, book="FBS") == pytest.approx(1942.0)
    assert calls == [("BTCUSD", "D1", "FBS")]


def test_stress_slippage_points_raises_when_no_override_and_data_unavailable(monkeypatch):
    """Red-team N6: never falls back to a fixed 1-point default -- a crypto/indices symbol
    with no override and no data lookup available must raise, not silently price its stress
    slippage as a no-op."""
    def fake_load_bars(symbol, timeframe, *, book=None, **kw):
        raise FileNotFoundError("no manifest.json in this environment")

    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = fake_load_bars
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    btc = _spec(symbol="BTCUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="BTC", quote_ccy="USD")
    with pytest.raises(ValueError, match="no pip/tick convention"):
        costs.stress_slippage_points(btc)


def test_stress_slippage_points_raises_on_non_positive_median(monkeypatch):
    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = lambda symbol, timeframe, *, book=None, **kw: pl.DataFrame(
        {"spread": [0.0, 0.0]}
    )
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    eth = _spec(symbol="ETHUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="ETH", quote_ccy="USD")
    with pytest.raises(ValueError, match="non-positive or NaN"):
        costs.stress_slippage_points(eth)


def test_stress_slippage_points_unknown_class_raises_without_override_or_data(monkeypatch):
    """Not FX (digits with a non-alpha symbol), not a metal, not WIN/WDO -- an index CFD
    this catalog doesn't carry yet. With no override and no data, this must raise rather
    than silently returning 1 point (the original N6 bug, generalised)."""
    def fake_load_bars(symbol, timeframe, *, book=None, **kw):
        raise KeyError(symbol)

    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = fake_load_bars
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    spx = _spec(symbol="SPX500", digits=1, point=0.1, tick_size=0.1, contract_size=1.0,
                base_ccy="SPX", quote_ccy="USD")
    with pytest.raises(ValueError, match="no pip/tick convention"):
        costs.stress_slippage_points(spx)


def test_stressed_forwards_book_and_computes_median_spread_for_crypto(monkeypatch):
    calls = []

    def fake_load_bars(symbol, timeframe, *, book=None, **kw):
        calls.append((symbol, timeframe, book))
        return pl.DataFrame({"spread": [1900.0, 1942.0, 2000.0]})

    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = fake_load_bars
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    btc = _spec(symbol="BTCUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="BTC", quote_ccy="USD")
    base = costs.CostModel(version_tag="fbs-v2")
    stressed = base.stressed(spec=btc)  # book defaults to "FBS"; no gates.py call site changes
    assert stressed.slippage_points == pytest.approx(1942.0)
    assert calls == [("BTCUSD", "D1", "FBS")]


def test_stressed_median_spread_points_override_skips_data_lookup(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("should not call data.load_bars when median_spread_points is given")

    fake_data = types.ModuleType("quantlab.data")
    fake_data.load_bars = fail
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    eth = _spec(symbol="ETHUSD", digits=2, point=0.01, tick_size=0.01, contract_size=1.0,
                base_ccy="ETH", quote_ccy="USD")
    base = costs.CostModel(version_tag="fbs-v2")
    stressed = base.stressed(spec=eth, median_spread_points=203.0)
    assert stressed.slippage_points == pytest.approx(203.0)


def test_stressed_b3_futures_use_one_tick_no_data_lookup_needed():
    win = _spec(symbol="WINZ26", digits=0, point=1.0, tick_size=5.0, contract_size=0.20,
                base_ccy="IBOV", quote_ccy="BRL", volume_min=1.0, volume_step=1.0)
    wdo = _spec(symbol="WDOZ26", digits=3, point=0.001, tick_size=0.5, contract_size=10.0,
                base_ccy="USD", quote_ccy="BRL", volume_min=1.0, volume_step=1.0)
    base = costs.CostModel(version_tag="fbs-v2")
    assert base.stressed(spec=win).slippage_points == pytest.approx(5.0)
    assert base.stressed(spec=wdo).slippage_points == pytest.approx(500.0)


# --------------------------------------------------------------------------- version (red-team N5)

def test_version_is_just_the_tag_when_every_knob_is_at_its_default():
    assert costs.CostModel(version_tag="fbs-v1").version == "fbs-v1"
    # DESIGN §8's own default tag; bumped v0->v1 for red-team M2 (engine slippage scope),
    # then v1->v2 for red-team N6 (per-asset-class stress slippage -- see costs.py's class
    # docstring for exactly what "the same nominal 1 pip stress" now means differently).
    assert costs.CostModel().version == "fbs-v2-uncalibrated"


def test_version_encodes_each_non_default_knob_in_a_fixed_order():
    """Deterministic, fixed order regardless of which knobs are set (red-team N5)."""
    m = costs.CostModel(
        version_tag="fbs-v1",
        spread_multiplier=1.2,
        extra_spread_points=0.5,
        slippage_points=2.0,
        swap_multiplier=0.5,
        stop_fill="bar_extreme",
    )
    assert m.version == "fbs-v1+spread_mult1.2+extra_spread0.5+slippage2+swap_mult0.5+stop_fill=bar_extreme"


def test_version_never_collides_across_different_knob_combinations():
    """The literal bug this fix closes: `CostModel(stop_fill="bar_extreme").version` used to be
    identical to the plain default's version, even though the two models fill stops very
    differently."""
    default = costs.CostModel()
    bar_extreme = costs.CostModel(stop_fill="bar_extreme")
    stressed = costs.CostModel(spread_multiplier=1.5)
    assert default.version != bar_extreme.version
    assert default.version != stressed.version
    assert bar_extreme.version != stressed.version
    assert bar_extreme.version == "fbs-v2-uncalibrated+stop_fill=bar_extreme"


def test_version_does_not_accumulate_stale_suffixes_across_repeated_stressed_calls():
    """Because `version` is computed fresh from the model's own current field values (never
    string-concatenated across `replace()`), stacking modifications can never produce a stale
    or duplicated suffix (red-team N5)."""
    base = costs.CostModel(version_tag="fbs-v1", spread_multiplier=1.2)
    once = base.stressed(spread_mult=1.5, extra_slippage=1.0)  # spread_multiplier -> 1.2*1.5 = 1.8
    twice = once.stressed(spread_mult=2.0, extra_slippage=0.5, stop_fill="bar_extreme")
    assert once.version == "fbs-v1+spread_mult1.8+slippage1+stop_fill=bar_extreme"
    # spread_multiplier -> 1.8*2.0 = 3.6 ; slippage_points -> 1.0+0.5 = 1.5 -- exactly one
    # occurrence of each knob, describing the FINAL state, never "spread_mult1.8+spread_mult3.6"
    assert twice.version == "fbs-v1+spread_mult3.6+slippage1.5+stop_fill=bar_extreme"
    assert twice.version.count("spread_mult") == 1
    assert twice.version.count("slippage") == 1


# --------------------------------------------------------------------------- TSV loading

_SYMBOLS_HEADER = [
    "symbol", "path", "description", "calc_mode", "trade_mode", "digits", "point",
    "tick_size", "tick_value", "tick_value_profit", "tick_value_loss", "contract_size",
    "currency_base", "currency_profit", "currency_margin",
    "volume_min", "volume_max", "volume_step",
    "spread_now_points", "spread_float", "stops_level", "freeze_level",
    "swap_mode", "swap_long", "swap_short", "swap_3day",
    "margin_initial", "margin_maintenance", "margin_hedged",
    "margin_rate_init_buy", "margin_rate_maint_buy", "margin_1lot_buy_now", "filling_mode",
]

_COMMISSIONS_HEADER = [
    "symbol", "deals", "lots_opened", "commission_total", "commission_per_lot_roundtrip",
    "swap_total", "fee_total", "first_deal", "last_deal",
]


def _write_symbols_row(path, **over):
    row = dict(
        symbol="EURUSD", path="Forex\\EURUSD", description="Euro vs US Dollar",
        calc_mode="SYMBOL_CALC_MODE_FOREX", trade_mode="SYMBOL_TRADE_MODE_FULL",
        digits=5, point=0.00001, tick_size=0.00001, tick_value=1.0, tick_value_profit=1.0,
        tick_value_loss=1.0, contract_size=100000.0, currency_base="EUR", currency_profit="USD",
        currency_margin="EUR", volume_min=0.01, volume_max=500.0, volume_step=0.01,
        spread_now_points=10, spread_float="true", stops_level=0, freeze_level=0,
        swap_mode="SYMBOL_SWAP_MODE_POINTS", swap_long=-2.5, swap_short=0.3, swap_3day="WEDNESDAY",
        margin_initial=0.0, margin_maintenance=0.0, margin_hedged=0.0,
        margin_rate_init_buy=0.0, margin_rate_maint_buy=0.0, margin_1lot_buy_now=0.0, filling_mode=1,
    )
    row.update(over)
    with open(path, "w") as fh:
        fh.write("\t".join(_SYMBOLS_HEADER) + "\n")
        fh.write("\t".join(str(row[c]) for c in _SYMBOLS_HEADER) + "\n")


def _write_commissions_row(path, **over):
    row = dict(
        symbol="EURUSD", deals=42, lots_opened=10.0, commission_total=70.0,
        commission_per_lot_roundtrip=7.0, swap_total=-12.5, fee_total=0.0,
        first_deal="2026.01.01 00:00:00", last_deal="2026.06.01 00:00:00",
    )
    row.update(over)
    with open(path, "w") as fh:
        fh.write("\t".join(_COMMISSIONS_HEADER) + "\n")
        fh.write("\t".join(str(row[c]) for c in _COMMISSIONS_HEADER) + "\n")


def test_load_instrument_from_broker_tsv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv")
    _write_commissions_row(tmp_path / "FBS_commissions.tsv")

    spec = costs.load_instrument("EURUSD", book="FBS")

    assert spec.calibrated is True
    assert spec.digits == 5
    assert spec.point == pytest.approx(0.00001)
    assert spec.contract_size == pytest.approx(100_000.0)
    assert spec.tick_size == pytest.approx(0.00001)
    assert (spec.volume_min, spec.volume_step, spec.volume_max) == pytest.approx((0.01, 0.01, 500.0))
    assert spec.base_ccy == "EUR" and spec.quote_ccy == "USD"
    assert spec.margin_ccy == "EUR"  # from currency_margin (red-team N1)
    assert spec.swap_mode == "points"  # normalised from SYMBOL_SWAP_MODE_POINTS
    assert spec.swap_ccy == "USD"  # points mode -> falls back to quote_ccy (red-team N1)
    assert spec.swap_long == pytest.approx(-2.5)
    assert spec.swap_short == pytest.approx(0.3)
    assert spec.swap_3day == 2  # WEDNESDAY -> Mon=0..Sun=6
    assert spec.commission_per_lot_rt == pytest.approx(7.0)
    assert spec.stops_level == pytest.approx(0.0)
    assert spec.swap_every_day is False  # EURUSD is not a crypto CFD


@pytest.mark.parametrize("raw_mode,expected_swap_ccy", [
    ("SYMBOL_SWAP_MODE_CURRENCY_SYMBOL", "EUR"),     # base_ccy
    ("SYMBOL_SWAP_MODE_CURRENCY_MARGIN", "GBP"),     # currency_margin
    ("SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT", "ACCOUNT"),
])
def test_load_instrument_from_broker_tsv_swap_ccy_per_mt5_money_mode(
    tmp_path, monkeypatch, raw_mode, expected_swap_ccy
):
    """Regression (red-team N1, MAJOR, p17 d): MT5's three "money" swap modes are NOT
    interchangeable currencies -- the loader must derive swap_ccy from the RAW enum
    string (before normalize_swap_mode collapses all three into "money"), using the
    row's own currency_margin (not just base/quote) for CURRENCY_MARGIN."""
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv", swap_mode=raw_mode, currency_margin="GBP")
    _write_commissions_row(tmp_path / "FBS_commissions.tsv")

    spec = costs.load_instrument("EURUSD", book="FBS")
    assert spec.swap_mode == "money"  # all 3 collapse to the same canonical mode
    assert spec.margin_ccy == "GBP"
    assert spec.swap_ccy == expected_swap_ccy


def test_load_instrument_from_broker_tsv_crypto_gets_swap_every_day(tmp_path, monkeypatch):
    """Regression (red-team minor-4): a crypto symbol loaded from the broker
    TSV must default to swap_every_day=True (the export itself has no
    explicit field for this; costs.py infers it from the symbol)."""
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv", symbol="BTCUSD",
                       currency_base="BTC", currency_profit="USD")
    _write_commissions_row(tmp_path / "FBS_commissions.tsv", symbol="BTCUSD")

    spec = costs.load_instrument("BTCUSD", book="FBS")
    assert spec.calibrated is True
    assert spec.swap_every_day is True


def test_load_instrument_tsv_normalizes_friday_swap_day(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv", swap_3day="FRIDAY")

    spec = costs.load_instrument("EURUSD", book="FBS")
    assert spec.swap_3day == 4


def test_load_instrument_tsv_missing_commissions_file_warns_and_zeros(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv")
    # no commissions.tsv written

    with pytest.warns(UserWarning, match="commissions.tsv"):
        spec = costs.load_instrument("EURUSD", book="FBS")
    assert spec.calibrated is True  # symbol terms are still real, just no commission history yet
    assert spec.commission_per_lot_rt == 0.0


def test_load_instrument_missing_symbol_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    _write_symbols_row(tmp_path / "FBS_symbols.tsv")  # only has EURUSD

    fake_data = types.ModuleType("quantlab.data")
    fake_data.infer_point = lambda symbol: 0.01
    fake_data.symbol_currencies = lambda symbol: ("XAU", "USD")
    monkeypatch.setitem(sys.modules, "quantlab.data", fake_data)

    spec = costs.load_instrument("XAUUSD", book="FBS")
    assert spec.calibrated is False


# --------------------------------------------------------------------------- fallback defaults

@pytest.fixture
def fake_data_module(monkeypatch):
    """Install a stand-in quantlab.data with the 2 functions costs.py is allowed to call."""
    points = {"EURUSD": 0.00001, "USDJPY": 0.001, "WINZ26": 1.0, "WDOZ26": 0.5, "BTCUSD": 1.0}
    currencies = {"EURUSD": ("EUR", "USD"), "USDJPY": ("USD", "JPY"), "BTCUSD": ("BTC", "USD")}
    mod = types.ModuleType("quantlab.data")
    mod.infer_point = lambda symbol: points[symbol]
    mod.symbol_currencies = lambda symbol: currencies[symbol]
    monkeypatch.setitem(sys.modules, "quantlab.data", mod)
    return mod


def test_fallback_fx_default_and_warns_once(tmp_path, monkeypatch, fake_data_module):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)  # empty: no TSVs exist

    with pytest.warns(UserWarning, match="uncalibrated"):
        spec = costs.load_instrument("EURUSD", book="FBS")
    assert spec.calibrated is False
    assert spec.contract_size == pytest.approx(100_000.0)
    assert spec.point == pytest.approx(0.00001)
    assert spec.base_ccy == "EUR" and spec.quote_ccy == "USD"
    assert spec.swap_mode == "points"
    assert spec.swap_every_day is False

    # second call for the SAME symbol must not warn again
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        costs.load_instrument("EURUSD", book="FBS")  # would raise if it warned


def test_fallback_crypto_default_sets_swap_every_day(tmp_path, monkeypatch, fake_data_module):
    """Regression (red-team minor-4): the uncalibrated fallback for a crypto
    symbol must default swap_every_day=True (FBS-like brokers charge crypto
    CFD swap on every night of the week, unlike FX/metals)."""
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)

    with pytest.warns(UserWarning, match="uncalibrated"):
        spec = costs.load_instrument("BTCUSD", book="FBS")
    assert spec.calibrated is False
    assert spec.swap_every_day is True


def test_fallback_b3_win_and_wdo_defaults(tmp_path, monkeypatch, fake_data_module):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)

    win = costs.load_instrument("WINZ26", book="B3")
    assert win.calibrated is False
    assert win.contract_size == pytest.approx(0.20)   # R$0.20/point
    assert win.tick_size == pytest.approx(5.0)
    assert (win.volume_min, win.volume_step) == pytest.approx((1.0, 1.0))
    assert win.quote_ccy == "BRL"

    wdo = costs.load_instrument("WDOZ26", book="B3")
    assert wdo.calibrated is False
    assert wdo.contract_size == pytest.approx(10.0)   # R$10/point
    assert wdo.tick_size == pytest.approx(0.5)
    assert wdo.quote_ccy == "BRL"


def test_fallback_b3_unknown_symbol_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BROKER_DIR", tmp_path)
    with pytest.raises(ValueError, match="unknown B3 symbol"):
        costs.load_instrument("INDFUT1", book="B3")
