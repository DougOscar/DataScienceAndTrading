"""Proves the DESIGN §7 performance fix to ``quantlab.engine.run_backtest``
changed nothing about its results (performance-engineer's mandate: "must not
change results ... outputs must match bit-for-bit").

The only line changed was how ``entry_ts``/``exit_ts`` get built:

    # before (O(n_bars) per call -- >90% of wall time on the 3.7M M1-bar
    # kernel benchmark was this one line converting *every* bar's timestamp
    # to a Python object just to look up entry_idx.shape[0] ~ O(n_trades) of
    # them):
    ts_list = bars["ts"].to_list()
    entry_ts = [ts_list[i] for i in entry_idx]
    exit_ts = [ts_list[i] for i in exit_idx]

    # after (O(n_trades) -- gather the needed rows *before* converting to
    # Python objects, instead of after):
    ts_col = bars["ts"]
    entry_ts = ts_col.gather(entry_idx).to_list()
    exit_ts = ts_col.gather(exit_idx).to_list()

Everything downstream of that (swap/nights, trades-frame assembly) is
untouched code operating on whatever entry_ts/exit_ts turn out to be, so
proving those two lists are identical between the two extraction strategies
is sufficient to prove the whole ``trades`` frame (and anything computed from
it, like ``sizing.daily_equity``) is unaffected. This is checked here on a
real, multi-year EURUSD H1+M1 window (not a toy fixture) using the *actual*
entry_idx/exit_idx a real run produces, so it also exercises realistic
gap/weekend/holiday patterns in the timestamp column.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "bench"))
from _strategy import sma_cross_atr_signals  # noqa: E402

from quantlab import costs, data
from quantlab.engine import run_backtest
from quantlab.sizing import apply_sizing, daily_equity

DEV_START = "2016-05-02"
DEV_END = "2025-05-14"


@pytest.mark.slow
def test_gather_ts_matches_pre_fix_full_to_list_reference():
    warnings.filterwarnings("ignore")
    symbol, htf = "EURUSD", "H1"
    bars = data.load_bars(symbol, htf, start=DEV_START, end=DEV_END)
    m1 = data.load_bars(symbol, "M1", start=DEV_START, end=DEV_END)
    sig = sma_cross_atr_signals(bars)
    spec = costs.load_instrument(symbol, book="FBS")
    cost = costs.CostModel(version="fbs-v0-uncalibrated")

    result = run_backtest(bars, sig, spec, cost, m1=m1)
    assert result.trades.height > 50, "need a real sample of trades for this to mean anything"

    # Reference extraction: the pre-fix algorithm, replayed against the exact
    # entry_idx/exit_idx this run produced. Those indices come out of the
    # untouched numba kernel, so this isolates exactly the two changed lines.
    ts_list_reference = bars["ts"].to_list()
    entry_idx = result.trades["entry_idx"].to_numpy()
    exit_idx = result.trades["exit_idx"].to_numpy()
    reference_entry_ts = [ts_list_reference[i] for i in entry_idx]
    reference_exit_ts = [ts_list_reference[i] for i in exit_idx]

    assert result.trades["entry_ts"].to_list() == reference_entry_ts
    assert result.trades["exit_ts"].to_list() == reference_exit_ts

    # Belt-and-suspenders: the two downstream consumers of entry_ts/exit_ts
    # (swap/nights -> trades columns, and the whole apply_sizing/daily_equity
    # chain) still run end-to-end and produce a sane, non-empty result.
    sized = apply_sizing(result.trades, spec, mode="fixed_fraction", risk_fraction=0.01,
                          account_ccy=spec.quote_ccy)
    daily = daily_equity(bars, result, sized)
    assert sized.height == result.trades.height
    assert daily.height > 0
    assert daily["equity"].null_count() == 0


@pytest.mark.slow
def test_gather_matches_to_list_on_a_second_real_symbol():
    """Same proof, second real symbol/timeframe/window shape (H4, not H1) --
    guards against the fix being coincidentally right only for H1-sized
    entry_idx/exit_idx patterns."""
    warnings.filterwarnings("ignore")
    symbol, htf = "XAUUSD", "H4"
    bars = data.load_bars(symbol, htf, start=DEV_START, end=DEV_END)
    m1 = data.load_bars(symbol, "M1", start=DEV_START, end=DEV_END)
    sig = sma_cross_atr_signals(bars)
    spec = costs.load_instrument(symbol, book="FBS")
    cost = costs.CostModel(version="fbs-v0-uncalibrated")

    result = run_backtest(bars, sig, spec, cost, m1=m1)
    assert result.trades.height > 20

    ts_list_reference = bars["ts"].to_list()
    entry_idx = result.trades["entry_idx"].to_numpy()
    exit_idx = result.trades["exit_idx"].to_numpy()
    assert result.trades["entry_ts"].to_list() == [ts_list_reference[i] for i in entry_idx]
    assert result.trades["exit_ts"].to_list() == [ts_list_reference[i] for i in exit_idx]
