"""S2 code red-team probe: mandatory audits at grid corners (synthetic + real dev EURUSD H4)."""
import itertools, time
from datetime import datetime
import polars as pl
from quantlab import data
from quantlab.strategies import toy_tsmom
from quantlab.strategies.toy_tsmom import ToyTsmom, ToyTsmomParams
from quantlab.testing import assert_no_lookahead, assert_engine_causal, lint_strategy_source, synthetic_bars
from quantlab.costs import InstrumentSpec
import quantlab.strategies.base as base

print("lint toy_tsmom:", lint_strategy_source(toy_tsmom.__file__))
print("lint base:", lint_strategy_source(base.__file__))
SPEC = InstrumentSpec(symbol="X", digits=5, point=1e-5, contract_size=1e5, tick_size=1e-5, volume_min=0.01,
    volume_step=0.01, volume_max=100.0, base_ccy="EUR", quote_ccy="USD", swap_mode="points", swap_long=0.0,
    swap_short=0.0, swap_3day=2, commission_per_lot_rt=0.0, stops_level=0.0, calibrated=True)
corners = [(10, 1.0, 6), (120, 4.0, 60), (10, 4.0, 60), (120, 1.0, 6), (60, 2.5, 30)]
syn = synthetic_bars(1500, seed=99, timeframe="H4")
for lb, sm, h in corners:
    f = (lambda lb=lb, sm=sm, h=h: ToyTsmom(ToyTsmomParams(lookback=lb, stop_mult=sm, hold=h)))
    t0 = time.time(); assert_no_lookahead(f, syn, n_checks=40, seed=3)
    assert_engine_causal(f(), syn, SPEC, timeframe="H4", n_checks=10, seed=4, min_history=200)
    print(f"synthetic {lb},{sm},{h}: clean ({time.time()-t0:.1f}s)")
real = data.load_bars("EURUSD", "H4", book="FBS", end=datetime(2025, 5, 15))
print("real rows", real.height, real["ts"].min(), real["ts"].max())
seg = real.slice(real.height - 2500, 2500)
for lb, sm, h in [(10, 1.0, 6), (60, 2.5, 30)]:
    f = (lambda lb=lb, sm=sm, h=h: ToyTsmom(ToyTsmomParams(lookback=lb, stop_mult=sm, hold=h)))
    t0 = time.time(); assert_no_lookahead(f, seg, n_checks=40, seed=5)
    print(f"real seg {lb},{sm},{h}: no_lookahead clean ({time.time()-t0:.1f}s)")
