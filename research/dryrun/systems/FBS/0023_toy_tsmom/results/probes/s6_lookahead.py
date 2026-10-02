"""S6 red team: look-ahead re-audit at the S5-selected config, the raw argmax, and the net/gross
best configs, on real dev EURUSD H4 (last 3000 bars) + engine causality with the M1 path."""
import warnings
warnings.filterwarnings("ignore")
from quantlab import data
from quantlab.costs import load_instrument
from quantlab.strategies.toy_tsmom import ToyTsmom, ToyTsmomParams
from quantlab.testing import assert_no_lookahead, assert_engine_causal, lint_strategy_source

if __name__ == "__main__":
    print("lint:", lint_strategy_source("quantlab/strategies/toy_tsmom.py"))
    bars = data.load_bars("EURUSD", "H4", book="FBS", end="2025-05-15").tail(3000)
    m1 = data.load_bars("EURUSD", "M1", book="FBS", start=bars["ts"][0], end="2025-05-15")
    spec = load_instrument("EURUSD", book="FBS")
    for p in [(80, 1.0, 18), (70, 1.0, 12), (90, 1.0, 18), (120, 4.0, 60), (10, 1.0, 6)]:
        f = (lambda p=p: ToyTsmom(ToyTsmomParams(*p)))
        assert_no_lookahead(f, bars, n_checks=40, seed=7)
        assert_engine_causal(f(), bars, spec, m1=m1, timeframe="H4", n_checks=8, seed=3, min_history=300)
        print("clean:", p)
