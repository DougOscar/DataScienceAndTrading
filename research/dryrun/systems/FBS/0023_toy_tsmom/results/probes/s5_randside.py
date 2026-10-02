"""Random-side variant of toy_tsmom for the S5 mechanism ablation (card: "Mechanism ablation").

Identical entry timestamps (every ROC zero-cross), ATR stop, `hold` time exit, reversal timing and
sizing; the side of each cross is replaced by an independent fair coin (seed `side_seed`).
side_seed < 0 reproduces toy_tsmom exactly (sanity check against the trial store).

Engine limitation (documented deviation from the card): when two consecutive crosses draw the same
side, the engine treats the second as "already in position" (no close + reopen, stop not reset);
the time-exit clock still restarts at the second cross (toy_tsmom's episode_start rule). The card
asks for close-and-reopen at every cross; the difference is one fewer round trip of spread
(the null pays slightly less cost) and a stop not re-anchored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from quantlab.contracts import RiskType
from quantlab.strategies.toy_tsmom import ToyTsmom, ToyTsmomParams


@dataclass(frozen=True)
class RandSideParams(ToyTsmomParams):
    side_seed: int = -1


class ToyTsmomRandSide:
    name = "toy_tsmom"          # same name: the evaluator describe() is not compared here (not gated)
    risk_type = RiskType.A
    params_cls = RandSideParams

    def __init__(self, params: RandSideParams | None = None) -> None:
        self.params = params or RandSideParams()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        base = ToyTsmom(ToyTsmomParams(lookback=p.lookback, stop_mult=p.stop_mult, hold=p.hold)).signals(bars)
        if p.side_seed < 0:
            return base
        coin = pl.Series("coin", np.random.default_rng(p.side_seed).choice(
            np.array([-1, 1], dtype=np.int8), size=base.height), dtype=pl.Int8)
        return base.with_columns(coin).select(
            pl.when(pl.col("signal").is_in([-1, 1])).then(pl.col("coin")).otherwise(pl.col("signal"))
            .cast(pl.Int8).alias("signal"),
            "stop_dist", "target_dist")
