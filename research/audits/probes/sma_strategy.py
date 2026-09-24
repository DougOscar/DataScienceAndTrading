"""Deterministic 20/50 SMA cross with a 2x ATR(14) stop -- used by the hand-check probe."""
from dataclasses import dataclass
from typing import ClassVar
import polars as pl
from quantlab.contracts import Params, RiskType


@dataclass(frozen=True)
class SmaParams(Params):
    fast: int = 20
    slow: int = 50
    atr_n: int = 14
    atr_mult: float = 2.0
    far_target: bool = False      # workaround for the short-without-target engine bug (p01)


class SmaCross:
    name: ClassVar[str] = "sma_cross_probe"
    risk_type: ClassVar[RiskType] = RiskType.A

    def __init__(self, params: SmaParams = SmaParams()):
        self.params = params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        prev_c = pl.col("close").shift(1)
        tr = pl.max_horizontal(pl.col("high") - pl.col("low"), (pl.col("high") - prev_c).abs(), (pl.col("low") - prev_c).abs())
        df = bars.select(
            pl.col("close").rolling_mean(p.fast).alias("f"),
            pl.col("close").rolling_mean(p.slow).alias("s"),
            tr.rolling_mean(p.atr_n).alias("atr"),
        )
        above = (pl.col("f") > pl.col("s"))
        cross_up = above & ~above.shift(1).fill_null(True)
        cross_dn = ~above & above.shift(1).fill_null(False) & pl.col("s").is_not_null()
        out = df.select(
            pl.when(cross_up).then(pl.lit(1, pl.Int8)).when(cross_dn).then(pl.lit(-1, pl.Int8)).otherwise(pl.lit(None, pl.Int8)).alias("signal"),
            (pl.col("atr") * p.atr_mult).alias("stop_dist"),
            (pl.col("atr") * 1000.0 if p.far_target else pl.lit(None, pl.Float64)).alias("target_dist"),
        )
        return out
