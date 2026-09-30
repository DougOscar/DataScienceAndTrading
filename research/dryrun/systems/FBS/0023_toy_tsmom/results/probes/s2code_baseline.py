"""S2 code red-team probe: reproduce and decompose the S3 baseline (mid-grid prior)."""
import numpy as np, polars as pl, warnings
warnings.filterwarnings("ignore")
from quantlab.evaluators import RuleEvaluator
from quantlab.strategies.toy_tsmom import ToyTsmom
from quantlab.costs import CostModel

ev = RuleEvaluator(ToyTsmom, symbol="EURUSD", timeframe="H4", book="FBS")
out = ev({})
t = out.trades
print(t.columns)
m = out.metrics
print({k: m[k] for k in m if k in ("sharpe", "n_trades", "trades_per_year", "cagr", "max_dd", "win_rate", "n_skipped")})
yrs = (t["exit_ts"].max() - t["entry_ts"].min()).total_seconds() / (365.25*86400) if "exit_ts" in t.columns else 9.04
print("trades", t.height, "years", round(yrs, 2), "per yr", round(t.height / yrs, 1))
t = t.with_columns((pl.col("pnl_points") + pl.col("spread_cost_points")).alias("gross"))
print("gross pts/trade mean", t["gross"].mean(), "sd", t["gross"].std(), "SE", t["gross"].std()/np.sqrt(t.height))
print("net pnl_points mean", t["pnl_points"].mean(), "spread mean", t["spread_cost_points"].mean(),
      "swap mean", t["swap_points"].mean() if "swap_points" in t.columns else None)
print(t.group_by("direction").agg(pl.len(), pl.col("gross").mean(), pl.col("spread_cost_points").mean()).sort("direction"))
print(t.group_by("exit_reason").agg(pl.len(), pl.col("gross").mean(), (pl.col("exit_idx")-pl.col("entry_idx")).mean().alias("bars")).sort("exit_reason"))
t = t.with_columns((pl.col("exit_idx")-pl.col("entry_idx")).alias("bars"))
print("holding bars quantiles", t["bars"].quantile(0.25), t["bars"].median(), t["bars"].quantile(0.75), "mean", t["bars"].mean())
print("hold-binding fraction (bars==30 & signal exit):", t.filter((pl.col("bars")==30)&(pl.col("exit_reason")=="signal")).height / t.height)
if "entry_ts" in t.columns:
    print(t.group_by(pl.col("entry_ts").dt.year().alias("y")).agg(pl.len(), pl.col("gross").sum()).sort("y"))
    print("entry hour share:", t.group_by(pl.col("entry_ts").dt.hour().alias("h")).agg(pl.len(), pl.col("spread_cost_points").mean()).sort("h"))
# drift: buy-and-hold points over window
b = ev._data().bars_eval
print("EURUSD dev drift pts:", (b["close"][-1] - b["open"][0]) / 1e-5, "=> per bar", (b["close"][-1]-b["open"][0])/1e-5/b.height)
# long exposure share
print("long bars share", t.filter(pl.col("direction")==1)["bars"].sum() / t["bars"].sum())
# stop fills & short stop spread
st = t.filter(pl.col("exit_reason")=="stop")
print("stops by dir", st.group_by("direction").agg(pl.len(), pl.col("spread_cost_points").mean()))
