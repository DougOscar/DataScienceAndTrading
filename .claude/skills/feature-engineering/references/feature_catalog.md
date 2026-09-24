# Feature Catalog

Complete catalog of features for trading ML models, organized by category.
Each entry includes the formula, typical lookback period, whether the feature
is stationary by construction, and a brief interpretation guide.

## Price Features

| # | Name | Formula | Lookback | Stationary | Interpretation |
|---|------|---------|----------|------------|----------------|
| 1 | `log_return` | `ln(close_t / close_{t-1})` | 1 | Yes | Core return measure. Symmetric, additive over time. |
| 2 | `abs_return` | `abs(log_return)` | 1 | Yes | Unsigned volatility proxy. Spikes on large moves either direction. |
| 3 | `return_volatility` | `std(log_return, N)` | 20 | Yes | Realized volatility over N bars. Higher = more risk/opportunity. |
| 4 | `momentum_5` | `close / close[5] - 1` | 5 | Yes | Short-term trend strength. Positive = uptrend. |
| 5 | `momentum_10` | `close / close[10] - 1` | 10 | Yes | Medium-term trend. Compare to momentum_5 for acceleration. |
| 6 | `momentum_20` | `close / close[20] - 1` | 20 | Yes | Longer-term trend. Divergence from short-term = potential reversal. |
| 7 | `acceleration` | `momentum_5_t - momentum_5_{t-5}` | 10 | Yes | Rate of change of momentum. Positive = trend strengthening. |
| 8 | `high_low_range` | `(high - low) / close` | 1 | Yes | Intrabar volatility as fraction of price. High = volatile bar. |
| 9 | `close_position` | `(close - low) / (high - low)` | 1 | Yes | Where close sits in bar range. 1.0 = closed at high (bullish). |
| 10 | `gap` | `open_t / close_{t-1} - 1` | 1 | Yes | Overnight/inter-bar gap. Large gaps may revert. |
| 11 | `rolling_skew` | `skew(log_return, 20)` | 20 | Yes | Return distribution asymmetry. Negative = more large drops. |
| 12 | `rolling_kurtosis` | `kurtosis(log_return, 20)` | 20 | Yes | Tail heaviness. High = more extreme moves than normal. |

## Volume Features

| # | Name | Formula | Lookback | Stationary | Interpretation |
|---|------|---------|----------|------------|----------------|
| 13 | `volume_ratio` | `volume / mean(volume, 20)` | 20 | Yes | Current volume relative to average. >2 = volume spike. |
| 14 | `volume_ma_ratio` | `sma(volume, 5) / sma(volume, 20)` | 20 | Yes | Short vs long volume trend. >1 = increasing activity. |
| 15 | `obv_slope` | `linregress(OBV, 10).slope` | 10 | Yes | On-Balance Volume trend. Positive + price up = confirmed trend. |
| 16 | `vwap_deviation` | `(close - VWAP) / VWAP` | Intraday | Yes | Distance from fair price. Positive = trading above fair value. |
| 17 | `volume_acceleration` | `volume_ratio_t - volume_ratio_{t-1}` | 21 | Yes | Rate of change of relative volume. Spike detection. |
| 18 | `buy_volume_ratio` | `buy_volume / total_volume` | 1 | Yes | Buy pressure. >0.5 = net buying. Requires trade-level data. |
| 19 | `dollar_volume` | `close * volume` | 1 | No* | Absolute liquidity measure. Normalize by rolling mean. |
| 20 | `volume_cv` | `std(volume, 20) / mean(volume, 20)` | 20 | Yes | Volume consistency. Low CV = steady trading. High = erratic. |

*`dollar_volume` is non-stationary in raw form. Use `dollar_volume / rolling_mean(dollar_volume, 20)` for a stationary version.

## Technical Features

All computed via standard indicator libraries (pandas-ta, ta-lib).

| # | Name | Source Indicator | Lookback | Stationary | Interpretation |
|---|------|-----------------|----------|------------|----------------|
| 21 | `rsi` | RSI(14) | 14 | Yes | Bounded 0-100. <30 oversold, >70 overbought. |
| 22 | `macd_histogram` | MACD(12,26,9) | 33 | Yes | Momentum oscillator. Positive = bullish momentum. |
| 23 | `bb_position` | Bollinger Bands(20,2) | 20 | Yes | Position within bands. 0 = at lower, 1 = at upper. |
| 24 | `bb_width` | Bollinger Bands(20,2) | 20 | Yes | Band width / midline. Narrow = low vol (squeeze). |
| 25 | `atr_ratio` | ATR(14) / close | 14 | Yes | Volatility as % of price. Comparable across price levels. |
| 26 | `adx` | ADX(14) | 14 | Yes | Trend strength 0-100. >25 = trending, <20 = ranging. |
| 27 | `stoch_k` | Stochastic(14,3) | 14 | Yes | Momentum oscillator 0-100. Similar to RSI but price-range based. |
| 28 | `cci` | CCI(20) | 20 | Yes | Mean reversion oscillator. >100 overbought, <-100 oversold. |
| 29 | `mfi` | MFI(14) | 14 | Yes | Volume-weighted RSI. Divergence from RSI = volume disagreement. |
| 30 | `supertrend_dir` | Supertrend(10,3) | 10 | Yes | Binary trend direction. +1 = uptrend, -1 = downtrend. |

## Spread / Execution Features

MT5 OHLC bars are Bid prices and include a `spread` (points) column — a
point-in-time, causal feature (reflects conditions at bar close, same as
OHLCV, so it carries no lookahead) capturing execution cost and liquidity.

| # | Name | Formula | Lookback | Stationary | Interpretation |
|---|------|---------|----------|------------|----------------|
| 31 | `spread_points` | raw `spread` column (points) | 1 | Yes | Current execution cost. Spikes in illiquid hours/news. |
| 32 | `spread_ratio` | `spread / mean(spread, 20)` | 20 | Yes | Spread relative to its own recent norm. >2 = abnormal widening. |

## Time Features

Cyclical encoding preserves the circular nature of time (hour 23 is near hour 0).

| # | Name | Formula | Lookback | Stationary | Interpretation |
|---|------|---------|----------|------------|----------------|
| 33 | `hour_sin` | `sin(2 * pi * hour / 24)` | 0 | Yes | Cyclical hour encoding (vertical component). |
| 34 | `hour_cos` | `cos(2 * pi * hour / 24)` | 0 | Yes | Cyclical hour encoding (horizontal component). |
| 35 | `day_of_week` | `sin(2 * pi * weekday / 7)` | 0 | Yes | Cyclical day encoding. Captures weekly patterns (FX/metals close weekends; B3 sessions don't span days). |

## Feature Interaction Notes

Some features are more informative in combination:

- **Volume + Price momentum**: Volume confirming price direction is stronger than
  either alone. High `volume_ratio` + positive `momentum_5` = confirmed breakout.
- **RSI + BB position**: RSI oversold + BB position near 0 = double confirmation
  of oversold condition.
- **ADX + momentum**: High ADX + strong momentum = trend trade. Low ADX +
  mean-reversion indicators = range trade.
- **Spread + volatility**: Elevated `spread_ratio` coinciding with elevated
  `return_volatility` often signals genuine liquidity stress (news, thin
  hours) rather than a tradeable dislocation — widen or skip entries.

## Recommended Starter Set

For a first model, use these 12 features (low correlation, diverse categories):

1. `log_return` — recent return
2. `return_volatility` — risk level
3. `momentum_10` — medium trend
4. `close_position` — bar structure
5. `volume_ratio` — activity level
6. `volume_ma_ratio` — volume trend
7. `rsi` — momentum oscillator
8. `bb_position` — mean reversion signal
9. `atr_ratio` — normalized volatility
10. `adx` — trend strength
11. `hour_sin` — time of day (vertical)
12. `hour_cos` — time of day (horizontal)
