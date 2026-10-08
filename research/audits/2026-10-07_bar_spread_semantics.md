# Audit: semantics of the `spread` field of MT5 bar data (2026-10-07)

Scope: FBS forex M1/H1/D1 (dev only, ts < 2025-05-15; holdout never read, `include_holdout` never used). B3 not read (see 2).
Scripts (scratchpad, not in repo): H1/D1 aggregation comparison, tick_vol gradient, 00:00 spread profile. No quantlab code or data modified; no corrections logged (none made).

## Verdict (severity: HIGH, for cost-model documentation; cost impact is mixed, see 3)

1. MT5's own higher-timeframe bar spread = **MIN of the constituent M1 spreads**, exact match 100.0% on every test.
2. What the M1 `spread` itself is (first / min / last tick) cannot be proven without FX ticks. Indirect evidence points to "minimum (or at least low-biased) over the minute", not "first tick". Treat as UNVERIFIED; MQL5 docs claim not checked (no WebFetch).
3. quantlab's definition "spread at bar open = first M1 spread" for resampled bars is NOT what MT5 calls a D1/H1 spread. It is a different (and, at D1 open, much larger) quantity.

## 1. Native H1/D1 exports vs the M1 archive (same broker/symbol)

Native bar joined to the M1 spreads inside its window (bin = ts truncated). Exact-match rate of native spread with each M1 aggregate:

| Dataset | bars | first | MIN | max | last | median | mode | mean |
|---|---|---|---|---|---|---|---|---|
| EURUSD H1 | 53,708 | 65.6% | **100.0%** | 34.1% | 62.6% | 62.7% | 65.7% | 59.2% |
| EURUSD D1 | 2,243 | 1.5% | **100.0%** | 0.04% | 1.7% | 52.1% | 55.0% | 1.6% |
| GBPJPY D1 | 2,243 | 0.4% | **100.0%** | 0.04% | 0.9% | 23.0% | 36.6% | 0.3% |
| NZDCHF D1 | 2,243 | 1.3% | **100.0%** | 0.04% | 1.4% | 11.0% | 23.8% | 0.1% |
| XAUUSD D1 | 2,199 | 5.6% | **100.0%** | 0.09% | 16.7% | 29.7% | 48.6% | 1.3% |

- Native spread is never above the M1 min (0 cases) and never below it. Tick_vol of native bars equals the sum of M1 tick_vol in 100% of bars (confirms alignment/timezone; no join errors).
- Since min composes (min of mins = min), this is what a tick-level "min" rule would also give. First/last/median/mode tick rules would not reproduce an exact min of M1 values. Caveat: it is equally consistent with "MT5 builds H1/D1 from stored M1 rates using min" with M1 spread being something else. So 100% match proves the aggregation rule, not the M1 rule.

## 2. B3 ticks vs B3 M1

Not done. B3 holdout starts 2025-04-29 (`quantlab/config.py`); WIN/WDO BidAsk ticks start 2025-07-14, i.e. entirely inside the B3 holdout. Not read. (No dev-period B3 ticks exist.) Open item: if a pre-2025-04-29 tick sample were ever exported, test first/min/last directly.

## 3. Indirect evidence on the M1 rule (EURUSD, 2022-01..2025-05-14, server hours 11-16)

Mean/median M1 spread by number of ticks in the minute (tick_vol):

| ticks | n | mean | median | p90 |
|---|---|---|---|---|
| 1 | 34 | 13.7 | 13.5 | 19 |
| 2-3 | 107 | 10.1 | 9 | 13 |
| 4-10 | 1,104 | 9.4 | 9 | 12 |
| 11-30 | 26,528 | 8.73 | 8 | 10 |
| 31+ | 286,101 | 8.46 | 8 | 10 |

Spread falls monotonically with ticks per minute and busy minutes sit on the floor (8). A single-tick bar has first = min = last, so it is the unbiased sample: ~13.5 vs ~8.5 for busy minutes. This is what a minimum over more samples predicts; a first-tick snapshot would show it only through liquidity confounding (confounding works the other way too: busy minutes usually widen). Suggestive, not conclusive. Also 4 of 1.25M EURUSD minutes have spread 0 and values 1-3 occur (consistent with any rule).

## 4. Impact on ManAHL baseline (D1 bars, entries/exits at 00:00 server; 693 trades, all legs at 00:00; window 2017-08-18..2025-05-14)

The engine charges `spread` of the D1 bar = the 00:00 M1 spread (rollover minute). Averages over all days / over the actual trade legs (points):

| Symbol | 00:00 M1 (used) | day-min = native D1 spread | median 01:00-20:00 | max M1 in 00:00-00:05 |
|---|---|---|---|---|
| EURUSD | 19.1 / 20.6 | 6.9 / 7.1 | 8.0 / 8.2 | 38.0 / 41.6 |
| GBPJPY | 76.8 / 77.5 | 22.6 / 22.7 | 28.7 / 28.2 | 139 / 136 |
| NZDCHF | 68.5 / 62.7 | 21.0 / 19.4 | 26.5 / 24.5 | 128 / 128 |
| XAUUSD | 37.2 / 37.6 | 23.6 / 23.5 | 33.2 / 35.2 | 42.1 / 44.4 |

Medians at 00:00: EURUSD 10, GBPJPY 41, NZDCHF 38, XAUUSD 34.

- The spread used is already the rollover-inflated one (2.4-3x the daytime median for FX majors/crosses; 1.1x for gold). Even if the field is a per-minute minimum, the 00:00 minute is wide, so costs are NOT understated relative to a day-min or daytime reading; they are overstated by ~2.4x (FX) vs daytime quotes.
- Downside risk: if the M1 field is a minimum, the true first-tick spread at 00:00 is >= the field. Proxy upper bound = max M1 spread of 00:00-00:05, about 2x the used value (EURUSD 41.6 vs 20.6; GBPJPY 136 vs 78; NZDCHF 128 vs 63; XAUUSD 44 vs 38 = 1.2x). So bound on cost per leg: [daytime ~0.4x] .. [used 1x] .. [~2x] for FX; gold [0.9x .. 1x .. 1.2x]. This is a rollover execution-timing issue (00:00 = 17:00 NY), not a bar-semantics issue.
- `spread_max` (max of M1 spreads in the bar) is a max of per-minute values; if M1 is a min it is still a lower bound of the true worst tick spread. Engine's stop-touch conservatism (uses spread_max) is thus mildly optimistic.
- Intraday strategies (M1/M5/H1 execution in busy hours): M1 field vs true quote spread gap is bounded by the tick gradient in section 3: ~5-10% in liquid hours for EURUSD, larger in thin minutes (+60% for 1-tick minutes). Not tick-verified.

## 5. Recommendations (no code edited)

1. contracts.py / data.py docstrings: say "spread = MT5 M1 `spread` field (semantics unverified; native H1/D1 equal the min of M1; likely a per-minute low-biased value); resampled `spread` = first M1 spread, NOT comparable to a native H1/D1 `spread`". Add `spread_min` (min of M1) to resample to reproduce native bars, and keep `spread_max`.
2. Cost estimator: do not use a single bar-open M1 spread for execution at rollover. Use per-leg `spread_eff = max(spread_open_M1, rolling hour-of-day p75 of M1 spread at that minute (dev data))` for entries at 00:00; for liquid-hour execution add a stress via existing `stressed()` (1.5x covers the 5-10% gap and ~the 2x rollover bound only partially: add a 2x rollover sensitivity run for D1-at-00:00 strategies).
3. For any rule that executes at D1 open, report a sensitivity table: cost at {daytime median, 00:00 M1 (used), max 00:00-00:05} (section 4 numbers) and require the verdict to hold at the upper one. Also consider executing at 01:00-02:00 (spread = daytime) as the notebook's what-if already does.
4. Never validate resampled spread against a native export's spread; compare `spread_min` instead (exact).
5. Resolve the open item by obtaining one tick export (MT5 FX ticks for a few dev days) and testing first/min/last on M1; log result here.
