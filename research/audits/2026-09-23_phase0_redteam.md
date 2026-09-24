# Phase 0 red-team audit — quantlab (2026-09-23)

Scope: `quantlab/{config,contracts,data,ledger,costs,engine,sizing,testing}.py` on branch
`feat/quantlab-phase0`, audited against DESIGN §4.1, §4.3, §5 and the §10 Phase 0 exit test
(look-ahead truncation test, known-answer backtests, hand-checked trades).

- Test suite: 92 passed (`.venv/bin/python -m pytest -q tests`).
- Library state when the probes ran: `research/audits/probes/_library_sha1_at_run.txt`. `engine.py` was
  edited during the audit by the perf work (the `ts.gather` change at 23:17). I re-read it and it has the
  same semantics. I re-ran every probe after that edit.
- Probes: `research/audits/probes/pNN_*.py`. Output for each is in the matching `.out` file.
  Run them from the repo root with `PYTHONPATH=. .venv/bin/python research/audits/probes/<probe>.py`.
- No data at or after 2025-05-15 was read by the auditor, apart from the guard attempts in p04. Every
  successful call there was reduced to `max(ts)`.

**Verdict: Phase 0 does NOT pass its exit test yet.** There is one BLOCKER in the engine (shorts), a
look-ahead in the engine's M1 mapping, and the look-ahead auditor has two blind spots.

---

## BLOCKER

### B1. Shorts without both a stop AND a target are closed on their entry bar at −spread
- **Code:** `engine.py::_run_core`, lines 199-200:
  ```python
  sl_c = sl if has_sl else -1.0e18
  tp_c = tp if has_tp else 1.0e18
  ```
  These sentinels are right for longs (`l <= -1e18` and `h >= 1e18` can never be true). They are wrong
  for shorts. `_chk_short` tests `ask_h >= sl` and `ask_l <= tp`, and both are always true when compared
  against these sentinels. The short is therefore "gap-stopped" or "gap-targeted" at `ask_o` on the same
  bar it opened.
- **Evidence (synthetic, `p01_short_no_stop.out`):**
  ```
  short no stop/no target  entry_idx=1 exit_idx=1 reason=gap_stop   pnl_pts=-10.0
  short stop only          entry_idx=1 exit_idx=1 reason=gap_target pnl_pts=-10.0
  short target only        entry_idx=1 exit_idx=1 reason=gap_stop   pnl_pts=-10.0
  short stop+target        entry_idx=1 exit_idx=49 reason=eod       pnl_pts=847.0
  ```
- **Evidence (real data, `p07_handcheck_run.out`):** EURUSD H1 2019, 20/50 SMA cross with a 2×ATR stop and
  no target. With M1, all 71 shorts exit as `gap_target` on their entry bar (`same_bar=71`), for a total
  of −755 points (= spread only). Adding a distant target changes the short book to 30 signal exits
  (+7 767 pts) and 41 stops (−8 389 pts).
- **Failure scenario:** every Type C system (no hard stop), and every system with a stop but no target,
  silently loses its short side. Those shorts are replaced with a guaranteed −spread trade. Long-only
  results are unaffected, so a system can look long-biased, or look cost-dominated, for no real reason.
  The unit tests missed this because every short test sets both `stop_dist` and `target_dist`.
- **Fix:** use per-direction sentinels (for a short, `sl_c=+1e18` and `tp_c=-1e18`), or pass
  `has_sl/has_tp` into the check functions. Add known-answer tests for short {stop only, target only,
  neither}, both with and without `m1`.

---

## MAJOR

### M1. The M1 path of the last HTF bar (and of any bar before a gap) uses M1 bars outside that bar, including future data
- **Code:** `engine.py::_map_m1_bounds`. It sets `ends[-1] = len(m1)`, and for every other bar
  `ends[j] = searchsorted(ts[j+1])`. That bound is not `ts[j] + timeframe`.
- **Evidence (`p02_m1_last_bar_leak.out`):**
  - A) Loading `D1` and `M1` with the same intraday `end=2019-06-14 12:00`: `load_bars` drops the
    partial D1 bar, but M1 keeps those minutes. The last D1 bar (06-13) gets 2 160 M1 rows, running up
    to 2019-06-14 11:59.
  - B) H1 bars ending 2019-06-01, with M1 loaded to 2019-07-01. A long is opened on the last H1 bar.
    Without M1 it exits `eod` at 1.1165. With M1 it exits `target` at 1.12528, triggered by the M1 bar
    at **2019-06-03 21:50**, which is after the backtest window. 28 846 M1 rows were mapped to one H1 bar.
  - C) With a session-filtered H1 frame (08-16h), 29 266 overnight/weekend M1 rows are mapped onto a
    single 16:00 bar.
- **Failure scenario:** walk-forward/CPCV code that loads M1 once and backtests fold slices of the HTF
  bars. The last open trade of every fold is resolved with prices from after the fold (days later),
  instead of `eod`. Across many folds this biases OOS numbers. For filtered frames, exits are also
  recorded on the wrong bar and date.
- **Fix:** bound each window by `ts[j] + span`, i.e. `ends[j] = min(searchsorted(ts[j+1]), searchsorted(ts[j]+span))`,
  and pass the timeframe to `run_backtest`. Or assert that `m1.ts` lies within `[ts[0], ts[-1]+span)`
  and that there are no interior gaps.

### M2. `assert_no_lookahead` passes sparse leaks and out-of-band data leaks
- **Evidence (`p09_lookahead_harness.out`, EURUSD H1 2019, 20 seeds each):**

  | leaky strategy | passes auditor | backtest |
  |---|---|---|
  | `sparse_peek`: an SMA cross is taken only if the *next* bar closes in its favour (`shift(-1)`, used on 143 rows) | **12/20** | 42% win, +3 353 pts |
  | `external_d1`: calls `data.load_bars(...,"D1")` inside `signals()` and joins today's D1 bar by date | **20/20** | 96.8% win, +67 207 pts |
  | `self_d1`: resamples its own input to D1 (control) | 0/20 | caught |
  | `global_z`: full-sample z-score (control) | 0/20 | caught by truncation |
- **Why it passes:**
  1. A k-bar look-ahead at row r is only visible when a cut point t falls in `[r, r+k)`. With 25
     random cut points and signals on about 2% of rows, the auditor usually never looks at one of those
     rows.
  2. The auditor truncates or perturbs only the `bars` argument. Data that the strategy fetches itself
     (`load_bars`, a cached frame, a calendar) is never truncated.
  3. The perturbation step is a permutation of future rows. It keeps global statistics (mean, std,
     max) unchanged, so it cannot catch full-sample normalisation. Only the truncation step catches that.
- **Stronger checks (suggested):**
  - (a) Always include every row where the full-run signal is non-null or changes (plus row−1) in the
    cut points. Or run an exhaustive streaming check over all t in chunks.
  - (b) During the check, monkeypatch `quantlab.data.load_bars`/`conversion_rate` so that they clamp
    `end` to `bars.ts[t] + span`, or raise. Better still, give strategies extra data only through a
    harness-owned context.
  - (c) Replace the future with a fresh random walk (level shift + rescale), not a permutation.
  - (d) Also run the truncation test at the engine level: trades closed before t must be identical
    when bars and m1 are cut at t. This would have caught M1.

### M3. Currency conversion used in sizing looks 2-3 h ahead (plus 1 minute)
- **Code:** `sizing.apply_sizing` calls `rate_fn(quote, acct, trades["entry_ts"])`. `entry_ts` is naive
  server time. The documented `rate_fn` contract (and `data.conversion_rate`) expects `ts_utc`.
  `conversion_rate` accepts a naive series and treats it as UTC, which shifts it +2 h (winter) or +3 h
  (summer).
- **Evidence (`p05_conversion_timing.out`):** for an entry at 2019-07-03 10:00 server, the naive series
  gives 1/107.710. That is the USDJPY close at 13:00, three hours later. The correct value is 1/107.625.
  `p04` shows the same effect from the other side: a naive `2025-05-14 22:30` request raises
  HoldoutLocked because it has been shifted into 05-15.
- **Second issue:** `_asof_closes` does a backward as-of on M1 `ts`. At time t it returns the close of
  the M1 bar that *opens* at t, which is only known at t+1 min.
- **Magnitude (`p14_conversion_impact.out`):** USDJPY 2019-2020, 1% risk. Lot sizes differ on 100/292
  trades. 43 trades exceed their risk target when checked against the true entry-time rate. Final equity
  moves by $27. The economic effect is small, but it is a look-ahead by construction, through the
  library's own interface.
- **Fix:** have `apply_sizing` convert `entry_ts` to UTC (book tz) before calling `rate_fn`, or pass
  `ts_utc`. Make `conversion_rate` reject naive input. Use the as-of of the *previous* M1 close
  (`ts + 1min <= t`) or the M1 open at t. Consider converting realised P&L at the exit-time rate.

### M4. Swap is effectively not modelled
- **Evidence:**
  - `data/broker/` does not exist, so every `load_instrument` call returns the uncalibrated fallback:
    `swap_long = swap_short = 0`, `commission = 0`, and contract sizes are guesses (`p07` prints the
    spec).
  - Separately, `apply_sizing` never reads `swap_money_per_lot`. For `swap_mode="money"`, P&L ignores
    swap entirely. `p11_sizing.out` (a): a points-mode spec gives total pnl −2 622.14, while the same
    swap in money mode gives −2 224.14. The −398 USD/lot swap is dropped.
- **Failure scenario:** every current backtest has zero carry cost. Swing (multi-night) FBS systems will
  look better than they are, and "share of costs from swap" (§5) is 0 by construction. When the MT5
  export arrives with `SYMBOL_SWAP_MODE_CURRENCY_*` symbols, those swaps will still be dropped silently.
- **Fix:** add `swap_money_per_lot * lots` (converted to account currency) in `apply_sizing`. Block
  §4.2 gate runs while `spec.calibrated is False`, or at least flag them.

### M5. MAE/MFE include price action after the exit
- **Code:** `_run_core` step 2 updates MAE/MFE from the whole HTF bar (`h[j]`, `l[j]`) before checking
  whether the stop or target was hit inside that bar.
- **Evidence (`p10_mae_mfe_and_misc.out`):**
  - (a) Synthetic long stopped at 1.0990 for −110 pts: `mfe_points=990` from a rally *after* the stop.
  - (b) Real EURUSD 2019: 47% of stop exits report MAE > 1.2× the realised loss (median 1.18, max 3.24).
    For example, short 2019-01-09 lost 323 pts at the stop but reports MAE 687.
  - Short MAE is also measured on Bid, while the short's stop triggers on Ask.
- **Failure scenario:** MAE/MFE are the core metrics for Type C (§5). Stop/target placement studies
  that use MAE/MFE get inflated excursions.
- **Fix:** update the excursions on the M1 sub-bars up to the exit sub-bar. Without M1, cap them at
  the exit price on the exit bar. Measure short excursions on Ask.

---

## MINOR

1. **Intra-minute liquidity gaps.**
   - A stop fills exactly at its level whenever the M1 bar *opens* above it.
   - USDJPY 2019-01-03 flash crash (`p12_flash_crash_and_weekend_gap.out`): the M1 bar at 00:35 opened
     at 108.125 and traded down to 106.299. A long stop at 107.891 fills at 107.891. The default
     slippage is 0, and the stress test adds only 1 pt.
   - Consider range-proportional slippage in the stress model.
2. **D1 ≠ 17:00 NY in the US/EU DST-mismatch weeks.**
   - In 2019, 20 of 259 D1 bars open at 18:00 NY (`p03_resample_and_tz.out`).
   - The broker's week close moves to 22:59 server in those weeks, so the Friday D1 bar is 23 h long.
     This matches MT5, but the DESIGN §1 and config wording ("D1 closes at 17:00 NY") is not literally
     true.
3. **Crypto `ts_utc` goes backwards or duplicates on EU spring-forward days 2017-2020.**
   - The feed has bars at the non-existent server hour 03:xx. The "shift forward" rule maps them onto
     04:xx, which gives 412 duplicate `ts_utc` values and 4 backward steps (`p15_crypto_dst_ts_utc.out`).
   - This contradicts the `data.py` docstring claim that `ts_utc` never steps backwards. The engine uses
     naive `ts`, so it is unaffected, but UTC-keyed joins (conversion, calendar) are off by 1 h there.
4. **Crypto swap nights.**
   - `_nights_weighted` never counts Saturday or Sunday nights, and the fallback `swap_3day` is
     Wednesday. Crypto CFDs are normally charged 7 nights per week.
   - This needs a per-symbol rule from the broker export.
5. **`infer_point` reads the raw file's tail (holdout region) without the guard.**
   - It returns only a decimal count, so the leak is negligible, but it breaks the rule that all data
     access goes through `load_bars`. Sample from the dev window instead.
6. **`source_file` is not checked against `symbol`.**
   - `load_bars("EURUSD", source_file=<XAUUSD file>)` silently returns gold data labelled EURUSD (`p04`).
7. **`conversion_rate` raises HoldoutLocked for legitimate dev timestamps.**
   - This happens in the last minute before the cutoff, because it requests `end = hi + 1min` (`p04`).
8. **The resample cache key has no code or schema version.**
   - The key is `(name, size, mtime, tf)`. A change to `_resample_full` semantics (for example from the
     ongoing perf work) is masked by stale caches.
   - Cache writes are also not atomic, so parallel workers can read partial files.
9. **The holdout unlock is enforced by convention only.**
   - Any agent can call `ledger.record_holdout_unlock`, which only requires a clean tree. DESIGN §0.4
     says "only the user can unlock".
   - Add a Claude Code deny rule for writes to `holdout_access.jsonl` and for direct calls. Or require a
     user-supplied secret or confirmation.
10. **Spread timing on some exit paths.**
    - `eod` and `force` short exits use `close + spread` of that bar's *open* (p10 d).
    - Without M1, short stop/target checks apply the bar-open spread across the whole bar. For D1 that
      is the 00:00 rollover spread: EURUSD median 26 vs 9 all-day (`p06_point_and_spread.out`).
    - `stop_dist` < spread gives an instant `gap_stop` at entry (p10 e). A warning would help.
11. **Contract semantics to document for strategy authors.**
    - Signals are *desired state*. A strategy that keeps emitting +1 re-enters on the next bar after
      every stop-out.

---

## Hand-check (DESIGN §10): 20/50 SMA cross + 2×ATR(14) stop, EURUSD H1 2019

- **How the check was done:** the strategy is in `probes/sma_strategy.py`. I first checked it with
  `assert_no_lookahead` (it passed). The engine outputs are in `p07_trades_m1.csv` (with M1) and
  `p07_trades_nom1.csv` (without).
- **Independent verification:** `p08_handcheck_verify.py` reads the raw M1 Parquet with pyarrow and
  pandas, with no quantlab code. It rebuilds H1, SMA and ATR from that, then walks M1 to find each exit.
- **Units and settings:** point = 1e-5, slippage 0, spread multiplier 1.
- **Distant target:** a far target was added (`far_target=True`) so that B1 does not wipe out the shorts.
- **No-M1 run:** 143 trades, identical to the M1 run. With a stop but no target, M1 only matters for
  gaps.

| # | Trade | Signal (bar i close) | Fill | Stop | Exit found in raw M1 | pnl_points (hand) | Engine |
|---|---|---|---|---|---|---|---|
| 1 | long 2019-01-04 07:00 | 06:00: fast 1.138386 > slow 1.138307; prev 1.138314 < 1.138412 → cross up. ATR 0.0014914 → sd 0.0029829 | Ask = open 1.13926 + 9×1e-5 = **1.13935** | 1.13935 − 0.0029829 = 1.1363671 | M1 15:30: o 1.14029 > SL, l 1.13602 ≤ SL → `stop` at 1.1363671 | (1.1363671 − 1.13935)/1e-5 = **−298.29** | stop, exit_ts 15:00, −298.2857 ✓ |
| 2 | short 2019-01-09 02:00 | 01:00: 1.144424 < 1.144434; prev 1.144332 > 1.144304. ATR 0.0016150 → sd 0.00323 | Bid = open **1.14567** (spread not charged on entry) | 1.14567 + 0.00323 = 1.14890 | M1 16:35: Ask-high 1.14947 + 0.00009 = 1.14956 ≥ SL; Ask-open 1.14802 < SL → `stop` 1.14890 | (1.14567 − 1.14890)/1e-5 = **−323.00** | stop, −323.0 ✓ |
| 3 | short 2019-01-11 09:00 | 08:00 cross down. sd 0.0023229 | Bid **1.15219** | 1.1545129 (never hit) | Opposite cross at bar 2019-01-18 14:00 → exit at 15:00 open Ask = 1.14014 + 0.00009 = **1.14023** | (1.15219 − 1.14023)/1e-5 = **+1196** | signal, +1196.0; nights 5 (Fri, Mon, Tue, Wed, Thu) ✓ |
| 4 | long Fri 2019-01-25 23:00 | 22:00 cross up. sd 0.0031043 | Ask = 1.14139 + 17×1e-5 = **1.14156** (late-Friday spread 17) | 1.1384557 (never hit) | Cross down at 2019-02-01 06:00 → exit Bid open 07:00 = **1.14412** | (1.14412 − 1.14156)/1e-5 = **+256** | signal, +256; nights 5 over the weekend ✓ |
| 5 | long 2019-02-15 09:00 (same-bar stop) | 08:00 cross up. sd 0.0012629 | Ask = 1.12876 + 0.00009 = **1.12885** | 1.1275871 | M1 09:38: o 1.12779 > SL, l 1.12736 ≤ SL → `stop` 1.1275871 | **−126.29** | stop, exit bar = entry bar, −126.2857 ✓ |
| 6 | short 2019-04-05 00:00 (rollover) | 2019-04-04 23:00 cross down. sd 0.0018914 | Bid **1.12144** (00:00 spread 29 not charged to a short entry) | 1.1233314 | M1 09:17: 1.12324 + 0.00011 = 1.12335 ≥ SL; open-Ask 1.12323 < SL → `stop` | (1.12144 − 1.1233314)/1e-5 = **−189.14**; spread_cost 11 (exit minute) | stop, −189.1429, spread_cost 11 ✓ |
| 7 | short 2019-06-06 03:00 (ambiguous bar, `p13`: stop 2×ATR, target 1×ATR) | — | Bid 1.12280 | SL 1.125999, TP 1.121201; H1 14:00 bar has h 1.12725 and l 1.12005, so both are inside the bar | TP first at M1 14:45 (1.12005 + 0.00009 = 1.12014 ≤ TP); SL only at 14:53 | M1: (1.1228 − 1.121201)/1e-5 = **+159.93**; no-M1 (stop-first rule): **−319.86** | target +159.93 / stop −319.86 ✓ |

All 7 trades match the engine to floating-point precision. That covers entry price, exit reason, exit
price, spread charged, pnl_points and nights. This was only possible on the stop+target and
far-target variants; the stop-only shorts are broken (B1).

---

## Checks that found nothing

1. **Resampling:**
   - 1 800 random H1/H4/D1 bars (EURUSD, XAUUSD, USDJPY, 2019-2020) were recomputed from raw M1 over
     `[ts, ts+span)`: OHLC, spread (first M1), spread_max and tick_vol. **0 mismatches.**
   - All labels sit on the server-midnight grid, and `ts` is the open time (`p03`).
2. **`ts_utc` for FX:** equal to the zoneinfo Europe/Helsinki conversion for every H1 bar from March to
   November 2019, including both US/EU mismatch windows. Week-open and week-close hours behave as
   documented (Friday closes 22:59 in mismatch weeks) (`p03`).
3. **Holdout guard:** 30 attempts, all blocked or returning only bars that close at or before
   2025-05-15 00:00 (`p04`). The attempts covered:
   - default end, end == cutoff, cutoff + 1 ms / + 1 µs;
   - tz-aware datetimes and strings (TypeError);
   - `include_holdout` without a system, a system without an unlock, and lowercase book names;
   - book mismatch;
   - native H1 `source_file` resampled to H1/H4/D1, including a relative path;
   - straddling H4 and D1 bars with an intraday end (dropped);
   - BTC D1;
   - `conversion_rate` with direct, triangulated and naive inputs.

   The cache is only returned through the filter.
4. **Engine fills vs §4.3:**
   - Long entry at Ask, long exit at Bid.
   - Short entry at Bid, short exit at Ask, on the signal, stop, target, gap, force and eod paths.
   - Short stops trigger on Ask, using each M1 minute's own spread.
   - Stops and targets are measured from the fill price.
   - Gaps fill at the (M1 or HTF) open with adverse slippage.
   - Same-bar entry and exit.
   - Reversal = exit + entry at the same open.
   - The entry bar's M1 path starts at the entry minute, so there is no pre-entry range. Checked by
     reading the code and by hand-checks 1-7.
   - Weekend gaps are detected at the Monday open (largest in 2019-20: EURUSD 90 pips on 2020-03-16,
     `p12`).
5. **Swap nights:**
   - Friday → Monday = 1 night.
   - Wednesday is weighted ×3; Monday → Monday = 7 weighted.
   - Christmas holds are counted normally.
   - A 00:00 entry does not pay that night.
   - The server-clock basis makes DST irrelevant (`p10` c, hand-checks 3-4).
6. **Sizing:**
   - Risk is measured from the fill price to the stop.
   - `floor_to_step` never rounds up (0.29999999 → 0.29, 600 → 500, below minimum → 0).
   - Risk-realisation error lies in [−0.43%, 0].
   - Stop-out loss / risk target lies in [0.997, 1.000].
   - The final daily equity reconciles with equity0 + Σpnl_ccy.
   - Daily equity up to 2019-06-30 is identical when bars after 2019-07-01 are removed (truncation
     invariant) (`p11`).
7. **`assert_no_lookahead`** does catch full-sample normalisation and self-resampling leaks (`p09`).
8. **`infer_point`** returns the correct point for all 35 FBS symbols. M1 spreads are plausible and show
   the 00:00 rollover spike: EURUSD 23-25 vs 8-9 pts, USDJPY 36-44 vs 21 (`p06`).
9. **The perf edit to `engine.py`** (`ts.gather` instead of `to_list`) is semantically identical. All
   probes were re-run after it.

---

## Re-verification 2026-09-24

**How this was checked**
- Library under test: hashes in `research/audits/probes/reverify_2026-09-24/_library_sha1.txt`. The files did not change while the probes ran.
- Test suite: 160 passed (`reverify_2026-09-24/pytest.out`).
- Every original probe was re-run. Outputs are in `research/audits/probes/reverify_2026-09-24/*.out`.
- p02, p05, p09 and p14 were adapted to the new APIs as `p02b`, `p05b`, `p09b` and `p14b`. Each tests the same failure as the original, plus new attacks.
- New probes:
  - `p16_crypto_clock_xcorr.py` checks the crypto server clock.
  - `p17_new_knobs.py` covers `stop_fill`, `spread_max`, residual MAE and money-mode swap currency.
  - `p18` checks crypto swap nights (run inline; output in `p18_crypto_nights.out`).

### Status of each finding

| ID | Finding | Status | Evidence |
|---|---|---|---|
| B1 | Shorts without both a stop and a target closed on their entry bar | **CLOSED** | `p01`: all four short variants now run to `eod` (+847 pts). `p07`: the stop-only real-data run has 30 signal exits and 41 stops on the short side, identical to the far-target run. |
| M1 | M1 path of the last bar, or of a bar before a gap, reads future M1 | **CLOSED** for the stated cases | `p02b` A: the last D1 bar maps 1 440 M1 rows, ending at 23:59 of its own day. B: the last-H1-bar long exits `eod` 1.11650 (pre-fix: `target` from 2019-06-03). C: a session-filtered frame maps at most 60 rows per bar. F: `assert_engine_causal` passes on H1 slices with full-year M1. Residuals N3/N4 below. |
| M2 | Look-ahead auditor misses sparse and out-of-band leaks | **PARTIALLY CLOSED — STILL OPEN** | `p09b`: see the table after this one. |
| M3 | Currency conversion looks ahead 2-3 h (+1 min) | **CLOSED** | `p05b`: a naive series raises. The rate at t is the close of the M1 bar that *closed* at t (107.623 at 10:00, not 107.625). The same holds at t+30 s and for the NZD→CHF / EUR→JPY paths. `p14b`: `apply_sizing` without `bars=` raises; `rate_fn` receives `Datetime[ms, UTC]`; every entry and exit rate matches an independent closed-bar as-of; max risk error is −1e-5. |
| M4 | Swap not modelled | **Money-mode inclusion CLOSED. Export still OPEN (by design). See NEW N1** | `p11` (a): money mode now books −398, the same as points mode (−2 622.14 for both). `data/broker/` is still absent, so all specs are uncalibrated with zero swap. |
| M5 | MAE/MFE include price action after the exit | **CLOSED on the whole-bar path. Residual on the M1 path (MINOR)** | `p10` (a): the post-stop rally no longer counts (MFE 990 → 10). `p10` (b): MAE > 1.2× loss dropped from 47% to 8% of stop exits. **Residual:** on the M1 path, the exit minute's full range is still included, so MAE is not capped at the fill. Worst case (`p17` c): the 2019-09-12 ECB-minute short had MAE 446 against a loss of 159. In a synthetic long (`p10` a), MAE was 210 against a loss of 110. Fix: cap the exit sub-bar's excursion at the fill price, as the whole-bar path already does. |
| minor 1 | Intra-minute liquidity gaps | **CLOSED as a knob. Two gaps: NEW N5** | `stop_fill="bar_extreme"` works (`p17` a: −2 224 → −4 010 pts with M1, −8 214 without M1). |
| minor 2 | D1 ≠ 17:00 NY wording | **CLOSED** | DESIGN §1 now says "normally 17:00 New York, 18:00 during the 2–3 US/EU DST-mismatch weeks". |
| minor 3 | Crypto `ts_utc` duplicates / steps backwards | **Symptom CLOSED; replacement is wrong — see NEW N2** | `p15`: 0 duplicates, 0 backward steps. |
| minor 4 | Crypto weekend swap nights | **CLOSED (with a caveat)** | `p18`: BTC Fri→Mon = 3 nights. Caveat: `swap_every_day` is combined with the Wednesday ×3, so a Mon→Mon hold is 9 weighted nights, not 7. That is 29% over-charged until the export confirms the rule (over-charging is the conservative direction). |
| minor 5 | `infer_point` reads the holdout | **CLOSED** | The code now filters `ts < holdout_start` before head/tail. |
| minor 6 | `source_file` not checked against `symbol` | **CLOSED** | `p04`: EURUSD with the XAUUSD file now raises ValueError. |
| minor 7 | `conversion_rate` raises for valid last-minute dev timestamps | **CLOSED** | `p04`: 2025-05-14 23:59:30 now resolves (last bar closes 00:00). |
| minor 8 | Cache key has no version; writes not atomic | **CLOSED** | `_RESAMPLE_SCHEMA_VERSION` is in the key; writes go through temp file + `os.replace`. `p03`: 1 800 bars re-checked, 0 mismatches. |
| minor 9 | Any agent can call `record_holdout_unlock` | **STILL OPEN** | Not fixed in code, as stated by the coordinator. |
| minor 10 | Spread timing on eod/force and no-M1 short checks | **CLOSED** | Uses the last-minute spread (with M1) or `spread_max`. On real data `spread_max ≥ spread` always, so it is conservative. `p17` b: H1 shorts go from −2 176 (M1) to −2 656 (no M1); D1 shows no difference. A tight-stop warning was added. |
| minor 11 | Signals are desired state (re-entry after a stop) | **CLOSED (documented)** | Engine module docstring. |

**M2 detail (`p09b`).** Each leaky strategy was run through `assert_no_lookahead`, then backtested.

| leaky strategy | passes auditor | backtest |
|---|---|---|
| `sparse_peek` (pre-fix escape) | 0/5 — caught | — |
| `veto_peek` (nulls out crosses that lose within 6 bars) | 0/5 — caught by the random-walk future at forced rows | +5 057 pts |
| `exit_peek` (exit one bar before a −20-pip move) | 0/5 — caught | — |
| `external_d1` via `data.load_bars` attribute | caught by the sandbox | — |
| **`from quantlab.data import load_bars`** (bound at import time) | **passes (1/1)** | 96.8% win, +67 207 pts |
| **`quantlab.data._load_resampled`** (private helper) | **passes** | same |
| **`pl.scan_parquet(<catalog file>)`** | **passes** | same |
| **D1 fetched in `__init__` and closed over** | **passes** | same |

Why the last four pass: the sandbox rebinds only the module attributes `data.load_bars` and `data.conversion_rate`. Any reference taken earlier, any other entry point, and any data captured before `signals()` runs are untouched.

### NEW findings

**N1 — MAJOR (latent until the broker export exists).** Money-mode swap is always treated as *quote-currency* money.
- MT5's three money modes are CURRENCY_SYMBOL (base currency), CURRENCY_MARGIN (margin currency) and CURRENCY_DEPOSIT (account currency). None of them means "quote currency" in general.
- `sizing.apply_sizing` multiplies `swap_money_per_lot` by the quote→account rate.
- `p17` (d): USDJPY in money mode at −5/lot/night gives −210 in swap. Only −1.91 USD is booked; it was divided by USDJPY.
- Failure scenario: any symbol exported as CURRENCY_DEPOSIT or CURRENCY_SYMBOL with base ≠ account currency has its swap scaled wrong by a factor of up to about 150.
- Fix: keep the raw MT5 enum (or the swap currency) in `InstrumentSpec` and convert from *that* currency.

**N2 — MAJOR (latent: hits any UTC-keyed join for crypto, including the planned economic-calendar join and BTC/ETH/LTC conversion).** The fixed-offset crypto localisation is wrong. The crypto server clock *does* observe EU DST, like FX.
- `p16`: cross-correlation of |1-min returns| between BTCUSD and XAUUSD/EURUSD, keyed by naive server time, peaks at **lag 0 in both summer and winter**:
  - 2022 summer: 0.138 / 0.146
  - 2024 summer: 0.161 / 0.146
- Keyed by the library's new `ts_utc`, the summer peak moves to **lag +60 min** (0.138 / 0.146 / 0.161 / 0.146 at +60). Winter is unchanged, as expected.
- So from late March to late October, crypto `ts_utc` is **1 hour later than the true instant**. A bar the library labels 12:30 UTC actually happened at 11:30 UTC.
- The fixer's inference in the `data.py` docstring misreads the evidence:
  - The 60 rows at the autumn repeated hour are what a DST-observing MT5 clock produces, because the repeated labels overwrite each other.
  - Spring-forward days from 2021 onward have essentially no 03:xx rows (4 rows vs 48-60 in 2019-2020). The clock did skip that hour.
- Failure scenario: a strategy that filters crypto bars on news events via an as-of join on `ts_utc` would see each release up to 1 h before it happened.
- Fix: return crypto to the zoneinfo rule. Handle only the 2017-2020 spring-forward days that have bars at the non-existent hour: flag them via `calendar_anomalies`, drop them, or give them explicit offsets.

**N3 — MINOR.** When `timeframe` is omitted, span inference overstates the span on sparse frames, which re-opens a last-bar look-ahead.
- `p02b` D: a frame keeping only the 08:00 H1 bar of each day infers a 1-day span. The last bar maps M1 through 23:59.
- A long opened on it exits at `target` 1.11425 using prices up to 15 h after the bar closed. With `timeframe="H1"` it is `eod` 1.11276.

**N4 — MINOR.** A mis-declared `timeframe` is accepted silently.
- `p02b` E: D1 bars run with `timeframe="H1"` restrict each day's M1 window to its first hour. Stops and targets later in the day are missed, and P&L changes from −4 545 to −8 059 with no error.
- Suggested check that catches both N3 and N4: raise if the declared span is below the minimum bar gap. Better still, assert that each HTF bar's high and low equal the max and min of its mapped M1 window, a cheap per-bar consistency test.

**N5 — MINOR.** The `stop_fill` knob is not part of `CostModel.version`.
- `CostModel(stop_fill="bar_extreme").version == "fbs-v0-uncalibrated"`, so two different cost models log the same version to the ledger (DESIGN §8).
- `stressed()` does not enable `bar_extreme`, although the docstring says it is "meant for the validation stress test".
- Also note: without M1, `bar_extreme` fills at the whole HTF bar's extreme. For D1 that is the day's high or low, which is very pessimistic.

**N6 — MINOR.** The auditor's cost grows quadratically with dense signals.
- `_forced_cut_points` forces a cut at every active row. A state-style strategy (±1 on every row) gets about n cut points × 4 `signals()` calls, each O(n).
- `p09b`: 768 bars took 8-47 s per strategy. A 10-year H1 frame (about 60k rows) is impractical, which invites auditing on short frames.
- Suggestion: force cuts at rows where the signal *changes* (plus the row before). Sample the rest.

### Verdict
The three Phase 0 exit-test items:

1. **Known-answer backtests: pass.** B1 is fixed and 160 tests are green.
2. **Hand-checked trades: pass.** All 7 re-verified trades (`p08`, `p13`) still match raw M1 to floating-point precision: fills, exit reasons and pnl_points. The stop-only shorts now match too.
3. **Look-ahead truncation test: passes at the strategy level for every leak that goes through the `bars` argument, and at the engine level (`assert_engine_causal`).** It is still escapable through data access outside the sandbox. The most common escape is the everyday `from quantlab.data import load_bars` import.

**Overall:** the FX/metals engine core now meets the exit test. Phase 0 is **not a clean pass**. Three MAJOR items remain:
- **M2 residual:** the sandbox escape.
- **N1:** money-swap currency.
- **N2:** crypto `ts_utc` off by 1 h in summer.

M4 (broker export) is also open by design.

Recommendation: pass Phase 0 for the **FX/metals, no-swap-dependent** scope only, once the M2 sandbox is hardened.
- Hardening: check a module-level "audit in progress" flag *inside* `load_bars`, `_load_resampled` and `conversion_rate` themselves, instead of rebinding the attributes.
- Add a red-team checklist item for raw parquet reads and data captured in `__init__`; no runtime sandbox can catch those.

Keep crypto (N2) and any swap-sensitive system (N1/M4) out of Phase 1+ gate runs until those are fixed.

---

## Re-verification round 2 (2026-09-24)

**How this was checked**
- Library under test: `research/audits/probes/reverify_2026-09-24/r2_library_sha1.txt`. It did not change while the probes ran.
- Test suite: **198 passed** (`r2_pytest.out`).
- Outputs are in `probes/reverify_2026-09-24/r2_*.out`.
- Probes adapted to the new APIs without weakening them:
  - `timeframe=` added wherever `m1=` is used: `r2_p07`, `r2_p08`, `r2_p10`, `r2_p12`, `r2_p13`, and `p02c`, which replaces `p02b`.
- New probes:
  - `p19` swap-currency mapping, end to end via a fake broker TSV;
  - `p20` sandbox escapes, each run in a fresh process;
  - `p21` forced-cut cap;
  - `p22` HTF/M1 consistency check;
  - `p23` unlock phrase.

### Items re-checked

| ID | Status | Evidence |
|---|---|---|
| B1 shorts without both levels | **CLOSED** (unchanged) | `r2_p01`. |
| Hand-checks (7 trades) | **still exact** | `r2_p08`: every fill, reason and pnl_points matches raw M1. `r2_p13`: the ambiguous-bar trade resolves to target +159.93 with M1, and to stop −319.86 without M1. |
| M1 / N3 / N4 M1 window and timeframe | **CLOSED** | See the `p02c` breakdown after this table. |
| M3 conversion timing | **CLOSED** (unchanged) | `r2_p05b`, `r2_p14b`: rates are the closed-bar as-of, `rate_fn` gets UTC times, and all rates match an independent recomputation. |
| M5 MAE/MFE (incl. M1 residual) | **CLOSED** | `r2_p10` (a): the synthetic long now shows MAE = MFE-capped = loss 110. (b): 0 of 86 stop exits have MAE > 1.2× loss, max ratio 1.00. `r2_p17` (c): the ECB-minute case is gone. |
| N1 money-swap currency | **CLOSED** (the export is still pending) | `p19` with a fake export, account currency USD, 1 lot. See the breakdown after this table. |
| N2 crypto `ts_utc` | **CLOSED** | `r2_p16`: in 2022 and 2024, BTC vs XAUUSD/EURUSD peaks at lag 0 on the library's `ts_utc` in summer and in winter. 2019 is too noisy to tell either way. `r2_p15`: 0 duplicates, 0 backward steps. `calendar_anomalies("BTCUSD")` lists the dropped artefact hours: 48-60 rows on each 2017-2020 spring-forward day, and 4 rows (03:00-03:03) on 2021-2025. |
| N5 cost-model version | **CLOSED** | `r2_p17` (a): `version` = `fbs-v0-uncalibrated+stop_fill=bar_extreme`; `stressed()` = `…+spread_mult1.5+slippage1+stop_fill=bar_extreme`. |
| N6 auditor cost | **CLOSED as a cost problem. Regression, see R2-2** | Dense strategies now finish in seconds. |
| minor 4 crypto nights | **CLOSED** | `r2_p18_crypto_nights.out`: BTC Mon→Mon = 7 nights / 7 weighted; Fri→Mon = 3; EURUSD Mon→Mon = 5 nights / 7 weighted. |
| minor 9 holdout unlock | **Improved, STILL OPEN (MINOR)** | `p23`: agent code can call `unlock_phrase(book, system)` and pass the result, and the unlock succeeds. The phrase is public and deterministic. `.claude/settings.local.json` has no deny rule for `record_holdout_unlock`, `unlock_phrase` or writes to `research/ledger/holdout_access.jsonl`. The confirmation is stored, so the unlock is auditable, but it is not enforced. |
| M2 look-ahead sandbox | **PARTIALLY CLOSED — STILL OPEN** | See R2-1 and R2-2. |

**`p02c` breakdown (M1 / N3 / N4):**
- A last-bar long with full M1 and `timeframe="H1"` exits `eod` 1.11650.
- Omitting `timeframe` with `m1` raises.
- A one-bar-per-day frame with `timeframe="H1"` exits `eod`; declared as `D1` it is rejected by the consistency check.
- D1 bars declared as H1 are rejected.
- An H1 WFO slice declared as D1, with M1 running past the slice, is rejected on its last bar.
- `assert_engine_causal(timeframe="H1")` passes at 40 cut points with full-year M1.

**`p19` breakdown (N1):** fake export, USD account, 1 lot.

| symbol and MT5 mode | swap currency | swap per lot | booked |
|---|---|---|---|
| USDJPY `CURRENCY_DEPOSIT` | `ACCOUNT` | −15.00 | −15.00 USD (was −1.91 before) |
| USDJPY `CURRENCY_DEPOSIT`, EUR account | `ACCOUNT` | −15.00 | −15.00 EUR (identity, correct) |
| EURUSD `CURRENCY_SYMBOL` | EUR | −15 EUR | −17.26 USD (×1.1507) |
| EURJPY `CURRENCY_MARGIN` | EUR | −30 EUR | −34.49 USD |
| GBPUSD `INTEREST_CURRENT` (−2%/yr) | USD | 1.26343 × 100 000 × 2% / 360 × 2 nights | −14.04 USD (hand arithmetic matches) |
| AUDUSD `POINTS` | USD | −15 pts | −15.00 USD |

**Does the HTF/M1 consistency check reject legitimate data? (`p22`)**
- **Accepted, correctly:**
  - EURUSD H1, H4 and D1 against M1 over the same window.
  - The broker's native H1 file against M1: 0 of 6 212 bars differ in high or low.
  - H1 with warm-up history from 2018-12 and M1 only from 2019-01-01. Bars without M1 coverage are skipped.
  - BTCUSD H1, H4 and D1 against M1 across the 2019, 2020 and 2021 spring-forward weeks and the 2019 fall-back week. The dropped artefact minutes are removed from both M1 and the resample, so there is no HTF/M1 mismatch.
- **Rejected, correctly:**
  - D1 declared as H1, and H4 declared as H1.
  - H1 bars paired with GBPUSD M1.
  - M1 with one bar deleted.
- **Rejected, strict but defensible:**
  - M1 that starts mid-bar (10:30 against an H1 bar at 10:00).
  - M1 that ends mid-day against a D1 bar.

  Walk-forward code must slice M1 on HTF bar boundaries. Partial coverage would otherwise silently truncate a bar's stop/target path, so rejecting it is the safer behaviour. Worth a sentence in the `run_backtest` docstring.
- **Accepted, harmless:** H1 bars declared as D1 over a same-window M1. The `min(next_start, ts+span)` bound caps every interior window. Where it matters (the last bar with M1 running further, or a session-gapped frame), the check does reject it (`p02c` G, D′).
- `spread_max` plays no part in the check, so it cannot cause false rejections.

### Findings still open or new in round 2

**R2-1 — MAJOR. The look-ahead sandbox is still escapable, including by two innocent-looking patterns.**

`p20` runs each attempt in a fresh process. The leak joins today's full D1 bar direction onto H1 bars, which wins 96.8% of trades in a backtest.

- **Caught** (`r2_p09b`, `p20`):
  - `quantlab.data.load_bars`, whether called via the module attribute or bound earlier with `from … import`;
  - `_load_resampled`;
  - `pl.scan_parquet` / `read_parquet` called via the attribute;
  - `pandas.read_parquet` (it goes through the patched `pq.read_table`);
  - `ThreadPoolExecutor` / `threading.Thread`, caught only because the reader patch is process-global. The contextvar is not inherited by new threads.
- **ESCAPED** (the auditor passes):

  | escape | innocent-looking? |
  |---|---|
  | data loaded at **module import time** into a module-level cache | yes |
  | a **pre-built instance** passed as `strategy=` with data fetched in `__init__` (`factory=` is optional) | yes |
  | `from polars import scan_parquet` bound at import time | fairly common |
  | `polars.io.parquet.functions.scan_parquet` | deliberate |
  | `pyarrow.parquet.ParquetFile(...).read()` | deliberate |
  | `pyarrow.dataset.dataset(...)` | deliberate |
  | `open(file, "rb")` → bytes → pyarrow | deliberate |
  | reading `data/_quantlab_cache/*.parquet` with `ParquetFile` | deliberate |
  | a `subprocess` running `load_bars` | deliberate |
- **Robustness bug:** `importlib.reload(quantlab.data)` inside `signals()` makes the auditor crash, which is safe. But the sandbox's `finally` runs `AUDIT_IN_PROGRESS.reset(token)` first, and that raises `ValueError` before the readers are restored. **`pl.scan_parquet`, `pl.read_parquet` and `pq.read_table` then stay patched for the rest of the process**, so every later `load_bars` fails (`r2_p20` last block).
- **Related:** the reader patch is process-global. Running an audit in one thread while backtests run in another makes their legitimate reads fail.
- **Fix:**
  1. Make `factory=` mandatory, or reject instances whose `__dict__` holds DataFrames or arrays.
  2. Restore the readers *before* `reset`, each in its own `try`.
  3. Add a static lint over strategy source files, run by the red team or the research-cycle skill. It should forbid module-level I/O, `polars`/`pyarrow`/`pandas` I/O imports, `open(`, `subprocess`, `importlib` and `quantlab.data`/`quantlab.config` imports in strategy modules.

  No runtime sandbox can be complete. The lint closes the deliberate escapes; the innocent ones need items 1 and 2.

**R2-2 — MAJOR (regression introduced by the N6 fix). Capping forced cut points at 200 hides sparse leaks in dense strategies.**
- `p21`: a 1-bar-momentum strategy (about 3 200 signal changes per year) that peeks one bar ahead on a sparse subset of rows.

  | leak rows | passes the auditor |
  |---|---|
  | 124 | 0/10 seeds (caught) |
  | **31** | **9/10 seeds** |
  | **10** | **10/10 seeds** |

- The 200 forced cuts, evenly spaced over the change points, plus 25 random cuts rarely land on a leak row.
- Before N6, every active row was forced (quadratic cost, but this leak was caught).
- **Fix:** keep the cap for routine use, but add `exhaustive=True`, required for the gate/validation run. It truncates at every row: O(n) `signals()` calls. For vectorised polars strategies that is minutes on a 10-year H1 frame. Alternatively, scale the cap to a time budget rather than a fixed 200.

**R2-3 — MINOR.** `swap_every_day` is hard-coded from the symbol name (`_is_crypto`) and is not read from the export. The same is true of the "no Wednesday triple on every-day symbols" rule. If FBS's crypto CFDs actually use a weekday-plus-triple schedule, the export cannot override it. Suggest exporting MT5's per-weekday swap multipliers (`SYMBOL_SWAP_SUNDAY` … `SYMBOL_SWAP_SATURDAY`) and using them.

**R2-4 — MINOR.** The consistency check rejects M1 frames that start or end mid-HTF-bar (`p22`). This is correct, but it is a new constraint for walk-forward code (slice M1 on HTF bar boundaries), and it is not mentioned in `run_backtest`'s docstring.

**minor 9 — MINOR (still open).** See the table above.

**M4 — open by design.** There is no broker export yet, so every spec is uncalibrated: zero swap, zero commission, guessed contract sizes.

### Checks that found nothing new in round 2
- **Holdout guard:** `r2_p04` shows 0 leaks.
- **Resample/`ts_utc` correctness for FX:** unchanged code path; round-1 `p03` still holds (cache schema v3 re-checked through `p22`'s consistency passes).
- **Sizing and equity invariants:** `r2_p11` shows money and points modes equal (−2 622.14), risk error ≤ 0, and the daily-equity truncation test unchanged.
- **Flash-crash and weekend-gap behaviour:** unchanged (`r2_p12`); `stop_fill="bar_extreme"` is available for stress.
- **Crypto nights:** 7/week with no triple; FX 5 nights / 7 weighted.
- **Engine causality:** `assert_engine_causal` with M1 passes.

### Round-2 verdict

The exit test:
1. **Known-answer backtests: PASS** (198 tests).
2. **Hand-checked trades: PASS** (7/7 exact).
3. **Look-ahead truncation test:** PASS at engine level (`assert_engine_causal`). At strategy level, pass for leaks that come through `bars`, `quantlab.data` or plain polars/pandas reads, but not a clean pass: R2-2 (sparse leak in a dense strategy) and R2-1 (import-time caches, pre-built instances, direct pyarrow/`open` reads) still get through.

**Phase 0 passes its exit test *conditionally*, for the FX/metals scope.** The engine, data, timezone, conversion and sizing layers are correct under every probe I could build. The remaining risk is concentrated in the look-ahead *auditor*. Both MAJOR items have cheap fixes: `factory=` mandatory + restore-order fix + a static strategy lint (R2-1), and an `exhaustive=True` gate mode (R2-2). I recommend doing those before any Phase 1 gate run relies on `assert_no_lookahead` as evidence.

**Still open:**

| item | severity |
|---|---|
| R2-1 sandbox escapes + reader-restore bug | MAJOR |
| R2-2 forced-cut cap regression | MAJOR |
| M4 broker export (swap/commission/contract terms uncalibrated) | MAJOR, open by design; blocks swap-sensitive systems |
| minor 9 unlock not enforced (no permission deny rule) | MINOR |
| R2-3 `swap_every_day` / triple rule not export-driven | MINOR |
| R2-4 M1 must be sliced on HTF bar boundaries, undocumented | MINOR |
| minor 1 intra-minute liquidity gaps (mitigated by `stop_fill` in stress; not modelled in base costs) | MINOR |
