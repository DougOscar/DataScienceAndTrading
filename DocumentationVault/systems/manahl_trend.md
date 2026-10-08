# ManAHL Trend (multi-lookback vote EA)
**Book:** FBS   **Status:** killed (S1 (pre-card power check), Undetectable: after-cost Sharpe 0.03 (0.27 gross, t≈0.7, gold-long only); gates need ≥1.0–1.34 on 7.7y; FX+metals trend prior 0.2–0.5, scope: EURUSD, GBPJPY, NZDCHF, XAUUSD D1)   **Issue:** none (exploratory, pre-card; no GitHub issue)
**Idea:** D1 trend vote over 20/60/120/240-day vol-normalised returns (score −4..4), half/full scaled positions, Kaufman ER entry filter, 4×ATR disaster stop, portfolio vol targeting.
**Result:** unprofitable — 0.015%/month (≈0) at 10% DD budget (after-cost daily returns scaled by 0.10/0.1398; eval 2017-08-18 to 2025-05-14)
**Risk profile:** Lumpy — MaxDD 14.0%, longest DD 63.8 months (1,339 days), max losing streak 13 trades, skew 0.27, CVaR95 -1.12%
**Robustness:** DSR 0.54 (N = 1, after costs) · CSCV OOS loss not run (killed at S1) · WFO OOS Sharpe not run (killed at S1) · holdout not unlocked
![[manahl_trend_sharpe_yearly.png]]
![[manahl_trend_sharpe_monthly.png]]

## Why killed / lessons
- Power (decisive): over 7.7 y the gates need an after-cost Sharpe of at least 1.0 (OOS gate), about 1.34 at 30 trials, and MinTRL rules out anything below about 0.6. The breadth formula, which matches this run (predicted 0.26 vs observed 0.27 gross), caps a 32-symbol FX+metals basket at 0.29-0.42 *before* costs. No parameter or indicator change closes a 4-5x gap. See notebook §10-11.
- Still worth trying, as NEW hypotheses: FX carry (needs the swap export), crypto trend, or a DESIGN path for low-Sharpe portfolio diversifiers.
- Costs: the EA trades at 00:00 server time, where rollover spreads (4-7x daytime on FX) ate the gross edge; execution hour matters (daytime-spread what-if Sharpe 0.24 vs base 0.03, frictionless 0.27, stressed -0.13, rollover-conservative -0.30). Audit: research/audits/2026-10-07_bar_spread_semantics.md.
- Concentration: per-symbol frictionless Sharpe XAUUSD 0.68, NZDCHF 0.14, EURUSD 0.03, GBPJPY -0.26; 56% positive years, best year 56% of P&L. "Lumpy" because total P&L is near zero (richest year exceeds 100% of it).
- Costs use fbs-v2-uncalibrated, specs uncalibrated (swap=0). 6 stop-outs in 693 legs.
- Post-hoc looks (run_record.json): daytime-spread what-if and 3 execution-hour probes (01:00, 02:00, 10:00). If revisited, declare as prior_trials.
- Reusable: quantlab/strategies/manahl_trend.py simulator. Sources: research/exploratory/manahl_trend/.
