# Tear sheet — fbs-0023-a1

**Class:** unprofitable — -0.0%/month at the 10% DD budget (bootstrap p5/p50/p95 = -0.08%/-0.02%/0.05%)  
**Gate verdict:** FAIL  
**Risk type:** A  

## Key risks

- Failed gate(s): Deflated Sharpe probability, CSCV OOS loss probability, OOS Sharpe (CPCV median), Walk-forward procedure OOS, Cost-stress Sharpe, Parameter plateau, Max single-year PnL share, Trade count vs MinTRL.
- Holdout not unlocked: the figures above are development-data only.

## Metrics applicability

| Metric | Shown | Why |
|---|---|---|
| cagr | yes | always reported (DESIGN §5) |
| mean_monthly | yes | always reported (DESIGN §5) |
| sharpe | yes | always reported (DESIGN §5) |
| sortino | yes | always reported (DESIGN §5) |
| calmar | yes | always reported (DESIGN §5) |
| max_dd | yes | always reported (DESIGN §5) |
| longest_dd_days | yes | always reported (DESIGN §5) |
| ulcer | yes | always reported (DESIGN §5) |
| cvar95 | yes | always reported (DESIGN §5) |
| worst_month | yes | always reported (DESIGN §5) |
| pct_time_underwater | yes | always reported (DESIGN §5) |
| psr0 | yes | always reported (DESIGN §5); recomputed here (deterministic, not a gate) |
| ret_at_budget | yes | always reported (DESIGN §5, §4.5 classification) |
| cost_pct_of_gross | yes | always reported (DESIGN §5); approximated from the trades frame's cost columns |
| break_even_spread_multiple | yes | always reported (DESIGN §5); approximated from the trades frame's cost columns |
| swap_share_of_costs | yes | always reported (DESIGN §5); approximated from the trades frame's cost columns |
| dsr | yes | logged gates event, gate_run 1 |
| cscv_oos_loss | yes | logged gates event, gate_run 1 |
| oos_sharpe | yes | logged gates event, gate_run 1 |
| wfo_oos | yes | logged gates event, gate_run 1 |
| plateau | yes | logged gates event, gate_run 1 |
| cost_stress_sharpe | yes | logged gates event, gate_run 1 |
| positive_years | yes | logged gates event, gate_run 1 |
| max_year_share | yes | logged gates event, gate_run 1 |
| holdout_status | yes | from the ledger's holdout_unlock/holdout_exam rows (never from fresh holdout bars) |
| pbo | yes | diagnostic only; recomputed from the trial store with stats.pbo_cscv |
| win_rate_profit_factor | yes | DESIGN §5, risk type A |
| holding_time | yes | DESIGN §5, risk type A |
| expectancy_r | yes | DESIGN §5, risk type A |
| max_losing_streak_r | yes | DESIGN §5, risk type A |
| risk_realisation_error | yes | DESIGN §5, risk type A |
| raw_points_per_trade | no | lot size varies with the stop, so raw points are not comparable across trades (use R-multiples) |
| pnl_ccy_distribution | no | shown as R-multiples instead (risk varies by design; currency P&L mixes that in) |
| risk_per_trade_distribution | no | risk type A, not B |
| mae_mfe | no | risk type A, not C |
| worst_trade | no | risk type A, not C |
| turnover_exposure_cost_per_turnover | no | type A is a discrete-position system: use R-multiples / trade stats instead |

## Metrics

| Metric | Value | Interpretation |
|---|---|---|
| cagr | -0.02553 | CAGR -2.6%: compound annual growth of the %-equity curve. |
| mean_monthly | -0.001426 | Average calendar-month return -0.14% (unlevered, all months equal weight). |
| sharpe | -0.1178 | Annualised Sharpe -0.12 from daily returns (periods/year = 260). |
| sortino | -0.183 | Sortino -0.18: like Sharpe but only penalises downside deviation. |
| calmar | -0.0509 | Calmar -0.05: CAGR / max drawdown magnitude. |
| max_dd | -0.5017 | Max drawdown -50.2% of the running peak (unlevered). |
| longest_dd_days | 2043 | Longest stretch below the running peak: 2043 trading days. |
| ulcer | 0.3302 | Ulcer index 0.330: RMS drawdown depth (penalises deep and long drawdowns). |
| cvar95 | -0.01988 | Daily CVaR95 -1.99%: mean return on the worst 5% of days. |
| worst_month | -0.105 | Worst calendar month -10.50%. |
| skew | 1.159 |  |
| kurtosis | 12.28 |  |
| pct_time_underwater | 0.9783 | 98% of trading days were below the running peak. |
| psr0 | 0.3622 | PSR(SR*=0) 0.36: probability the true per-period Sharpe exceeds 0, given its skew/kurtosis over 2352 days -- NOT deflated for multiple testing (see DSR under Robustness for that). |
| cost_pct_of_gross | 18.08 | Costs are 1808.4% of gross P&L (gross 1,221, net -20,863, cost 22,084, 100k nominal). |
| break_even_spread_multiple | nan | Break-even spread multiple: not defined (net P&L <= 0 or no spread cost). |
| swap_share_of_costs | -0 | Swap is -0% of total cost. |
| gate_verdict | FAIL |  |
| dsr | 0.000197 | Deflated Sharpe probability 0.000197 (>= 0.95, status FAIL). |
| cscv_oos_loss | 0.7887 | CSCV OOS loss probability 0.789 (< 0.1, status FAIL). |
| oos_sharpe | -0.6826 | OOS Sharpe (CPCV median) -0.683 (>= 1.0, status FAIL). |
| wfo_oos | 0.06736 | Walk-forward procedure OOS 0.07 (>= 0.5, status FAIL); recent third (recomputed from the stored WFO OOS series, same 33% window the gate uses) -0.07. |
| plateau | 0 | Parameter plateau 0.00 (>= 0.6, status FAIL): weakest parameter axis 'lookback' keeps >= half the peak Sharpe on 0% of its judge-run perturbations. |
| cost_stress_sharpe | -0.9174 | Cost-stress Sharpe -0.917 (> 0.5, status FAIL). |
| positive_years | 0.6 | Positive years 0.6 (>= 0.6, status PASS). |
| max_year_share | None | Max single-year PnL share — (<= 0.4, status FAIL). |
| wfo_oos_recent | -0.07403 |  |
| holdout_status | not unlocked | Holdout: not unlocked (checkpoint B has not run). |
| pbo | 0.5312 | PBO 0.53 (diagnostic, not a v1.2 gate — replaced by CSCV OOS loss): probability the in-sample-best configuration ranks at or below the out-of-sample median. |
| win_rate | 0.294 | Win rate 29% over 670 trades. |
| profit_factor | 0.9304 | Profit factor 0.93 (gross win / gross loss). |
| n_trades | 670 |  |
| expectancy_points | -12.29 |  |
| hold_days_p95 | 3 |  |
| expectancy_r | -0.02273 | Expectancy -0.02 R over 670 trades (mean P&L / planned risk). |
| max_losing_streak_r | 11 | Longest losing streak: 11 trades in a row with a negative R-multiple. |

## Gates (as logged in the ledger)

| Gate | Value | Threshold | Status |
|---|---|---|---|
| Deflated Sharpe probability | 0.000197 | >= 0.95 | FAIL |
| CSCV OOS loss probability | 0.789 | < 0.1 | FAIL |
| OOS Sharpe (CPCV median) | -0.683 | >= 1.0 | FAIL |
| Walk-forward procedure OOS | 0.0674 | >= 0.5 | FAIL |
| Cost-stress Sharpe | -0.917 | > 0.5 | FAIL |
| Parameter plateau | 0 | >= 0.6 | FAIL |
| Positive years | 0.6 | >= 0.6 | PASS |
| Max single-year PnL share | — | <= 0.4 | FAIL |
| Trade count vs MinTRL | 670 | >= MinTRL | FAIL |
| Mechanism check | — | manual None | MANUAL |

## Risk profile

**Trend-like** — MaxDD 50.2%, longest DD 97.3 months, max losing streak 11, skew 1.16, CVaR95 -1.99%
