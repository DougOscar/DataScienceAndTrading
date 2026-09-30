# Toy H4 time-series momentum
**Book:** FBS   **Status:** killed (S5, DRY RUN: S5 FAIL — DSR 0.0002 (need >=0.95), CSCV OOS-loss prob 0.79 (need <0.10), CPCV OOS Sharpe -0.68 (need >=1.0), WFO OOS 0.07/recent -0.07 (need >=0.5/>0), cost-stress Sharpe -0.92 (need >0.5), plateau 0.0 (need >=0.6); S1 had already said KILL-EARLY and S3 baseline was negative.)   **Issue:** #23
**Idea:** DRY RUN (pipeline test, not a real system): EURUSD H4 ROC-sign cross, ATR stop, time exit, 1% fixed-fraction risk.
**Result:** unprofitable — -0.0%/month at 10% DD budget
**Risk profile:** Trend-like — MaxDD 50.2%, longest DD 97.3 months, max losing streak 11, skew 1.16, CVaR95 -1.99%
**Robustness:** DSR 0.00 · CSCV OOS loss 0.79 · WFO OOS Sharpe 0.07 (recent third -0.07) · holdout not unlocked
![[toy_tsmom_sharpe_yearly.png]]
![[toy_tsmom_sharpe_monthly.png]]
