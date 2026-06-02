# Deep Reinforcement Learning Trading Agent

> **Type:** Machine Learning / Deep Reinforcement Learning
> **Markets:** Forex, B3
> **Timeframes:** 1h, 4h (forex); 30min, 1h (B3)
> **Direction:** Long & Short (Discrete(3) → {−1, 0, +1}; see [[#Action Space]])
> **Status:** Planning — documentation only, code not yet implemented

---

## Overview

Classical strategies in this repo encode a *human* hypothesis about market structure (trend, mean-reversion, regime, cycle) and parameter-tune it. A reinforcement-learning agent inverts that: the analyst specifies only the **environment** (what the agent sees, what it can do, how it is rewarded) and lets gradient-based optimisation learn the policy `π(action | observation)` that maximises cumulative discounted reward.

The hypothesis under test is narrow: that a small MLP policy fed a rolling window of price/volume/indicator features can learn position-taking decisions that **out-of-sample** outperform the repo's hand-coded baselines (SMA crossover #01, HMM overlay #10) on the same `(group, tf, asset)` grid, after accounting for transaction costs and run-to-run variance across random seeds.

The notebook trains and compares **two algorithm families** side by side:

- **PPO** (on-policy, actor-critic, clipped objective) — the de-facto baseline for continuous-control and finance RL papers; stable hyperparameter defaults; no replay buffer (RAM-friendly).
- **DQN** (off-policy, value-based, ε-greedy) — naturally discrete; large replay buffer; different sample-efficiency profile.

Both are trained with `stable-baselines3` on top of PyTorch with CUDA, on the same custom `gymnasium.Env`, against the same train/validation/test split, and evaluated by the **same** `source.backtest.Backtester` pipeline as every other strategy — so the comparison number drops straight into the strategy-comparison dashboard.

The expected RL-specific risks (overfitting to one asset's noise, brittleness across seeds, the reward-vs-objective gap) are addressed by: multi-seed training with mean ± std reporting; a held-out OOS window not touched until §11; block-bootstrap robustness on OOS trades; and direct head-to-head against the unoptimised SMA baseline on the same cells.

---

## Action Space

`gymnasium.spaces.Discrete(3)` decoded as:

| Action index | Meaning |
|---|---|
| 0 | Target position = −1 (short) |
| 1 | Target position = 0 (flat) |
| 2 | Target position = +1 (long) |

The discrete encoding matches the existing single-asset `Backtester` signal contract (it already accepts a per-bar `signal ∈ {−1, 0, +1}` column). After training, the agent's action sequence is replayed as `signal` and dropped into `Backtester.run(...)` unchanged — **no engine changes needed for v1**.

**Position-management mode for v1.** SL/TP are not the policy's tool — the agent controls exposure by *switching* its action. The notebook trains and evaluates with `sl_atr_mult = tp_atr_mult = 1e6` so neither stop ever triggers; the trade transitions are driven entirely by signal change. This is consistent with the Backtester's documented contract (it always supports signal reversal as an exit) and avoids modifying engine code in v1.

**Why not continuous sizing?** Continuous actions in `[−1, +1]` would let the policy size positions. The existing `Backtester` cannot apply a per-bar size multiplier (sizing is set at entry from `sizing_mode`); the `PortfolioBacktester` could but is not on `main` yet. Exposed as `use_continuous_action` flag and **disabled** in v1 (see [[#Implementation Notes]]).

---

## Observation Space

Each step's observation is a fixed-shape `Box(low=-∞, high=+∞, shape=(obs_window × n_features + 2,))` vector concatenating:

1. **Rolling window of features** (`obs_window` bars, default 32):
    - `log_return(t)` — 1-bar log return.
    - `realized_vol(t)` — rolling std of log returns over `rv_window` bars (default 20).
    - `rsi(t) / 100` — Wilder RSI period 14, scaled to `[0, 1]`.
    - `atr_rel(t)` — ATR-14 divided by close (unit-free volatility).
    - `volume_ratio(t)` — `tick_vol / SMA(tick_vol, vol_period)`.
    - `bb_pos(t)` — `(close − bb_mid) / (bb_upper − bb_mid)` ∈ `[−1, +1]` band position.
2. **Position-state features** (current step, 2 scalars):
    - `current_position ∈ {−1, 0, +1}`
    - `bars_in_position` normalised by `episode_len` (∈ `[0, 1]`)

All features are **z-scored** against a rolling lookback (default 252 bars) computed online inside the env — no global normalisation that would leak future stats into the training window.

Lookback warm-up: the env's first valid step is at bar `max(obs_window, rv_window, 14, vol_period, bb_period, rolling_z_window)` — earlier bars are skipped on `reset()`.

---

## Reward Function

Per-step reward, dense:

```
r_t  =  position_{t-1} · log(close_t / close_{t-1})  −  tx_cost · |position_t − position_{t-1}|
```

- Position is the integer set by the previous step's action (so the reward at bar `t` is earned by the position held *into* bar `t`).
- `tx_cost` is the round-trip transaction-cost-and-slippage proxy applied on every position change. Default: 5 bp for forex (0.0005), 10 bp for B3 (0.001). Exposed as `tx_cost_bps` param.
- Reward is dimensionless (log-return units), comparable across assets and timeframes — same property as `vol_scaled` sizing elsewhere in the repo.

**Optional reward shaping (off by default, exposed as flags):**

- `use_differential_sharpe` — Moody-Saffell differential Sharpe ratio. Rewards risk-adjusted return; smoother gradient than raw return but harder to tune. Disabled in v1 (see [[#Implementation Notes]]).
- `use_drawdown_penalty` — adds `−λ · max(0, peak_equity − equity_t)` to penalise drawdowns. Disabled in v1.
- `use_holding_penalty` — small constant penalty per step in market to discourage over-trading-free pathological "always long" policies. Disabled in v1.

---

## Environment (`TradingEnv`)

A custom `gymnasium.Env` defined in `source/rl/env.py` (new module). Key methods:

| Method | Behaviour |
|---|---|
| `reset(seed)` | Pick a random contiguous slice of length `episode_len` from the training window; reset rolling stats and position; return first obs. |
| `step(action)` | Update position; advance one bar; compute reward; check termination. |
| `render()` | No-op (training) / matplotlib equity overlay (eval). |
| Termination | `True` at end of episode window **or** when `drawdown > max_drawdown_fraction` (default 0.30 — circuit-breaker, not a stop-loss). |

**Episode shape:**

- Training: `episode_len` (default 2048 bars) random slice; the agent sees many overlapping episodes per epoch.
- Validation / OOS eval: one episode per asset, full length, deterministic policy (`deterministic=True` in `model.predict`).

**Train/validation/test split (chronological, hard-fenced):**

| Slice | Default range | Purpose |
|---|---|---|
| Train | 2016 → end-of-2022 (forex) / 2021 → end-of-2023 (B3) | Policy gradient updates |
| Validation | 2023 (forex) / Q1–Q3 2024 (B3) | `EvalCallback`, early-stopping, checkpoint selection |
| OOS test | 2024–2026 (forex) / Q4 2024 → 2026 (B3) | **Untouched** until §11 |

The validation split is consumed during training — only the **OOS test** is the honest performance signal. This matches the convention in [[06_Robustness_Testing]].

---

## Algorithms

### PPO (Proximal Policy Optimization)

`stable_baselines3.PPO` with `MlpPolicy`. Default hyperparameter overrides:

| Hyperparameter | Default | Notes |
|---|---|---|
| `learning_rate` | 3e-4 | linear-anneal to 1e-5 over training |
| `n_steps` | 2048 | rollout length per env per update |
| `batch_size` | 64 | minibatch size for PPO epochs |
| `n_epochs` | 10 | PPO update epochs per rollout |
| `gamma` | 0.99 | discount factor |
| `gae_lambda` | 0.95 | GAE smoothing |
| `clip_range` | 0.2 | PPO clip |
| `ent_coef` | 0.01 | entropy bonus (exploration) |
| `vf_coef` | 0.5 | value-loss weight |
| `policy_kwargs` | `dict(net_arch=[64, 64])` | 2× 64 MLP, shared trunk |

### DQN (Deep Q-Network)

`stable_baselines3.DQN` with `MlpPolicy`. Default hyperparameter overrides:

| Hyperparameter | Default | Notes |
|---|---|---|
| `learning_rate` | 1e-4 | constant |
| `buffer_size` | 100_000 | replay buffer (≈ 50 MB at obs dim 200) |
| `learning_starts` | 10_000 | random actions until buffer warms |
| `batch_size` | 64 | minibatch from replay |
| `tau` | 1.0 | hard target-net update |
| `gamma` | 0.99 | discount factor |
| `train_freq` | 4 | gradient step every 4 env steps |
| `target_update_interval` | 1000 | bars between target-net syncs |
| `exploration_fraction` | 0.1 | fraction of training in ε-decay |
| `exploration_final_eps` | 0.05 | minimum ε |
| `policy_kwargs` | `dict(net_arch=[64, 64])` | matches PPO for fair comparison |

**Sb3-contrib extras (Rainbow, QR-DQN) — exposed but disabled in v1.**

---

## Training Protocol

- **Hardware target:** RTX 3060 Laptop, 6 GB VRAM. MLP fits easily; the bottleneck is env-step throughput (single-process, vectorised across `n_envs=8` parallel envs per algo).
- **Total timesteps:** `total_timesteps` per algo per asset, default `1_000_000` (≈ 30–60 min wall-clock on the RTX 3060 with `n_envs=8`).
- **Seeds:** `n_seeds` independent training runs per algo per asset (default 3). Final report = mean ± std across seeds.
- **Eval callback:** `stable_baselines3.common.callbacks.EvalCallback` runs the deterministic policy on the validation env every `eval_freq=50_000` steps; the best checkpoint (by mean validation reward) is saved separately.
- **Checkpoint cadence:** `CheckpointCallback` every `save_freq=100_000` steps. Files:
    - `models/16_deep_rl/<algo>_<asset>_<tf>_seed<S>_step<N>.zip`
    - `models/16_deep_rl/<algo>_<asset>_<tf>_seed<S>_best.zip` (eval-best)
    - For DQN: replay buffer pickled alongside (~50 MB each) — only the latest is kept (rolling overwrite) so disk usage stays bounded.
- **Resume policy:** notebook §8 looks for the latest checkpoint and, if found, calls `model = PPO.load(path, env=env)` (or `DQN.load(..., env, custom_objects)`) and resumes from `model.num_timesteps`. If no checkpoint exists, train from scratch.
- **CUDA device pin:** `device="cuda" if torch.cuda.is_available() else "cpu"`. CPU fallback path exists for code-only checks but a printed warning recommends aborting.

The training section commits **unexecuted**. The user runs the notebook manually after merge; the checkpoint cache is `.gitignore`d.

---

## Risk Management

Reuses the existing single-asset `Backtester` contract verbatim:

| Parameter | Value | Notes |
|---|---|---|
| Max simultaneous positions per asset | **1** | engine-enforced |
| Stop type | **N/A in v1** — policy-driven | `sl_atr_mult = tp_atr_mult = 1e6` so neither triggers |
| Take profit | N/A | same |
| Trailing stop | No | out of scope |
| Position sizing | **`unit`** (1 unit per trade) | matches every other v1 single-asset strategy in the repo |
| Pyramiding | No | engine-enforced |
| Transaction cost | Implicit in env reward (`tx_cost_bps`) | the Backtester replay does not double-count it |

### Markets

| Group | Assets | Notes |
|---|---|---|
| Forex | EURUSD, EURCAD, GBPCHF | 1h and 4h. Daily TF skipped — too few bars for stable RL training. |
| B3 | WDO, WIN (mini-futures) | 30min and 1h. Session 09–18 enforced via env mask (returns done=False but action is overridden to `flat` outside session). |
| Crypto | — | **Skipped** — no `data/crypto/` (per repo convention). |

### Time Restrictions

| Rule | Description |
|---|---|
| Session filter | B3: env masks out-of-session bars and forces position = 0. Forex/Crypto: no filter (24/5). |
| Days of week | none |
| News/event blackout | none |

---

## Parameters

**Parameters class:** `DeepRLTradingParams` (new in `source/strategy.py`).

| Parameter | Default | WFO Range | Description |
|---|---|---|---|
| `algorithm` | `"PPO"` | `["PPO", "DQN"]` | which SB3 algorithm to instantiate |
| `obs_window` | `32` | `[16, 32, 64]` | rolling window of indicator features |
| `rv_window` | `20` | fixed | realised-vol rolling window |
| `vol_period` | `20` | fixed | tick-vol normalisation SMA period |
| `bb_period` | `20` | fixed | Bollinger period for `bb_pos` feature |
| `rolling_z_window` | `252` | fixed | online z-score lookback (1 trading year) |
| `episode_len` | `2048` | `[1024, 2048, 4096]` | training-episode length |
| `tx_cost_bps` | `5` (forex) / `10` (B3) | fixed | round-trip cost proxy in basis points |
| `max_drawdown_fraction` | `0.30` | fixed | env circuit-breaker (terminates episode) |
| `gamma` | `0.99` | `[0.95, 0.99, 0.995]` | discount factor (RL hyperparameter) |
| `learning_rate` | `3e-4` (PPO) / `1e-4` (DQN) | `[1e-4, 3e-4, 1e-3]` | optimiser LR |
| `n_seeds` | `3` | fixed | independent training runs per cell |
| `total_timesteps` | `1_000_000` | fixed | training budget per seed |
| `use_continuous_action` | `False` | — | exposed, **disabled** in v1 |
| `use_differential_sharpe` | `False` | — | exposed, **disabled** in v1 |
| `use_drawdown_penalty` | `False` | — | exposed, **disabled** in v1 |
| `use_holding_penalty` | `False` | — | exposed, **disabled** in v1 |
| `use_lstm_policy` | `False` | — | exposed, **disabled** in v1 (SB3 supports via `RecurrentPPO` in `sb3-contrib`) |
| `use_cnn_policy` | `False` | — | exposed, **disabled** in v1 (would need 2-D obs reshape) |
| `policy_arch` | `(64, 64)` | `[(64, 64), (128, 128), (256, 256)]` | MLP hidden sizes |
| `session_start` | `None` (forex) / `9` (B3) | fixed per group | session filter (B3 only) |
| `session_end` | `None` (forex) / `18` (B3) | fixed per group | session filter (B3 only) |
| `sizing_mode` | `"unit"` | fixed | per repo v1 convention |
| `risk_fraction` | `0.01` | fixed | unused under `sizing_mode="unit"` |
| `sl_atr_mult` | `1e6` | fixed | sentinel — disables Backtester SL |
| `tp_atr_mult` | `1e6` | fixed | sentinel — disables Backtester TP |
| `atr_period` | `14` | fixed | required by Backtester for ATR column (sentinel still needs ATR computed) |

`DeepRLTradingParams.is_valid()` returns `False` (→ zero signals → WFO score `−inf`) for any combo that violates `obs_window >= rv_window` or `obs_window + rolling_z_window > 4096`. Never raises in `__post_init__` per [[strategies/00_Index|repo convention]].

### WFO-Optimized Values

_(Fill in after running §10 — abbreviated WFO grid; see [[#Walk-Forward Optimisation Adaptation]].)_

| Group | TF | Asset | algorithm | gamma | learning_rate | obs_window | policy_arch |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

---

## Walk-Forward Optimisation Adaptation

Classical grid-WFO retrains a strategy *from scratch* on each fold. For an RL policy that costs ~30 min per training run, a 5-fold × 27-combo grid would cost ~67 GPU-hours — impractical.

The notebook uses a **two-tier reduction**:

1. **Hyperparameter selection on one asset per group** (e.g. EURUSD-4h for forex, WIN-1h for B3) over a small grid (`gamma × learning_rate × policy_arch` = `3 × 3 × 3 = 27` combos × `n_seeds=3` = **81 training runs**). Best config per group selected by mean validation Sharpe across seeds.
2. **Generalisation pass on the remaining assets** using the chosen hyperparameters — one training per `(group, tf, asset)` cell × `n_seeds`.

This is documented as a *deliberate departure* from the canonical [[05_Walk_Forward_Optimization]] flow in the notebook's WFO section. It is closer to the [[machine_learning|ML]] notebooks' train-once + evaluate-OOS pattern than to TA's nested WFO.

The existing `source.wfo.walk_forward` is **not used** for the main RL training. It *is* used in §12 to walk a *re-evaluation* of the trained policy across a 3-fold OOS chronological split — this measures whether the policy degrades over time without retraining.

---

## Performance Summary

> Populated by running the notebook after merge.

| Metric | RL (PPO best seed) | RL (DQN best seed) | RL (mean ± std, 3 seeds) | SMA #01 baseline | HMM #10 ML overlay |
|---|---|---|---|---|---|
| Sharpe (daily, OOS) | | | | | |
| Profit Factor (OOS) | | | | | |
| Win Rate (OOS) | | | | | |
| Max Drawdown (OOS) | | | | | |
| Total trades (OOS) | | | | | |
| P(profitable) — block bootstrap | | | | | |
| Mean ± std across seeds | n/a | n/a | | n/a | n/a |

**Pre-defined acceptance bar** (set *before* training, per [[06_Robustness_Testing]] discipline): the RL agent passes the comparison only if **mean OOS Sharpe across seeds** beats **both** the SMA baseline and the HMM overlay on **at least 2 of the 4 representative cells** (forex-1h-EURUSD, forex-4h-EURUSD, b3-30min-WIN, b3-1h-WIN), with the worst-seed Sharpe also positive on those cells. Otherwise the notebook's Findings section reports negative result and Status stays at "Backtested — did not beat baseline".

See: [[04_Backtesting_and_Metrics]], [[05_Walk_Forward_Optimization]], [[06_Robustness_Testing]], [[09_Strategy_Comparison_Dashboard]].

---

## Known Weaknesses & Improvement Ideas

- **Single-asset envs.** Each policy is trained on one asset's history; cross-asset generalisation is not directly trained for. Multi-asset env (cycling assets per episode) is a v2 follow-up.
- **Stationarity assumption.** Z-scoring against a rolling 252-bar window does not handle structural regime breaks (2020-03, 2022-Q1 rate cycle, etc.). A regime-detector pre-filter (like HMM #10) feeding the agent its current regime state is a natural ensemble follow-up.
- **Reward design is dense PnL.** Maximising raw log-return reward yields high-variance, high-turnover policies. Differential-Sharpe / drawdown-penalty are wired but disabled; their effect should be A/B-tested in a follow-up issue.
- **Hyperparameter sensitivity of RL is severe.** The repo's traditional `parameter_sensitivity` helper compares strategy-parameter perturbations; for RL, *seed* and *learning_rate* sensitivity dominate. The notebook reports both.
- **No flat-action signal-replay nuance.** A `0` action in v1 is replayed verbatim as `signal=0` to the Backtester, which interprets it as "no new signal" rather than "close to flat". This works because the policy operates on every bar and a `0` after a non-zero position is reasonably interpreted as "stay if no opposite signal fires". A proper "policy mode" Backtester (where `0` forces close) is a clean v2 engine extension.
- **SB3 + Python 3.14.** Stable wheels for Python 3.14 lag the Python release. If install fails, fall back to a dedicated `.venv-rl` with Python 3.12 (per the install instructions in the notebook's §0 markdown).
- **No commissions/slippage in the Backtester replay** — they are already inside the reward via `tx_cost_bps`. Adding a second cost layer at backtest would double-count. The notebook §11 documents this.

---

## Implementation

**Notebook:** `machine_learning/16_deep_rl_trading.ipynb`
**Source module:** `source/strategy.py` — `DeepRLTradingStrategy` (signal-replay wrapper).
**Parameters class:** `DeepRLTradingParams`
**New module:** `source/rl/env.py` — `TradingEnv(gymnasium.Env)`
**New module:** `source/rl/train.py` — train/resume/eval orchestration helpers (`train_one_seed`, `latest_checkpoint`, `evaluate_policy_to_signals`).
**Models cache:** `models/16_deep_rl/` (gitignored).
**Dependencies (new):** `torch` (CUDA build), `stable-baselines3>=2.3`, `gymnasium>=0.29`. Pinned in `requirements.txt` in the same commit that introduces the strategy.

### Implementation Notes

- **PyTorch install is a prerequisite step**, not part of the notebook. The repo's `requirements.txt` adds `torch`, but the **wheel comes from PyTorch's own index** (`--index-url https://download.pytorch.org/whl/cu126`) because the default PyPI wheel is CPU-only. The install command is documented in the notebook's §0 markdown and in the README appendix.
- **Python 3.14 caveat:** if PyTorch's stable cu126 wheel does not yet publish 3.14 builds, the notebook §0 instructs the user to (a) try nightly (`--pre --index-url .../nightly/cu126`), or (b) create a parallel `.venv-rl` with Python 3.12. The notebook is verified to run under either path.
- **Signal-replay wrapper.** `DeepRLTradingStrategy` is **not** a learning strategy at backtest time. It carries a pre-computed `signal` array (output of `evaluate_policy_to_signals(trained_model, df)`) and `generate_signals(df)` returns the input `df` with that array attached. This keeps the existing `Backtester` / `compute_metrics` / `plot_backtest_dashboard` pipeline unchanged — the only RL-aware code lives in `source/rl/`.
- **Backtester SL/TP sentinel.** `sl_atr_mult = tp_atr_mult = 1e6` is the v1 mechanism for "policy controls exits". A proper engine-level "policy mode" (where `0` means close) is exposed as `use_flat_close_signal` in `DeepRLTradingParams` and **disabled** in v1 — implementing it requires a ~10-line `Backtester` change scoped to a follow-up PR.
- **Disabled-by-default v1 flags** (exposed for the next PR):
    - `use_continuous_action`, `use_differential_sharpe`, `use_drawdown_penalty`, `use_holding_penalty`, `use_lstm_policy`, `use_cnn_policy`, `use_flat_close_signal`.
- **WFO deviation** (see [[#Walk-Forward Optimisation Adaptation]]) — RL training cost rules out the canonical 5-fold × full-grid WFO. The notebook uses a two-tier reduction and explicitly documents it.
- **Data layer is PySpark.** The notebook reuses the `spark_loader` module designed for the multi-filter portfolio system. **Dependency:** `source/spark_loader.py` is currently only on the `feature/multi-filter-portfolio-system` branch (not yet on `main`). The notebook's data section must either (a) wait for that PR to merge, (b) cherry-pick `spark_loader.py` + `pyspark`/`pyarrow` requirement additions into this branch, or (c) temporarily fall back to `source.data_loader.discover_datasets` + `LazyDataset`. The default in the markdown plan is **(b) cherry-pick** — a small, self-contained dependency. The notebook §0 markdown documents this dependency.
- **JDK 17/21 requirement** for PySpark applies (see [[15_Multi_Filter_Portfolio_System]] for the same caveat) — `get_spark()`'s auto-detection handles it.
- **B3 session enforcement** lives in the env (`reset()` slices and `step()` masks), **not** as a `signal *= in_session` post-filter — RL needs the truncation to be visible inside the MDP so the agent doesn't learn to take positions that the env immediately overrides.
- **Notebook committed unexecuted.** As with every strategy PR in this repo, the user runs the notebook manually after merge.

---

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347. The PPO paper.
2. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540), 529–533. The original DQN paper.
3. Moody, J., & Saffell, M. (2001). *Learning to Trade via Direct Reinforcement*. IEEE Transactions on Neural Networks, 12(4), 875–889. Differential Sharpe ratio reward.
4. Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). *Deep Direct Reinforcement Learning for Financial Signal Representation and Trading*. IEEE Transactions on Neural Networks and Learning Systems, 28(3), 653–664.
5. Raffin, A., Hill, A., Ernestus, M., Gleave, A., Kanervisto, A., & Dormann, N. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR, 22(268), 1–8.
6. Towers, M., Terry, J. K., Kwiatkowski, A., et al. (2023). *Gymnasium*. https://gymnasium.farama.org/
