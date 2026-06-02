Implement a documented trading strategy as a runnable Jupyter notebook end-to-end: branch, read docs, plan, code, execute, debug, commit, and open a PR.

## Your inputs

Arguments: "$ARGUMENTS" — a strategy identifier. Accept any of these forms:
- `02` (two-digit number prefix)
- `02_RSI_Mean_Reversion`
- `DocumentationVault/strategies/02_RSI_Mean_Reversion.md` (full or relative path)

Working directory: `/home/douglaso/Finances/DataScienceAndTrading/`
Repo handle: `DougOscar/DataScienceAndTrading`

Resolve the doc:
- Find the matching file in `DocumentationVault/strategies/` whose basename starts with the two-digit prefix.
- If multiple candidates match, list them and stop.
- If none match, stop and ask the user to clarify.

From the resolved doc, extract:
- `NN` — two-digit prefix (e.g. `02`)
- `slug_lower` — lowercase filename (e.g. `rsi_mean_reversion`) — strip the `NN_` prefix and lowercase the rest
- `class_name` — from the doc's "Implementation" section (e.g. `RSIMeanReversionStrategy`). If absent, derive PascalCase from the slug + `Strategy`.
- `params_class` — from the doc's "Implementation" section (e.g. `StrategyParams` or `RSIMeanReversionParams`).
- `target_groups` — from the "Markets" / "Approved Markets" tables (e.g. `forex`, `b3`, `crypto`).
- `target_timeframes` — per-group from the "Timeframes" / "Markets" sections.
- `param_grid` — from the "Parameters" table's `WFO Range` / `WFO Grid` columns. Skip params marked `fixed` or `—`.

## Step-by-step

### 1. Create the feature branch

```bash
git checkout main
git pull origin main
git checkout -b feature/strategy-<NN>-<slug_lower>
```

If the branch already exists, switch to it and pull. If there are uncommitted changes in the working tree, stop and tell the user.

### 2. Read the strategy doc

Read the full strategy file. From it, lock in:
- Indicator formulas and lookback requirements
- Entry / exit / filter rules and exit priority
- Risk management (stops, sizing, session)
- Parameter defaults and WFO grid
- Any constraints (e.g. `fast < slow`, `rsi_lower < 50 < rsi_upper`)

If the doc is malformed (missing entry/exit/parameters sections), stop and tell the user which sections are missing — do not guess.

### 3. Read the pipeline documentation

Read all of `DocumentationVault/general/`:
- `00_Overview.md` — pipeline shape and source layout
- `01_Data_Loading.md` — `load_all`, `DatasetMeta`, file naming convention
- `02_MultiTimeframe_Preparation.md` — `resample_ohlc`, group→timeframe mapping
- `03_Strategy_Definition.md` — strategy interface contract
- `04_Backtesting_and_Metrics.md` — `Backtester`, `compute_metrics`, `metrics_table`, multi-asset portfolio
- `05_Walk_Forward_Optimization.md` — `walk_forward`, `pick_best_params`, OOS stitching
- `06_Robustness_Testing.md` — `monte_carlo_trades`, `block_bootstrap_trades`, `subperiod_analysis`, `parameter_sensitivity`

Also read `source/parallel.py`, `source/runner.py` and the `LazyDataset`
section of `source/data_loader.py`. Every notebook in this repo is expected
to use the lazy-loading + parallel-runner patterns — see §6 below.

Also skim the existing baseline notebook `technical_analysis/01_baseline_sma_crossover.ipynb` and `source/strategy.py`, `source/backtest.py`, `source/__init__.py` so the new notebook reuses the same infrastructure and matches the same shape.

The expected notebook structure (mirrors §1–§6 of the baseline):

| § | Title | Content |
|---|-------|---------|
| 1 | Load the data | `load_all(REPO_ROOT / "data")`, split by group |
| 2 | Multi-timeframe preparation | resample to per-group target timeframes |
| 3 | Strategy definition + baseline backtest | define `<class_name>` params, run baseline across all (group, tf) — display portfolio metrics tables and per-asset dashboards |
| 4 | Walk-Forward Optimization | run `walk_forward` per (group, tf, asset), combine, display OOS metrics; pick best params per (group, tf, asset) and run optimized full-history backtest |
| 5 | Robustness | on the best-OOS-Sharpe TF per group: Monte Carlo, block bootstrap, sub-period (yearly), parameter sensitivity |
| 6 | Takeaways | brief written summary of best config, regime sensitivity, and what to do next |

Section 7 (Extensions: session filter, sizing modes, portfolio weighting) is **out of scope** for this command — it is added in a follow-up.

### 4. Decide where the strategy code lives

Default: add the new `class_name` and (if needed) a new `<...>Params` dataclass to `source/strategy.py`, then export them from `source/__init__.py`.

Make a `params_class` judgment call:
- If the doc lists `params_class = StrategyParams` and the new strategy needs only fields already on `StrategyParams` (`atr_period`, `sl_atr_mult`, `tp_atr_mult`, `session_start`, `session_end`, `sizing_mode`, `risk_fraction`), reuse it.
- Otherwise create a new dataclass `<ClassName>Params` that exposes the same field names the `Backtester` reads (`sl_atr_mult`, `tp_atr_mult`, `session_start`, `session_end`, `sizing_mode`, `risk_fraction`, `as_dict()`) plus the strategy-specific fields. This keeps `Backtester` and `walk_forward` working unchanged.

The strategy class must implement the same interface as `SMACrossoverStrategy`:
- `__init__(self, params)` storing `self.params`
- `compute_indicators(df) -> DataFrame` adding any indicator columns plus an `atr` column (Backtester reads `atr` for SL/TP sizing — even if the strategy logic doesn't use ATR for entries, ATR must be present for stop placement)
- `generate_signals(df) -> DataFrame` returning the indicator DataFrame with an integer `signal` column (`+1` long, `-1` short, `0` no action) and respecting any session filter

Implement all indicators exactly as the doc specifies (formula, lookback, smoothing). Do **not** invent variations.

### 5. Plan the notebook in markdown first

Build the notebook by writing a Python helper script that constructs the `.ipynb` JSON via `nbformat`. Do **not** hand-write JSON.

Order: every section starts with one or more markdown cells stating *what* the section does and *why*, then the code cells. Markdown text should match the strategy doc's terminology so a reader can map sections back to it.

Required markdown cells (at minimum):
- Title cell with strategy name and link to the doc (`[[strategies/NN_Name]]`)
- §1 description, §2 description (state per-group target timeframes), §3 description (state entry/exit rules and default params), §4 description (state WFO grid and `n_splits` / `oos_ratio`), §5 description (state which TF was selected and why), §6 (takeaways — fill in after results are observed)

### 6. Implement the notebook code

Reuse the existing `source/` building blocks. Do not re-implement Backtester, WFO, or robustness logic.

**Performance + memory contract.** Every section that fans out across (group,
timeframe, asset) MUST use the lazy-loading + parallel-runner pattern. Failure
modes you are explicitly avoiding:

* **RAM** — never call `load_all(...)` and then iterate the full result. CSVs
  are large (~200 MB per forex M1 file); materialising every (asset, tf) at
  once will OOM as the data dir grows. Prefer `discover_datasets` /
  `build_lazy_grid` and let workers materialise a single frame at a time.
* **Wall clock** — never use sequential nested for-loops over (group, tf, asset)
  for backtests, WFOs, or sensitivity sweeps. These are independent units of
  work — fan them out via `run_backtest_grid`, `run_wfo_grid`,
  `run_backtests_with_params`, or `parallel_map`.

§1 — Load data:
```python
from source import discover_datasets, build_lazy_grid
metas = discover_datasets(REPO_ROOT / "data")          # metadata only, no I/O
group_tfs = build_lazy_grid(REPO_ROOT / "data",
                            group_timeframes=GROUP_TIMEFRAMES)
```
Skip target groups for which no data is present (e.g. crypto), but warn in a
printed message. If **none** of the doc's target groups have local data, stop
and report — do not silently substitute a different group.

§2 — Multi-timeframe prep: `build_lazy_grid(...)` returns
`{group: {tf: {asset: LazyDataset}}}`. Don't call `.load()` here — the
runners do it inside workers.

§3 — Strategy + baseline:
- Construct `baseline_params = <ParamsClass>(<all defaults from the doc>)`
- Run baseline across the entire grid in parallel:
  ```python
  flat = run_backtest_grid(group_tfs, baseline_params,
                           strategy_cls=<ClassName>, n_jobs="auto")
  baseline_per_asset = reshape_grid_results(flat)
  baseline_portfolio = {g: {tf: build_portfolio(by_asset)
                             for tf, by_asset in tfs.items()}
                        for g, tfs in baseline_per_asset.items()}
  ```
- Display `metrics_comparison` tables per group across timeframes
- Plot `plot_backtest_dashboard` per (group, tf) portfolio

§4 — WFO:
- Build `param_grid` dict using the WFO ranges from the doc
- Run WFO across the entire grid in parallel:
  ```python
  flat_wfo = run_wfo_grid(group_tfs, param_grid=param_grid,
                          n_splits=5, oos_ratio=0.25,
                          strategy_cls=<ClassName>, params_cls=<ParamsClass>,
                          n_jobs="auto", inner_n_jobs=1)
  wfo_per_asset = reshape_grid_results(flat_wfo)
  wfo_combined = {g: {tf: build_combined_wfo(by_asset)
                       for tf, by_asset in tfs.items()}
                  for g, tfs in wfo_per_asset.items()}
  ```
  Use `inner_n_jobs=1` when the outer grid is already ≥ #cores; raise it only
  when you have ≤ 2 cells (e.g. a single asset / single timeframe sweep).
- Combine OOS trades across assets per (group, tf), display OOS metrics
- Use `pick_best_params` (port the helper from the baseline notebook) to
  choose the most-frequent param combo per (group, tf, asset). To re-run
  full-history backtests with **per-cell** optimised params, use
  `run_backtests_with_params(group_tfs, params_by_key, ..., n_jobs="auto")` —
  it accepts a `{(group, tf, asset): params}` dict and dispatches one job
  per cell.

§5 — Robustness on the best-OOS-Sharpe TF per group:
- `monte_carlo_trades`, `monte_carlo_summary` (vectorised — n_runs=1 000 is
  millisecond-cheap, no parallelism needed)
- `block_bootstrap_trades` + summary (also vectorised)
- `subperiod_analysis(trades, freq="YE")`
- `parameter_sensitivity(df, base_params, variations, strategy_cls=<X>,
  n_jobs="auto")` — pass `n_jobs="auto"` so the (param, value) sweep
  parallelises across the process pool
- `plot_robustness_dashboard` per group

For each (group, tf) where you need a raw price DataFrame (e.g. the
sensitivity slice), use the `LazyDataset.using()` context manager:
```python
with group_tfs[group][tf][asset].using() as df:
    sensitivity_results[group] = parameter_sensitivity(df, ..., n_jobs="auto")
gc.collect()
```
This guarantees the frame is dropped immediately after the section finishes.

§6 — Takeaways: a short markdown cell summarizing best config, key risks, and regime sensitivity. Fill this **after** the notebook has executed successfully so the numbers are real.

Match the baseline notebook's variable names where possible (`group_tfs`, `baseline_per_asset`, `baseline_portfolio`, `wfo_combined`, `optimized_portfolio`, `mc_results`, `sensitivity_results`, etc.) so a reader can compare the two notebooks side by side.

**Other places to look for parallelism.** The (group, tf, asset) loop is the
main lever, but the same pattern fits anywhere the work is independent:

* Section 7 extension comparisons (sizing modes, session filters, weighting
  schemes) — each *configuration* is independent of the others. Wrap each
  configuration in a `run_backtest_grid` call rather than re-running a
  nested for-loop.
* Synthetic asset generation — each synthetic asset is independent. If the
  generator becomes a bottleneck, fan it out with `parallel_map`.
* ML training / cross-validation folds — same shape as WFO folds; reuse
  `parallel_map` from `source.parallel`.

Default to `n_jobs="auto"` (= cpu_count − 1). Drop to `1` only when
debugging a worker-side traceback (which is otherwise hidden behind the
process boundary).

### 7. Execute the notebook

Use the project's venv. nbconvert may not be installed:

```bash
.venv/bin/python -c "import nbconvert" 2>/dev/null \
  || .venv/bin/python -m pip install nbconvert
```

Then execute in place:

```bash
.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  technical_analysis/<NN>_<slug_lower>.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Set `--ExecutePreprocessor.timeout=1800` (30 min) — the WFO grid can be large.

### 8. Debug

If execution fails:
- Read the traceback from the failing cell's output. Locate the actual cause — do **not** comment out cells or wrap broken code in `try/except` to make it pass.
- Common failures and where to look:
  - `KeyError: 'atr'` → strategy's `compute_indicators` must add an `atr` column even if entries don't use ATR
  - `AttributeError: 'XParams' object has no attribute 'sl_atr_mult'` → params dataclass missing fields the `Backtester` reads
  - WFO returns empty trades on every fold → param grid is too narrow or warm-up exceeds fold size; check `n_splits` vs. data length
  - Indicator NaN runs longer than expected → lookback / `min_periods` mismatch with the doc's stated warm-up
  - Constraint violations in WFO grid (e.g. `fast >= slow`) → filter the param grid before search, or add a guard inside the strategy that returns no signals
- Re-run after each fix. Repeat until the full notebook executes cleanly.

If a single (group, tf) combination is fundamentally incompatible with the strategy (e.g. lookback exceeds available bars), skip it explicitly with a printed warning rather than letting the cell crash.

### 9. Sanity-check the results

Before committing, glance at the executed notebook outputs and confirm:
- Trade counts are non-zero on at least one (group, tf) combination
- Equity curves rendered (no empty `<Figure>` placeholders)
- WFO produced OOS metrics for at least one (group, tf, asset)
- Robustness section ran on the selected best-OOS TF

If any of these are empty across the board, the strategy is likely mis-implemented — debug before continuing.

### 10. Commit

Stage only files you created or modified:

```bash
rtk git add source/strategy.py source/__init__.py technical_analysis/<NN>_<slug_lower>.ipynb
rtk git commit -m "$(cat <<'EOF'
feat(strategy): implement <Strategy Name> baseline notebook (#<NN>)

Adds <ClassName> and <ParamsClass?> in source/strategy.py and the
end-to-end notebook technical_analysis/<NN>_<slug>.ipynb covering
data load, multi-timeframe prep, baseline backtest, WFO, and
robustness — wired to the spec in DocumentationVault/strategies/<NN>_<Name>.md.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Do not commit `.ipynb_checkpoints/` or notebook outputs that contain absolute paths to user-private data.

### 11. Push and open the PR

```bash
rtk git push -u origin feature/strategy-<NN>-<slug_lower>
```

```bash
gh pr create \
  --repo DougOscar/DataScienceAndTrading \
  --title "feat(strategy): implement <Strategy Name> baseline notebook" \
  --body "$(cat <<'EOF'
Implements strategy <NN> per `DocumentationVault/strategies/<NN>_<Name>.md`.

## Summary
- Adds `<ClassName>` (and `<ParamsClass>` if new) in `source/strategy.py`
- New notebook `technical_analysis/<NN>_<slug>.ipynb` covering §1 load → §2 multi-TF → §3 baseline → §4 WFO → §5 robustness → §6 takeaways
- Reuses `Backtester`, `walk_forward`, robustness helpers — no changes to shared infrastructure

## Spec compliance checklist
- [ ] Indicators implemented per the doc (formula, lookback, smoothing)
- [ ] Entry / exit / filter rules match the doc's exit priority
- [ ] Default params match the doc's "Parameters" table
- [ ] WFO grid matches the doc's "WFO Range" column
- [ ] Constraints (e.g. `fast < slow`) enforced or filtered
- [ ] Notebook executes end-to-end without errors

## Results snapshot
<paste the §3 baseline metrics_comparison output and the §4 OOS metrics_comparison output for each group>

## Out of scope
- Section 7 extensions (session filter, sizing modes, portfolio weighting) — follow-up

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

### 12. Confirm

Print the PR URL. Then point the user at the executed notebook for human review:

> "PR opened: <pr-url>. Notebook ready for review: `technical_analysis/<NN>_<slug>.ipynb`."
