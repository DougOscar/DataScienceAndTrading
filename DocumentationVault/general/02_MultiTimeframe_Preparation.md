# Step 2 — Multi-Timeframe Preparation

**Source:** `source/data_loader.py` (`resample_ohlc`) | **Notebook section:** §2

## Rationale

Different markets suit different analysis horizons. B3 mini-futures trade in an ~8-hour daily session — daily bars would be too sparse. Forex runs 24/5 and has enough bars even at daily resolution.

## Target Timeframes by Group

```python
GROUP_TIMEFRAMES = {
    "forex": ["1h", "4h", "1D"],
    "b3":    ["1min", "5min", "15min", "30min"],
}
```

## How Resampling Works

`resample_ohlc(df, target_tf)` converts a higher-frequency OHLC DataFrame to a lower frequency using standard OHLC aggregation:

- `open` → first bar open
- `high` → max of highs
- `low` → min of lows
- `close` → last bar close

If the source timeframe is already at or above the target (e.g. source is `1D`, target is `1h`), the raw DataFrame is returned unchanged.

## Output Structure

```python
group_tfs: dict[group, dict[tf, dict[asset, pd.DataFrame]]]
# e.g. group_tfs["forex"]["4h"]["EURUSD"] → DataFrame
```

## Bar Counts After Resampling

| Group | TF | Asset | Bars | Range |
|-------|----|-------|------|-------|
| Forex | 1h | EURUSD | 62,346 | 2016–2026 |
| Forex | 4h | EURUSD | 15,615 | 2016–2026 |
| Forex | 1D | EURUSD | 2,609 | 2016–2026 |
| B3 | 1min | WDO | 695,501 | 2021–2026 |
| B3 | 5min | WDO | 139,143 | 2021–2026 |
| B3 | 15min | WDO | 46,388 | 2021–2026 |
| B3 | 30min | WDO | 23,197 | 2021–2026 |

## Notes

- 1D bars for B3 are not in scope — the session structure (intraday open/close) makes daily resampling misleading without session-aware aggregation.
- Resampled DataFrames feed directly into [[03_Strategy_Definition]] without additional transformation.
- The session filter in [[07_Extensions]] operates on the resampled data, not at this stage.
