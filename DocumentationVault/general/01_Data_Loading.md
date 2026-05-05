# Step 1 — Data Loading

**Source:** `source/data_loader.py` | **Notebook section:** §1

## What It Does

`load_all(data_dir)` scans a directory for OHLC CSV files in both the root and named subdirectories. The subdirectory name becomes a `group` tag on each dataset.

## File Naming Convention

```
<ASSET>_<TIMEFRAME>_<STARTYYYYMMDDHHMM>_<ENDYYYYMMDDHHMM>.csv
```

Examples:
- `data/forex/EURUSD_M1_201604110000_202604212358.csv`
- `data/b3/WDO_M1_202104300900_202604291829.csv`

## Return Structure

```python
datasets: dict[(asset, timeframe), (DatasetMeta, pd.DataFrame)]
```

`DatasetMeta` carries: `asset`, `timeframe`, `start`, `end`, `group`.

The DataFrame index is a `DatetimeIndex`; columns are `open`, `high`, `low`, `close`.

## Splitting by Group

```python
forex_raw = {k: v for k, v in datasets.items() if v[0].group == "forex"}
b3_raw    = {k: v for k, v in datasets.items() if v[0].group == "b3"}
```

## Current Datasets

| Group | Asset | TF | Rows | Date Range |
|-------|-------|----|------|------------|
| b3 | WDO | M1 | 695,501 | 2021-04-30 → 2026-04-29 |
| b3 | WIN | M1 | 689,104 | 2021-04-30 → 2026-04-29 |
| forex | EURCAD | M1 | 3,719,603 | 2016-04-11 → 2026-04-21 |
| forex | EURUSD | M1 | 3,717,726 | 2016-04-11 → 2026-04-21 |
| forex | GBPCHF | M1 | 3,715,005 | 2016-04-11 → 2026-04-21 |

## Adding New Data

1. Place the CSV in `data/<group>/` with the correct filename format.
2. Re-run `load_all` — new files are picked up automatically.
3. Update [[02_MultiTimeframe_Preparation]] if a new group needs different target timeframes.

## Known Limitations / Next Steps

- No volume data currently — strategies relying on volume require schema extension.
- All data is raw 1-minute bars; data quality checks (gaps, spikes) are not yet automated.
- B3 data has significantly fewer years than Forex — sub-period analysis will have fewer folds.
