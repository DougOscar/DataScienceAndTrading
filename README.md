# DataScienceAndTrading

A collection of Jupyter notebooks showcasing data science work and experiments with different models applied to trading, ranging from classical technical analysis to machine learning.

## Repository layout

```
.
├── data/                 # Datasets used by the notebooks (mostly gitignored)
├── technical_analysis/   # Notebooks exploring classical TA indicators and strategies
├── machine_learning/     # Notebooks for ML / DL model experiments
└── source/               # Reusable Python classes and helpers shared across notebooks
```

### `data/`
Input datasets (CSV, Parquet, etc.). Bulky files are ignored via `.gitignore`; only the folder itself is tracked so notebook paths stay valid after cloning.

### `technical_analysis/`
Notebooks that study indicators (moving averages, RSI, MACD, Bollinger bands, ...) and rule-based strategies.

### `machine_learning/`
Notebooks that train and evaluate ML/DL models on market data (regression, classification, time-series models, ...).

### `source/`
Python modules with custom classes and utilities intended to be reused by notebooks here and potentially by other projects. It is currently empty and will be populated as patterns emerge from the notebooks.

## Getting started

1. Clone the repo.
2. Create a virtual environment and install the libraries you need (pandas, numpy, scikit-learn, jupyter, ...).
3. Launch Jupyter and open any notebook under `technical_analysis/` or `machine_learning/`.

## Status

Early work-in-progress. This README will be expanded as the project grows.
