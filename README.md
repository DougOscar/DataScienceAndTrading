# DataScienceAndTrading

A collection of Jupyter notebooks showcasing data science work and experiments with different models applied to trading, ranging from classical technical analysis to machine learning.

## Repository layout

```
.
├── data/                 # Datasets used by the notebooks (mostly gitignored)
├── technical_analysis/   # Notebooks exploring classical TA indicators and strategies
├── machine_learning/     # Notebooks for ML / DL model experiments
├── source/               # Reusable Python classes and helpers shared across notebooks
└── requirements.txt      # Python dependencies
```

### `data/`
Input datasets (CSV, Parquet, etc.). Bulky files are ignored via `.gitignore`; only the folder itself is tracked so notebook paths stay valid after cloning. CSV files are expected to follow the naming convention `<ASSET>_<TIMEFRAME>_<STARTYYYYMMDDHHMM>_<ENDYYYYMMDDHHMM>.csv` (e.g. `EURUSD_M1_201401012300_202604221434.csv`) and contain OHLC(V) columns with a datetime/date+time/timestamp column.

### `technical_analysis/`
Notebooks that study indicators (moving averages, RSI, MACD, Bollinger bands, ...) and rule-based strategies. The baseline notebook `01_baseline_sma_crossover.ipynb` wires together the full load → backtest → metrics → dashboards → walk-forward optimization → robustness pipeline.

### `machine_learning/`
Notebooks that train and evaluate ML/DL models on market data (regression, classification, time-series models, ...).

### `source/`
Python package with reusable classes: `data_loader`, `strategy`, `backtest`, `metrics`, `dashboard`, `wfo`, `robustness`. Notebooks add the repo root to `sys.path` and import directly from `source`.

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/DougOscar/DataScienceAndTrading.git
cd DataScienceAndTrading
```

### 2. Create an isolated Python environment

Pick whichever you prefer — both work the same way.

**Option A – venv (stdlib, no extra tools):**

```bash
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

**Option B – conda:**

```bash
conda create -n datasci-trading python=3.11 -y
conda activate datasci-trading
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Register a Jupyter kernel for this environment

This step tells Jupyter (and VS Code) that the environment you just created is a kernel it can run notebooks against.

```bash
python -m ipykernel install --user --name datasci-trading --display-name "Python (datasci-trading)"
```

- `--name` is the internal identifier (no spaces).
- `--display-name` is what you'll see in the kernel picker.
- To remove it later: `jupyter kernelspec uninstall datasci-trading`.
- To list registered kernels: `jupyter kernelspec list`.

### 5. (Optional) Launch JupyterLab

```bash
jupyter lab
```

Open any notebook under `technical_analysis/` and select **Python (datasci-trading)** from the kernel picker.

## Using the kernel in VS Code

1. Install the **Python** and **Jupyter** extensions from the VS Code marketplace.
2. Open this repository (`File → Open Folder…`).
3. Open any `.ipynb` under `technical_analysis/`.
4. In the top-right corner of the notebook, click the kernel selector (it will say something like *Select Kernel* or show the current kernel name).
5. Choose **Python Environments…** → pick the `datasci-trading` venv/conda env **or** choose **Jupyter Kernel…** → pick **Python (datasci-trading)** (the kernelspec registered in step 4 above).
6. Run the first cell. If imports like `from source import ...` work, you're set.

Tip — to make VS Code default to this interpreter for the whole workspace, open the Command Palette (`Ctrl/Cmd+Shift+P`) → **Python: Select Interpreter** → pick the venv/conda env. VS Code stores this in `.vscode/settings.json` (already gitignored).

## Status

Early work-in-progress. This README will be expanded as the project grows.
