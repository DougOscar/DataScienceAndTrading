---
name: financial-ml
description: Machine learning for financial time series with PyTorch and TensorFlow — feature/label engineering, leakage-proof cross-validation (purged k-fold, embargo, CPCV), meta-labeling, feature importance, sequence models (LSTM/TCN/Transformers), and the model lifecycle (training, evaluation, registry, reproducibility). Use when designing or documenting an ML pipeline, forecasting models, or any task applying neural nets / gradient-boosting to market data.
---

# Financial Machine Learning (PyTorch / TensorFlow)

Financial ML fails differently from ordinary ML: signal-to-noise is tiny, data is non-stationary and serially correlated, and the most natural validation schemes leak. Default to skepticism — a model that looks great almost always has a leak or is overfit. Apply `quant-trading-research` and `applied-math-quant` alongside this.

## Framing the problem

- **Features (X):** must be strictly causal (computed from data ≤ decision time). Engineer from prices/returns (momentum, vol, range), microstructure (spread, tick imbalance), cross-sectional ranks, and regime indicators. Consider **fractional differentiation** to make series stationary while preserving memory.
- **Labels (y):** prefer the **triple-barrier method** (first of profit-take / stop / time barrier) over fixed-horizon returns; these reflect realistic exits and the path. Add **meta-labeling**: a primary model (or rule) sets direction, a secondary ML model predicts P(the primary call is correct) and thus position size. Meta-labeling improves precision and is where ML adds the most value.
- **Sample weights:** down-weight overlapping labels by **uniqueness**; optionally weight by absolute return (attention to the consequential samples) and apply time-decay.
- **Bars:** consider information-driven bars (volume, dollar, or imbalance bars) instead of time bars — they have better statistical properties (closer to iid, less heteroskedastic) for ML.

## Validation (this is where most edge evaporates)

- Plain k-fold and a single train/test split both **leak** because of serial correlation and label overlap. Use **purged k-fold with an embargo**: purge training samples whose label windows overlap the test fold, and embargo a gap after each test fold.
- Use **Combinatorial Purged Cross-Validation (CPCV)** to obtain many OOS paths and a *distribution* of performance, then compute **PBO**.
- **Never select hyperparameters on the test fold.** Tune on a purged validation split; report on untouched OOS. Track every configuration tried — the trial count feeds the Deflated Sharpe Ratio.
- Walk-forward (expanding/rolling) for the final temporal validation; confirm performance is not concentrated in one regime.

## Feature importance & interpretation

- **MDI** (impurity) is in-sample and biased toward high-cardinality features — diagnostic only.
- **MDA** (permutation importance, out-of-sample) is the workhorse; permute within purged folds.
- **SHAP** for local/global attributions and sanity-checking that the model uses defensible structure, not artifacts.
- A model whose importance is dominated by one suspicious feature is a leakage alarm.

## Models

- **Gradient-boosted trees (XGBoost/LightGBM)** are the right *first* model for tabular features — strong, fast, robust, interpretable. Beat this baseline before reaching for deep nets.
- **Sequence models** (PyTorch or TF/Keras): LSTM/GRU, Temporal Convolutional Networks (dilated causal convolutions — strictly no look-ahead), Transformers, and forecasting-specific architectures (N-BEATS, N-HiTS, Temporal Fusion Transformer). They need a lot of data and heavy regularization; on small/noisy financial sets they usually *underperform* boosted trees unless the structure is genuinely sequential.
- Regularize hard: dropout, weight decay, early stopping on a purged validation set, small models. Prefer probabilistic/quantile outputs (you care about uncertainty and tails, not point forecasts).

## PyTorch vs TensorFlow

- **PyTorch** is the default for research/experimentation (dynamic graphs, ecosystem, Lightning for training loops). **TensorFlow/Keras** is fine where a team standard, TF-Serving, or existing TF assets justify it. Keep model code framework-agnostic behind a small interface (e.g. `IForecaster` / `IModel`) so trading logic doesn't depend on either framework directly.
- Determinism: set seeds for python/NumPy/framework, enable deterministic kernels where feasible, pin library versions; record them in the run manifest. GPU non-determinism must be acknowledged in results.

## Lifecycle & reproducibility

- Treat every trained model as an artifact with a **manifest**: data snapshot hash, feature spec, label spec, CV scheme, hyperparameters, code version, seeds, metrics, and trial count.
- Use a **model registry / experiment tracker** (e.g. MLflow) so results are queryable and the multiple-testing budget is auditable.
- Separate *fit* (offline, infrastructure) from *inference* (the domain consumes predictions through an interface) so the trading logic never imports a DL framework directly.
