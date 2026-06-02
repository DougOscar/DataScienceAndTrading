"""Reinforcement-learning building blocks for strategy #16.

This subpackage is intentionally import-light at the top level: importing
``source`` (or ``source.rl``) must not pull in ``torch`` /
``stable_baselines3`` so the non-RL notebooks keep working on the base venv.
The heavy deps are imported lazily inside :mod:`source.rl.train`.

``source.rl.env`` only needs ``gymnasium`` + numpy/pandas.
"""

from .env import TradingEnv, compute_feature_panel, FEATURE_COLUMNS

__all__ = ["TradingEnv", "compute_feature_panel", "FEATURE_COLUMNS"]
