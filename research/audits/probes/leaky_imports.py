"""Helper module for p09b: binds load_bars at import time (the most common import style)."""
from quantlab.data import load_bars as imported_load_bars  # noqa: F401
from quantlab import data as qd  # noqa: F401
