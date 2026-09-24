"""Interfaces shared by every quantlab module.  Keep this file small and stable.

Data model
----------
Bars are ``polars.DataFrame`` with columns::

    ts        Datetime[ms]  bar OPEN time, naive broker server time (MT5 convention)
    ts_utc    Datetime[ms, UTC]
    open, high, low, close   Float64   **Bid** prices (MT5 bars are Bid)
    spread    Int32/Float64  spread in points at bar open (M1: as exported; resampled: first M1 bar)
    spread_max Float64       max M1 spread inside the bar (resampled bars only; = spread for M1)
    tick_vol  Int64

A bar with open time ``ts`` covers ``[ts, ts + timeframe)`` and is *closed* at
``ts + timeframe``.  A signal computed on bar ``i`` is known at its close and
executes at the **open of bar i+1** (DESIGN §4.3).

Strategy contract
-----------------
A strategy maps bars → a signal frame (same length/order as bars)::

    signal       Int8 or null   desired position after this bar closes:
                                 +1 long, -1 short, 0 flat, null = keep current state
    stop_dist    Float64|null   stop distance in PRICE units from the fill price (null = no stop)
    target_dist  Float64|null   target distance in PRICE units from the fill price (null = no target)

Stops/targets are attached when a position opens and stay fixed (v1).  The
engine never lets a strategy see data after bar i when producing row i; the
``quantlab.testing.assert_no_lookahead`` helper proves it per strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol, runtime_checkable

import polars as pl


class RiskType(str, Enum):
    """Risk semantics declared by each system (DESIGN §5)."""

    A = "A"   # fixed-fraction risk, price-based stop → dynamic lots
    B = "B"   # fixed lots, variable stop distance
    C = "C"   # no hard stop (signal/time exit)
    D = "D"   # continuous / vol-targeted position (ML, RL)


SIGNAL_COLUMNS = ("signal", "stop_dist", "target_dist")
BAR_COLUMNS = ("ts", "ts_utc", "open", "high", "low", "close", "spread", "spread_max", "tick_vol")


@dataclass(frozen=True)
class Params:
    """Base class for strategy parameter sets (frozen, hashable, JSON-able)."""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Strategy(Protocol):
    name: ClassVar[str]
    risk_type: ClassVar[RiskType]
    params: Params

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        """Return a frame with ``SIGNAL_COLUMNS``, one row per input bar, same order."""
        ...


class HoldoutLocked(PermissionError):
    """Raised when code asks for holdout data without a ledger unlock (DESIGN §4.1)."""


# --------------------------------------------------------------------------- Phase 1: studies
# Daily returns convention (used by stats, opt, gates, portfolio):
#   pl.DataFrame(date: Date, ret: Float64) — daily simple return of equity on the book's
#   nominal account (sizing.daily_equity 'ret'), one row per server trading day of the
#   evaluated window, sorted, no gaps filled.  Wide matrices: 'date' + one column per trial.

@dataclass
class Outcome:
    """One evaluated parameter set over a window (normally the full dev window)."""

    daily: pl.DataFrame                 # date, ret
    trades: pl.DataFrame                # engine trades + sizing columns
    metrics: dict[str, float]


@runtime_checkable
class Evaluator(Protocol):
    """Maps a parameter set to an Outcome.  Rule-based systems are evaluated ONCE per
    parameter set on the full dev window; CV/walk-forward then slice the daily return
    matrix (valid because signals at t depend only on data <= t).  Systems that must be
    re-fitted per window (ML) set ``requires_refit = True`` and implement ``fit_window``
    (Phase 1 only needs the rule-based path; the flag keeps the API honest)."""

    book: str
    periods_per_year: float
    requires_refit: bool
    cost: Any                           # base costs.CostModel (gates derive the stressed model from it)

    def __call__(self, params: dict[str, Any], *, cost: Any = None) -> Outcome: ...


@dataclass
class StudyResult:
    """Everything a study produces; input to the gate evaluation (DESIGN §4.2)."""

    study_id: str
    param_names: tuple[str, ...]
    trials: pl.DataFrame        # trial_id, status, param_<name>..., m_<metric>...
    returns: pl.DataFrame       # wide: date + 't<trial_id>' daily returns (full dev window)
    selected_params: dict[str, Any]
    selection: dict[str, Any]   # method, plateau_score, neighbourhood, objective values
    cpcv_paths: pl.DataFrame    # date, path_id, ret — OOS returns of the *selection procedure*
    wfo_oos: pl.DataFrame       # date, ret, refit_id, n_trades (entries; null = unavailable) — re-optimisation schedule
    wfo_params: pl.DataFrame    # refit_id, refit_date, train_start, train_end, param_<name>...
    meta: dict[str, Any]        # cv scheme, schedule, n_trials, seeds, runtime, cost version, book/system/
                                # issue/attempt, space, candidate_set(_data_dependent), embargo_capped
