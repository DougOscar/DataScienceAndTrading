"""Everything the tear sheet reads from the ledger / trial store.

Nothing here recomputes a gate: gate values always come from the study's latest logged
``gates`` event (DESIGN §8), read verbatim.  Nothing here reads holdout *bars* either (that
stays behind ``/unlock-holdout``); the only holdout information used is the already-logged
``holdout_unlock`` / ``holdout_exam`` rows in ``holdout_access.jsonl``, i.e. ledger metadata,
not price data.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .. import ledger
from ..contracts import RiskType


class ReportError(ValueError):
    """The tear sheet cannot be built honestly from what is in the ledger / trial store."""


def created_row(study_id: str, ledger_dir: Path | None) -> dict[str, Any]:
    row = ledger.created_row(study_id, ledger_dir=ledger_dir)
    if row is None:
        raise ReportError(f"unknown study {study_id!r}: no study_created row in the ledger "
                          f"(ledger_dir={ledger_dir}); the tear sheet only reports on studies "
                          f"the ledger knows about")
    return row


def selection_event(study_id: str, ledger_dir: Path | None) -> dict[str, Any] | None:
    ev = ledger.study_events(study_id, "selection", ledger_dir=ledger_dir)
    return ev[-1] if ev else None


def latest_gates_event(study_id: str, ledger_dir: Path | None) -> dict[str, Any] | None:
    ev = ledger.study_events(study_id, "gates", ledger_dir=ledger_dir)
    return ev[-1] if ev else None


def resolve_studies_dir(row: dict[str, Any], studies_dir: Path | None) -> Path | None:
    if studies_dir is not None:
        return Path(studies_dir)
    sd = row.get("studies_dir")
    return Path(sd) if sd else None


def selected_trial_id(sel_event: dict[str, Any] | None) -> int | None:
    if not sel_event:
        return None
    sel = sel_event.get("selection") or {}
    tid = sel.get("trial_id", sel.get("selected_trial"))
    return None if tid is None else int(tid)


def load_trial_returns_matrix(study_id: str, studies_dir: Path | None) -> pl.DataFrame:
    """Wide ``date`` + ``t<trial_id>`` matrix from the trial store; empty frame if none."""
    return ledger.load_trial_returns(study_id, studies_dir=studies_dir)


def selected_daily(study_id: str, trial_id: int | None, studies_dir: Path | None) -> pl.DataFrame | None:
    """The selected configuration's daily returns, from the trial store (never from a fresh
    evaluator call, so this is exactly the series the gates/DSR used).  None when the trial
    store has no returns, or no column for ``trial_id``."""
    if trial_id is None:
        return None
    wide = load_trial_returns_matrix(study_id, studies_dir)
    col = f"t{trial_id}"
    if wide.is_empty() or col not in wide.columns:
        return None
    return (wide.select("date", pl.col(col).alias("ret")).drop_nulls("ret")
            .sort("date").with_columns(pl.col("date").cast(pl.Date)))


def resolve_trades(evaluator: Any, selected_params: dict[str, Any] | None) -> pl.DataFrame | None:
    """Trades of the selected configuration from ``evaluator(selected_params)`` (DESIGN task:
    "for trades, from evaluator(selected_params) when an evaluator is given").  None without an
    evaluator, or when the call raises (reported as unavailable, never crashes the tear sheet)."""
    if evaluator is None or not selected_params:
        return None
    try:
        outcome = evaluator(dict(selected_params))
    except Exception:  # noqa: BLE001 -- trade-level metrics degrade to "not available"
        return None
    return getattr(outcome, "trades", None)


def resolve_risk_type(risk_type: Any, evaluator: Any, row: dict[str, Any]) -> RiskType:
    """``risk_type=`` argument > the evaluator's strategy class > the ledger's evaluator
    description > raise (DESIGN task order; the report never guesses)."""
    if risk_type is not None:
        return RiskType(risk_type)
    strategy_cls = getattr(evaluator, "strategy_cls", None)
    rt = getattr(strategy_cls, "risk_type", None)
    if rt is not None:
        return RiskType(rt)
    ev_desc = row.get("evaluator") or {}
    if isinstance(ev_desc, dict) and ev_desc.get("risk_type") is not None:
        return RiskType(ev_desc["risk_type"])
    raise ReportError(f"cannot resolve risk_type for {row.get('study_id')!r}: pass risk_type=, or an "
                      f"evaluator whose strategy_cls declares one, or log 'risk_type' in the ledger's "
                      f"evaluator description (DESIGN §5)")


def holdout_info(row: dict[str, Any], ledger_dir: Path | None, override: Any,
                 study_id: str) -> dict[str, Any]:
    """Holdout status for the card/tear sheet, from already-logged ledger metadata only (never
    from fresh holdout bars).  ``override`` (a dict with at least a ``status`` key, e.g. the
    return of ``stats.holdout_check`` or a raw ``holdout_exam`` row) wins when given."""
    if isinstance(override, dict):
        status = override.get("status")
        return {"status": status, "unlocked": True, "detail": override}
    if isinstance(override, str):
        return {"status": override, "unlocked": True, "detail": {}}
    book, system = row.get("book"), row.get("system")
    if not book or not system:
        return {"status": None, "unlocked": False, "detail": {}}
    state = ledger.holdout_state(book, system, ledger_dir, study_id=study_id)
    if state["n_exams"] == 0:
        return {"status": None, "unlocked": state["n_unlocks"] > 0, "detail": state}
    return {"status": state["last_status"], "unlocked": True, "detail": state}


def pbo_diagnostic(study_id: str, studies_dir: Path | None, n_splits: int = 16) -> dict[str, Any]:
    """PBO (probability of backtest overfitting) as a **diagnostic only** (DESIGN v1.2: the gate
    is ``cscv_oos_loss``, not PBO) -- computed with the exact same ``stats.pbo_cscv`` the
    statistician uses, from the trial store's full return matrix.  Never a gate value: no
    ledger event carries it, so there is nothing to read back and compare."""
    from .. import stats as st
    wide = load_trial_returns_matrix(study_id, studies_dir)
    if wide.is_empty() or wide.width < 3:
        return {"pbo": float("nan"), "available": False,
                "why": "trial store has fewer than 2 trial columns (or is missing)"}
    try:
        res = st.pbo_cscv(wide, n_splits=n_splits)
    except Exception as e:  # noqa: BLE001
        return {"pbo": float("nan"), "available": False, "why": f"pbo_cscv failed: {e!r}"}
    return {"pbo": res.pbo, "prob_oos_loss": res.prob_oos_loss, "available": True}


def load_artifact(study_id: str, name: str, studies_dir: Path | None) -> pl.DataFrame | None:
    df = ledger.load_study_artifact(study_id, name, studies_dir=studies_dir)
    return None if df is None or df.is_empty() else df


def cost_sensitivity_curve(evaluator: Any, selected_params: dict[str, Any] | None, ppy: float,
                          mults: tuple[float, ...] = (1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    """Annualised Sharpe of the selected configuration at increasing spread multipliers
    (display-only diagnostic, never the cost-stress *gate* -- that single worst-case number is
    read from the ledger).  None when there is no evaluator, no cost model on it, or every
    call fails."""
    from dataclasses import replace
    from .. import metrics as m
    if evaluator is None or not selected_params:
        return None
    base = getattr(evaluator, "cost", None)
    if base is None or not hasattr(base, "stressed"):
        return None
    xs, ys = [], []
    for mult in mults:
        try:
            cm = replace(base, spread_multiplier=base.spread_multiplier * mult)
            outcome = evaluator(dict(selected_params), cost=cm)
            xs.append(mult)
            ys.append(m.sharpe(outcome.daily, ppy))
        except Exception:  # noqa: BLE001 -- a stress point that errors is just dropped
            continue
    if not xs:
        return None
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def wfo_recent_sharpe(study_id: str, studies_dir: Path | None, ppy: float) -> dict[str, Any]:
    """Recent-third annualised Sharpe of the stored WFO OOS series (display-only; the *gate*
    value read from the ledger is the whole-span Sharpe — this reproduces the same
    ``gates.WFO_RECENT_FRACTION`` window the gate used, from the same stored series, so it is
    never a second, disagreeing measurement of the gate itself)."""
    from .. import metrics as m
    from ..gates import WFO_RECENT_FRACTION
    wfo = load_artifact(study_id, "wfo_oos", studies_dir)
    if wfo is None or "ret" not in wfo.columns or wfo.height == 0:
        return {"available": False}
    w = wfo.sort("date")
    r = w["ret"].cast(pl.Float64).fill_null(0.0).to_numpy()
    n_rec = max(2, int(math.ceil(r.size * WFO_RECENT_FRACTION)))
    return {"available": True, "sharpe_recent": m.sharpe(r[-n_rec:], ppy), "n_recent_days": n_rec}
