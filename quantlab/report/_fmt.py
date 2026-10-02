"""Shared number -> text helpers, so a missing or non-finite value never leaks into the tear
sheet / card as a literal Python repr ("nan", "None", "-0%") instead of something a reader can
act on (dry-run #23 findings #61, #62, #63, #67).

``is_missing`` -- and therefore :func:`dash` / :func:`fmt_sig` -- treats only ``None`` and NaN as
"no value". +-inf is a real (if extreme) result, not a missing one: it gets its own plain-language
translation at the call site (finding #52, e.g. profit factor with no losing trades, MinTRL at a
non-positive Sharpe) rather than being swallowed into a dash.
"""

from __future__ import annotations

import math
from typing import Any

DASH = "—"  # em dash: "no value" everywhere in the tear sheet / card (#62)


def is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def is_inf(v: Any) -> bool:
    return isinstance(v, float) and math.isinf(v)


def dash(v: Any, spec: str = "") -> str:
    """``format(v, spec)``, or an em dash when ``v`` is None/NaN (#62). ``spec=""`` just
    ``str()``s a non-missing value, so this also works for text fields (e.g. a gate status)."""
    if is_missing(v):
        return DASH
    return format(v, spec) if spec else str(v)


def fmt_metric_value(v: Any) -> str:
    """One cell of the tear sheet's Metrics table: dash for None/NaN, unicode infinity for
    +-inf (never the literal "inf" -- #52), ``%.4g`` for any other number, ``str()`` otherwise
    (e.g. the ``gate_verdict`` string metric)."""
    if is_missing(v):
        return DASH
    if is_inf(v):
        return "∞" if v > 0 else "-∞"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.4g}"
    return str(v)


def gate_threshold_text(thr: tuple[str, Any] | None) -> str:
    """``(op, value)`` -> ``"op value"``, or just ``"op"`` when the gate has no numeric
    threshold to print (e.g. ``mechanism``'s ``("manual", None)`` -- printing that tuple
    verbatim used to render the literal word "None")."""
    if not thr:
        return ""
    op, val = thr
    return op if val is None else f"{op} {val}"


def sig_round(x: float, sig: int = 2) -> float:
    """Round ``x`` to ``sig`` significant figures (never a fixed number of decimals, so a small
    number like -0.0148 doesn't collapse to "-0.0" at ``.1f`` -- #67)."""
    if not math.isfinite(x) or x == 0.0:
        return 0.0
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


def fmt_sig(x: float | None, sig: int = 2, suffix: str = "") -> str:
    """``x`` to ``sig`` significant figures as a fixed-point string (never scientific
    notation), or an em dash when missing (#62, #67). +-0 always prints as plain "0" (no
    signed zero -- #61)."""
    if is_missing(x):
        return DASH
    if is_inf(x):
        return ("∞" if x > 0 else "-∞") + suffix
    r = sig_round(float(x), sig)
    if r == 0.0:
        return f"0{suffix}"
    d = max(0, sig - int(math.floor(math.log10(abs(r)))) - 1)
    return f"{r:.{d}f}{suffix}"
