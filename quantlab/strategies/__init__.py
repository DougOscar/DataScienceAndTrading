"""Home for system strategy code, one module per system (DESIGN §1, §3 S2).

``research/systems/<book>/<issue#>_<slug>/hypothesis.md`` names a ``slug``; the matching
strategy implementation lives at ``quantlab/strategies/<slug>.py`` and is written by the
strategy-engineer at S2, never by this package. This package is only the lookup
(:func:`get_strategy`) plus, in :mod:`quantlab.strategies.base`, the handful of indicator
helpers already duplicated across strategy modules.

Every module under here must pass ``quantlab.testing.assert_strategy_source_clean`` (no data
I/O, no import-time work) -- see that function's docstring and ``contracts.py`` for the
``Strategy`` protocol these modules implement.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["get_strategy"]


def _looks_like_strategy(obj: Any, *, module: Any = None) -> bool:
    """Best-effort structural check for :class:`quantlab.contracts.Strategy`: a class, defined
    in ``module`` when one is given (so an imported helper class doesn't get picked up as a
    stray second candidate), exposing ``signals`` + ``risk_type`` + ``name``."""
    if not isinstance(obj, type):
        return False
    if module is not None and obj.__module__ != module.__name__:
        return False
    return hasattr(obj, "signals") and hasattr(obj, "risk_type") and hasattr(obj, "name")


def get_strategy(slug: str) -> type:
    """Import ``quantlab.strategies.<slug>`` and return its :class:`contracts.Strategy` class.

    Resolution order inside the module:

    1. an explicit ``STRATEGY = <cls>`` module attribute;
    2. an attribute literally named ``Strategy``;
    3. the single class defined *in that module* (not merely imported into it) that looks like
       a strategy (``signals`` + ``risk_type`` + ``name``).

    Raises ``ValueError`` if the module doesn't exist, or if none / more than one class
    matches unambiguously (add ``STRATEGY = <cls>`` to disambiguate).
    """
    mod_name = f"{__name__}.{slug}"
    try:
        module = importlib.import_module(mod_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"get_strategy({slug!r}): no module {mod_name!r} (expected quantlab/strategies/{slug}.py, "
            "written by the strategy-engineer at S2)"
        ) from exc

    cls = getattr(module, "STRATEGY", None)
    if cls is None:
        named = getattr(module, "Strategy", None)
        cls = named if isinstance(named, type) else None
    if cls is None:
        candidates = [obj for obj in vars(module).values() if _looks_like_strategy(obj, module=module)]
        if len(candidates) == 1:
            cls = candidates[0]
        elif not candidates:
            raise ValueError(
                f"{mod_name}: no Strategy class found (define a class named `Strategy`, or set "
                "`STRATEGY = <cls>`)"
            )
        else:
            raise ValueError(
                f"{mod_name}: {len(candidates)} candidate Strategy classes {candidates}; "
                "disambiguate with `STRATEGY = <cls>`"
            )
    if not _looks_like_strategy(cls):
        raise ValueError(
            f"{mod_name}: {cls!r} does not look like a quantlab Strategy (needs `signals`, "
            "`risk_type`, `name`)"
        )
    return cls
