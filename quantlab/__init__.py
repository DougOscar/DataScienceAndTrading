"""quantlab — research library for the pro research team (see research/DESIGN.md).

Modules: config · contracts · data · ledger · costs · engine · sizing · testing · metrics ·
stats · gates · opt · evaluators (portfolio and report arrive in later phases).
"""

from .config import BOOKS, get_book
from .contracts import HoldoutLocked, Params, RiskType, Strategy

__all__ = ["BOOKS", "get_book", "HoldoutLocked", "Params", "RiskType", "Strategy"]
