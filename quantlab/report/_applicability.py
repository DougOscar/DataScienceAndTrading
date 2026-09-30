"""DESIGN §5 rule-aware metric applicability: which trade-level metrics apply to which risk
type, and the one-line reason when a metric is excluded or reinterpreted for this system.

Units note baked into every trade-level row: points/pips are symbol-specific (not comparable
across symbols); the engine does not tag a trade with the ATR at entry, so the ATR-unit
cross-symbol view DESIGN §5 asks for is never available yet -- every trade-level metric below
carries that same exclusion alongside its risk-type-specific one.
"""

from __future__ import annotations

from ..contracts import RiskType

ATR_NOTE = "not produced by engine yet: trades carry no ATR-at-entry column, so the ATR-unit view is unavailable"


def not_relevant_reason(risk_type: RiskType, metric: str) -> str:
    reasons = {
        "expectancy_r": {
            RiskType.B: "fixed lot size, variable stop: R is not a constant unit across trades",
            RiskType.C: "no hard stop: there is no risk unit to express a multiple of",
            RiskType.D: "continuous position, no discrete per-trade risk unit",
        },
        "max_losing_streak_r": {
            RiskType.B: "no constant R unit at fixed lot size",
            RiskType.C: "no hard stop to define R",
            RiskType.D: "no discrete trades",
        },
        "risk_realisation_error": {
            RiskType.B: "sizing is fixed-lot: there is no risk target to compare a realisation against",
            RiskType.C: "no hard stop: no risk target",
            RiskType.D: "continuous sizing, not lot/stop based",
        },
        "raw_points_per_trade": {
            RiskType.A: "lot size varies with the stop, so raw points are not comparable across trades "
                        "(use R-multiples)",
        },
        "pnl_ccy_distribution": {
            RiskType.A: "shown as R-multiples instead (risk varies by design; currency P&L mixes that in)",
        },
        "risk_per_trade_distribution": {},
        "mae_mfe": {},
        "worst_trade": {},
        "holding_time": {},
        "win_rate_profit_factor": {
            RiskType.D: "not relevant to a continuous / vol-targeted position (DESIGN §5)",
        },
        "turnover_exposure_cost_per_turnover": {
            RiskType.A: "type A is a discrete-position system: use R-multiples / trade stats instead",
            RiskType.B: "type B is a discrete-position system: use P&L / risk-per-trade instead",
            RiskType.C: "type C is a discrete-position system: use MAE/MFE / holding time instead",
        },
    }
    return reasons.get(metric, {}).get(risk_type, "")
