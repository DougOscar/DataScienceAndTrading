#!/usr/bin/env python3
"""Position size calculator using multiple sizing methods.

Calculates position size using fixed fractional, volatility-adjusted,
Kelly criterion, and MT5 lot-constrained rounding. Shows the binding
constraint and provides a formatted report.

Usage:
    python scripts/size_calculator.py

    Or with environment variables:
    ACCOUNT_SIZE=10000 ENTRY_PRICE=1.0850 STOP_LOSS=1.0820 python scripts/size_calculator.py

Dependencies:
    None (pure math, no external packages required)

Environment Variables:
    ACCOUNT_SIZE:    Account value in USD (default: 10000)
    ENTRY_PRICE:     Entry price per unit (default: 1.0850, e.g. EURUSD)
    STOP_LOSS:       Stop loss price per unit (default: 1.0820, a 30-pip stop)
    WIN_RATE:        Historical win rate 0-1 (default: 0.55)
    AVG_WIN_RATIO:   Average win / average loss ratio (default: 1.8)
    UNITS_PER_LOT:   Contract size — units per 1.0 lot (default: 100000, a standard FX lot)
    VOLUME_MIN:      Smallest tradable lot (default: 0.01)
    VOLUME_STEP:     Lot increment (default: 0.01)
    LEVERAGE:        Margin leverage, e.g. 30 = 30:1 (default: 30)
    ATR_VALUE:       14-period ATR in price units (default: 0.0045)
    TARGET_VOL_PCT:  Target daily portfolio vol as decimal (default: 0.02)
    SPREAD_RATE:     One-way spread cost as decimal of price (default: 0.00009, ~2 pips on EURUSD)
    SLIPPAGE_EST:    Estimated slippage as decimal (default: 0.00005)
"""

import os
import sys
from typing import Optional


# ── Configuration ───────────────────────────────────────────────────

def get_float_env(name: str, default: float) -> float:
    """Read a float from an environment variable with a default."""
    val = os.getenv(name, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        print(f"Warning: {name}='{val}' is not a valid number, using default {default}")
        return default


ACCOUNT_SIZE = get_float_env("ACCOUNT_SIZE", 10_000.0)
ENTRY_PRICE = get_float_env("ENTRY_PRICE", 1.0850)
STOP_LOSS = get_float_env("STOP_LOSS", 1.0820)
WIN_RATE = get_float_env("WIN_RATE", 0.55)
AVG_WIN_RATIO = get_float_env("AVG_WIN_RATIO", 1.8)
UNITS_PER_LOT = get_float_env("UNITS_PER_LOT", 100_000.0)
VOLUME_MIN = get_float_env("VOLUME_MIN", 0.01)
VOLUME_STEP = get_float_env("VOLUME_STEP", 0.01)
LEVERAGE = get_float_env("LEVERAGE", 30.0)
ATR_VALUE = get_float_env("ATR_VALUE", 0.0045)
TARGET_VOL_PCT = get_float_env("TARGET_VOL_PCT", 0.02)
SPREAD_RATE = get_float_env("SPREAD_RATE", 0.00009)
SLIPPAGE_EST = get_float_env("SLIPPAGE_EST", 0.00005)


# ── Fixed Fractional Sizing ────────────────────────────────────────

def fixed_fractional(
    account: float,
    risk_pct: float,
    entry: float,
    stop: float,
    spread_rate: float = 0.0,
    slippage: float = 0.0,
) -> dict:
    """Calculate position size using fixed fractional method.

    Args:
        account: Total account value.
        risk_pct: Fraction of account to risk (e.g., 0.02 for 2%).
        entry: Entry price per unit.
        stop: Stop loss price per unit.
        spread_rate: One-way spread cost as a fraction of price (applied to
            entry and exit; for MT5 this approximates the `spread` column —
            use the actual points x point value for a precise figure).
        slippage: Estimated slippage as fraction of entry price.

    Returns:
        Dictionary with units, value, risk_amount, and effective_risk.
    """
    risk_amount = account * risk_pct
    price_risk = abs(entry - stop)
    spread_cost = entry * spread_rate * 2  # round-trip spread
    slippage_cost = entry * slippage
    effective_risk = price_risk + spread_cost + slippage_cost

    if effective_risk <= 0:
        return {"units": 0.0, "value": 0.0, "risk_amount": risk_amount, "effective_risk": 0.0}

    units = risk_amount / effective_risk
    value = units * entry

    return {
        "units": units,
        "value": value,
        "risk_amount": risk_amount,
        "effective_risk": effective_risk,
        "price_risk": price_risk,
        "spread_cost": spread_cost,
        "slippage_cost": slippage_cost,
    }


# ── Volatility-Adjusted Sizing ─────────────────────────────────────

def volatility_adjusted(
    account: float,
    target_vol_pct: float,
    atr: float,
    entry: float,
) -> dict:
    """Calculate position size scaled by volatility.

    Args:
        account: Total account value.
        target_vol_pct: Target daily portfolio volatility as decimal.
        atr: Average True Range (14-period) in price units.
        entry: Current price per unit.

    Returns:
        Dictionary with units, value, and volatility metrics.
    """
    if atr <= 0:
        return {"units": 0.0, "value": 0.0, "daily_vol_pct": 0.0, "expected_daily_pnl": 0.0}

    target_daily_pnl = account * target_vol_pct
    daily_vol_pct = atr / entry
    units = target_daily_pnl / atr
    value = units * entry
    expected_daily_pnl = units * atr

    return {
        "units": units,
        "value": value,
        "daily_vol_pct": daily_vol_pct,
        "expected_daily_pnl": expected_daily_pnl,
        "target_daily_pnl": target_daily_pnl,
    }


# ── Kelly Criterion Sizing ─────────────────────────────────────────

def kelly_criterion(
    account: float,
    win_rate: float,
    payoff_ratio: float,
    entry: float,
    stop: float,
    fraction: float = 0.25,
) -> dict:
    """Calculate position size using Kelly criterion.

    Args:
        account: Total account value.
        win_rate: Probability of winning trade (0-1).
        payoff_ratio: Average win / average loss.
        entry: Entry price per unit.
        stop: Stop loss price per unit.
        fraction: Kelly fraction to use (0.25 recommended).

    Returns:
        Dictionary with full Kelly, fractional Kelly, units, and value.
    """
    q = 1.0 - win_rate
    if payoff_ratio <= 0:
        return {
            "full_kelly": 0.0, "fractional_kelly": 0.0, "fraction_used": fraction,
            "units": 0.0, "value": 0.0, "has_edge": False,
        }

    full_kelly = (win_rate * payoff_ratio - q) / payoff_ratio
    has_edge = full_kelly > 0
    fractional_kelly = max(0.0, full_kelly * fraction)

    price_risk = abs(entry - stop)
    if price_risk <= 0 or not has_edge:
        units = 0.0
    else:
        risk_amount = account * fractional_kelly
        units = risk_amount / price_risk

    value = units * entry

    return {
        "full_kelly": full_kelly,
        "fractional_kelly": fractional_kelly,
        "fraction_used": fraction,
        "units": units,
        "value": value,
        "has_edge": has_edge,
        "expected_growth_fraction": fractional_kelly * (2 - fractional_kelly / max(full_kelly, 1e-9)),
    }


# ── MT5 Lot-Constrained Rounding ───────────────────────────────────

def mt5_lot_constrained(
    raw_units: float,
    entry: float,
    stop: float,
    units_per_lot: float,
    volume_min: float,
    volume_step: float,
) -> dict:
    """Round a raw unit-based position size to the broker's MT5 lot grid.

    Args:
        raw_units: Unrounded position size in instrument units.
        entry: Entry price per unit.
        stop: Stop loss price per unit.
        units_per_lot: Contract size — units per 1.0 lot.
        volume_min: Smallest tradable lot.
        volume_step: Increment between allowed lot sizes.

    Returns:
        Dictionary with lots, rounded_units, rounded_value, and actual_risk.
    """
    if units_per_lot <= 0 or volume_step <= 0 or raw_units <= 0:
        return {"lots": 0.0, "rounded_units": 0.0, "rounded_value": 0.0, "actual_risk": 0.0}

    raw_lots = raw_units / units_per_lot
    step_count = round(raw_lots / volume_step)
    lots = max(volume_min, step_count * volume_step)
    rounded_units = lots * units_per_lot
    price_risk = abs(entry - stop)

    return {
        "lots": lots,
        "raw_lots": raw_lots,
        "rounded_units": rounded_units,
        "rounded_value": rounded_units * entry,
        "actual_risk": rounded_units * price_risk,
    }


# ── R:R Targets ────────────────────────────────────────────────────

def calculate_rr_targets(
    entry: float,
    stop: float,
    units: float,
) -> list:
    """Calculate reward-to-risk targets for a given position.

    Args:
        entry: Entry price.
        stop: Stop loss price.
        units: Number of units in position.

    Returns:
        List of dicts with R multiple, target price, and P&L.
    """
    risk_per_unit = abs(entry - stop)
    direction = 1 if entry > stop else -1  # long if stop below entry
    targets = []

    for r_multiple in [1.0, 1.5, 2.0, 3.0, 5.0]:
        target_price = entry + direction * risk_per_unit * r_multiple
        pnl = units * direction * (target_price - entry)
        targets.append({
            "r_multiple": r_multiple,
            "target_price": round(target_price, 6),
            "pnl": round(pnl, 2),
        })

    return targets


# ── Report Formatting ──────────────────────────────────────────────

def format_number(val: float, decimals: int = 2) -> str:
    """Format a number with commas and specified decimals."""
    return f"{val:,.{decimals}f}"


def print_header(title: str) -> None:
    """Print a section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_row(label: str, value: str, width: int = 40) -> None:
    """Print a label-value row."""
    print(f"  {label:<{width}} {value}")


def print_report(
    account: float,
    entry: float,
    stop: float,
    win_rate: float,
    payoff_ratio: float,
    units_per_lot: float,
    volume_min: float,
    volume_step: float,
    atr: float,
    target_vol_pct: float,
    spread_rate: float,
    slippage_est: float,
    leverage: float,
) -> None:
    """Print the complete position sizing report.

    Args:
        account: Account value in USD.
        entry: Entry price per unit.
        stop: Stop loss price per unit.
        win_rate: Historical win rate (0-1).
        payoff_ratio: Avg win / avg loss.
        units_per_lot: Contract size — units per 1.0 lot.
        volume_min: Smallest tradable lot.
        volume_step: Lot increment.
        atr: 14-period ATR in price units.
        target_vol_pct: Target daily vol as decimal.
        spread_rate: One-way spread cost as a fraction of price.
        slippage_est: Estimated slippage fraction.
        leverage: Margin leverage (e.g. 30 = 30:1) used to convert
            notional position value into margin required.
    """
    # ── Input Summary ───────────────────────────────────────────
    print_header("POSITION SIZE CALCULATOR")
    print_row("Account Size", f"${format_number(account)}")
    print_row("Entry Price", f"${format_number(entry, 6)}")
    print_row("Stop Loss", f"${format_number(stop, 6)}")
    print_row("Price Risk", f"${format_number(abs(entry - stop), 6)} ({abs(entry - stop) / entry * 100:.2f}%)")
    print_row("Win Rate", f"{win_rate * 100:.1f}%")
    print_row("Payoff Ratio", f"{payoff_ratio:.2f}x")
    print_row("Contract Size (units/lot)", f"{format_number(units_per_lot, 0)}")
    print_row("Lot Min / Step", f"{volume_min:.2f} / {volume_step:.2f}")
    print_row("Leverage", f"{leverage:.0f}:1")
    print_row("ATR(14)", f"${format_number(atr, 6)}")
    print_row("Spread Rate (one-way)", f"{spread_rate * 100:.4f}%")
    print_row("Slippage Estimate", f"{slippage_est * 100:.4f}%")

    # ── Fixed Fractional ────────────────────────────────────────
    print_header("METHOD 1: FIXED FRACTIONAL")
    results_ff = {}
    for risk_pct in [0.01, 0.02, 0.03]:
        ff = fixed_fractional(account, risk_pct, entry, stop, spread_rate, slippage_est)
        results_ff[risk_pct] = ff
        label = f"{risk_pct * 100:.0f}% risk"
        print_row(
            label,
            f"{format_number(ff['units'])} units | ${format_number(ff['value'])} value | ${format_number(ff['risk_amount'])} at risk",
        )
    ff_default = results_ff[0.02]
    print(f"\n  Cost-adjusted risk per unit: ${format_number(ff_default['effective_risk'], 6)}")
    print(f"  (Price risk ${format_number(ff_default['price_risk'], 6)} + spread ${format_number(ff_default['spread_cost'], 6)} + slippage ${format_number(ff_default['slippage_cost'], 6)})")

    # ── Volatility-Adjusted ─────────────────────────────────────
    print_header("METHOD 2: VOLATILITY-ADJUSTED")
    va = volatility_adjusted(account, target_vol_pct, atr, entry)
    print_row("Daily Vol (ATR/Price)", f"{va['daily_vol_pct'] * 100:.1f}%")
    print_row("Target Daily PnL", f"${format_number(va['target_daily_pnl'])}")
    print_row("Position Size", f"{format_number(va['units'])} units")
    print_row("Position Value", f"${format_number(va['value'])}")
    print_row("Expected Daily PnL Range", f"+/- ${format_number(va['expected_daily_pnl'])}")

    # ── Kelly Criterion ─────────────────────────────────────────
    print_header("METHOD 3: KELLY CRITERION")
    kelly_results = {}
    for frac, label in [(1.0, "Full Kelly"), (0.5, "Half Kelly"), (0.25, "Quarter Kelly")]:
        kc = kelly_criterion(account, win_rate, payoff_ratio, entry, stop, frac)
        kelly_results[frac] = kc
        edge_str = "" if kc["has_edge"] else " [NO EDGE]"
        print_row(
            f"{label} ({frac:.0%})",
            f"{kc['fractional_kelly'] * 100:.1f}% risk | {format_number(kc['units'])} units | ${format_number(kc['value'])}{edge_str}",
        )
    full_kc = kelly_results[1.0]
    print(f"\n  Full Kelly fraction: {full_kc['full_kelly'] * 100:.2f}%")
    if not full_kc["has_edge"]:
        print("  *** NEGATIVE KELLY: No statistical edge detected. Do not trade. ***")
    else:
        print("  Recommendation: Use Quarter Kelly (0.25x) as default")

    # ── Combined Recommendation ─────────────────────────────────
    print_header("RECOMMENDATION (BINDING CONSTRAINT)")

    candidates = {
        "Fixed Fractional (2%)": ff_default["units"],
        "Volatility-Adjusted": va["units"],
        "Quarter Kelly": kelly_results[0.25]["units"],
    }

    # Filter out zero/negative
    valid = {k: v for k, v in candidates.items() if v > 0}
    if not valid:
        print("  No valid position size found. Check inputs.")
        return

    binding_method = min(valid, key=valid.get)
    recommended_units = valid[binding_method]
    recommended_value = recommended_units * entry
    margin_pct_of_account = (recommended_value / leverage / account) * 100
    target_risk = account * 0.02  # matches the 2%-risk candidate above

    print_row("Binding Constraint", binding_method)
    print_row("Raw Size (pre-lot-rounding)", f"{format_number(recommended_units)} units")
    print_row("Notional Value", f"${format_number(recommended_value)}")
    print_row(f"Margin Required (at {leverage:.0f}:1)", f"${format_number(recommended_value / leverage)} ({margin_pct_of_account:.1f}% of account)")

    print("\n  All methods compared:")
    for method, units in sorted(candidates.items(), key=lambda x: x[1]):
        marker = " <-- BINDING" if method == binding_method else ""
        val = units * entry
        print(f"    {method:<30} {format_number(units):>12} units  ${format_number(val):>12}{marker}")

    # ── MT5 Lot-Constrained Rounding ─────────────────────────────
    print_header("METHOD 4: MT5 LOT-CONSTRAINED ROUNDING")
    for step in sorted({volume_step, 0.10}):
        lc = mt5_lot_constrained(recommended_units, entry, stop, units_per_lot, volume_min, step)
        risk_error_pct = (
            (lc["actual_risk"] - target_risk) / target_risk * 100 if target_risk > 0 else 0.0
        )
        print_row(
            f"volume_step={step:.2f}",
            f"{lc['lots']:.2f} lots | ${format_number(lc['actual_risk'])} actual risk "
            f"({risk_error_pct:+.1f}% vs ${format_number(target_risk)} target)",
        )
    final_lot = mt5_lot_constrained(recommended_units, entry, stop, units_per_lot, volume_min, volume_step)
    print(f"\n  Tradable size at volume_step={volume_step:.2f}: {final_lot['lots']:.2f} lots "
          f"({format_number(final_lot['rounded_units'])} units)")

    # Check portfolio limits against margin used (not notional), since FX/metals
    # positions are leveraged — notional value is routinely a large multiple of
    # account equity even for a conservatively risk-managed trade.
    final_margin_pct = (final_lot["rounded_value"] / leverage / account) * 100
    print("\n  Portfolio limit checks (post lot-rounding, margin-based):")
    single_ok = final_margin_pct <= 10
    print(f"    Single position margin < 10%: {'PASS' if single_ok else 'FAIL'} ({final_margin_pct:.1f}%)")

    # ── R:R Targets ─────────────────────────────────────────────
    print_header("R:R TARGETS (at tradable lot-rounded size)")
    risk_per_unit = abs(entry - stop)
    recommended_units = final_lot["rounded_units"]
    risk_total = recommended_units * risk_per_unit
    targets = calculate_rr_targets(entry, stop, recommended_units)

    print_row("Risk per trade", f"${format_number(risk_total)}")
    print()
    print(f"  {'R:R':<8} {'Target Price':<16} {'P&L':<16} {'% of Account'}")
    print(f"  {'-' * 56}")
    for t in targets:
        pct = (t["pnl"] / account) * 100
        print(f"  {t['r_multiple']:<8.1f} ${format_number(t['target_price'], 6):<14} ${format_number(t['pnl']):<14} {pct:+.2f}%")


# ── Validation ──────────────────────────────────────────────────────

def validate_inputs(
    account: float,
    entry: float,
    stop: float,
    win_rate: float,
    payoff_ratio: float,
    units_per_lot: float,
    volume_min: float,
    volume_step: float,
) -> list:
    """Validate inputs and return list of warning messages.

    Args:
        account: Account size.
        entry: Entry price.
        stop: Stop loss price.
        win_rate: Win rate (0-1).
        payoff_ratio: Avg win / avg loss.
        units_per_lot: Contract size.
        volume_min: Smallest tradable lot.
        volume_step: Lot increment.

    Returns:
        List of warning/error strings. Empty list means all OK.
    """
    errors: list = []
    if account <= 0:
        errors.append("Account size must be positive")
    if entry <= 0:
        errors.append("Entry price must be positive")
    if stop <= 0:
        errors.append("Stop loss must be positive")
    if entry == stop:
        errors.append("Entry and stop loss cannot be the same price")
    if not 0 < win_rate < 1:
        errors.append(f"Win rate must be between 0 and 1, got {win_rate}")
    if payoff_ratio <= 0:
        errors.append("Payoff ratio must be positive")
    if units_per_lot <= 0:
        errors.append("Contract size (units per lot) must be positive")
    if volume_min <= 0 or volume_step <= 0:
        errors.append("Lot minimum and step must be positive")

    # Warnings (non-fatal)
    risk_pct = abs(entry - stop) / entry * 100
    if risk_pct > 30:
        errors.append(f"Warning: Stop distance is {risk_pct:.1f}% from entry (very wide)")

    return errors


# ── Main ────────────────────────────────────────────────────────────

def main() -> None:
    """Run the position size calculator with configured parameters."""
    issues = validate_inputs(
        ACCOUNT_SIZE, ENTRY_PRICE, STOP_LOSS,
        WIN_RATE, AVG_WIN_RATIO, UNITS_PER_LOT, VOLUME_MIN, VOLUME_STEP,
    )

    fatal = [i for i in issues if not i.startswith("Warning")]
    warnings = [i for i in issues if i.startswith("Warning")]

    if fatal:
        print("Input errors:")
        for e in fatal:
            print(f"  - {e}")
        sys.exit(1)

    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  - {w}")

    print_report(
        account=ACCOUNT_SIZE,
        entry=ENTRY_PRICE,
        stop=STOP_LOSS,
        win_rate=WIN_RATE,
        payoff_ratio=AVG_WIN_RATIO,
        units_per_lot=UNITS_PER_LOT,
        volume_min=VOLUME_MIN,
        volume_step=VOLUME_STEP,
        atr=ATR_VALUE,
        target_vol_pct=TARGET_VOL_PCT,
        spread_rate=SPREAD_RATE,
        slippage_est=SLIPPAGE_EST,
        leverage=LEVERAGE,
    )


if __name__ == "__main__":
    main()
