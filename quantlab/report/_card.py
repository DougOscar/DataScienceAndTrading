"""Risk-profile labelling (DESIGN §6) and the Obsidian card writer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .. import metrics as m

MAX_IDEA_CHARS = 200


# --------------------------------------------------------------------------- risk-profile label
def _losing_streak_by_sign(x: np.ndarray) -> int:
    best = cur = 0
    for v in x < 0:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _block_concentration(r: np.ndarray, block: int = 63) -> float:
    """Share of total (additive) return carried by the single richest ``block``-length
    contiguous stretch (a calendar-agnostic stand-in for ``stats.time_stability``'s
    max-year-share, used when ``daily`` has no date column)."""
    if r.size < block:
        return float("nan")
    n_blocks = r.size // block
    sums = r[: n_blocks * block].reshape(n_blocks, block).sum(axis=1)
    total = sums.sum()
    return float(sums.max() / total) if total > 0 else float("nan")


def risk_profile_label(daily: Any, trades: pl.DataFrame | None = None
                       ) -> tuple[str, dict[str, Any]]:
    """DESIGN §6: Steady / Grinder / Trend-like / Lumpy, from shape (never from leverage --
    every system is compared at the same 10% DD budget).

    ``daily``: a return series (array-like, or a ``date``/``ret`` frame -- a frame lets the
    concentration check use real calendar years via ``stats.time_stability``; otherwise it
    falls back to a 63-trading-day block share, labelled as such in the evidence).
    ``trades``: sized trades frame (optional) -- used for the win rate and the losing streak
    (type A: in R-multiples; else by the sign of ``pnl_ccy``/``pnl_points``); without it both
    fall back to a same-sign-day streak / daily up-day rate on ``daily``, labelled as such.

    Rule (documented, not a statistical gate -- purely descriptive):
    1. **Lumpy** if the richest single year (or 63-day block) carries > 55% of total PnL.
    2. else **Trend-like** if skew > +0.15 and the win rate < 50%.
    3. else **Grinder** if skew < -0.15 and the win rate >= 50%.
    4. else **Steady**.
    """
    r = m.as_array(daily)
    if r.size < 2:
        raise ValueError("risk_profile_label needs at least 2 daily returns")
    max_dd = abs(m.max_drawdown(r))
    longest_dd_days = m.longest_drawdown_periods(r)
    skew, _kurt = m.skew_kurt(r)
    cvar95 = m.cvar(r, 0.95)

    if isinstance(daily, pl.DataFrame) and "date" in daily.columns:
        from .. import stats as st
        ts = st.time_stability(daily)
        max_share, share_kind = ts["max_year_share"], "year"
    else:
        max_share, share_kind = _block_concentration(r), "63-day block"

    if trades is not None and trades.height > 0:
        col = "pnl_ccy" if "pnl_ccy" in trades.columns else ("pnl_points" if "pnl_points" in trades.columns else None)
        t = trades.filter(~pl.col("skipped")) if "skipped" in trades.columns else trades
        win_rate = float(np.mean(t[col].to_numpy().astype(float) > 0)) if col and t.height else float("nan")
        r_mult = m.r_multiples(trades)
        if r_mult.size:
            streak, streak_kind = m.max_losing_streak_r(trades), "R-multiple"
        else:
            streak = _losing_streak_by_sign(t[col].to_numpy().astype(float)) if col else 0
            streak_kind = f"{col} sign" if col else "unavailable"
    else:
        win_rate, streak = float(np.mean(r > 0)), _losing_streak_by_sign(r)
        streak_kind = "up/down day (no trade data)"

    if np.isfinite(max_share) and max_share > 0.55:
        label = "Lumpy"
    elif np.isfinite(skew) and skew > 0.15 and np.isfinite(win_rate) and win_rate < 0.50:
        label = "Trend-like"
    elif np.isfinite(skew) and skew < -0.15 and np.isfinite(win_rate) and win_rate >= 0.50:
        label = "Grinder"
    else:
        label = "Steady"

    evidence = {"max_dd": max_dd, "longest_dd_days": longest_dd_days, "skew": skew, "cvar95": cvar95,
               "max_period_share": max_share, "max_period_share_kind": share_kind,
               "win_rate": win_rate, "max_losing_streak": streak, "max_losing_streak_kind": streak_kind}
    return label, evidence


# --------------------------------------------------------------------------- Obsidian card
def _truncate_idea(idea: str) -> str:
    idea = str(idea).strip()
    return idea if len(idea) <= MAX_IDEA_CHARS else idea[: MAX_IDEA_CHARS - 1].rstrip() + "…"


def _status_line(status: str, stage: str | None, reason: str | None) -> str:
    if status == "killed":
        detail = ", ".join(x for x in (stage, reason) if x)
        return f"killed ({detail})" if detail else "killed"
    return status


def write_card(ts: Any, *, slug: str, name: str, idea: str, status: str, issue: int, book: str,
               stage: str | None = None, reason: str | None = None,
               vault_dir: Path | None = None) -> Path:
    """Write the DESIGN §6 Obsidian card (exactly the template) and its two chart attachments.

    Written for **killed** systems too, with ``stage``/``reason`` folded into the status line,
    so a dead idea is recorded instead of silently forgotten (DESIGN §3 / report-builder charter).
    """
    from .. import config
    if status == "killed" and not (stage and reason):
        raise ValueError("write_card(status='killed') needs both stage= and reason=")
    vdir = Path(vault_dir) if vault_dir is not None else getattr(config, "VAULT_DIR",
                                                                 config.ROOT / "DocumentationVault" / "systems")
    att_dir = vdir / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)
    figs = ts.figures(att_dir)
    yearly_src, monthly_src = figs.get("sharpe_yearly"), figs.get("sharpe_monthly")
    yearly_dst, monthly_dst = att_dir / f"{slug}_sharpe_yearly.png", att_dir / f"{slug}_sharpe_monthly.png"
    for src, dst in ((yearly_src, yearly_dst), (monthly_src, monthly_dst)):
        if src is not None and Path(src) != dst:
            dst.write_bytes(Path(src).read_bytes())

    headline = ts.headline
    rp_label, rp = ts.risk_profile
    result_line = (f"{headline.get('class', 'unclassified')} — "
                   f"{headline.get('mean_monthly_pct', float('nan')):.1f}%/month at 10% DD budget")

    def _fmt(v, spec=".2f"):
        return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)

    robust = headline.get("robustness", {})
    longest_dd_months = rp.get("longest_dd_days", float("nan")) / 21.0 if rp else float("nan")
    lines = [
        f"# {name}",
        f"**Book:** {book}   **Status:** {_status_line(status, stage, reason)}   **Issue:** #{issue}",
        f"**Idea:** {_truncate_idea(idea)}",
        f"**Result:** {result_line}",
        f"**Risk profile:** {rp_label} — MaxDD {_fmt(rp.get('max_dd'), '.1%')}, "
        f"longest DD {_fmt(longest_dd_months, '.1f')} months, "
        f"max losing streak {_fmt(rp.get('max_losing_streak'), '.0f')}, "
        f"skew {_fmt(rp.get('skew'))}, CVaR95 {_fmt(rp.get('cvar95'), '.2%')}",
        f"**Robustness:** DSR {_fmt(robust.get('dsr'))} · CSCV OOS loss {_fmt(robust.get('cscv_oos_loss'))} · "
        f"WFO OOS Sharpe {_fmt(robust.get('wfo_oos'))} (recent third {_fmt(robust.get('wfo_oos_recent'))}) · "
        f"holdout {robust.get('holdout_status', 'not unlocked')}",
        f"![[{slug}_sharpe_yearly.png]]",
        f"![[{slug}_sharpe_monthly.png]]",
    ]
    text = "\n".join(lines) + "\n"
    out = vdir / f"{slug}.md"
    vdir.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
