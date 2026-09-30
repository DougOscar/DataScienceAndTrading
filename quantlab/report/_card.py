"""Risk-profile labelling (DESIGN §6) and the Obsidian card writer."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .. import metrics as m
from ._fmt import dash, fmt_sig

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


def _status_line(status: str, stage: str | None, reason: str | None, scope: str | None = None) -> str:
    if status == "killed":
        detail = ", ".join(x for x in (stage, reason) if x)
        if scope:
            detail = f"{detail}, scope: {scope}" if detail else f"scope: {scope}"
        return f"killed ({detail})" if detail else "killed"
    return status


def write_card(ts: Any, *, slug: str, name: str, idea: str, status: str, issue: int, book: str,
               stage: str | None = None, reason: str | None = None, scope: str | None = None,
               vault_dir: Path | None = None) -> Path:
    """Write the DESIGN §6 Obsidian card (exactly the template) and its two chart attachments.

    Written for **killed** systems too, with ``stage``/``reason`` folded into the status line,
    so a dead idea is recorded instead of silently forgotten (DESIGN §3 / report-builder charter).
    ``scope`` (e.g. ``"EURUSD H4"``, #74) records what the kill actually covers -- a red-team
    finding may apply to one symbol/timeframe and not another -- and, when given, is folded into
    that same killed-status line.

    Only the two §6 figures (``sharpe_yearly``, ``sharpe_monthly``) are ever written into the
    shared vault ``attachments/`` folder, and always slug-prefixed (#66): every other figure
    ``ts.figures()`` can build is rendered into a private scratch directory that is discarded
    once the two needed PNGs are copied out, so a second system's card can never overwrite a
    first system's attachments under a generic name (e.g. two systems both writing
    ``plateau_heatmap.png``).
    """
    from .. import config
    if status == "killed" and not (stage and reason):
        raise ValueError("write_card(status='killed') needs both stage= and reason=")
    vdir = Path(vault_dir) if vault_dir is not None else getattr(config, "VAULT_DIR",
                                                                 config.ROOT / "DocumentationVault" / "systems")
    att_dir = vdir / "attachments"
    att_dir.mkdir(parents=True, exist_ok=True)
    yearly_dst, monthly_dst = att_dir / f"{slug}_sharpe_yearly.png", att_dir / f"{slug}_sharpe_monthly.png"
    with tempfile.TemporaryDirectory(prefix=f"tear_sheet_figs_{slug}_") as scratch:
        figs = ts.figures(scratch)                       # every figure the study has data for
        for name_, dst in (("sharpe_yearly", yearly_dst), ("sharpe_monthly", monthly_dst)):
            src = figs.get(name_)
            if src is not None:
                dst.write_bytes(Path(src).read_bytes())

    headline = ts.headline
    rp_label, rp = ts.risk_profile
    result_line = (f"{headline.get('class', 'unclassified')} — "
                   f"{fmt_sig(headline.get('mean_monthly_pct'), 2, '%')}/month at 10% DD budget")

    robust = headline.get("robustness", {})
    longest_dd_days = rp.get("longest_dd_days") if rp else None
    longest_dd_months = longest_dd_days / 21.0 if longest_dd_days is not None else None
    lines = [
        f"# {name}",
        f"**Book:** {book}   **Status:** {_status_line(status, stage, reason, scope)}   **Issue:** #{issue}",
        f"**Idea:** {_truncate_idea(idea)}",
        f"**Result:** {result_line}",
        f"**Risk profile:** {rp_label} — MaxDD {dash(rp.get('max_dd'), '.1%')}, "
        f"longest DD {dash(longest_dd_months, '.1f')} months, "
        f"max losing streak {dash(rp.get('max_losing_streak'), '.0f')}, "
        f"skew {dash(rp.get('skew'), '.2f')}, CVaR95 {dash(rp.get('cvar95'), '.2%')}",
        f"**Robustness:** DSR {dash(robust.get('dsr'), '.2f')} · "
        f"CSCV OOS loss {dash(robust.get('cscv_oos_loss'), '.2f')} · "
        f"WFO OOS Sharpe {dash(robust.get('wfo_oos'), '.2f')} "
        f"(recent third {dash(robust.get('wfo_oos_recent'), '.2f')}) · "
        f"holdout {robust.get('holdout_status') or 'not unlocked'}",
        f"![[{slug}_sharpe_yearly.png]]",
        f"![[{slug}_sharpe_monthly.png]]",
    ]
    text = "\n".join(lines) + "\n"
    out = vdir / f"{slug}.md"
    vdir.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
