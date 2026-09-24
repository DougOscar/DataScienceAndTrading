"""Backtest execution engine (DESIGN §4.3, §10 Phase 0).

``run_backtest`` turns a bar frame + a signal frame (contracts.py) into fills
that respect MT5's Bid-bar reality: longs buy the Ask and sell the Bid, shorts
sell the Bid and buy back the Ask, stops slip, gaps fill at the open, and (for
H1+ systems) the M1 path resolves same-bar SL/TP ambiguity.

Semantic decisions (see the module docstrings below and the final report for
the full list):

* A signal on row ``i`` is executed at the **open of bar i+1**. Within that
  fill bar, in order: (1) the pending entry/exit from the signal happens at
  the open, (2) the resulting position (new or carried over) is checked for
  SL/TP across that bar's own range (or its M1 path), (3) if still open and
  ``force_exit`` is set for this bar, it is flattened at the bar's close.
* **Contract for strategy authors (red-team minor-11): a signal is the
  *desired position*, not a one-shot event.** The engine only acts when
  ``signal[i-1] != position`` (see Step 1), so a strategy that keeps emitting
  ``+1`` after being stopped out re-enters at the very next bar's open --
  there is no implicit "one entry per cross". A strategy that wants exactly
  one entry per signal change must emit ``null`` (keep current state) on
  every row after the entry, not repeat the same non-null value.
* **Spread convention**: a long pays its entry-bar's spread once, fixed at
  entry (`Ask = open + eff_spread*point`); its exits are always raw Bid — no
  second charge. A short pays spread once, on the leg that closes it (its
  entry is raw Bid; every exit style — stop, target, signal, force, eod — is
  `Ask = price + eff_spread*point` using *that exit bar's own* spread). This
  keeps "one round-trip spread per trade" exactly true while letting a
  short's realised cost float with the spread at the moment it actually
  closes (DESIGN §4.3: "shorts enter at Bid and exit at Ask").
* **SL/TP sentinels (a missing stop or target must never trigger)**: a
  position with no stop and/or no target is modelled with a per-direction
  price sentinel that *that* direction's own comparisons can never satisfy --
  longs: stop -1e18 / target +1e18 (tested against raw Bid low/high); shorts:
  stop +1e18 / target -1e18 (tested against Ask high/low, which are always
  positive prices). Using the *same* sentinel pair for both directions is
  wrong for shorts (Ask high/low are always ≥ 0, so ``ask_h >= -1e18`` and
  ``ask_l <= +1e18`` are trivially true) and silently "gap-stops" every short
  with no hard stop and/or no target on its own entry bar (red-team B1,
  BLOCKER).
* **SL/TP same-bar ambiguity**: checked in bar (or M1 sub-bar) order; if a
  single bar/sub-bar satisfies both conditions at once, the stop wins
  (adverse assumption, per DESIGN §4.3 / task spec).
* **Gaps**: if the checking bar's own *open* (adjusted for that bar's spread,
  for shorts) is already through the level, the fill is at that open (with
  adverse slippage for stops; targets never get slippage, so a gapped target
  fills at the open even though that's *better* than the target itself — a
  faithful model of a resting limit order).
* **Intra-minute liquidity gaps cannot be modelled from M1 data (red-team
  minor-1).** Within a single M1 bar that merely *trades through* a stop
  (rather than gapping past it on its own open), we cannot see whether the
  broker's fill would have slipped further before the next real tick. The
  default (``CostModel.stop_fill="level"``) fills such a touch at the stop
  level itself (+ ``slippage_points``) -- optimistic. Passing
  ``CostModel(stop_fill="bar_extreme")`` instead fills at the worst of that
  M1 (or, without M1, that whole HTF) bar's own high/low -- a pessimistic
  stand-in meant for the validation stress test, not for the base cost
  model. Targets are never affected: a resting limit order never fills worse
  than its own price.
* **M1 window per HTF bar j is exactly ``[ts_j, ts_j + timeframe)``** --
  bounded by the HTF bar's own span (an explicit ``timeframe=`` kwarg, or
  inferred from the smallest gap between consecutive ``bars.ts`` values when
  omitted), and *never* by the next bar's ``ts`` alone and *never* by the end
  of the ``m1`` frame. A session/weekend/holiday gap right after bar j, or an
  ``m1`` frame that runs past the last HTF bar (e.g. "load M1 once, backtest
  many walk-forward fold slices of HTF bars"), must not leak prices from
  *after* bar j's own close into bar j's SL/TP check (red-team M1). See
  :func:`_map_m1_bounds`.
* **MAE/MFE only reflect prices the still-open position actually saw
  (red-team M5).** On every bar/sub-bar that does *not* close the trade, the
  bar's own full high/low (Ask-adjusted for shorts, see below) is used, since
  the position genuinely was open for the whole bar. On the bar/sub-bar that
  *does* close the trade: with M1, sub-bars are walked and their excursions
  accumulated in order, stopping at (and including) the sub-bar that
  triggers the exit -- later sub-bars in the same HTF bar are simply never
  looked at. Without M1 (or with an empty M1 window for that bar), the exit
  bar's contribution is capped at the fill price itself on both sides,
  because we only know a hit happened *somewhere* in the bar's range and
  cannot rule out that price kept moving (favourably *or* adversely) only
  after the position had already closed -- e.g. a post-stop rally must not
  inflate MFE. **Shorts' excursions are measured on Ask** (``high``/``low`` +
  that bar's effective spread), matching the price a short's stop/target
  actually triggers on, not on raw Bid.
* Exit reasons: ``signal`` (flat/reversal driven by the strategy),
  ``stop`` / ``gap_stop``, ``target`` / ``gap_target``, ``force`` (session
  flattening), ``eod`` (end of data).
* ``entry_ts``/``exit_ts`` are always the **HTF bar's** ``ts`` (open time) of
  the bar in which the fill happened — even when the M1 path resolved *which*
  price triggered first inside that bar. M1 only disambiguates order/price,
  not the recorded timestamp granularity.
* **Spread on eod/force short exits, and on no-M1 short SL/TP checks
  (red-team minor-10).** An eod/force short exit is priced with the *last* M1
  bar's own spread inside that HTF bar when ``m1`` is given, else with the
  bar's own ``spread_max`` (conservative stand-in for "the spread somewhere
  in this bar", since the bar-*open* spread used everywhere else can be a
  rollover/session-open spike unrepresentative of the rest of the bar).
  Likewise, a no-M1 short's intrabar SL/TP *touch* test (``ask_h``/``ask_l``)
  uses ``spread_max``, not the open spread -- the conservative choice in both
  directions (it can only make a stop trigger *more* easily and a target
  trigger *less* easily). Gap detection and gap fills still use the bar's own
  (exactly known) *open* spread. A warning (default "once" via Python's
  ``warnings`` de-duplication) is raised when any trade's stop distance is
  narrower than its entry bar's effective spread plus ``spec.stops_level``
  points -- MT5 would reject that order in reality; the backtest still fills
  it as modelled.
* Nights/swap counting is documented in :func:`_nights_weighted`; whether
  swap accrues on Friday/Saturday nights too (crypto CFDs at FBS-like
  brokers) is ``spec.swap_every_day`` (see ``quantlab.costs.InstrumentSpec``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import numpy as np
import polars as pl
from numba import njit

from .contracts import BAR_COLUMNS, SIGNAL_COLUMNS
from .costs import CostModel, InstrumentSpec

__all__ = ["BacktestResult", "run_backtest"]

_NO_SIGNAL = np.int8(-9)  # sentinel for "keep" in the int8 signal array (valid domain is {-1,0,1})

_REASON_NAMES = {
    0: "signal", 1: "stop", 2: "gap_stop", 3: "target", 4: "gap_target", 5: "force", 6: "eod",
}

_TIMEFRAME_MINUTES: dict[str, int] = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}

# Warned-once-per-symbol set for the minor-10 tight-stop warning (mirrors costs._WARNED_SYMBOLS).
# Deliberately *not* keyed on the per-call trade counts: a message that varies every call would
# defeat Python's own (message-keyed) warning de-duplication and, worse, print on every single
# run_backtest call in perf-sensitive loops (walk-forward, Optuna studies, ProcessPoolExecutor
# workers) -- exactly the O(n_bars)-per-call regression DESIGN §7 exists to prevent.
_WARNED_TIGHT_STOP: set[str] = set()


@dataclass(frozen=True)
class BacktestResult:
    trades: pl.DataFrame
    position: np.ndarray
    meta: dict[str, Any]


# --------------------------------------------------------------------------- numba core

@njit(cache=True)
def _chk_long(o: float, h: float, l: float, sl: float, tp: float, slip: float, bar_extreme: bool):
    """One bar/sub-bar's long SL/TP check. SL is tested first -> adverse tie-break.

    ``bar_extreme``: on a non-gap stop touch, fill at the bar's own low
    instead of the stop level itself -- see the module docstring's
    "intra-minute liquidity gaps" note (red-team minor-1). Targets are
    unaffected: a resting limit order never fills worse than its own price.
    """
    if l <= sl:
        gap = o <= sl
        if gap:
            return True, o - slip, 2
        fill = l if bar_extreme else sl
        return True, fill - slip, 1
    if h >= tp:
        gap = o >= tp
        price = o if gap else tp
        return True, price, (4 if gap else 3)
    return False, 0.0, 0


@njit(cache=True)
def _chk_short(o: float, h: float, l: float, spread_open_pts: float, spread_touch_pts: float,
               point: float, sl: float, tp: float, slip: float, bar_extreme: bool):
    """One bar/sub-bar's short SL/TP check (Ask-triggered). Returns spread_pts paid on hit.

    ``spread_open_pts`` prices the bar's own (exactly known) open, for gap
    detection and gap fills. ``spread_touch_pts`` prices the high/low used to
    *detect* an intrabar touch, and is reported back as the spread paid on a
    non-gap touch. On the M1 path both are that same minute's own spread; on
    the no-M1 whole-bar path ``spread_touch_pts`` is the bar's ``spread_max``
    -- the conservative stand-in documented in the module docstring
    (red-team minor-10). ``bar_extreme`` mirrors :func:`_chk_long`.
    """
    ask_o = o + spread_open_pts * point
    ask_h = h + spread_touch_pts * point
    ask_l = l + spread_touch_pts * point
    if ask_h >= sl:
        gap = ask_o >= sl
        if gap:
            return True, ask_o + slip, 2, spread_open_pts
        fill = ask_h if bar_extreme else sl
        return True, fill + slip, 1, spread_touch_pts
    if ask_l <= tp:
        gap = ask_o <= tp
        price = ask_o if gap else tp
        return True, price, (4 if gap else 3), (spread_open_pts if gap else spread_touch_pts)
    return False, 0.0, 0, 0.0


@njit(cache=True)
def _wholebar_long(o: float, h: float, l: float, ep: float, sl: float, tp: float, slip: float,
                    point: float, bar_extreme: bool):
    """No-M1 (or M1-empty-window) long SL/TP check + this bar's MAE/MFE contribution.

    If the bar closes the trade, the excursion contribution is capped at the
    fill price (red-team M5): we only know a hit happened somewhere in the
    bar's range, and price action *after* the fill -- e.g. a rally after a
    stop -- never happened to an already-closed position. If the bar does
    *not* close the trade, its own full high/low is used, since the position
    genuinely was open for the whole bar.
    """
    hit, price, reason = _chk_long(o, h, l, sl, tp, slip, bar_extreme)
    if hit:
        adverse = (ep - price) / point
        favorable = (price - ep) / point
    else:
        adverse = (ep - l) / point
        favorable = (h - ep) / point
    return hit, price, reason, adverse, favorable


@njit(cache=True)
def _wholebar_short(o: float, h: float, l: float, spread_open_pts: float, spread_touch_pts: float,
                     point: float, ep: float, sl: float, tp: float, slip: float, bar_extreme: bool):
    """Short counterpart of :func:`_wholebar_long`; excursions measured on Ask (red-team M5)."""
    hit, price, reason, scost = _chk_short(
        o, h, l, spread_open_pts, spread_touch_pts, point, sl, tp, slip, bar_extreme
    )
    if hit:
        adverse = (price - ep) / point
        favorable = (ep - price) / point
    else:
        ask_h = h + spread_touch_pts * point
        ask_l = l + spread_touch_pts * point
        adverse = (ask_h - ep) / point
        favorable = (ep - ask_l) / point
    return hit, price, reason, scost, adverse, favorable


@njit(cache=True)
def _run_core(
    o, h, l, c, eff_spread, eff_spread_max, point,
    signal_i8, stop_dist, target_dist, force_exit,
    m1_start, m1_end, m1_o, m1_h, m1_l, m1_spread, use_m1,
    slippage_points, bar_extreme,
):
    n = o.shape[0]
    slip_price = slippage_points * point
    max_trades = n + 2

    entry_idx = np.empty(max_trades, dtype=np.int64)
    exit_idx = np.empty(max_trades, dtype=np.int64)
    entry_price = np.empty(max_trades, dtype=np.float64)
    exit_price = np.empty(max_trades, dtype=np.float64)
    stop_price_a = np.empty(max_trades, dtype=np.float64)
    target_price_a = np.empty(max_trades, dtype=np.float64)
    direction_a = np.empty(max_trades, dtype=np.int8)
    reason_a = np.zeros(max_trades, dtype=np.int8)
    spread_cost_a = np.empty(max_trades, dtype=np.float64)
    mae_a = np.zeros(max_trades, dtype=np.float64)
    mfe_a = np.zeros(max_trades, dtype=np.float64)
    position_track = np.zeros(n, dtype=np.int8)

    n_trades = 0
    position = 0
    entry_i = -1
    ep = 0.0
    sl = 0.0
    tp = 0.0
    has_sl = False
    has_tp = False
    cur_spread_cost = 0.0
    cur_mae = 0.0
    cur_mfe = 0.0

    for j in range(n):
        # ---- Step 1: pending signal from row j-1 fills at the open of bar j
        if j > 0 and signal_i8[j - 1] != _NO_SIGNAL and signal_i8[j - 1] != position:
            desired = signal_i8[j - 1]
            if position != 0:
                if position == 1:
                    xprice = o[j]
                else:
                    xprice = o[j] + eff_spread[j] * point
                    cur_spread_cost = eff_spread[j]
                entry_idx[n_trades] = entry_i
                exit_idx[n_trades] = j
                entry_price[n_trades] = ep
                exit_price[n_trades] = xprice
                stop_price_a[n_trades] = sl if has_sl else np.nan
                target_price_a[n_trades] = tp if has_tp else np.nan
                direction_a[n_trades] = position
                reason_a[n_trades] = 0
                spread_cost_a[n_trades] = cur_spread_cost
                mae_a[n_trades] = cur_mae
                mfe_a[n_trades] = cur_mfe
                n_trades += 1
                position = 0

            blocked = force_exit[j]
            if desired != 0 and not blocked:
                position = desired
                entry_i = j
                if desired == 1:
                    ep = o[j] + eff_spread[j] * point
                    cur_spread_cost = eff_spread[j]
                else:
                    ep = o[j]
                    cur_spread_cost = 0.0
                sd = stop_dist[j - 1]
                td = target_dist[j - 1]
                has_sl = not np.isnan(sd)
                has_tp = not np.isnan(td)
                sl = (ep - sd) if (has_sl and desired == 1) else ((ep + sd) if has_sl else 0.0)
                tp = (ep + td) if (has_tp and desired == 1) else ((ep - td) if has_tp else 0.0)
                cur_mae = 0.0
                cur_mfe = 0.0
            else:
                position = 0

        # ---- Step 2: SL/TP check across bar j's range (or its M1 path)
        if position != 0:
            # Per-direction sentinels for "no stop"/"no target" (red-team B1,
            # BLOCKER): a short's checks compare against Ask highs/lows, which
            # are always positive prices, so the *long* sentinels (-1e18/+1e18)
            # would be trivially satisfied and gap-exit every unstopped short
            # on its own entry bar. See the module docstring.
            if position == 1:
                sl_c = sl if has_sl else -1.0e18
                tp_c = tp if has_tp else 1.0e18
            else:
                sl_c = sl if has_sl else 1.0e18
                tp_c = tp if has_tp else -1.0e18

            hit = False
            price = 0.0
            reason = 0
            scost = 0.0

            if use_m1:
                s0 = m1_start[j]
                s1 = m1_end[j]
                if s1 > s0:
                    for t in range(s0, s1):
                        # Excursion from this sub-bar, applied *before* the hit
                        # check, for every sub-bar actually walked -- the loop
                        # breaks at the hit sub-bar, so later sub-bars in this
                        # HTF bar are never seen (red-team M5).
                        if position == 1:
                            sub_adverse = (ep - m1_l[t]) / point
                            sub_favorable = (m1_h[t] - ep) / point
                        else:
                            sub_ask_h = m1_h[t] + m1_spread[t] * point
                            sub_ask_l = m1_l[t] + m1_spread[t] * point
                            sub_adverse = (sub_ask_h - ep) / point
                            sub_favorable = (ep - sub_ask_l) / point
                        if sub_adverse > cur_mae:
                            cur_mae = sub_adverse
                        if sub_favorable > cur_mfe:
                            cur_mfe = sub_favorable

                        if position == 1:
                            hit, price, reason = _chk_long(
                                m1_o[t], m1_h[t], m1_l[t], sl_c, tp_c, slip_price, bar_extreme
                            )
                        else:
                            hit, price, reason, scost = _chk_short(
                                m1_o[t], m1_h[t], m1_l[t], m1_spread[t], m1_spread[t], point,
                                sl_c, tp_c, slip_price, bar_extreme
                            )
                        if hit:
                            break
                else:
                    if position == 1:
                        hit, price, reason, adverse, favorable = _wholebar_long(
                            o[j], h[j], l[j], ep, sl_c, tp_c, slip_price, point, bar_extreme
                        )
                    else:
                        hit, price, reason, scost, adverse, favorable = _wholebar_short(
                            o[j], h[j], l[j], eff_spread[j], eff_spread_max[j], point,
                            ep, sl_c, tp_c, slip_price, bar_extreme
                        )
                    if adverse > cur_mae:
                        cur_mae = adverse
                    if favorable > cur_mfe:
                        cur_mfe = favorable
            else:
                if position == 1:
                    hit, price, reason, adverse, favorable = _wholebar_long(
                        o[j], h[j], l[j], ep, sl_c, tp_c, slip_price, point, bar_extreme
                    )
                else:
                    hit, price, reason, scost, adverse, favorable = _wholebar_short(
                        o[j], h[j], l[j], eff_spread[j], eff_spread_max[j], point,
                        ep, sl_c, tp_c, slip_price, bar_extreme
                    )
                if adverse > cur_mae:
                    cur_mae = adverse
                if favorable > cur_mfe:
                    cur_mfe = favorable

            if hit:
                if position == -1:
                    cur_spread_cost = scost
                entry_idx[n_trades] = entry_i
                exit_idx[n_trades] = j
                entry_price[n_trades] = ep
                exit_price[n_trades] = price
                stop_price_a[n_trades] = sl if has_sl else np.nan
                target_price_a[n_trades] = tp if has_tp else np.nan
                direction_a[n_trades] = position
                reason_a[n_trades] = reason
                spread_cost_a[n_trades] = cur_spread_cost
                mae_a[n_trades] = cur_mae
                mfe_a[n_trades] = cur_mfe
                n_trades += 1
                position = 0

        # ---- Step 3: force-flatten at bar j's close
        if position != 0 and force_exit[j]:
            if position == 1:
                xprice = c[j]
            else:
                if use_m1 and m1_end[j] > m1_start[j]:
                    exit_spread = m1_spread[m1_end[j] - 1]
                else:
                    exit_spread = eff_spread_max[j]
                xprice = c[j] + exit_spread * point
                cur_spread_cost = exit_spread
            entry_idx[n_trades] = entry_i
            exit_idx[n_trades] = j
            entry_price[n_trades] = ep
            exit_price[n_trades] = xprice
            stop_price_a[n_trades] = sl if has_sl else np.nan
            target_price_a[n_trades] = tp if has_tp else np.nan
            direction_a[n_trades] = position
            reason_a[n_trades] = 5
            spread_cost_a[n_trades] = cur_spread_cost
            mae_a[n_trades] = cur_mae
            mfe_a[n_trades] = cur_mfe
            n_trades += 1
            position = 0

        position_track[j] = position

    if position != 0:
        j = n - 1
        if position == 1:
            xprice = c[j]
        else:
            if use_m1 and m1_end[j] > m1_start[j]:
                exit_spread = m1_spread[m1_end[j] - 1]
            else:
                exit_spread = eff_spread_max[j]
            xprice = c[j] + exit_spread * point
            cur_spread_cost = exit_spread
        entry_idx[n_trades] = entry_i
        exit_idx[n_trades] = j
        entry_price[n_trades] = ep
        exit_price[n_trades] = xprice
        stop_price_a[n_trades] = sl if has_sl else np.nan
        target_price_a[n_trades] = tp if has_tp else np.nan
        direction_a[n_trades] = position
        reason_a[n_trades] = 6
        spread_cost_a[n_trades] = cur_spread_cost
        mae_a[n_trades] = cur_mae
        mfe_a[n_trades] = cur_mfe
        n_trades += 1
        position = 0
        position_track[j] = 0

    return (
        entry_idx[:n_trades], exit_idx[:n_trades], entry_price[:n_trades], exit_price[:n_trades],
        stop_price_a[:n_trades], target_price_a[:n_trades], direction_a[:n_trades], reason_a[:n_trades],
        spread_cost_a[:n_trades], mae_a[:n_trades], mfe_a[:n_trades], position_track,
    )


# --------------------------------------------------------------------------- swap / nights

def _nights_weighted(entry_ts: datetime, exit_ts: datetime, swap_3day: int,
                      swap_every_day: bool = False) -> tuple[int, float]:
    """Server-midnight rollovers held between entry and exit (DESIGN §4.3).

    A "night" is named by the calendar date it starts on. Night ``D`` counts
    iff the rollover at ``D+1`` 00:00 server time falls inside
    ``(entry_ts, exit_ts]`` **and** (``D`` is Mon-Fri, **or** ``swap_every_day``
    is set -- red-team minor-4: crypto CFDs at FBS-like brokers are charged
    swap every night of the week, including Friday->Saturday and
    Saturday->Sunday, unlike FX/metals, where MT5 does not roll swap at the
    weekend-starting nights and instead prices the weekend gap into the
    ``swap_3day`` weekday's x3 multiplier). This makes a FX Friday-open ->
    Monday-close hold count exactly 1 night (Friday's, charged at Saturday
    00:00) while a *crypto* Friday-open -> Monday-close hold counts 3 (Fri,
    Sat, Sun), each at its own weight -- the triple-day multiplier still
    applies by weekday regardless of ``swap_every_day`` (brokers may or may
    not triple a crypto symbol's Wednesday night too; this is a per-symbol
    fact the broker export should eventually confirm, so the same rule is
    applied uniformly here until then). A Mon-open -> Thu-close FX hold counts
    3 nights with the swap_3day one (default Wednesday) weighted x3.
    """
    if exit_ts <= entry_ts:
        return 0, 0.0
    raw = 0
    weighted = 0.0
    d = entry_ts.date()
    while True:
        midnight = datetime.combine(d + timedelta(days=1), datetime.min.time())
        if midnight > exit_ts:
            break
        if swap_every_day or d.weekday() <= 4:  # Mon=0 .. Fri=4
            raw += 1
            weighted += 3.0 if d.weekday() == swap_3day else 1.0
        d += timedelta(days=1)
    return raw, weighted


# --------------------------------------------------------------------------- helpers

def _to_bool_array(x, bars: pl.DataFrame, n: int) -> np.ndarray:
    if x is None:
        return np.zeros(n, dtype=np.bool_)
    if isinstance(x, str):
        s = bars[x]
    elif isinstance(x, pl.Series):
        s = x
    else:
        s = pl.Series(x)
    arr = s.fill_null(False).cast(pl.Boolean).to_numpy()
    if arr.shape[0] != n:
        raise ValueError(f"force_exit has {arr.shape[0]} rows, expected {n}")
    return arr


def _infer_span(htf_np: np.ndarray) -> np.timedelta64:
    """Fallback HTF bar span when ``timeframe`` isn't passed to :func:`run_backtest`
    explicitly: the smallest positive gap between consecutive ``bars['ts']``
    values. This is correct for the ordinary case (uniformly-spaced bars,
    possibly with a handful of larger session/weekend/holiday gaps -- the
    *smallest* gap is still the true timeframe). Pass ``timeframe=`` explicitly
    for any frame whose gaps are *all* larger than the true span (e.g. a
    2-bar Friday+Monday D1 slice), where this inference would overstate it.
    """
    if htf_np.shape[0] < 2:
        raise ValueError(
            "cannot infer the HTF bar span from a single-bar frame; pass timeframe=... to run_backtest"
        )
    diffs = np.diff(htf_np)
    pos = diffs[diffs > np.timedelta64(0, "ns")]
    if pos.shape[0] == 0:
        raise ValueError("cannot infer the HTF bar span: bars['ts'] is not strictly increasing")
    return pos.min()


def _map_m1_bounds(htf_ts: pl.Series, m1_ts: pl.Series, span: np.timedelta64) -> tuple[np.ndarray, np.ndarray]:
    """M1 row range ``[start, end)`` feeding HTF bar j, bounded by bar j's own
    window ``[ts_j, ts_j + span)``.

    Never bounded by ``ts[j+1]`` alone (a session/weekend/holiday gap right
    after bar j would let its SL/TP check see M1 rows from *after* its own
    close -- genuine look-ahead: red-team M1) and never by ``len(m1)`` for the
    last bar (which would let it see *every* remaining M1 row, no matter how
    far past the HTF frame's end that reaches -- the audit's "load M1 once,
    backtest many walk-forward fold slices of HTF bars" failure mode). Taking
    the ``min`` of the ``ts[j+1]``-based bound and the ``ts[j]+span`` bound
    keeps the ordinary, non-gapped case byte-identical to before (the two
    bounds coincide) while clipping to bar j's own window whenever they don't.
    """
    n = htf_ts.len()
    htf_np = htf_ts.to_numpy()
    m1_np = m1_ts.to_numpy()
    starts = np.searchsorted(m1_np, htf_np, side="left").astype(np.int64)
    span_ends = np.searchsorted(m1_np, htf_np + span, side="left").astype(np.int64)
    if n > 1:
        next_starts = np.searchsorted(m1_np, htf_np[1:], side="left").astype(np.int64)
        ends = span_ends.copy()
        ends[:-1] = np.minimum(next_starts, span_ends[:-1])
    else:
        ends = span_ends
    return starts, ends


def _check_columns(df: pl.DataFrame, required: tuple[str, ...], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} frame is missing columns: {missing}")


# --------------------------------------------------------------------------- public API

def run_backtest(
    bars: pl.DataFrame,
    signals: pl.DataFrame,
    spec: InstrumentSpec,
    cost: CostModel = CostModel(),
    *,
    m1: Optional[pl.DataFrame] = None,
    timeframe: Optional[str] = None,
    force_exit: Optional[Union[str, pl.Series, np.ndarray, list]] = None,
) -> BacktestResult:
    """Run the single-position fill/stop/target loop (DESIGN §4.3).

    ``bars``/``signals`` follow the contracts in ``quantlab.contracts`` (same
    length/order; Bid OHLC + points spread; +1/-1/0/null signal with PRICE-unit
    stop_dist/target_dist). ``m1`` (optional) is an M1 bar frame with the same
    schema, used to resolve same-HTF-bar SL/TP ambiguity. ``timeframe``
    (optional, one of ``"M1"/"M5"/"M15"/"M30"/"H1"/"H4"/"D1"``) declares
    ``bars``' own timeframe so the M1 window per bar can be bounded exactly
    (see :func:`_map_m1_bounds`); if omitted, it is inferred from ``bars``
    itself (the smallest gap between consecutive ``ts`` values) -- pass it
    explicitly for any ``bars`` slice whose only gaps could be mistaken for
    the timeframe (e.g. a 2-row Friday+Monday D1 slice). Ignored when ``m1``
    is not given. ``force_exit`` is an optional per-bar boolean (column name
    in ``bars``, a Series, or an array-like) used for B3 session flattening:
    on a ``True`` bar, any open position is closed at that bar's close (after
    that bar's own SL/TP is checked), and no *new* entry is opened on that bar
    even if the signal asks for one.
    """
    _check_columns(bars, BAR_COLUMNS, "bars")
    _check_columns(signals, SIGNAL_COLUMNS, "signals")
    if bars.height != signals.height:
        raise ValueError(f"bars has {bars.height} rows but signals has {signals.height}")
    if spec.point <= 0:
        raise ValueError("spec.point must be > 0")
    if timeframe is not None and timeframe not in _TIMEFRAME_MINUTES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {sorted(_TIMEFRAME_MINUTES)}")

    n = bars.height
    o = bars["open"].cast(pl.Float64).to_numpy()
    h = bars["high"].cast(pl.Float64).to_numpy()
    l = bars["low"].cast(pl.Float64).to_numpy()
    c = bars["close"].cast(pl.Float64).to_numpy()
    raw_spread = bars["spread"].cast(pl.Float64).to_numpy()
    raw_spread_max = bars["spread_max"].cast(pl.Float64).to_numpy()
    eff_spread = cost.effective_spread(raw_spread)
    eff_spread_max = cost.effective_spread(raw_spread_max)
    point = float(spec.point)
    bar_extreme = cost.stop_fill == "bar_extreme"

    sig_arr = signals["signal"].fill_null(_NO_SIGNAL).cast(pl.Int8).to_numpy()
    stop_dist = signals["stop_dist"].fill_null(float("nan")).cast(pl.Float64).to_numpy()
    target_dist = signals["target_dist"].fill_null(float("nan")).cast(pl.Float64).to_numpy()
    force_exit_arr = _to_bool_array(force_exit, bars, n)

    use_m1 = m1 is not None
    if use_m1:
        _check_columns(m1, BAR_COLUMNS, "m1")
        m1_o = m1["open"].cast(pl.Float64).to_numpy()
        m1_h = m1["high"].cast(pl.Float64).to_numpy()
        m1_l = m1["low"].cast(pl.Float64).to_numpy()
        m1_raw_spread = m1["spread"].cast(pl.Float64).to_numpy()
        m1_spread = cost.effective_spread(m1_raw_spread)
        if timeframe is not None:
            span = np.timedelta64(_TIMEFRAME_MINUTES[timeframe], "m")
        else:
            span = _infer_span(bars["ts"].to_numpy())
        m1_start, m1_end = _map_m1_bounds(bars["ts"], m1["ts"], span)
    else:
        m1_o = m1_h = m1_l = m1_spread = np.zeros(0, dtype=np.float64)
        m1_start = np.zeros(n, dtype=np.int64)
        m1_end = np.zeros(n, dtype=np.int64)

    (entry_idx, exit_idx, entry_price, exit_price, stop_price, target_price,
     direction, reason_code, spread_cost_points, mae_points, mfe_points,
     position_track) = _run_core(
        o, h, l, c, eff_spread, eff_spread_max, point,
        sig_arr, stop_dist, target_dist, force_exit_arr,
        m1_start, m1_end, m1_o, m1_h, m1_l, m1_spread, use_m1,
        float(cost.slippage_points), bar_extreme,
    )

    # Perf (DESIGN §7): materialising the *whole* `ts` column to a Python list
    # (`bars["ts"].to_list()`) and then indexing it costs O(n_bars) even though
    # only O(n_trades) timestamps are ever needed -- on the 3.7M-row M1 kernel
    # benchmark this one line was >90% of total wall time. `Series.gather`
    # extracts just the needed rows *before* the Series->Python conversion, so
    # only `n_trades` datetimes are ever built. Result is identical: gather()
    # preserves row order and dtype, so `.to_list()` on the gathered Series
    # yields the exact same `datetime` objects as `[full_list[i] for i in idx]`
    # (see tests/test_perf_equivalence.py).
    ts_col = bars["ts"]
    n_trades = entry_idx.shape[0]
    entry_ts = ts_col.gather(entry_idx).to_list()
    exit_ts = ts_col.gather(exit_idx).to_list()

    if n_trades > 0 and spec.symbol not in _WARNED_TIGHT_STOP:
        has_stop = ~np.isnan(stop_price)
        if has_stop.any():
            entry_spread_at_fill = eff_spread[entry_idx[has_stop]]
            stop_dist_pts = np.abs(entry_price[has_stop] - stop_price[has_stop]) / point
            too_tight = stop_dist_pts < (entry_spread_at_fill + spec.stops_level)
            if too_tight.any():
                _WARNED_TIGHT_STOP.add(spec.symbol)
                warnings.warn(
                    f"{spec.symbol}: at least one stop-bearing trade has a stop distance narrower than "
                    "its entry bar's effective spread + spec.stops_level "
                    f"({spec.stops_level:g} pts) -- MT5 would reject an order this tight "
                    "(TRADE_RETCODE_INVALID_STOPS); the backtest still fills it as modelled "
                    "(red-team minor-10). Warned once per symbol per process.",
                    stacklevel=2,
                )

    nights = np.empty(n_trades, dtype=np.int64)
    weighted_nights = np.empty(n_trades, dtype=np.float64)
    for k in range(n_trades):
        nights[k], weighted_nights[k] = _nights_weighted(
            entry_ts[k], exit_ts[k], spec.swap_3day, spec.swap_every_day
        )

    swap_points = np.zeros(n_trades, dtype=np.float64)
    swap_money_per_lot = np.zeros(n_trades, dtype=np.float64)
    if spec.swap_mode == "points":
        per_night = np.where(direction == 1, spec.swap_long, spec.swap_short)
        swap_points = weighted_nights * per_night * cost.swap_multiplier
    elif spec.swap_mode == "money":
        per_night = np.where(direction == 1, spec.swap_long, spec.swap_short)
        swap_money_per_lot = weighted_nights * per_night * cost.swap_multiplier
    elif spec.swap_mode == "interest":
        if n_trades > 0 and np.any(weighted_nights > 0):
            warnings.warn(
                f"{spec.symbol}: swap_mode='interest' is not modelled (needs notional + rate curve); "
                "swap_points and swap_money_per_lot are 0 for held-overnight trades. Flag any system "
                "whose edge depends on swap for this symbol.",
                stacklevel=2,
            )
    # "disabled" -> both stay 0.

    trades = pl.DataFrame({
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_reason": [_REASON_NAMES[r] for r in reason_code],
        "bars_held": exit_idx - entry_idx,
        "pnl_points": np.where(direction == 1, exit_price - entry_price, entry_price - exit_price) / point,
        "spread_cost_points": spread_cost_points,
        "mae_points": mae_points,
        "mfe_points": mfe_points,
        "nights": nights,
        "swap_points": swap_points,
        "swap_money_per_lot": swap_money_per_lot,
        "commission_per_lot": np.full(n_trades, spec.commission_per_lot_rt),
    })

    meta = {
        "spec": spec,
        "cost_model_version": cost.version,
        "m1_used": use_m1,
        "n_bars": n,
    }
    return BacktestResult(trades=trades, position=position_track, meta=meta)
