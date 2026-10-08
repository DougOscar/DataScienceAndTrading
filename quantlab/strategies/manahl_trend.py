"""``manahl_trend``: Python port of the MT5 ``ManAhl_EA`` v2 (multi-lookback trend vote).

Source of truth: ``MQL5/Experts/ManAhl_EA.mq5`` + ``MQL5/Indicators/ManAHL_Trend.mq5`` +
``MQL5/Indicators/EffRatio.mq5`` (v2.00, 2026-10-07). Every rule below mirrors that code; the
docstrings name the MQL5 function each piece ports so a parity check can go line by line.

Pieces
------
* :func:`features` -- the three indicators on closed bars (causal, row ``t`` uses rows ``<= t``):
  ``score`` (ManAHL_Trend buffer 0), ``sigma`` (buffer 2), ``er`` (EffRatio), ``atr`` (MT5
  ``iATR`` = simple mean of true range, :func:`base.atr`).
* :func:`target_fraction` -- the EA's position rule (``TargetFraction``).
* :class:`ManAhlTrend` -- the ``contracts.Strategy`` view used by the look-ahead audit and the
  engine parity check: binary position modes only, stops not fed back into the state.
* :func:`simulate` -- multi-symbol daily simulator for the full EA (scaled positions,
  vol-target sizing, stop-outs feeding back into the state, swap). Fill conventions follow
  ``quantlab.engine`` (Bid bars; longs pay the entry-bar spread, shorts pay it on exit;
  stops fill at the level or the gap open; adverse slippage on every market fill).

No data I/O here (``assert_strategy_source_clean``): conversion rates and instrument specs are
passed in by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from ..contracts import Params, RiskType
from ..costs import CostModel, InstrumentSpec
from ..sizing import floor_to_step
from . import base

__all__ = ["ManAhlParams", "ManAhlTrend", "SymbolInput", "SimResult", "features",
           "target_fraction", "simulate", "yearly_sharpe_score", "STRATEGY"]


# --------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class ManAhlParams(Params):
    """EA inputs (same names minus the ``Inp`` prefix, snake_case; enums as strings).

    Defaults are the EA v2 defaults. ``days_per_year`` is 260 (the project's FX convention,
    ``metrics.PERIODS_PER_YEAR``); the EA's own default is 252 -- set it to match when
    comparing against an MT5 report.
    """

    # signal (ManAHL_Trend)
    lookback_mode: str = "scaled"        # "scaled": L, 3L, 6L, 12L | "custom": lookbacks below
    lookback_base: int = 20
    lookbacks: tuple[int, int, int, int] = (20, 60, 120, 240)
    thresh_mode: str = "volz"            # "pct" | "volz"
    pct_up: float = 0.0
    pct_down: float = 0.0
    z_min: float = 0.0
    vol_period: int = 60
    # position logic
    position_mode: str = "scaled"        # "binary_half" | "binary_full" | "scaled"
    wait_after_stop: bool = True
    # efficiency filter
    er_mode: str = "absolute"            # "off" | "absolute" | "directional"
    er_period: int = 10
    er_min: float = 30.0
    # stop
    stop_mode: str = "atr"               # "none" | "atr" | "swing_atr"
    atr_period: int = 20
    stop_atr_mult: float = 4.0
    swing_strength: int = 2
    swing_lookback: int = 200
    swing_atr_pad: float = 0.5
    min_stop_atr: float = 1.0
    # sizing
    sizing_mode: str = "vol_target"      # "vol_target" | "risk" | "fixed"
    portfolio_vol_pct: float = 12.0
    assumed_corr: float = 0.3
    risk_percent: float = 1.0
    fixed_lots: float = 0.10
    max_leverage: float = 5.0
    # annualisation
    days_per_year: float = 260.0

    def resolved_lookbacks(self) -> tuple[int, int, int, int]:
        if self.lookback_mode == "scaled":
            L = self.lookback_base
            return (L, 3 * L, 6 * L, 12 * L)
        return tuple(self.lookbacks)  # type: ignore[return-value]

    def validate(self) -> None:
        enums = {
            "lookback_mode": (self.lookback_mode, ("scaled", "custom")),
            "thresh_mode": (self.thresh_mode, ("pct", "volz")),
            "position_mode": (self.position_mode, ("binary_half", "binary_full", "scaled")),
            "er_mode": (self.er_mode, ("off", "absolute", "directional")),
            "stop_mode": (self.stop_mode, ("none", "atr", "swing_atr")),
            "sizing_mode": (self.sizing_mode, ("vol_target", "risk", "fixed")),
        }
        for name, (value, allowed) in enums.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        if min(self.resolved_lookbacks()) < 1 or self.vol_period < 2 or self.er_period < 2:
            raise ValueError("lookbacks >= 1, vol_period > 1 and er_period > 1 required")
        if self.sizing_mode == "risk" and self.stop_mode == "none":
            raise ValueError("risk-per-trade sizing needs a stop (EA OnInit rule)")


# --------------------------------------------------------------------------- indicators
def features(bars: pl.DataFrame, p: ManAhlParams) -> pl.DataFrame:
    """``score``, ``sigma``, ``er``, ``atr`` per bar (null during warm-up), causal.

    * ``sigma`` (``ManAHL_Trend.SigmaAt``): sample st.dev. of the last ``vol_period`` 1-bar log
      returns; valid from row ``vol_period``.
    * ``score`` (``ManAHL_Trend.ScoreAt``): sum of four +1/0/-1 votes; valid from row
      ``max(max lookback, vol_period)``. ``volz`` vote: ``z = ln(c_t / c_{t-L}) /
      (sigma_t sqrt(L))`` with ``z >= z_min`` (strictly non-zero); ``pct`` vote: % change vs
      ``p_up`` / ``p_down``. A zero ``sigma`` casts no ``volz`` votes (MQL5 ``continue``).
    * ``er`` (``EffRatio``): ``100 (c_t - c_{t-k}) / sum |c_j - c_{j-1}|``; null if the
      denominator is 0.
    * ``atr``: MT5 ``iATR`` (simple mean of true range), :func:`base.atr`.
    """
    lbs = p.resolved_lookbacks()
    first = max(max(lbs), p.vol_period)
    c = pl.col("close")
    log_ret = (c / c.shift(1)).log()
    sigma = log_ret.rolling_std(p.vol_period, ddof=1)

    votes = []
    for L in lbs:
        ref = c.shift(L)
        if p.thresh_mode == "pct":
            chg = 100.0 * (c - ref) / ref
            vote = (pl.when((chg > 0) & (chg >= p.pct_up)).then(1)
                    .when((chg < 0) & (-chg >= p.pct_down)).then(-1).otherwise(0))
        else:
            z = (c / ref).log() / (sigma * math.sqrt(L))
            vote = (pl.when(sigma <= 0).then(0)
                    .when((z > 0) & (z >= p.z_min)).then(1)
                    .when((z < 0) & (-z >= p.z_min)).then(-1).otherwise(0))
        votes.append(vote)
    row = pl.int_range(pl.len())
    score = pl.when(row >= first).then(pl.sum_horizontal(votes)).otherwise(None).cast(pl.Int8)

    k = p.er_period
    den = (c - c.shift(1)).abs().rolling_sum(k)
    er = pl.when(den > 0).then(100.0 * (c - c.shift(k)) / den).otherwise(None)

    return bars.select(
        score.alias("score"),
        pl.when(row >= p.vol_period).then(sigma).otherwise(None).alias("sigma"),
        er.alias("er"),
        base.atr(p.atr_period).alias("atr"),
    )


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def target_fraction(score: int, cur: float, position_mode: str) -> float:
    """Signed target fraction of a full position (EA ``TargetFraction``).

    * Exit: the score is 0 or on the other side of the current position.
    * ``scaled``: |score| >= 4 -> 1, >= 2 -> 1/2; |score| = 1 on our side holds, capped at 1/2.
    * ``binary_*``: hold until the score reaches 0; enter at |score| >= 2 (half) or 4 (full).
    """
    sgn, a, cs = _sign(score), abs(score), _sign(cur)
    if cs != 0 and sgn != cs:
        cs = 0
    if position_mode == "scaled":
        if a >= 4:
            return float(sgn)
        if a >= 2:
            return 0.5 * sgn
        if cs != 0:
            return cs * min(abs(cur), 0.5)
        return 0.0
    threshold = 4 if position_mode == "binary_full" else 2
    if cs != 0:
        return cur
    if a >= threshold:
        return float(sgn)
    return 0.0


def _passes_er(er: float | None, direction: int, p: ManAhlParams) -> bool:
    """EA ``PassEfficiencyFilter``; a missing ER value fails the filter."""
    if p.er_mode == "off":
        return True
    if er is None or not np.isfinite(er):
        return False
    if p.er_mode == "absolute":
        return abs(er) >= p.er_min
    return direction * er >= p.er_min


# --------------------------------------------------------------------------- Strategy view
class ManAhlTrend:
    """``contracts.Strategy`` view: the binary-mode state machine as a desired-position signal.

    Used for (1) the look-ahead audit and (2) parity against ``quantlab.engine``. Stops are
    attached (``stop_dist``) but cannot feed back into this causal state, so exact parity with
    :func:`simulate` holds only for ``stop_mode="none"``. Scaled positions and vol-target
    sizing (risk type D) exist only in :func:`simulate`.
    """

    name = "manahl_trend"
    risk_type = RiskType.D
    params_cls = ManAhlParams

    def __init__(self, params: ManAhlParams | None = None) -> None:
        self.params = params or ManAhlParams(position_mode="binary_full", stop_mode="none",
                                             sizing_mode="fixed")
        self.params.validate()

    def signals(self, bars: pl.DataFrame) -> pl.DataFrame:
        p = self.params
        if p.position_mode == "scaled":
            raise ValueError("ManAhlTrend.signals: scaled positions need simulate(); use a binary mode")
        f = features(bars, p)
        score = f["score"].to_list()
        er = f["er"].to_list()
        out: list[int | None] = []
        pos = 0.0
        for s, e in zip(score, er):
            if s is None:
                out.append(None)
                continue
            tgt = target_fraction(int(s), pos, p.position_mode)
            tdir = _sign(tgt)
            increase = tdir != 0 and (tdir != _sign(pos) or abs(tgt) > abs(pos) + 1e-9)
            if increase and not _passes_er(e, tdir, p):
                tgt = pos if tdir == _sign(pos) else 0.0
            pos = tgt
            out.append(_sign(pos))
        stop = (f["atr"] * p.stop_atr_mult) if p.stop_mode == "atr" else pl.Series([None] * bars.height,
                                                                                   dtype=pl.Float64)
        return pl.DataFrame({
            "signal": pl.Series(out, dtype=pl.Int8),
            "stop_dist": stop.cast(pl.Float64),
            "target_dist": pl.Series([None] * bars.height, dtype=pl.Float64),
        })


STRATEGY = ManAhlTrend


# --------------------------------------------------------------------------- simulator
@dataclass
class SymbolInput:
    """One symbol for :func:`simulate`: bars (``contracts.BAR_COLUMNS``), its spec and the
    quote->account conversion rate known at each bar's open and close (as-of, causal).
    ``cost`` overrides :func:`simulate`'s cost model for this symbol (e.g. a stressed model
    whose "+1 pip" is expressed in this symbol's own points)."""

    symbol: str
    bars: pl.DataFrame
    spec: InstrumentSpec
    rate_open: np.ndarray
    rate_close: np.ndarray
    cost: CostModel | None = None


@dataclass
class SimResult:
    daily: pl.DataFrame        # date, equity, ret, gross_leverage, n_open
    trades: pl.DataFrame       # one row per closed leg (or closed part of a leg)
    positions: pl.DataFrame    # date, symbol, lots (signed), frac -- end of day
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Leg:
    direction: int
    lots: float
    entry_price: float
    stop: float                # nan = no stop
    entry_ts: Any
    frac_target: float


def _find_swing(lows_or_highs: np.ndarray, t: int, p: ManAhlParams, must_beat: float, low: bool) -> float | None:
    """EA ``FindSwingLow``/``FindSwingHigh`` on closed bars before ``t`` (shift i = row t-i).

    A swing low at shift i is strictly below the S newer bars and not above the S older ones;
    the most recent one beyond the price wins, else the window's extreme (shifts 1..).
    """
    S = p.swing_strength
    need = p.swing_lookback + 2 * S + 2
    got = min(need, t + 1)
    if got < 2 * S + 2:
        return None
    x = lows_or_highs
    for i in range(S + 1, got - S):
        v = x[t - i]
        ok = True
        for j in range(1, S + 1):
            newer, older = x[t - i + j], x[t - i - j]
            if low and (v >= newer or v > older):
                ok = False
                break
            if not low and (v <= newer or v < older):
                ok = False
                break
        if ok and ((low and v < must_beat) or (not low and v > must_beat)):
            return float(v)
    window = x[t - got + 1: t]          # shifts 1 .. got-1
    if window.size == 0:
        return None
    ext = float(window.min() if low else window.max())
    return ext if ((low and ext < must_beat) or (not low and ext > must_beat)) else None


def simulate(symbols: list[SymbolInput], p: ManAhlParams, cost: CostModel = CostModel(), *,
             equity0: float = 100_000.0, periods_per_year: float | None = None) -> SimResult:
    """Run the EA on daily bars for a basket, bar by bar (EA ``ProcessSymbol`` per bar open).

    Order of events on each server day ``d`` (for every symbol with a bar that day):

    1. **Open** -- decide from features of the previous closed bar (EA reads shift 1): reconcile
       the state with real legs, clear the stop-out block, compute the target fraction, gate
       any increase with the ER filter / stop-out block, then close / reduce / open legs at the
       open. Sizing uses the portfolio equity at the previous close.
    2. **Intrabar** -- every open leg (new ones included) is checked against its stop with the
       bar's range (``engine._chk_long``/``_chk_short`` rules, no M1 path).
    3. **Close** -- swap for legs held through the rollover; mark-to-market equity.

    Not modelled (documented simplifications): free-margin cap (the 5x leverage cap binds
    first at FBS leverage), the EA's intra-day timing of other symbols' first ticks, and
    requotes/rejections.
    """
    p.validate()
    ppy = periods_per_year or p.days_per_year
    n_sym = len(symbols)
    instr_vol = p.portfolio_vol_pct / 100.0 / math.sqrt(n_sym * (1.0 + (n_sym - 1) * p.assumed_corr))

    # per-symbol arrays
    S: list[dict[str, Any]] = []
    for si in symbols:
        b = si.bars
        f = features(b, p)
        spec = si.spec
        cm = si.cost or cost
        if spec.swap_mode not in ("points", "money", "disabled"):
            raise NotImplementedError(f"{si.symbol}: swap_mode {spec.swap_mode!r} not modelled")
        S.append(dict(
            sym=si.symbol, spec=spec, pt=spec.point,
            ts=b["ts"].to_list(), date=b["ts"].dt.date().to_list(),
            o=b["open"].to_numpy(), h=b["high"].to_numpy(), l=b["low"].to_numpy(), c=b["close"].to_numpy(),
            spr=b["spread"].cast(pl.Float64).to_numpy(), spr_max=b["spread_max"].cast(pl.Float64).to_numpy(),
            eff=cm.effective_spread(b["spread"].cast(pl.Float64).to_numpy()),
            eff_max=cm.effective_spread(b["spread_max"].cast(pl.Float64).to_numpy()),
            cm=cm, slip=cm.slippage_points,
            score=f["score"].to_numpy(), sigma=f["sigma"].to_numpy(), er=f["er"].to_numpy(),
            atr=f["atr"].to_numpy(), rate_o=np.asarray(si.rate_open, float),
            rate_c=np.asarray(si.rate_close, float),
            legs=[], frac=0.0, blocked=0, last_close=None, last_mark=0.0,
        ))
        S[-1]["row_of"] = {d: i for i, d in enumerate(S[-1]["date"])}

    calendar = sorted({d for s in S for d in s["date"]})
    cash = equity0
    equity_prev = equity0
    trades: list[dict[str, Any]] = []
    daily: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []

    def notional_per_lot(s, t, price):
        return s["spec"].contract_size * price * s["rate_o"][t]

    # swap accrued per leg (booked to cash daily, attributed to the trade row on close)
    leg_swap: dict[int, float] = {}

    def book_close(s, leg: _Leg, lots: float, price: float, ts, rate: float, reason: str):
        nonlocal cash
        spec = s["spec"]
        pnl = leg.direction * (price - leg.entry_price) * spec.contract_size * lots * rate
        comm = -spec.commission_per_lot_rt * lots
        cash += pnl + comm
        acc = leg_swap.get(id(leg), 0.0)
        share = lots / leg.lots if leg.lots > 0 else 1.0
        swap_part = acc * share
        leg_swap[id(leg)] = acc - swap_part
        trades.append(dict(symbol=s["sym"], direction=leg.direction, lots=lots, entry_ts=leg.entry_ts,
                           entry_price=leg.entry_price, stop=leg.stop, exit_ts=ts, exit_price=price,
                           reason=reason, frac_target=leg.frac_target, pnl_price=pnl, swap=swap_part,
                           commission=comm, pnl=pnl + swap_part + comm))

    def calc_stop(s, t, d, bid, ask):
        if p.stop_mode == "none":
            return float("nan")
        atr = s["atr"][t - 1]
        if not np.isfinite(atr) or atr <= 0:
            return None
        if p.stop_mode == "atr":
            sl = bid - p.stop_atr_mult * atr if d > 0 else ask + p.stop_atr_mult * atr
        else:
            arr = s["l"] if d > 0 else s["h"]
            swing = _find_swing(arr, t, p, bid if d > 0 else ask, low=d > 0)
            if swing is None:
                return None
            sl = (min(swing - p.swing_atr_pad * atr, bid - p.min_stop_atr * atr) if d > 0
                  else max(swing + p.swing_atr_pad * atr, ask + p.min_stop_atr * atr))
        spec = s["spec"]
        min_dist = (spec.stops_level + 1) * spec.point
        if d > 0 and bid - sl < min_dist:
            sl = bid - min_dist
        if d < 0 and sl - ask < min_dist:
            sl = ask + min_dist
        ts_ = spec.tick_size if spec.tick_size > 0 else spec.point
        return math.floor(sl / ts_) * ts_ if d > 0 else math.ceil(sl / ts_) * ts_

    def full_lots(s, t, d, entry, sl):
        spec = s["spec"]
        if p.sizing_mode == "fixed":
            return p.fixed_lots
        npl = notional_per_lot(s, t, entry)
        if npl <= 0:
            return 0.0
        if p.sizing_mode == "vol_target":
            sig = s["sigma"][t - 1]
            if not np.isfinite(sig) or sig <= 0:
                return 0.0
            lots = equity_prev * instr_vol / (npl * sig * math.sqrt(ppy))
        else:
            loss = abs(entry - sl) * spec.contract_size * s["rate_o"][t]
            if not np.isfinite(loss) or loss <= 0:
                return 0.0
            lots = equity_prev * p.risk_percent / 100.0 / loss
        return min(lots, p.max_leverage * equity_prev / npl)

    def open_leg(s, t, d, lots, frac_target):
        spec = s["spec"]
        o, pt = s["o"][t], s["pt"]
        bid, ask = o, o + s["eff"][t] * pt
        sl = calc_stop(s, t, d, bid, ask)
        if sl is None:
            return
        n = floor_to_step(lots, spec.volume_min, spec.volume_step, spec.volume_max)
        if n <= 0:
            return
        entry = ask + s["slip"] * pt if d > 0 else bid - s["slip"] * pt
        s["legs"].append(_Leg(d, n, entry, sl, s["ts"][t], frac_target))

    def exit_price_open(s, t, d):
        o, pt = s["o"][t], s["pt"]
        return (o - s["slip"] * pt) if d > 0 else (o + s["eff"][t] * pt + s["slip"] * pt)

    for day in calendar:
        # ---- 1. decisions at the open
        for s in S:
            t = s["row_of"].get(day)
            if t is None or t == 0:
                continue
            score = s["score"][t - 1]
            if score is None or not np.isfinite(score):
                continue
            score = int(score)
            net = sum(l.direction * l.lots for l in s["legs"])
            if abs(net) < 1e-12:
                s["frac"] = 0.0
            elif _sign(net) != _sign(s["frac"]):
                s["frac"] = _sign(net) * (0.5 if p.position_mode == "scaled" and abs(score) < 4 else 1.0)
            if s["blocked"] != 0 and score * s["blocked"] <= 0:
                s["blocked"] = 0
            cur = s["frac"]
            tgt = target_fraction(score, cur, p.position_mode)
            tdir = _sign(tgt)
            increase = tdir != 0 and (tdir != _sign(cur) or abs(tgt) > abs(cur) + 1e-9)
            er_prev = s["er"][t - 1]
            if increase and (s["blocked"] == tdir or not _passes_er(er_prev if np.isfinite(er_prev) else None, tdir, p)):
                tgt = cur if tdir == _sign(cur) else 0.0
            if abs(tgt - cur) < 1e-9:
                continue
            ts, rate = s["ts"][t], s["rate_o"][t]
            if _sign(tgt) != _sign(cur):
                for leg in list(s["legs"]):
                    book_close(s, leg, leg.lots, exit_price_open(s, t, leg.direction), ts, rate,
                               "flip" if _sign(tgt) != 0 else "signal_exit")
                s["legs"].clear()
                if _sign(tgt) != 0:
                    d = _sign(tgt)
                    o, pt = s["o"][t], s["pt"]
                    sl = calc_stop(s, t, d, o, o + s["eff"][t] * pt)
                    entry = o + s["eff"][t] * pt if d > 0 else o
                    if sl is not None:
                        open_leg(s, t, d, abs(tgt) * full_lots(s, t, d, entry, sl), tgt)
            elif abs(tgt) > abs(cur):
                d = _sign(tgt)
                o, pt = s["o"][t], s["pt"]
                sl = calc_stop(s, t, d, o, o + s["eff"][t] * pt)
                entry = o + s["eff"][t] * pt if d > 0 else o
                if sl is not None:
                    add = abs(tgt) * full_lots(s, t, d, entry, sl) - abs(net)
                    if add > 0:
                        open_leg(s, t, d, add, tgt)
            else:
                spec = s["spec"]
                red = abs(net) * (1.0 - abs(tgt) / abs(cur))
                red = math.floor(red / spec.volume_step + 1e-9) * spec.volume_step
                if red >= spec.volume_min:
                    for leg in reversed(list(s["legs"])):      # newest legs first
                        if red < spec.volume_min - 1e-12:
                            break
                        px = exit_price_open(s, t, leg.direction)
                        if leg.lots <= red + 1e-9:
                            book_close(s, leg, leg.lots, px, ts, rate, "scale_down")
                            red -= leg.lots
                            s["legs"].remove(leg)
                        else:
                            part = floor_to_step(red, spec.volume_min, spec.volume_step, spec.volume_max)
                            if part <= 0:
                                break
                            book_close(s, leg, part, px, ts, rate, "scale_down")
                            leg.lots = round(leg.lots - part, 8)
                            red -= part
            s["frac"] = tgt

        # ---- 2. intrabar stops, 3. swap + mark at the close
        unreal = 0.0
        gross = 0.0
        n_open = 0
        for s in S:
            t = s["row_of"].get(day)
            if t is None:
                unreal += s["last_mark"]
                continue
            spec, pt, cm, slip = s["spec"], s["pt"], s["cm"], s["slip"]
            o, h, l, c = s["o"][t], s["h"][t], s["l"][t], s["c"][t]
            for leg in list(s["legs"]):
                if not np.isfinite(leg.stop):
                    continue
                if leg.direction > 0 and l <= leg.stop:
                    fill = (o if o <= leg.stop else (l if cm.stop_fill == "bar_extreme" else leg.stop)) - slip * pt
                elif leg.direction < 0 and h + s["eff_max"][t] * pt >= leg.stop:
                    ask_o = o + s["eff"][t] * pt
                    fill = (ask_o if ask_o >= leg.stop else
                            (h + s["eff_max"][t] * pt if cm.stop_fill == "bar_extreme" else leg.stop)) + slip * pt
                else:
                    continue
                book_close(s, leg, leg.lots, fill, s["ts"][t], s["rate_c"][t], "stop")
                s["legs"].remove(leg)
                if p.wait_after_stop:
                    s["blocked"] = leg.direction
            # swap for legs held through tonight's rollover (FX/metals: Mon-Fri nights, x3 on swap_3day)
            # (none on the symbol's last bar: the end-of-data exit happens at its close)
            wd = s["date"][t].weekday()
            if s["legs"] and wd <= 4 and spec.swap_mode != "disabled" and t < len(s["ts"]) - 1:
                w = (3.0 if wd == spec.swap_3day else 1.0) * cm.swap_multiplier
                for leg in s["legs"]:
                    rate_sw = s["rate_c"][t]
                    sw = spec.swap_long if leg.direction > 0 else spec.swap_short
                    if spec.swap_mode == "points":
                        amt = sw * pt * spec.contract_size * leg.lots * rate_sw * w
                    else:   # money per lot, in swap_ccy
                        if spec.swap_ccy == spec.quote_ccy:
                            amt = sw * leg.lots * rate_sw * w
                        elif spec.swap_ccy in ("ACCOUNT", "USD"):
                            amt = sw * leg.lots * w
                        else:
                            raise NotImplementedError(f"{s['sym']}: money swap in {spec.swap_ccy} not modelled")
                    cash += amt
                    leg_swap[id(leg)] = leg_swap.get(id(leg), 0.0) + amt
            mark = 0.0
            for leg in s["legs"]:
                px = c if leg.direction > 0 else c + s["spr"][t] * pt
                mark += leg.direction * (px - leg.entry_price) * spec.contract_size * leg.lots * s["rate_c"][t]
                gross += spec.contract_size * c * leg.lots * s["rate_c"][t]
                n_open += 1
            s["last_mark"] = mark
            unreal += mark
            positions.append(dict(date=day, symbol=s["sym"],
                                  lots=float(sum(lg.direction * lg.lots for lg in s["legs"])), frac=float(s["frac"])))
        equity = cash + unreal
        daily.append(dict(date=day, equity=equity, ret=equity / equity_prev - 1.0,
                          gross_leverage=gross / equity if equity > 0 else float("nan"), n_open=n_open))
        equity_prev = equity

    # ---- end of data: flatten at the last close
    for s in S:
        t = len(s["ts"]) - 1
        for leg in list(s["legs"]):
            # engine convention: a short's end-of-data exit pays the bar's max spread
            px = s["c"][t] if leg.direction > 0 else s["c"][t] + s["eff_max"][t] * s["pt"]
            book_close(s, leg, leg.lots, px, s["ts"][t], s["rate_c"][t], "eod")
        s["legs"].clear()
    if daily:
        daily[-1]["equity"] = cash
        prev = daily[-2]["equity"] if len(daily) > 1 else equity0
        daily[-1]["ret"] = cash / prev - 1.0

    trade_schema = {"symbol": pl.Utf8, "direction": pl.Int8, "lots": pl.Float64, "entry_ts": pl.Datetime("ms"),
                    "entry_price": pl.Float64, "stop": pl.Float64, "exit_ts": pl.Datetime("ms"),
                    "exit_price": pl.Float64, "reason": pl.Utf8, "frac_target": pl.Float64,
                    "pnl_price": pl.Float64, "swap": pl.Float64, "commission": pl.Float64, "pnl": pl.Float64}
    return SimResult(
        # explicit schemas: inference from the first (all-flat) rows would type lots as Int64
        daily=pl.DataFrame(daily, schema={"date": pl.Date, "equity": pl.Float64, "ret": pl.Float64,
                                          "gross_leverage": pl.Float64, "n_open": pl.Int64}),
        trades=pl.DataFrame(trades, schema=trade_schema) if trades else pl.DataFrame(schema=trade_schema),
        positions=pl.DataFrame(positions, schema={"date": pl.Date, "symbol": pl.Utf8, "lots": pl.Float64,
                                                  "frac": pl.Float64}),
        meta=dict(params=p.as_dict(), cost_model_version={s["sym"]: s["cm"].version for s in S}, equity0=equity0,
                  instr_vol=instr_vol, symbols=[s["sym"] for s in S], periods_per_year=ppy),
    )


# --------------------------------------------------------------------------- OnTester score
def yearly_sharpe_score(daily: pl.DataFrame, *, mode: str = "mean_minus_std", penalty: float = 0.5,
                        periods_per_year: float = 260.0, min_days: int = 20) -> dict[str, Any]:
    """EA ``OnTester``: annualised Sharpe per calendar year of daily returns, then aggregated
    (``mean`` | ``mean_minus_std`` | ``median`` | ``min``). A flat year counts as Sharpe 0."""
    yearly = (daily.with_columns(pl.col("date").dt.year().alias("year"))
              .group_by("year").agg(pl.col("ret").mean().alias("mu"), pl.col("ret").std().alias("sd"),
                                    pl.len().alias("days"))
              .filter(pl.col("days") >= max(min_days, 2)).sort("year")
              .with_columns(pl.when(pl.col("sd") > 1e-10)
                            .then(pl.col("mu") / pl.col("sd") * math.sqrt(periods_per_year))
                            .otherwise(0.0).alias("sharpe")))
    sh = yearly["sharpe"].to_numpy()
    if sh.size == 0:
        return dict(score=float("nan"), yearly=yearly)
    agg = dict(mean=float(sh.mean()), sd=float(sh.std(ddof=1)) if sh.size > 1 else 0.0,
               median=float(np.median(sh)), min=float(sh.min()))
    score = {"mean": agg["mean"], "mean_minus_std": agg["mean"] - penalty * agg["sd"],
             "median": agg["median"], "min": agg["min"]}[mode]
    return dict(score=score, yearly=yearly, **agg)
