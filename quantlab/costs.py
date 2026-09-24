"""Per-book cost models (DESIGN §4.3, §10 Phase 0).

Two things live here:

* :class:`InstrumentSpec` — the static contract/tick/swap/commission facts about
  one symbol, either read from a real MT5 export (``tools/mql5/ExportBrokerSpecs.mq5``,
  see ``data/broker/<Company>_{symbols,commissions}.tsv``) or, until that export
  exists, a documented fallback with ``calibrated=False``.
* :class:`CostModel` — the *calibratable* knobs applied on top of a spec's raw
  numbers: a spread multiplier + additive floor (the bar's own spread column is
  optimistic — DESIGN §4.3), slippage (in ``slippage_points``, the engine's own
  unit — see ``engine._run_core``) applied adversely to **every market fill**
  (entries, signal/time/session-flatten exits, and stop fills; targets never
  slip — red-team M2), and a swap sensitivity multiplier (only *current* swap
  rates are ever known).  ``CostModel.stressed()`` implements the cost-stress
  gate (DESIGN §4.2: Sharpe > 0.5 at 1.5x spread + 1 pip adverse slippage on
  every market fill); :func:`pip_points` converts the stress's pip count to
  ``slippage_points`` per instrument.

Versioning: every ``CostModel`` carries a ``version`` string that studies record
in the ledger (DESIGN §8).  ``version`` is computed from a human-chosen
``version_tag`` (e.g. ``"fbs-v1"``) plus a deterministic encoding of every
calibratable knob that isn't at its class default (red-team N5) — see
``CostModel.version``.  Never mutate the meaning of an existing ``version_tag``
in place — bump the string instead (``fbs-v1`` -> ``fbs-v2``) so old studies
keep their recorded numbers meaningful.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from typing import Optional

import polars as pl

from . import config

__all__ = [
    "InstrumentSpec",
    "CostModel",
    "load_instrument",
    "normalize_swap_mode",
    "swap_currency_for",
    "pip_points",
]

# --------------------------------------------------------------------------- InstrumentSpec

_VALID_SWAP_MODES = ("points", "money", "interest", "disabled")


@dataclass(frozen=True)
class InstrumentSpec:
    """Static contract/tick/swap/commission facts about one symbol.

    ``swap_mode`` is normalised to one of ``{"points", "money", "interest",
    "disabled"}`` (see :func:`normalize_swap_mode` for the MT5 enum mapping).
    Only ``"points"`` is fully modelled by :mod:`quantlab.engine` today; the
    others keep ``swap_points`` at 0 in trades and record ``swap_money_per_lot``
    instead (documented in ``engine.run_backtest``).

    ``swap_3day`` is the weekday (Mon=0 .. Sun=6) whose overnight swap is
    tripled to cover the weekend (MT5 ``SYMBOL_SWAP_ROLLOVER3DAYS``); brokers
    default to Wednesday (2).

    ``swap_every_day`` (red-team minor-4): whether swap accrues on *every*
    calendar night, including Friday->Saturday and Saturday->Sunday. FX/metals
    at MT5 brokers do not roll swap on those two nights (the weekend gap is
    priced into the ``swap_3day`` weekday's x3 multiplier instead -- the
    default, ``False``). Crypto CFDs at FBS-like brokers are commonly charged
    every night of the week, so :func:`load_instrument`'s fallback defaults
    this to ``True`` for crypto symbols; a real broker export should confirm
    it per-symbol when available. See ``engine._nights_weighted`` (which also
    does *not* triple the ``swap_3day`` weekday when ``swap_every_day`` is set
    -- that x3 exists specifically to compensate FX/metals for the two nights
    it skips, so applying it on top of an *every-day* charge would double
    count the weekend: 9 weighted nights across a week instead of the correct
    7 -- red-team minor-4 follow-up).

    ``margin_ccy`` (red-team N1) is the symbol's margin currency
    (``SYMBOL_CURRENCY_MARGIN``); ``None`` (the default, used by every code
    path that doesn't have a real broker export yet) resolves to ``base_ccy``
    in :meth:`__post_init__`, the common case for plain FX.

    ``swap_ccy`` (red-team N1, MAJOR) is the currency ``swap_money_per_lot``
    (``swap_mode="money"``) or the interest-mode swap amount is denominated
    in. MT5's three "money" modes are NOT interchangeable -- see
    :func:`swap_currency_for`, which is how :func:`load_instrument` derives
    this from the raw ``ENUM_SYMBOL_SWAP_MODE`` string. ``None`` (the default)
    resolves to ``quote_ccy`` in :meth:`__post_init__`: a documented
    best-effort guess (matching this module's pre-N1 behaviour) for any spec
    built without going through the real MT5 enum -- e.g. every uncalibrated
    fallback spec, and any test/spec built directly. The one real currency
    code that is never a plain ISO code is the sentinel ``"ACCOUNT"``
    (``SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT``): the amount is already in
    whatever currency the account itself uses, so ``apply_sizing`` must never
    try to convert it again.

    ``commission_per_lot_rt`` is a round-trip commission already expressed in
    **account currency** (as exported by ``ExportBrokerSpecs.mq5``'s deal-history
    aggregation), so no currency conversion is needed for it in sizing.py.
    """

    symbol: str
    digits: int
    point: float
    contract_size: float
    tick_size: float
    volume_min: float
    volume_step: float
    volume_max: float
    base_ccy: str
    quote_ccy: str
    swap_mode: str
    swap_long: float
    swap_short: float
    swap_3day: int = 2
    swap_every_day: bool = False
    margin_ccy: Optional[str] = None
    swap_ccy: Optional[str] = None
    commission_per_lot_rt: float = 0.0
    stops_level: float = 0.0
    calibrated: bool = False

    def __post_init__(self) -> None:
        if self.swap_mode not in _VALID_SWAP_MODES:
            raise ValueError(f"swap_mode must be one of {_VALID_SWAP_MODES}, got {self.swap_mode!r}")
        if not 0 <= self.swap_3day <= 6:
            raise ValueError(f"swap_3day must be a weekday int Mon=0..Sun=6, got {self.swap_3day!r}")
        if self.point <= 0:
            raise ValueError(f"point must be > 0, got {self.point!r}")
        if self.volume_step <= 0 or self.volume_min <= 0:
            raise ValueError("volume_min and volume_step must be > 0")
        if self.margin_ccy is None:
            object.__setattr__(self, "margin_ccy", self.base_ccy)
        if self.swap_ccy is None:
            object.__setattr__(self, "swap_ccy", self.quote_ccy)

    @property
    def value_per_point_per_lot(self) -> float:
        """Quote-currency P&L per point of price movement, for a 1.0-lot position."""
        return self.contract_size * self.point


# --------------------------------------------------------------------------- CostModel

_VALID_STOP_FILL = ("level", "bar_extreme")


@dataclass(frozen=True)
class CostModel:
    """Calibratable execution-cost knobs layered on top of an :class:`InstrumentSpec`.

    ``version_tag`` is the human-chosen identifier for this book/calibration
    (e.g. ``"fbs-v1"``); ``version`` (a computed property, red-team N5) is
    what studies actually record in the ledger (DESIGN §8) -- ``version_tag``
    plus a deterministic suffix for every knob below that is not at its class
    default (fixed order: ``spread_multiplier``, ``extra_spread_points``,
    ``slippage_points``, ``swap_multiplier``, ``stop_fill``). Two
    ``CostModel``s can therefore only ever share a ``version`` string if every
    one of these fields is identical -- e.g.
    ``CostModel(stop_fill="bar_extreme").version`` is
    ``"fbs-v1-uncalibrated+stop_fill=bar_extreme"``, never the plain
    ``"fbs-v1-uncalibrated"`` a stress-agnostic reader might mistake it for.
    (The class default was bumped ``fbs-v0-uncalibrated`` -> ``fbs-v1-uncalibrated``
    for red-team M2: the *engine's* interpretation of ``slippage_points`` changed
    -- adverse slippage now applies to every market fill, not only stops -- so a
    ledger row tagged with the old default could not otherwise be told apart from
    one written before the fix, even though ``stressed()`` can produce the exact
    same field values either way when called with no ``spec=``.)
    Because it is computed fresh from the model's own current field values
    (never string-concatenated across ``replace()`` calls), stacking
    modifications -- e.g. ``.stressed()`` on top of an already non-default
    model -- can never accumulate duplicate or stale suffixes.

    Effective spread for a bar (in points, DESIGN §4.3) is
    ``spread * spread_multiplier + extra_spread_points`` — see
    :meth:`effective_spread`.

    ``stop_fill`` (red-team minor-1): M1 data cannot resolve intra-minute
    liquidity gaps -- within a single M1 bar that merely *trades through* a
    stop (rather than gapping past it on its own open), there is no way to
    know whether a real broker's fill slipped further before the next tick.
    ``"level"`` (default) fills such a touch at the stop level itself
    (+ ``slippage_points``) -- optimistic, and the right default for the base
    cost model. ``"bar_extreme"`` instead fills at the worst of that M1 (or,
    without M1, that whole HTF) bar's own high/low -- a pessimistic stand-in
    meant for the validation stress test (:meth:`stressed` turns it on by
    default -- see there). Targets are never affected (a resting limit order
    never fills worse than its own price). See ``engine._chk_long`` /
    ``engine._chk_short``. Note: without M1, ``"bar_extreme"`` fills at the
    *whole HTF bar's* own extreme -- for D1 that is the day's high or low,
    a very pessimistic stand-in worth keeping in mind when reading a D1
    stress-test result.

    ``slippage_points`` (red-team M2) is applied **adversely on every market
    fill** by ``engine._run_core``: entries (a long entry buys, so it fills
    *higher*; a short entry sells, so it fills *lower*), signal/time exits and
    session-flatten (``force_exit``)/end-of-data (``eod``) exits (the mirror
    image -- closing a long sells *lower*, closing a short buys *higher*), and
    stop fills (unchanged from before -- fills at the stop/gap-open, ± this
    same slippage). Target/limit fills never slip (a resting limit order never
    fills worse than its own price). The base cost model's default,
    ``slippage_points=0.0``, is therefore a no-op everywhere and produces
    bit-identical fills to the pre-M2 engine.
    """

    version_tag: str = "fbs-v1-uncalibrated"
    spread_multiplier: float = 1.0
    extra_spread_points: float = 0.0
    slippage_points: float = 0.0
    swap_multiplier: float = 1.0
    stop_fill: str = "level"

    def __post_init__(self) -> None:
        if self.stop_fill not in _VALID_STOP_FILL:
            raise ValueError(f"stop_fill must be one of {_VALID_STOP_FILL}, got {self.stop_fill!r}")

    @property
    def version(self) -> str:
        """``version_tag`` + a deterministic suffix for every non-default knob (red-team N5).

        Computed fresh from the model's current fields every time it is read
        -- never stored/mutated -- so it is impossible for two ``CostModel``s
        with different ``(spread_multiplier, extra_spread_points,
        slippage_points, swap_multiplier, stop_fill)`` to log the same
        ``version`` while sharing a ``version_tag``, and impossible for
        repeated ``replace()``/``.stressed()`` calls to accumulate stale or
        duplicate suffixes.
        """
        parts = [self.version_tag]
        if self.spread_multiplier != 1.0:
            parts.append(f"spread_mult{self.spread_multiplier:g}")
        if self.extra_spread_points != 0.0:
            parts.append(f"extra_spread{self.extra_spread_points:g}")
        if self.slippage_points != 0.0:
            parts.append(f"slippage{self.slippage_points:g}")
        if self.swap_multiplier != 1.0:
            parts.append(f"swap_mult{self.swap_multiplier:g}")
        if self.stop_fill != "level":
            parts.append(f"stop_fill={self.stop_fill}")
        return "+".join(parts)

    def effective_spread(self, spread_points):
        """Effective spread in points; works elementwise on scalars, numpy arrays or Series."""
        return spread_points * self.spread_multiplier + self.extra_spread_points

    def stressed(self, spread_mult: float = 1.5, extra_slippage_pips: float = 1.0,
                 stop_fill: str = "bar_extreme", *, spec: Optional["InstrumentSpec"] = None,
                 extra_slippage: Optional[float] = None) -> "CostModel":
        """DESIGN §4.2 cost-stress gate: 1.5x spread + 1 pip adverse slippage by default.

        ``spread_mult`` multiplies the *current* spread_multiplier (relative
        stress on top of whatever calibration is already applied).

        ``extra_slippage_pips`` (red-team M2) is a **pip** count, added to
        ``slippage_points`` -- the field ``engine._run_core`` actually reads,
        applied adversely on every market fill (see the class docstring), not
        just stops. A pip is a market convention, not a fixed number of
        points (10 points on a 5-digit FX quote, 1 on a 4-digit one, DESIGN's
        own ``0.1`` USD for gold, ...), so converting it needs the
        instrument's own spec: pass ``spec=`` to get the real per-symbol
        conversion via :func:`pip_points`. Without one (``spec=None``, the
        default -- e.g. a cost model built with no symbol in view) this
        degrades to treating 1 pip as 1 point, i.e. exactly this method's
        pre-M2 numeric default, so a caller that hasn't been updated to pass
        ``spec=`` yet (the M2 fix plan's follow-up wiring) sees no change in
        ``slippage_points`` from this fix alone -- only the engine's now-wider
        fill-type scope.

        ``extra_slippage`` (deprecated but kept working -- red-team M2) is the
        pre-M2 kwarg: *raw points*, added to ``slippage_points`` directly,
        bypassing the pip conversion entirely. Takes precedence over
        ``extra_slippage_pips``/``spec`` when given (not ``None``).

        ``stop_fill`` (red-team N5) defaults to ``"bar_extreme"``: the §4.2
        stress gate should assume the pessimistic intra-minute-liquidity-gap
        fill (see the class docstring), not silently inherit the base
        model's optimistic ``"level"`` default just because the caller didn't
        think to stress it too. Pass ``stop_fill=self.stop_fill`` explicitly
        to keep the base model's own choice through the stress instead.

        Returns a new model; ``self`` is never mutated, and (since
        ``version`` is computed from the model's own fields, not
        string-concatenated) the returned model's ``.version`` reflects
        exactly this stressed field combination -- no double-suffixing even
        if ``self`` was itself already non-default.
        """
        if extra_slippage is not None:
            added_points = extra_slippage
        else:
            added_points = extra_slippage_pips * (pip_points(spec) if spec is not None else 1.0)
        return replace(
            self,
            spread_multiplier=self.spread_multiplier * spread_mult,
            slippage_points=self.slippage_points + added_points,
            stop_fill=stop_fill,
        )


# --------------------------------------------------------------------------- pip conventions (red-team M2)

def pip_points(spec: InstrumentSpec) -> float:
    """Points per conventional "pip" for ``spec`` -- see :meth:`CostModel.stressed`.

    A pip is a retail-broker market convention, not a fixed price increment,
    so it must be derived per instrument:

    * **FX** (a plain 6-letter-alpha symbol whose ``base_ccy``/``quote_ccy``
      are both 3-letter currency codes -- metals and crypto are excluded
      below before this branch is reached): the standard convention prices a
      pip at the 4th decimal place for non-JPY pairs / 2nd for JPY pairs. A
      **3- or 5-digit** quote (this catalog's usual convention -- ``digits``
      counts one extra fractional "pipette" decimal past the pip) prices a
      pip at **10 points**; a legacy **2- or 4-digit** whole-pip quote has no
      pipette, so 1 pip is already 1 point.
    * **XAUUSD / XAGUSD / XPTUSD / XPDUSD**: no single convention exists
      across brokers. We define 1 pip = **0.1 quote-currency unit** (USD) for
      all four -- the task spec's own documented example for gold, applied
      uniformly to the other three metals rather than guessing a separate
      number for each.
    * **Everything else** (crypto CFDs, B3 futures WIN/WDO, indices, ...):
      1 pip = 1 point = 1 tick, unless a real broker spec says otherwise
      (none of ours currently do).
    """
    sym = spec.symbol.upper()
    if sym.startswith(("XAU", "XAG", "XPT", "XPD")):
        return 0.1 / spec.point
    if sym.startswith(_CRYPTO_PREFIXES):
        return 1.0
    if (
        len(sym) == 6 and sym.isalpha()
        and len(spec.base_ccy) == 3 and spec.base_ccy.isalpha()
        and len(spec.quote_ccy) == 3 and spec.quote_ccy.isalpha()
    ):
        return 10.0 if spec.digits in (3, 5) else 1.0
    return 1.0


# --------------------------------------------------------------------------- MT5 enum parsing
# Column names / enum spellings below match tools/mql5/ExportBrokerSpecs.mq5 exactly.

_SWAP_MODE_MAP = {
    "SYMBOL_SWAP_MODE_DISABLED": "disabled",
    "SYMBOL_SWAP_MODE_POINTS": "points",
    "SYMBOL_SWAP_MODE_CURRENCY_SYMBOL": "money",
    "SYMBOL_SWAP_MODE_CURRENCY_MARGIN": "money",
    "SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT": "money",
    "SYMBOL_SWAP_MODE_INTEREST_CURRENT": "interest",
    "SYMBOL_SWAP_MODE_INTEREST_OPEN": "interest",
    "SYMBOL_SWAP_MODE_REOPEN_CURRENT": "points",
    "SYMBOL_SWAP_MODE_REOPEN_BID": "points",
}

_WEEKDAY_MAP = {
    "MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
    "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6,
}


def normalize_swap_mode(raw: str) -> str:
    """Map ``EnumToString(SYMBOL_SWAP_MODE)`` to one of the 4 canonical modes."""
    key = raw.strip().upper()
    mode = _SWAP_MODE_MAP.get(key)
    if mode is None:
        warnings.warn(f"unknown swap_mode {raw!r}; assuming 'points'", stacklevel=2)
        return "points"
    return mode


_MONEY_SWAP_CCY_SOURCE = {
    "SYMBOL_SWAP_MODE_CURRENCY_SYMBOL": "base",
    "SYMBOL_SWAP_MODE_CURRENCY_MARGIN": "margin",
    "SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT": "account",
}


def swap_currency_for(raw: str, base_ccy: str, margin_ccy: str, quote_ccy: str) -> str:
    """Currency an MT5 swap mode's amount is denominated in (red-team N1, MAJOR).

    MT5's three "money" swap modes collapse to the same canonical
    ``swap_mode="money"`` in :func:`normalize_swap_mode`, but they are **not**
    interchangeable currencies:

    * ``SYMBOL_SWAP_MODE_CURRENCY_SYMBOL`` prices the swap in the instrument's
      own **base** currency (``SYMBOL_CURRENCY_BASE``).
    * ``SYMBOL_SWAP_MODE_CURRENCY_MARGIN`` prices it in the instrument's
      **margin** currency (``SYMBOL_CURRENCY_MARGIN``) -- usually equal to the
      base currency for plain FX, but can differ for CFDs.
    * ``SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT`` means the amount is already in the
      **account's own deposit currency** -- returned here as the sentinel
      ``"ACCOUNT"`` so ``apply_sizing`` knows never to convert it again,
      regardless of what ``account_ccy`` it is actually running with.

    Every other mode (``POINTS``, the two ``INTEREST_*`` modes, and the two
    ``REOPEN_*`` modes) prices its swap off the instrument's own
    quote-currency price, so this falls back to ``quote_ccy``.

    Real-world failure this fixes (``p17`` d): a USDJPY spec with
    ``swap_mode="money"`` at -5/lot/night, previously always assumed to be
    JPY (quote currency) and converted via the JPY->USD rate (~1/110) to
    -1.91 USD/lot/night; if the export actually says
    ``SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT`` on a USD account, the correct
    booked amount is the full -5.00 USD/lot/night (a ~150x error at typical
    USDJPY rates).
    """
    key = raw.strip().upper()
    source = _MONEY_SWAP_CCY_SOURCE.get(key)
    if source == "base":
        return base_ccy
    if source == "margin":
        return margin_ccy
    if source == "account":
        return "ACCOUNT"
    return quote_ccy


def _parse_weekday(raw: str) -> int:
    """Map ``EnumToString(ENUM_DAY_OF_WEEK)`` (e.g. 'WEDNESDAY') to Mon=0..Sun=6."""
    key = raw.strip().upper()
    wd = _WEEKDAY_MAP.get(key)
    if wd is None:
        warnings.warn(f"unknown swap_3day weekday {raw!r}; defaulting to Wednesday", stacklevel=2)
        return 2
    return wd


# --------------------------------------------------------------------------- broker TSV loader

_CRYPTO_PREFIXES = ("BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "SOL", "DOGE", "DOT", "LINK")


def _is_crypto(symbol: str) -> bool:
    """True for any of this catalog's crypto CFD symbols (see ``swap_every_day``,
    red-team minor-4). Used both for the TSV loader (the export itself does not
    carry an explicit "charges swap every day" field) and the uncalibrated
    fallback below.
    """
    return symbol.upper().startswith(_CRYPTO_PREFIXES)


def _load_from_tsv(symbol: str, book: config.Book) -> Optional[InstrumentSpec]:
    sym_path = config.BROKER_DIR / f"{book.broker}_symbols.tsv"
    if not sym_path.exists():
        return None
    symbols_df = pl.read_csv(sym_path, separator="\t", infer_schema_length=10_000)
    match = symbols_df.filter(pl.col("symbol") == symbol)
    if match.height == 0:
        return None
    r = match.row(0, named=True)

    commission = 0.0
    comm_path = config.BROKER_DIR / f"{book.broker}_commissions.tsv"
    if comm_path.exists():
        comm_df = pl.read_csv(comm_path, separator="\t", infer_schema_length=10_000)
        crow = comm_df.filter(pl.col("symbol") == symbol)
        val = crow[0, "commission_per_lot_roundtrip"] if crow.height > 0 else None
        if val is not None:
            commission = float(val)
        else:
            warnings.warn(
                f"no commission history for {symbol!r} in {comm_path.name}; assuming 0.0",
                stacklevel=2,
            )
    else:
        warnings.warn(f"{comm_path.name} not found; commission assumed 0.0 for {symbol!r}", stacklevel=2)

    raw_swap_mode = str(r["swap_mode"])
    base_ccy = str(r["currency_base"])
    quote_ccy = str(r["currency_profit"])
    margin_ccy = str(r["currency_margin"])

    return InstrumentSpec(
        symbol=symbol,
        digits=int(r["digits"]),
        point=float(r["point"]),
        contract_size=float(r["contract_size"]),
        tick_size=float(r["tick_size"]),
        volume_min=float(r["volume_min"]),
        volume_step=float(r["volume_step"]),
        volume_max=float(r["volume_max"]),
        base_ccy=base_ccy,
        quote_ccy=quote_ccy,
        margin_ccy=margin_ccy,
        swap_mode=normalize_swap_mode(raw_swap_mode),
        swap_long=float(r["swap_long"]),
        swap_short=float(r["swap_short"]),
        swap_3day=_parse_weekday(str(r["swap_3day"])),
        swap_every_day=_is_crypto(symbol),
        # red-team N1: derived from the RAW enum string (before normalize_swap_mode collapses
        # the 3 "money" variants into one canonical mode) -- see swap_currency_for.
        swap_ccy=swap_currency_for(raw_swap_mode, base_ccy, margin_ccy, quote_ccy),
        commission_per_lot_rt=commission,
        stops_level=float(r["stops_level"]),
        calibrated=True,
    )


# --------------------------------------------------------------------------- fallback defaults

_WARNED_SYMBOLS: set[tuple[str, str]] = set()


def _default_spec_for(symbol: str, book_name: str) -> InstrumentSpec:
    """Documented fallback used until a broker export exists for this symbol.

    Numbers here are best-effort industry-typical defaults, **not** this
    account's real terms — every fallback spec has ``calibrated=False`` and
    fires a one-time warning. Real values must come from
    ``data/broker/<Company>_symbols.tsv`` (see :func:`load_instrument`).
    """
    book = config.get_book(book_name)
    sym_u = symbol.upper()

    # Resolve which fallback template applies *before* touching quantlab.data: an
    # unrecognised symbol should raise a clear "no fallback" error rather than
    # whatever the catalog lookup happens to say about it.
    if book.name == "B3":
        if sym_u.startswith("WIN"):
            contract_size, tick_size = 0.20, 5.0
            base_ccy, quote_ccy = "IBOV", "BRL"
        elif sym_u.startswith("WDO"):
            contract_size, tick_size = 10.0, 0.5
            base_ccy, quote_ccy = "USD", "BRL"
        else:
            raise ValueError(
                f"no fallback default for unknown B3 symbol {symbol!r}; "
                "extend costs._default_spec_for or provide a broker export"
            )
    elif not (sym_u.startswith(("XAU", "XAG", "XPT", "XPD")) or sym_u.startswith(_CRYPTO_PREFIXES)
              or (len(sym_u) == 6 and sym_u.isalpha())):
        raise ValueError(
            f"no fallback default for unrecognised {book.name} symbol {symbol!r}; "
            "extend costs._default_spec_for or provide a broker export"
        )

    from .data import infer_point, symbol_currencies  # lazy: data.py is a sibling deliverable

    point = infer_point(symbol)

    if book.name == "B3":
        spec = InstrumentSpec(
            symbol=symbol, digits=0, point=point, contract_size=contract_size, tick_size=tick_size,
            volume_min=1.0, volume_step=1.0, volume_max=1000.0,
            base_ccy=base_ccy, quote_ccy=quote_ccy,
            swap_mode="disabled", swap_long=0.0, swap_short=0.0, swap_3day=2,
            commission_per_lot_rt=0.0, stops_level=0.0, calibrated=False,
        )
    else:
        base_ccy, quote_ccy = symbol_currencies(symbol)
        if sym_u.startswith("XAU"):
            contract_size = 100.0
        elif sym_u.startswith("XAG"):
            contract_size = 5_000.0
        elif sym_u.startswith(("XPT", "XPD")):
            contract_size = 100.0
        elif sym_u.startswith(_CRYPTO_PREFIXES):
            contract_size = 1.0
        else:
            contract_size = 100_000.0
        digits = max(0, round(-math.log10(point))) if point > 0 else 5
        spec = InstrumentSpec(
            symbol=symbol, digits=digits, point=point, contract_size=contract_size, tick_size=point,
            volume_min=0.01, volume_step=0.01, volume_max=500.0,
            base_ccy=base_ccy, quote_ccy=quote_ccy,
            swap_mode="points", swap_long=0.0, swap_short=0.0, swap_3day=2,
            swap_every_day=_is_crypto(symbol),  # red-team minor-4: crypto swaps every night, incl. weekends
            commission_per_lot_rt=0.0, stops_level=0.0, calibrated=False,
        )

    key = (book.name, symbol)
    if key not in _WARNED_SYMBOLS:
        _WARNED_SYMBOLS.add(key)
        warnings.warn(
            f"no calibrated broker export for {book.name}/{symbol}; using an uncalibrated "
            "fallback InstrumentSpec (DESIGN §10 Phase 0). Run tools/mql5/ExportBrokerSpecs.mq5 "
            "and re-run to calibrate.",
            stacklevel=3,
        )
    return spec


def load_instrument(symbol: str, book: str = "FBS") -> InstrumentSpec:
    """Load real broker terms from ``data/broker/`` if present, else a flagged fallback."""
    b = config.get_book(book)
    spec = _load_from_tsv(symbol, b)
    if spec is not None:
        return spec
    return _default_spec_for(symbol, book)
