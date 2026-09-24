"""Per-book cost models (DESIGN §4.3, §10 Phase 0).

Two things live here:

* :class:`InstrumentSpec` — the static contract/tick/swap/commission facts about
  one symbol, either read from a real MT5 export (``tools/mql5/ExportBrokerSpecs.mq5``,
  see ``data/broker/<Company>_{symbols,commissions}.tsv``) or, until that export
  exists, a documented fallback with ``calibrated=False``.
* :class:`CostModel` — the *calibratable* knobs applied on top of a spec's raw
  numbers: a spread multiplier + additive floor (the bar's own spread column is
  optimistic — DESIGN §4.3), slippage on stop fills, and a swap sensitivity
  multiplier (only *current* swap rates are ever known).  ``CostModel.stressed()``
  implements the cost-stress gate (DESIGN §4.2: Sharpe > 0.5 at 1.5x spread + 1pt
  slippage).

Versioning: every ``CostModel`` carries a ``version`` string that studies record
in the ledger (DESIGN §8).  Never mutate the meaning of an existing version in
place — bump the string instead (``fbs-v1`` -> ``fbs-v2``) so old studies keep
their recorded numbers meaningful.
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
    it per-symbol when available. See ``engine._nights_weighted``.

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

    @property
    def value_per_point_per_lot(self) -> float:
        """Quote-currency P&L per point of price movement, for a 1.0-lot position."""
        return self.contract_size * self.point


# --------------------------------------------------------------------------- CostModel

_VALID_STOP_FILL = ("level", "bar_extreme")


@dataclass(frozen=True)
class CostModel:
    """Calibratable execution-cost knobs layered on top of an :class:`InstrumentSpec`.

    ``version`` is the identifier studies record in the ledger (e.g.
    ``"fbs-v1"``). Effective spread for a bar (in points, DESIGN §4.3) is
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
    meant for the validation stress test. Targets are never affected (a
    resting limit order never fills worse than its own price). See
    ``engine._chk_long`` / ``engine._chk_short``.
    """

    version: str = "fbs-v0-uncalibrated"
    spread_multiplier: float = 1.0
    extra_spread_points: float = 0.0
    slippage_points: float = 0.0
    swap_multiplier: float = 1.0
    stop_fill: str = "level"

    def __post_init__(self) -> None:
        if self.stop_fill not in _VALID_STOP_FILL:
            raise ValueError(f"stop_fill must be one of {_VALID_STOP_FILL}, got {self.stop_fill!r}")

    def effective_spread(self, spread_points):
        """Effective spread in points; works elementwise on scalars, numpy arrays or Series."""
        return spread_points * self.spread_multiplier + self.extra_spread_points

    def stressed(self, spread_mult: float = 1.5, extra_slippage: float = 1.0) -> "CostModel":
        """DESIGN §4.2 cost-stress gate: 1.5x spread + 1pt extra slippage by default.

        ``spread_mult`` multiplies the *current* spread_multiplier (relative
        stress on top of whatever calibration is already applied);
        ``extra_slippage`` is *added* to the current slippage_points. Returns a
        new model whose version records the stress applied — it never mutates
        ``self`` or silently reuses the base version string.
        """
        return replace(
            self,
            spread_multiplier=self.spread_multiplier * spread_mult,
            slippage_points=self.slippage_points + extra_slippage,
            version=f"{self.version}+stress(x{spread_mult:g}spread,+{extra_slippage:g}slip)",
        )


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

    return InstrumentSpec(
        symbol=symbol,
        digits=int(r["digits"]),
        point=float(r["point"]),
        contract_size=float(r["contract_size"]),
        tick_size=float(r["tick_size"]),
        volume_min=float(r["volume_min"]),
        volume_step=float(r["volume_step"]),
        volume_max=float(r["volume_max"]),
        base_ccy=str(r["currency_base"]),
        quote_ccy=str(r["currency_profit"]),
        swap_mode=normalize_swap_mode(str(r["swap_mode"])),
        swap_long=float(r["swap_long"]),
        swap_short=float(r["swap_short"]),
        swap_3day=_parse_weekday(str(r["swap_3day"])),
        swap_every_day=_is_crypto(symbol),
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
