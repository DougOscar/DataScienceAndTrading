"""Book-level configuration: brokers, timezones, holdout boundaries, nominal accounts.

A *book* is an independent research universe (DESIGN §0.5): FBS (forex, metals,
crypto CFDs) and B3 (WIN/WDO futures, intraday only).  Books never share data,
cost models, holdouts or portfolios.

All timestamps in the Parquet store are **naive broker server time**.  The
``server_tz`` below is the IANA zone that reproduces each broker's clock:

* FBS runs on EET/EEST with EU DST rules (verified 2026-09-22 from weekly
  open/close patterns: week opens Mon 00:00, closes Fri 23:59 server time; in
  the March US/EU DST-mismatch weeks the close shifts to 22:59).
* Clear (B3) runs on Brasília time.  Brazil abolished DST in 2019, before our
  B3 data starts (2021).

Holdout boundaries are expressed in server time and are **inclusive starts**:
any bar with ``ts >= holdout_start`` is locked (DESIGN §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.json"
BROKER_DIR = DATA_DIR / "broker"          # ExportBrokerSpecs.mq5 outputs (*.tsv)
RESEARCH_DIR = ROOT / "research"
LEDGER_DIR = RESEARCH_DIR / "ledger"
STUDIES_DIR = RESEARCH_DIR / "studies"
CACHE_DIR = DATA_DIR / "_quantlab_cache"


@dataclass(frozen=True)
class Book:
    name: str                    # "FBS" | "B3"
    broker: str                  # broker label as written in data/manifest.json
    markets: tuple[str, ...]     # data/<market>/ folders belonging to the book
    server_tz: ZoneInfo
    holdout_start: datetime      # naive, server time, inclusive
    account_currency: str
    nominal_equity: float        # DESIGN §5: % equity is computed on this nominal account
    intraday_only: bool          # B3: positions must be flat before the session close

    @property
    def tz_name(self) -> str:
        return self.server_tz.key

    @property
    def locked_end(self) -> datetime:
        """End of the locked 12-month holdout (DESIGN §4.1).  Data after this is "newer data":
        the renewing holdout that the exam also uses once it exists (DESIGN §4.4)."""
        return self.holdout_start.replace(year=self.holdout_start.year + 1)


BOOKS: dict[str, Book] = {
    "FBS": Book(
        name="FBS",
        broker="FBS",
        markets=("forex", "crypto"),
        server_tz=ZoneInfo("Europe/Helsinki"),
        holdout_start=datetime(2025, 5, 15),
        account_currency="USD",
        nominal_equity=100_000.0,
        intraday_only=False,
    ),
    "B3": Book(
        name="B3",
        broker="Clear",
        markets=("b3",),
        server_tz=ZoneInfo("America/Sao_Paulo"),
        # Shortest futures series (WIN M1) ends 2026-04-29 → 12 months before.
        holdout_start=datetime(2025, 4, 29),
        account_currency="BRL",
        nominal_equity=100_000.0,
        intraday_only=True,
    ),
}

MARKET_TO_BOOK: dict[str, str] = {m: b.name for b in BOOKS.values() for m in b.markets}


def get_book(name: str) -> Book:
    try:
        return BOOKS[name.upper()]
    except KeyError:
        raise KeyError(f"unknown book {name!r}; expected one of {sorted(BOOKS)}") from None
