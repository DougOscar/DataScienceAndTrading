"""Shared fixtures for quantlab.data tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import polars as pl
import pytest

from quantlab import config, data

RAW_M1_SCHEMA = ["ts", "open", "high", "low", "close", "tick_vol", "volume", "spread"]


@pytest.fixture(autouse=True)
def isolated_cache_dir(monkeypatch, tmp_path: Path) -> Path:
    """Every test writes its resample cache under its own tmp dir, never into ``data/_quantlab_cache``."""
    cache_dir = tmp_path / "_quantlab_cache"
    monkeypatch.setattr(config, "CACHE_DIR", cache_dir)
    return cache_dir


@pytest.fixture
def clean_git_tree(monkeypatch):
    """Fake a clean, committed tree so ``ledger.record_holdout_unlock`` doesn't refuse on dirtiness."""
    monkeypatch.setattr("quantlab.ledger.git_commit", lambda: "abc1234")


def write_m1_parquet(path: Path, rows: list[tuple]) -> Path:
    """Write a hand-built raw M1 bars file (the on-disk schema ``data.py`` reads from)."""
    df = pl.DataFrame(rows, schema=RAW_M1_SCHEMA, orient="row")
    df = df.with_columns(pl.col("ts").cast(pl.Datetime("ms")), pl.col("spread").cast(pl.Int32))
    df.write_parquet(path)
    return path


@pytest.fixture
def install_catalog(monkeypatch) -> Callable[[list[dict[str, Any]]], pl.DataFrame]:
    """Replace ``data.catalog`` with a fixed, hand-built table for the duration of one test."""

    def _install(rows: list[dict[str, Any]]) -> pl.DataFrame:
        fixed = pl.DataFrame(rows)

        def _fake_catalog(book: str | None = None, *, include_ticks: bool = False) -> pl.DataFrame:
            out = fixed if include_ticks else fixed.filter(pl.col("kind") == "bars")
            if book is not None:
                out = out.filter(pl.col("book") == config.get_book(book).name)
            return out

        monkeypatch.setattr(data, "catalog", _fake_catalog)
        return fixed

    return _install


def catalog_row(*, file: str | Path, symbol: str, market: str = "forex", timeframe: str = "M1",
                 kind: str = "bars", start: datetime, end: datetime, rows: int) -> dict[str, Any]:
    """One :func:`data.catalog`-shaped row for :func:`install_catalog`."""
    return {
        "symbol": symbol, "market": market, "book": config.MARKET_TO_BOOK[market],
        "timeframe": timeframe, "kind": kind, "file": str(file), "start": start, "end": end, "rows": rows,
    }
