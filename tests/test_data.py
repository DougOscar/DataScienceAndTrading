"""quantlab.data: catalog, resampling, the holdout guard, timezones, currencies."""

from __future__ import annotations

import dataclasses
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from quantlab import config, contracts, data, ledger
from tests.conftest import catalog_row, write_m1_parquet

FBS = config.BOOKS["FBS"]
B3 = config.BOOKS["B3"]


# --------------------------------------------------------------------------- catalog
def test_catalog_is_bars_only_by_default_with_absolute_paths():
    cat = data.catalog()
    assert set(cat["kind"].unique().to_list()) == {"bars"}
    assert cat.filter(pl.col("symbol") == "WDO").height == 2   # DESIGN: two M1 files for WDO
    for file in cat["file"].to_list():
        assert Path(file).is_absolute()
        assert Path(file).exists()


def test_catalog_include_ticks_and_book_filter():
    with_ticks = data.catalog(include_ticks=True)
    assert with_ticks.filter(pl.col("kind") == "ticks").height == 2

    fbs_only = data.catalog(book="FBS")
    assert set(fbs_only["market"].unique().to_list()) <= {"forex", "crypto"}
    b3_only = data.catalog(book="B3")
    assert set(b3_only["market"].unique().to_list()) == {"b3"}


def test_wdo_default_picks_latest_end_and_source_file_overrides():
    cat = data.catalog()
    default_row = data._resolve_source(cat, "WDO", None)
    both = cat.filter(pl.col("symbol") == "WDO").sort("end")
    earliest_file, latest_file = both["file"][0], both["file"][1]
    assert default_row["file"] == latest_file

    overridden = data._resolve_source(cat, "WDO", earliest_file)
    assert overridden["file"] == earliest_file


# --------------------------------------------------------------------------- holdout guard
def test_default_end_stops_strictly_before_holdout():
    bars = data.load_bars("EURUSD", "H1")
    assert bars.height > 0
    assert bars["ts"].max() < FBS.holdout_start
    assert bars["ts"].max() == FBS.holdout_start - timedelta(hours=1)   # the very last dev H1 bar


def test_explicit_end_past_holdout_raises_without_unlock():
    with pytest.raises(contracts.HoldoutLocked):
        data.load_bars("EURUSD", "H1", end="2025-06-01")


def test_include_holdout_without_a_recorded_unlock_still_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path)
    with pytest.raises(contracts.HoldoutLocked):
        data.load_bars("EURUSD", "H1", end="2025-06-01", include_holdout=True, system="unlocked_test_system")


def test_unlock_recorded_in_ledger_grants_holdout_access(monkeypatch, tmp_path, clean_git_tree):
    monkeypatch.setattr(config, "LEDGER_DIR", tmp_path)
    ledger.create_study(book="FBS", system="eurusd_probe", issue=1, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="fbs-v0",
                        cv_scheme="CPCV(10,2)", study_id="fbs-0001-a1")

    with pytest.raises(contracts.HoldoutLocked):
        data.load_bars("EURUSD", "H1", end="2025-06-01", include_holdout=True, system="eurusd_probe")

    ledger.record_holdout_unlock(book="FBS", system="eurusd_probe", study_id="fbs-0001-a1",
                                 pass_band={"sharpe_p10": 0.1})
    bars = data.load_bars("EURUSD", "H1", end="2025-06-01", include_holdout=True, system="eurusd_probe")
    assert bars["ts"].max() >= FBS.holdout_start
    assert bars["ts"].max() < datetime(2025, 6, 1)


def test_book_mismatch_raises():
    with pytest.raises(ValueError, match="book mismatch"):
        data.load_bars("EURUSD", "H1", book="B3")


def test_source_file_must_belong_to_the_requested_symbol():
    """Minor 6: a source_file for a different symbol must raise, not silently mislabel data."""
    xau_m1 = data.catalog().filter(
        (pl.col("symbol") == "XAUUSD") & (pl.col("timeframe") == "M1")
    ).row(0, named=True)["file"]
    with pytest.raises(ValueError, match="XAUUSD"):
        data.load_bars("EURUSD", "D1", start="2025-05-01", source_file=xau_m1)


def test_cannot_upsample_past_the_native_timeframe():
    h1_file = data.catalog().filter(
        (pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "H1")
    ).row(0, named=True)["file"]
    with pytest.raises(ValueError, match="cannot resample"):
        data.load_bars("EURUSD", "M1", source_file=h1_file)


# --------------------------------------------------------------------------- resampling: known answers
@pytest.fixture
def known_answer_bars(tmp_path, install_catalog):
    """5 sparse M1 rows on one calendar day; hand-computable H1/H4/D1 aggregates."""
    rows = [
        (datetime(2021, 1, 4, 0, 0), 1.00, 1.05, 0.95, 1.02, 100, 0, 10),
        (datetime(2021, 1, 4, 0, 30), 1.02, 1.10, 1.00, 1.05, 150, 0, 8),
        (datetime(2021, 1, 4, 1, 15), 1.05, 1.06, 1.01, 1.03, 80, 0, 12),
        (datetime(2021, 1, 4, 5, 0), 2.00, 2.10, 1.90, 2.05, 200, 0, 5),
        (datetime(2021, 1, 4, 5, 45), 2.05, 2.20, 2.00, 2.15, 90, 0, 7),
    ]
    path = write_m1_parquet(tmp_path / "TESTX_M1.parquet", rows)
    install_catalog([catalog_row(file=path, symbol="TESTX", start=datetime(2021, 1, 4),
                                 end=datetime(2021, 1, 4, 6), rows=len(rows))])
    return path


def test_m1_passthrough_spread_max_equals_spread(known_answer_bars):
    bars = data.load_bars("TESTX", "M1", book="FBS")
    assert bars.height == 5
    assert (bars["spread"] == bars["spread_max"]).all()
    assert bars["spread"].to_list() == [10.0, 8.0, 12.0, 5.0, 7.0]


def test_h1_known_answer(known_answer_bars):
    h1 = data.load_bars("TESTX", "H1", book="FBS").sort("ts")
    assert h1["ts"].to_list() == [datetime(2021, 1, 4, 0), datetime(2021, 1, 4, 1), datetime(2021, 1, 4, 5)]
    bucket0 = h1.row(0, named=True)
    assert bucket0["open"] == 1.00 and bucket0["high"] == 1.10 and bucket0["low"] == 0.95
    assert bucket0["close"] == 1.05 and bucket0["spread"] == 10.0 and bucket0["spread_max"] == 10.0
    assert bucket0["tick_vol"] == 250
    bucket1 = h1.row(1, named=True)
    assert (bucket1["open"], bucket1["high"], bucket1["low"], bucket1["close"]) == (1.05, 1.06, 1.01, 1.03)
    assert bucket1["spread"] == 12.0 and bucket1["spread_max"] == 12.0 and bucket1["tick_vol"] == 80
    bucket2 = h1.row(2, named=True)
    assert bucket2["open"] == 2.00 and bucket2["high"] == 2.20 and bucket2["low"] == 1.90
    assert bucket2["close"] == 2.15 and bucket2["spread"] == 5.0 and bucket2["spread_max"] == 7.0
    assert bucket2["tick_vol"] == 290


def test_h4_known_answer_aligned_to_server_midnight(known_answer_bars):
    h4 = data.load_bars("TESTX", "H4", book="FBS").sort("ts")
    assert h4["ts"].to_list() == [datetime(2021, 1, 4, 0), datetime(2021, 1, 4, 4)]
    b0 = h4.row(0, named=True)
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (1.00, 1.10, 0.95, 1.03)
    assert b0["spread"] == 10.0 and b0["spread_max"] == 12.0 and b0["tick_vol"] == 330
    b1 = h4.row(1, named=True)
    assert (b1["open"], b1["high"], b1["low"], b1["close"]) == (2.00, 2.20, 1.90, 2.15)
    assert b1["spread"] == 5.0 and b1["spread_max"] == 7.0 and b1["tick_vol"] == 290


def test_d1_known_answer(known_answer_bars):
    d1 = data.load_bars("TESTX", "D1", book="FBS")
    assert d1.height == 1
    row = d1.row(0, named=True)
    assert row["ts"] == datetime(2021, 1, 4)
    assert (row["open"], row["high"], row["low"], row["close"]) == (1.00, 2.20, 0.95, 2.15)
    assert row["spread"] == 10.0 and row["spread_max"] == 12.0 and row["tick_vol"] == 100 + 150 + 80 + 200 + 90


# --------------------------------------------------------------------------- resampling: straddling bar
@pytest.fixture
def straddling_bars(tmp_path, install_catalog, monkeypatch):
    """One dev-only H4 bucket, and a second H4 bucket that straddles a mid-day holdout cutoff."""
    rows = [
        (datetime(2021, 1, 4, 0, 0), 1.00, 1.05, 0.95, 1.02, 100, 0, 10),   # H4 bucket [00:00,04:00), all dev
        (datetime(2021, 1, 4, 1, 0), 1.02, 1.07, 1.00, 1.04, 120, 0, 9),
        (datetime(2021, 1, 4, 4, 0), 1.10, 1.15, 1.05, 1.12, 200, 0, 5),    # H4 bucket [04:00,08:00): dev half
        (datetime(2021, 1, 4, 7, 0), 9.99, 9.99, 9.99, 9.99, 999, 0, 1),    # ... holdout half (contaminant)
        (datetime(2021, 1, 4, 9, 0), 8.88, 8.88, 8.88, 8.88, 111, 0, 2),    # fully holdout
    ]
    path = write_m1_parquet(tmp_path / "TESTY_M1.parquet", rows)
    install_catalog([catalog_row(file=path, symbol="TESTY", start=datetime(2021, 1, 4),
                                 end=datetime(2021, 1, 4, 9, 1), rows=len(rows))])
    custom_fbs = dataclasses.replace(FBS, holdout_start=datetime(2021, 1, 4, 6, 30))
    monkeypatch.setitem(config.BOOKS, "FBS", custom_fbs)
    return path


def _no_bar_reaches_holdout(bars: pl.DataFrame, span: timedelta, holdout_start: datetime) -> bool:
    return bool(((bars["ts"] + span) <= holdout_start).all())


def test_straddling_h4_bar_is_dropped_entirely(straddling_bars):
    h4 = data.load_bars("TESTY", "H4", book="FBS")
    assert h4["ts"].to_list() == [datetime(2021, 1, 4, 0)]        # the 04:00 bucket (dev+holdout) is gone
    assert _no_bar_reaches_holdout(h4, timedelta(hours=4), datetime(2021, 1, 4, 6, 30))


def test_h1_keeps_the_dev_half_of_what_h4_must_drop(straddling_bars):
    # Finer timeframes lose less near-boundary dev data than coarser ones, for the same cutoff.
    h1 = data.load_bars("TESTY", "H1", book="FBS")
    assert h1["ts"].to_list() == [datetime(2021, 1, 4, 0), datetime(2021, 1, 4, 1), datetime(2021, 1, 4, 4)]
    assert _no_bar_reaches_holdout(h1, timedelta(hours=1), datetime(2021, 1, 4, 6, 30))


def test_d1_drops_the_whole_day_when_the_cutoff_is_mid_day(straddling_bars):
    d1 = data.load_bars("TESTY", "D1", book="FBS")
    assert d1.height == 0


# --------------------------------------------------------------------------- real-data properties
def test_d1_real_eurusd_is_weekdays_only_2022():
    d1 = data.load_bars("EURUSD", "D1")
    year = d1.filter(pl.col("ts").dt.year() == 2022)
    assert year.height > 200
    weekdays = set(year["ts"].dt.weekday().unique().to_list())
    assert weekdays <= {1, 2, 3, 4, 5}                # ISO: 1=Mon .. 7=Sun, no weekend bars


def test_h1_resample_performance_cold_then_warm():
    cache = config.CACHE_DIR
    for f in cache.glob("*"):
        f.unlink()
    t0 = time.perf_counter()
    cold = data.load_bars("EURUSD", "H1")
    cold_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    warm = data.load_bars("EURUSD", "H1")
    warm_s = time.perf_counter() - t0
    assert cold.equals(warm)
    assert cold_s < 3.0, f"cold H1 resample took {cold_s:.2f}s (budget: 3s, DESIGN §7)"
    assert warm_s < 0.5, f"warm H1 resample took {warm_s:.2f}s (budget: 0.5s, DESIGN §7)"


# --------------------------------------------------------------------------- timezone / DST
def test_known_dst_dates_map_correctly_in_europe_helsinki():
    naive = pl.DataFrame({"ts": [
        datetime(2023, 3, 26, 1, 30),   # before the spring gap: EET, +2
        datetime(2023, 3, 26, 3, 30),   # inside the spring gap (non-existent): shift-forward rule
        datetime(2023, 3, 26, 5, 0),    # after the gap: EEST, +3
        datetime(2023, 10, 29, 1, 30),  # before the autumn ambiguity: EEST, +3
        datetime(2023, 10, 29, 3, 30),  # inside the repeated hour (ambiguous): resolved "earliest" -> +3
        datetime(2023, 10, 29, 5, 0),   # after the repeated hour: EET, +2
    ]}).with_columns(pl.col("ts").cast(pl.Datetime("ms")))
    out = naive.with_columns(data._localize_expr("ts", "Europe/Helsinki").alias("ts_utc"))["ts_utc"]
    expected = [
        datetime(2023, 3, 25, 23, 30, tzinfo=timezone.utc),
        datetime(2023, 3, 26, 1, 30, tzinfo=timezone.utc),
        datetime(2023, 3, 26, 2, 0, tzinfo=timezone.utc),
        datetime(2023, 10, 28, 22, 30, tzinfo=timezone.utc),
        datetime(2023, 10, 29, 0, 30, tzinfo=timezone.utc),
        datetime(2023, 10, 29, 3, 0, tzinfo=timezone.utc),
    ]
    assert out.to_list() == expected
    assert out.is_sorted()                                # no crash, no backward step, even across the gap


def test_crypto_window_across_fall_back_is_strictly_increasing():
    bars = data.load_bars("BTCUSD", "M1", start="2023-10-28", end="2023-10-30")
    diffs = bars["ts_utc"].diff().drop_nulls()
    assert (diffs.cast(pl.Int64) > 0).all()               # strictly increasing, never a dup/negative step


def test_crypto_window_across_spring_forward_does_not_crash():
    bars = data.load_bars("BTCUSD", "M1", start="2023-03-25", end="2023-03-27")
    assert bars.height > 0
    diffs = bars["ts_utc"].diff().drop_nulls()
    assert (diffs.cast(pl.Int64) > 0).all()


def _last_sunday(year: int, month: int) -> datetime:
    next_month = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
    last_day = next_month - timedelta(days=1)
    last_sunday = last_day - timedelta(days=(last_day.weekday() - 6) % 7)   # Mon=0..Sun=6
    return datetime(last_sunday.year, last_sunday.month, last_sunday.day)


@pytest.mark.parametrize("symbol", ["BTCUSD", "ETHUSD", "LTCUSD"])
def test_crypto_ts_utc_strictly_increasing_across_every_dst_transition_weekend(symbol):
    """Minor 3: root cause was the spring-forward "shift the naive clock forward" rule
    colliding with real crypto bars already sitting in the skipped hour (see the module
    docstring's DST section and research/audits/probes/p15_crypto_dst_ts_utc.py). Checks
    every EU DST transition weekend across the symbol's own dev-period history, kept fast
    by only loading the one transition day each time (not the whole multi-year range).
    """
    start_year = data.catalog().filter(pl.col("symbol") == symbol).row(0, named=True)["start"].year
    holdout_start = FBS.holdout_start
    windows_checked = 0
    for year in range(start_year, holdout_start.year + 1):
        for month in (3, 10):          # EU spring-forward (Mar) / fall-back (Oct)
            day_start = _last_sunday(year, month)
            if day_start >= holdout_start:
                continue
            day_end = min(day_start + timedelta(days=1), holdout_start)
            bars = data.load_bars(symbol, "M1", start=day_start, end=day_end)
            if bars.height < 2:
                continue
            diffs = bars["ts_utc"].diff().drop_nulls()
            assert (diffs.cast(pl.Int64) > 0).all(), f"{symbol} {year}-{month:02d}: ts_utc not increasing"
            assert bars["ts_utc"].is_duplicated().sum() == 0, f"{symbol} {year}-{month:02d}: duplicate ts_utc"
            windows_checked += 1
    assert windows_checked > 0


# --------------------------------------------------------------------------- point size
@pytest.mark.parametrize("symbol, expected_point", [
    ("EURUSD", 1e-5), ("USDJPY", 1e-3), ("XAUUSD", 1e-2),
])
def test_infer_point(symbol, expected_point):
    assert data.infer_point(symbol) == pytest.approx(expected_point)


def test_infer_point_only_reads_dev_period_data(tmp_path, install_catalog, monkeypatch):
    """Minor 5: infer_point must not leak precision information from the holdout tail.

    Dev rows are coarse (2 decimals); holdout rows (appended after the cutoff) are finer
    (5 decimals). The old implementation read the raw file's head+tail with no guard, so
    with a small `sample` its tail was entirely holdout rows and it reported the *finer*
    (holdout) precision instead of the true dev-period one.
    """
    cutoff = datetime(2021, 1, 5)
    dev_rows = [
        (datetime(2021, 1, 4, 0, m), 100.12, 100.12, 100.12, 100.12, 10, 0, 1) for m in range(5)
    ]
    holdout_rows = [
        (datetime(2021, 1, 5, 0, m), 100.12345, 100.12345, 100.12345, 100.12345, 10, 0, 1)
        for m in range(50)
    ]
    path = write_m1_parquet(tmp_path / "TESTPT_M1.parquet", dev_rows + holdout_rows)
    install_catalog([catalog_row(file=path, symbol="TESTPT", start=datetime(2021, 1, 4),
                                 end=datetime(2021, 1, 5, 1), rows=len(dev_rows) + len(holdout_rows))])
    monkeypatch.setitem(config.BOOKS, "FBS", dataclasses.replace(FBS, holdout_start=cutoff))

    assert data.infer_point("TESTPT", sample=5) == pytest.approx(1e-2)


# --------------------------------------------------------------------------- resample cache (Minor 8)
def test_resample_cache_key_changes_with_the_schema_version(monkeypatch, tmp_path):
    path = write_m1_parquet(tmp_path / "TESTV_M1.parquet", [
        (datetime(2021, 1, 4, 0, 0), 1.0, 1.0, 1.0, 1.0, 10, 0, 1),
    ])
    before = data._cache_file(path, "H1")
    monkeypatch.setattr(data, "_RESAMPLE_SCHEMA_VERSION", "some-other-schema-version")
    after = data._cache_file(path, "H1")
    assert before != after


def test_resample_cache_write_is_atomic_and_leaves_no_temp_files(known_answer_bars):
    data.load_bars("TESTX", "H1", book="FBS")   # triggers a real resample + cache write
    cache_dir = config.CACHE_DIR
    assert list(cache_dir.glob("*.tmp")) == []
    assert len(list(cache_dir.glob("*.parquet"))) >= 1
    # a second call must hit the now-cached file and return identical data
    again = data.load_bars("TESTX", "H1", book="FBS")
    assert data.load_bars("TESTX", "H1", book="FBS").equals(again)


# --------------------------------------------------------------------------- calendar anomalies
def test_calendar_anomalies_finds_the_known_eurusd_sunday_minutes():
    anomalies = data.calendar_anomalies("EURUSD")
    assert anomalies.height == 6
    assert int(anomalies["n_minutes"].sum()) == 220
    assert set(anomalies["date"].to_list()) == {
        date(2016, 5, 22), date(2016, 5, 29), date(2016, 6, 5),
        date(2016, 6, 12), date(2016, 6, 19), date(2024, 10, 27),
    }


def test_calendar_anomalies_empty_but_typed_for_a_clean_symbol(known_answer_bars):
    anomalies = data.calendar_anomalies("TESTX")
    assert anomalies.height == 0
    assert anomalies.schema["date"] == pl.Date
    assert anomalies.schema["n_minutes"] == pl.UInt32


def test_d1_resample_keeps_the_tiny_sunday_bar_that_calendar_anomalies_flags():
    """DESIGN: stay faithful to the raw feed -- anomalous bars are exposed, not dropped."""
    a_date = data.calendar_anomalies("EURUSD").sort("date")["date"][0]
    day_start = datetime(a_date.year, a_date.month, a_date.day)
    d1 = data.load_bars("EURUSD", "D1", start=day_start, end=day_start + timedelta(days=1))
    assert d1.height == 1
    assert d1["ts"][0] == day_start
    assert d1["ts"][0].isoweekday() == 7                 # a genuine (tiny) Sunday D1 bar


# --------------------------------------------------------------------------- currencies
@pytest.mark.parametrize("symbol, expected", [
    ("EURUSD", ("EUR", "USD")), ("USDJPY", ("USD", "JPY")), ("XAUUSD", ("XAU", "USD")),
    ("Palladium", ("XPD", "USD")), ("Platinum", ("XPT", "USD")), ("BTCUSD", ("BTC", "USD")),
])
def test_symbol_currencies_fx_and_crypto(symbol, expected):
    assert data.symbol_currencies(symbol) == expected


def test_symbol_currencies_b3_quotes_in_brl():
    assert data.symbol_currencies("PETR4") == ("PETR4", "BRL")


def test_conversion_rate_same_currency_is_one():
    ts = pl.Series([datetime(2020, 1, 10, 12, 0, tzinfo=timezone.utc)])
    assert data.conversion_rate("EUR", "EUR", ts).to_list() == [1.0]


def test_conversion_rate_empty_input():
    ts = pl.Series([], dtype=pl.Datetime("ms", "UTC"))
    out = data.conversion_rate("JPY", "USD", ts)
    assert len(out) == 0


def test_conversion_rate_raises_on_naive_timestamps():
    """M3: a naive series must never be silently treated as UTC (that shifts every lookup
    2-3h into the future -- a look-ahead through this exact interface, per the audit)."""
    ts_naive = pl.Series([datetime(2020, 1, 10, 12, 0)]).cast(pl.Datetime("ms"))
    with pytest.raises(ValueError, match="timezone-aware"):
        data.conversion_rate("JPY", "USD", ts_naive)


def test_conversion_rate_does_not_spuriously_lock_a_dev_timestamp_near_the_cutoff():
    """Minor 7: the internal M1 lookup window used to over-fetch by 1 minute, which could
    push it past the holdout cutoff for a perfectly legitimate dev-period timestamp."""
    # 2025-05-14 23:59:30 FBS server time (Europe/Helsinki, EEST/+3) = 20:59:30 UTC,
    # 30 seconds before the FBS holdout cutoff (2025-05-15 00:00:00 server).
    ts_edge = pl.Series([datetime(2025, 5, 14, 20, 59, 30, tzinfo=timezone.utc)])
    rate = data.conversion_rate("EUR", "USD", ts_edge)   # must not raise HoldoutLocked
    assert len(rate) == 1 and rate[0] > 0


def test_conversion_rate_direct_pair():
    ts = pl.Series([datetime(2020, 1, 10, 12, 0, tzinfo=timezone.utc)])
    rate = data.conversion_rate("EUR", "JPY", ts)
    # M3: the as-of match is on each M1 bar's CLOSE time (ts_utc + 1 minute), not its open
    # -- a bar whose window still contains `ts` hasn't closed yet, so its close isn't known.
    eurjpy = data.load_bars("EURJPY", "M1").filter((pl.col("ts_utc") + timedelta(minutes=1)) <= ts[0]).tail(1)["close"][0]
    assert rate.to_list() == [eurjpy]


def test_conversion_rate_inverse_pair_matches_manual_1_over_close():
    ts = pl.Series([datetime(2020, 1, 10, 12, 0, tzinfo=timezone.utc)])
    rate = data.conversion_rate("JPY", "USD", ts)
    usdjpy_close = (
        data.load_bars("USDJPY", "M1").filter((pl.col("ts_utc") + timedelta(minutes=1)) <= ts[0]).tail(1)["close"][0]
    )
    assert rate.to_list() == pytest.approx([1.0 / usdjpy_close])


def test_conversion_rate_triangulates_through_usd():
    ts = pl.Series([datetime(2020, 1, 10, 12, 0, tzinfo=timezone.utc)])
    direct = data.conversion_rate("BTC", "JPY", ts)[0]
    btc_usd = data.conversion_rate("BTC", "USD", ts)[0]
    usd_jpy = data.conversion_rate("USD", "JPY", ts)[0]
    assert direct == pytest.approx(btc_usd * usd_jpy)


def test_conversion_rate_is_backward_as_of_never_looks_ahead():
    """M3: at an instant strictly inside bar k's own minute, only bar k-1 has actually
    closed -- the rate must come from *its* close, never bar k's own (not yet known)."""
    m1 = data.load_bars("USDJPY", "M1")
    bar = m1.row(1000, named=True)
    prev_close = m1.row(999, named=True)["close"]
    just_after_open = bar["ts_utc"] + timedelta(seconds=30)     # strictly inside bar 1000's minute
    rate = data.conversion_rate("JPY", "USD", pl.Series([just_after_open]))
    assert rate.to_list() == pytest.approx([1.0 / prev_close])
    assert rate.to_list() != pytest.approx([1.0 / bar["close"]])   # not the still-forming bar's close


def test_conversion_rate_at_exact_bar_close_uses_that_bar_not_the_next():
    """The boundary is inclusive: an instant exactly at bar k's close already knows it."""
    m1 = data.load_bars("USDJPY", "M1")
    bar = m1.row(1000, named=True)
    at_close = bar["ts_utc"] + timedelta(minutes=1)
    rate = data.conversion_rate("JPY", "USD", pl.Series([at_close]))
    assert rate.to_list() == pytest.approx([1.0 / bar["close"]])


def test_conversion_rate_respects_the_holdout_guard():
    ts = pl.Series([datetime(2025, 8, 1, 10, 0, tzinfo=timezone.utc)])
    with pytest.raises(contracts.HoldoutLocked):
        data.conversion_rate("JPY", "USD", ts)
