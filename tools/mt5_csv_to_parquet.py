"""Convert MetaTrader 5 CSV exports (bars or ticks) to zstd Parquet.

MT5 exports are tab-separated, CRLF, with ``<DATE>`` = ``yyyy.MM.dd`` and
``<TIME>`` = ``HH:MM:SS`` (bars) or ``HH:MM:SS.fff`` (ticks).  Timestamps are
kept as naive *broker server time* — timezone normalisation belongs to the
loader, once each broker's offset/DST rule is verified.

Output keeps the source stem (``EURUSD_M1_2016..._2026....parquet``) next to
the CSV, and a ``data/manifest.json`` entry per file.  Conversion is streamed
(``sink_parquet``) so multi-GB tick files never load fully into RAM.

Usage::

    python tools/mt5_csv_to_parquet.py data/forex/EURUSD_M1_*.csv ...
    python tools/mt5_csv_to_parquet.py --all --delete-verified
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MANIFEST = DATA / "manifest.json"

BAR_HEADER = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
TICK_HEADER = ["<DATE>", "<TIME>", "<BID>", "<ASK>", "<LAST>", "<VOLUME>", "<FLAGS>"]

BAR_SCHEMA = {
    "<DATE>": pl.String, "<TIME>": pl.String,
    "<OPEN>": pl.Float64, "<HIGH>": pl.Float64, "<LOW>": pl.Float64, "<CLOSE>": pl.Float64,
    "<TICKVOL>": pl.Int64, "<VOL>": pl.Int64, "<SPREAD>": pl.Int32,
}
TICK_SCHEMA = {
    "<DATE>": pl.String, "<TIME>": pl.String,
    "<BID>": pl.Float64, "<ASK>": pl.Float64, "<LAST>": pl.Float64,
    "<VOLUME>": pl.Float64, "<FLAGS>": pl.Int32,
}
BROKER_BY_MARKET = {"forex": "FBS", "crypto": "FBS", "b3": "Clear"}


def _header(path: Path) -> list[str]:
    with path.open("r", newline="") as fh:
        return fh.readline().rstrip("\r\n").split("\t")


def _count_data_lines(path: Path) -> int:
    """Independent row count (``wc -l`` minus header) to cross-check the parser."""
    out = subprocess.run(["wc", "-l", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.split()[0]) - 1


def _last_line(path: Path) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 4096))
        lines = [ln for ln in fh.read().decode().splitlines() if ln.strip()]
        return lines[-1].strip()


def _first_data_line(path: Path) -> str:
    with path.open("r", newline="") as fh:
        fh.readline()
        return fh.readline().strip()


def convert(path: Path) -> dict:
    header = _header(path)
    if header == BAR_HEADER:
        kind, schema, fmt = "bars", BAR_SCHEMA, "%Y.%m.%d %H:%M:%S"
        rename = {"<OPEN>": "open", "<HIGH>": "high", "<LOW>": "low", "<CLOSE>": "close",
                  "<TICKVOL>": "tick_vol", "<VOL>": "volume", "<SPREAD>": "spread"}
    elif header == TICK_HEADER:
        kind, schema, fmt = "ticks", TICK_SCHEMA, "%Y.%m.%d %H:%M:%S%.3f"
        rename = {"<BID>": "bid", "<ASK>": "ask", "<LAST>": "last",
                  "<VOLUME>": "volume", "<FLAGS>": "flags"}
    else:
        raise ValueError(f"{path.name}: unrecognised header {header}")

    out = path.with_suffix(".parquet")
    lf = (
        pl.scan_csv(path, separator="\t", schema=schema, has_header=True)
        .with_columns(
            (pl.col("<DATE>") + " " + pl.col("<TIME>"))
            .str.to_datetime(fmt, time_unit="ms", strict=True)
            .alias("ts")
        )
        .drop("<DATE>", "<TIME>")
        .rename(rename)
        .select("ts", *rename.values())
    )
    lf.sink_parquet(out, compression="zstd", compression_level=6, statistics=True)

    # --- verification -------------------------------------------------------
    pq = pl.scan_parquet(out)
    stats = pq.select(
        pl.len().alias("rows"),
        pl.col("ts").min().alias("start"),
        pl.col("ts").max().alias("end"),
        (pl.col("ts").diff() < pl.duration(milliseconds=0)).sum().alias("non_monotonic"),
        pl.col("ts").null_count().alias("null_ts"),
    ).collect().row(0, named=True)
    expected_rows = _count_data_lines(path)
    head = pq.head(1).collect().row(0)
    tail = pq.tail(1).collect().row(0)

    def _fmt_row(row) -> list[str]:
        return [row[0].strftime(fmt.replace("%.3f", "")), *[str(v) for v in row[1:]]]

    first_csv = _first_data_line(path).split("\t")
    last_csv = _last_line(path).split("\t")
    # Compare numerically (CSV "1.00000000" vs parquet 1.0); empty CSV fields ↔ None.
    def _same(csv_fields: list[str], row) -> bool:
        for raw, val in zip(csv_fields[2:], row[1:]):
            if raw == "":
                if val is not None:
                    return False
            elif val is None or abs(float(raw) - float(val)) > 1e-9 * max(1.0, abs(float(raw))):
                return False
        return True

    ok = (
        stats["rows"] == expected_rows
        and stats["null_ts"] == 0
        and _same(first_csv, head)
        and _same(last_csv, tail)
    )
    stem_parts = path.stem.split("_")
    entry = {
        "file": str(out.relative_to(ROOT)),
        "source_csv": path.name,
        "market": path.parent.name,
        "broker": BROKER_BY_MARKET.get(path.parent.name, "unknown"),
        "symbol": stem_parts[0],
        "timeframe": stem_parts[1] if kind == "bars" else "tick",
        "kind": kind,
        "rows": stats["rows"],
        "start": str(stats["start"]),
        "end": str(stats["end"]),
        "non_monotonic_steps": stats["non_monotonic"],
        "tz": "broker server time (naive, unverified)",
        "csv_bytes": path.stat().st_size,
        "parquet_bytes": out.stat().st_size,
        "verified": bool(ok),
    }
    if not ok:
        entry["verify_detail"] = {
            "expected_rows": expected_rows, "rows": stats["rows"],
            "first_csv": first_csv, "first_pq": _fmt_row(head),
            "last_csv": last_csv, "last_pq": _fmt_row(tail),
        }
    return entry


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="convert every CSV under data/<market>/")
    ap.add_argument("--delete-verified", action="store_true", help="remove the CSV once its parquet verifies")
    args = ap.parse_args()

    files = sorted(DATA.glob("*/*.csv")) if args.all else args.files
    # Smallest first: cheap files validate the pipeline before the 19 GB tick file.
    files = sorted(files, key=lambda p: p.stat().st_size)
    manifest = _load_manifest()
    failures = 0
    for f in files:
        f = f.resolve()
        try:
            entry = convert(f)
        except Exception as exc:  # keep going; report at the end
            failures += 1
            print(f"FAIL  {f.name}: {exc}", flush=True)
            continue
        manifest[entry["file"]] = entry
        MANIFEST.write_text(json.dumps(manifest, indent=2, default=str))
        ratio = entry["csv_bytes"] / max(entry["parquet_bytes"], 1)
        status = "OK  " if entry["verified"] else "BAD "
        print(f"{status} {f.name}: {entry['rows']:,} rows, {ratio:.1f}x smaller", flush=True)
        if entry["verified"] and args.delete_verified:
            f.unlink()
        elif not entry["verified"]:
            failures += 1
    print(f"done: {len(files) - failures} ok, {failures} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
