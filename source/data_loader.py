"""Load OHLC CSV files that follow the `<ASSET>_<TF>_<START>_<END>.csv` scheme."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


FILENAME_RE = re.compile(
    r"^(?P<asset>[A-Za-z0-9]+)"
    r"_(?P<timeframe>[A-Za-z0-9]+)"
    r"_(?P<start>\d{12})"
    r"_(?P<end>\d{12})\.csv$"
)

_OHLC_COLS = ("open", "high", "low", "close")

COL_NAMES = {
    '<DATE>': 'date',
    '<TIME>': 'time',
    '<OPEN>': 'open',
    '<HIGH>': 'high',
    '<LOW>': 'low',
    '<CLOSE>': 'close',
    '<TICKVOL>': 'tick_vol',
    '<VOL>': 'volume',
    '<SPREAD>': 'spread'
}


@dataclass
class DatasetMeta:
    asset: str
    timeframe: str
    start: pd.Timestamp
    end: pd.Timestamp
    path: Path
    group: str = ""   # subdirectory name, e.g. "forex", "b3"


def _parse_filename(path: Path, group: str = "") -> DatasetMeta | None:
    m = FILENAME_RE.match(path.name)
    if m is None:
        return None
    return DatasetMeta(
        asset=m["asset"],
        timeframe=m["timeframe"],
        start=pd.to_datetime(m["start"], format="%Y%m%d%H%M"),
        end=pd.to_datetime(m["end"], format="%Y%m%d%H%M"),
        path=path,
        group=group,
    )


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    lowered = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for target in ("datetime", "date", "time", "timestamp",
                   "open", "high", "low", "close", "volume"):
        if target in lowered:
            rename[lowered[target]] = target
    df = df.rename(columns=COL_NAMES)
    if "datetime" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["date"].astype(str).str.strip() + " "
                + df["time"].astype(str).str.strip(),
                errors="coerce",
            )
            df.drop(["date", "time"], axis=1, inplace=True)
        elif "timestamp" in df.columns:
            df["datetime"] = pd.to_datetime(df["timestamp"], errors="coerce")
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            raise ValueError("No datetime column found (expected datetime/date+time/timestamp).")

    df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
    for col in _OHLC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = [c for c in _OHLC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {missing}")
    df = df.dropna(subset=list(_OHLC_COLS))
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV and return a clean OHLC(V) DataFrame indexed by datetime."""
    path = Path(path)
    last_err: Exception | None = None
    for sep in (",", ";", "\t", "|"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] >= 4:
                break
            else:
                print("Incorrect Sep...")
                continue
        except Exception as e:
            last_err = e
    if df.shape[1] >= 4:
        return _normalize(df)
    else:
        raise ValueError(f"Could not parse {path}: {last_err}")


def load_all(data_dir: str | Path = "data") -> Dict[Tuple[str, str], Tuple[DatasetMeta, pd.DataFrame]]:
    """Discover every CSV in `data_dir` (and subdirectories) matching the naming scheme.

    Root-level CSVs get ``group=""``.  CSVs inside a subdirectory get
    ``group=<subdirectory_name>`` (e.g. ``"forex"``, ``"b3"``).
    """
    data_dir = Path(data_dir)
    out: Dict[Tuple[str, str], Tuple[DatasetMeta, pd.DataFrame]] = {}
    if not data_dir.exists():
        return out

    def _scan(directory: Path, group: str) -> None:
        for path in sorted(directory.glob("*.csv")):
            meta = _parse_filename(path, group)
            if meta is None:
                continue
            df = load_csv(path)
            out[(meta.asset, meta.timeframe)] = (meta, df)

    _scan(data_dir, "")  # root-level files (no group)
    for subdir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        _scan(subdir, subdir.name)

    return out


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample an OHLC(V) frame to a coarser frequency (e.g. '1H', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "sum"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    if "tick_vol" in df.columns:
        agg["tick_vol"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=list(_OHLC_COLS))
