"""Probe 20 (round 2): try to escape the new look-ahead sandbox (AUDIT_IN_PROGRESS contextvar + patched
polars/pyarrow readers + factory=). Each strategy reads the full dev-period D1 bars some way and joins
today's D1 direction onto every H1 bar (a 96.8%-win leak). PASS of the auditor == escape."""
import sys, os, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from dataclasses import dataclass
from typing import ClassVar
import polars as pl
import leaky_imports_r2 as LI
from quantlab import data
from quantlab.contracts import Params, RiskType
from quantlab.testing import assert_no_lookahead

bars = data.load_bars("EURUSD", "H1", start="2019-01-01", end="2019-02-15")
FILE = data.catalog().filter((pl.col("symbol") == "EURUSD") & (pl.col("timeframe") == "M1"))["file"][0]
LO, HI = pl.datetime(2018, 12, 1), pl.datetime(2019, 3, 1)

@dataclass(frozen=True)
class P(Params):
    pass

def d1_from_m1(m1):
    return m1.sort("ts").group_by_dynamic("ts", every="1d").agg(pl.col("open").first(), pl.col("close").last())

def join(b, d1):
    d1 = d1.select(pl.col("ts").dt.date().alias("d"), (pl.col("close") > pl.col("open")).alias("up"))
    j = b.select(pl.col("ts").dt.date().alias("d")).join(d1, on="d", how="left", maintain_order="left")
    return j.select(pl.when(pl.col("up")).then(1).otherwise(-1).cast(pl.Int8).alias("signal"),
                    pl.lit(0.05).alias("stop_dist"), pl.lit(0.05).alias("target_dist"))

def reader_bound_polars():
    return d1_from_m1(LI.bound_scan_parquet(FILE).filter(pl.col("ts").is_between(LO, HI)).collect())
def reader_bound_pyarrow():
    return d1_from_m1(pl.from_arrow(LI.bound_read_table(FILE, filters=[("ts", ">=", pl.Series([0]).cast(pl.Datetime("ms")).dt.offset_by("1d").item().replace(year=2018, month=12, day=1))])).select("ts", "open", "close"))
def reader_parquetfile():
    import pyarrow.parquet as pq
    t = pq.ParquetFile(FILE).read(columns=["ts", "open", "close"])
    return d1_from_m1(pl.from_arrow(t).filter(pl.col("ts").is_between(LO, HI)))
def reader_dataset():
    import pyarrow.dataset as ds
    t = ds.dataset(FILE).to_table(columns=["ts", "open", "close"])
    return d1_from_m1(pl.from_arrow(t).filter(pl.col("ts").is_between(LO, HI)))
def reader_pandas():
    import pandas as pd
    return d1_from_m1(pl.from_pandas(pd.read_parquet(FILE, columns=["ts", "open", "close"])).filter(pl.col("ts").is_between(LO, HI)))
def reader_open_bytes():
    import io, pyarrow.parquet as pq
    with open(FILE, "rb") as fh:
        buf = io.BytesIO(fh.read())
    return d1_from_m1(pl.from_arrow(pq.ParquetFile(buf).read(columns=["ts", "open", "close"])).filter(pl.col("ts").is_between(LO, HI)))
def reader_polars_submodule():
    from polars.io.parquet.functions import scan_parquet as sp
    return d1_from_m1(sp(FILE).filter(pl.col("ts").is_between(LO, HI)).collect())
def reader_bound_load_bars():
    return LI.bound_load_bars("EURUSD", "D1", start="2018-12-01")
def reader_reload():
    import importlib, quantlab.data as qd
    qd2 = importlib.reload(qd)
    return qd2.load_bars("EURUSD", "D1", start="2018-12-01")
def reader_thread():
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(data.load_bars, "EURUSD", "D1", start="2018-12-01").result()
def reader_raw_thread():
    import threading
    box = {}
    th = threading.Thread(target=lambda: box.update(d=data.load_bars("EURUSD", "D1", start="2018-12-01")))
    th.start(); th.join()
    return box["d"]
def reader_subprocess():
    import subprocess, io
    code = ("import polars as pl, sys; from quantlab import data; "
            "data.load_bars('EURUSD','D1',start='2018-12-01').write_ipc(sys.stdout.buffer)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, check=True,
                         env={**os.environ, "PYTHONPATH": "."}).stdout
    return pl.read_ipc(io.BytesIO(out))
def reader_import_time_cache():
    return LI.IMPORT_TIME_D1
def reader_cache_dir():
    from quantlab import config
    import glob, pyarrow as pa
    f = sorted(glob.glob(str(config.CACHE_DIR / "EURUSD_M1*_D1_*.parquet")))[0]
    return pl.read_ipc(pa.BufferReader(b"")) if False else pl.from_arrow(__import__("pyarrow.parquet", fromlist=["x"]).ParquetFile(f).read())

READERS = [reader_bound_polars, reader_parquetfile, reader_dataset, reader_pandas, reader_open_bytes,
           reader_polars_submodule, reader_bound_load_bars, reader_reload, reader_thread, reader_raw_thread,
           reader_subprocess, reader_import_time_cache, reader_cache_dir]

class Leaky:
    name: ClassVar[str] = "leaky"; risk_type: ClassVar[RiskType] = RiskType.C; params = P()
    def __init__(self, reader, in_init):
        self.reader, self.cached = reader, (reader() if in_init else None)
    def signals(self, b):
        return join(b, self.cached if self.cached is not None else self.reader())

if len(sys.argv) > 1:          # run a subset in a FRESH process (a crashed audit can poison process state)
    READERS = [r for r in READERS if r.__name__ in sys.argv[1:]]
for rd in READERS:
    for in_init in (False, True):
        t0 = time.time()
        try:
            assert_no_lookahead(factory=lambda rd=rd, ii=in_init: Leaky(rd, ii), bars=bars, n_checks=5)
            verdict = "ESCAPED (auditor passed)"
        except AssertionError as e:
            verdict = "caught: " + str(e).split(":")[0][:60] + " | " + str(e)[str(e).find(":")+1:][:70].strip()
        except Exception as e:
            verdict = f"errored {type(e).__name__}: {str(e)[:80]}"
        where = "__init__ (factory=)" if in_init else "signals()"
        print(f"{rd.__name__:26s} in {where:20s} -> {verdict}  [{time.time()-t0:.0f}s]", flush=True)

# process-state check: are the polars/pyarrow readers restored after the audits above?
import pyarrow.parquet as _pq
print("after audits: pl.scan_parquet restored =", getattr(pl.scan_parquet, "__name__", "") != "_raise_raw_read",
      "| pq.read_table restored =", getattr(_pq.read_table, "__name__", "") != "_raise_raw_read")
if "--skip-prebuilt" in sys.argv:
    sys.exit(0)
# also: is factory= optional? a pre-built instance with data fetched in __init__
try:
    assert_no_lookahead(Leaky(reader_bound_load_bars, True), bars, n_checks=5)
    print("pre-built instance (strategy=, data fetched in __init__ before the audit) -> ESCAPED (auditor passed)")
except AssertionError as e:
    print("pre-built instance -> caught", str(e)[:80])
# sanity: after all this, is the reloaded data module still guarded?
import quantlab.data as qd
print("AUDIT_IN_PROGRESS object id stable after reload test:", qd.AUDIT_IN_PROGRESS is data.AUDIT_IN_PROGRESS)
