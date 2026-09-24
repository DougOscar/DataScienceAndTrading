"""Helper for p20: things a strategy module can do at IMPORT time, before any audit starts."""
from polars import scan_parquet as bound_scan_parquet            # reader bound at import time
from pyarrow.parquet import read_table as bound_read_table        # idem (pyarrow)
from quantlab.data import load_bars as bound_load_bars            # guarded internally now
from quantlab import data as _qd

# module-level cache populated at import time (e.g. a "context" D1 frame for the strategy)
IMPORT_TIME_D1 = _qd.load_bars("EURUSD", "D1", start="2018-12-01")
