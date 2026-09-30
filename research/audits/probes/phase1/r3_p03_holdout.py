"""R3-P03 (30b23b6) -- holdout life cycle after R2-2 / R2-3 + round-3 side effects.  SYNTHETIC only:
config.MANIFEST_PATH -> a hand-written manifest (files never opened), data.catalog -> synthetic M1 files in
a temp dir, LEDGER_DIR / CACHE_DIR -> temp, ledger.git_commit patched.  No real holdout bar is read.

 H2   band re-roll at S5 (frozen?), n_boot / seed / symbols overrides.
 H2f  a band with the right seed / n_boot / symbols but DOCTORED thresholds: log_gates (first run) and
      register_holdout_band (newer data) -> unlock -> zero-edge holdout exam.
 H3   family keying: rename after FAIL (Probe/42, probe_v2/42, probe_v2/43 new issue, issue None),
      false positives (other book same issue; unrelated system sharing a generic issue).
 H5   load_bars scope: other symbol, impersonation by system string, PASS -> horizon.
 LOG  holdout_read volume / cost.
 H6   span tolerance now; H7 horizon with legs.
"""
import json, math, sys, tempfile, time, warnings
from datetime import date, datetime, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import polars as pl
from quantlab import config, data, ledger, gates as G, stats as S, contracts
from quantlab.ledger import LedgerError
from test_gates import make_study, register   # before the manifest patch (it loads the EURUSD spec)

TMP = Path(tempfile.mkdtemp(prefix="rt_r3_holdout_"))
config.CACHE_DIR = TMP / "cache"
ledger.git_commit = lambda: "abc1234"
LOCKED_END = datetime(2026, 5, 15, 10, 36)
man = TMP / "manifest.json"


def set_manifest(entries):
    man.write_text(json.dumps({f"data/forex/{s}_M1_DOES_NOT_EXIST.parquet": {
        "file": f"data/forex/{s}_M1_DOES_NOT_EXIST.parquet", "market": "forex", "broker": "x", "symbol": s,
        "timeframe": "M1", "kind": "bars", "rows": 1, "start": "2016-05-02 00:00:00", "end": str(e)}
        for s, e in entries.items()}))
    config.MANIFEST_PATH = man


class WfoStudy:
    def __init__(self, sr, n=1500, seed=3, sd=0.006):
        z = np.random.default_rng(seed).standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        self.wfo_oos = pl.DataFrame({"date": [date(2018, 1, 1) + timedelta(days=i) for i in range(n)],
                                     "ret": sr / math.sqrt(260) * sd + sd * z})
        self.cpcv_paths = None


def reg_band(sid, sr=2.8, symbols=("EURUSD",)):
    return S.holdout_band(WfoStudy(sr), data.holdout_horizon("FBS", list(symbols)), 260.0,
                          n_boot=S.HOLDOUT_BAND_N_BOOT, n_power=S.HOLDOUT_BAND_N_POWER, trades_per_day=0.4,
                          seed=S.holdout_band_seed(sid)).as_dict()


def new(ld, sid, system, issue, *, sr=2.8, symbols=("EURUSD",), book="FBS", band=True):
    ledger.create_study(ledger_dir=ld, study_id=sid, book=book, system=system, issue=issue, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t",
                        symbols=list(symbols), conversion_legs=[])
    if not band:
        return None
    b = reg_band(sid, sr, symbols)
    ledger.register_holdout_band(study_id=sid, band=b, reason="S5", ledger_dir=ld)
    return b


def unlock(ld, sid, system, b):
    return ledger.record_holdout_unlock(book="FBS", system=system, study_id=sid, pass_band=b,
                                        user_confirmation=ledger.unlock_phrase("FBS", system), ledger_dir=ld)


def attempt(label, fn):
    try:
        out = fn()
        print(f"  {label:70s} ACCEPTED{'' if out is None else ' -> ' + str(out)[:90]}")
        return out
    except Exception as e:
        print(f"  {label:70s} refused ({type(e).__name__}: {str(e)[:110]})")


set_manifest({"EURUSD": LOCKED_END, "GBPUSD": LOCKED_END})

print("== H2: band re-roll at S5")
study, trades_ = make_study(seed=4)
ld = register(study, TMP / "led_h2", symbols=["EURUSD"], conversion_legs=[])
bands = []
for i in range(3):
    rep = G.evaluate_gates(study, None, periods_per_year=260.0, selected_trades=trades_, ledger_dir=ld)
    G.log_gates(rep, ledger_dir=ld)
    bands.append(rep.holdout_band_run["sharpe_lo"])
reg = ledger.registered_holdout_band(study.study_id, ledger_dir=ld)
print(f"  3 gate runs: sharpe_lo per run {[round(x, 4) for x in bands]}; registered = first: "
      f"{reg['sharpe_lo'] == bands[0]}; events with holdout_band_diagnostic: "
      f"{sum('holdout_band_diagnostic' in e for e in ledger.study_events(study.study_id, 'gates', ledger_dir=ld))}")
attempt("evaluate_gates(n_boot=60)", lambda: G.evaluate_gates(study, None, periods_per_year=260.0,
        selected_trades=trades_, ledger_dir=ld, n_boot=60))
attempt("evaluate_gates(seed=7)", lambda: G.evaluate_gates(study, None, periods_per_year=260.0,
        selected_trades=trades_, ledger_dir=ld, seed=7))
attempt("rebuild_holdout_band(symbols=['GBPUSD'])", lambda: G.rebuild_holdout_band(study, periods_per_year=260.0,
        reason="x", symbols=["GBPUSD"], ledger_dir=ld))
attempt("evaluate_gates(holdout_symbols='GBPUSD') on an EURUSD study", lambda: G.evaluate_gates(
    study, None, periods_per_year=260.0, selected_trades=trades_, ledger_dir=ld, holdout_symbols="GBPUSD").holdout_band_run["symbols"])

print("== H2f: doctored band thresholds (right seed / n_boot / symbols)")
ld = TMP / "led_h2f"
sid = "fbs-0061-a1"
new(ld, sid, "forged", 61, band=False)
honest = reg_band(sid, sr=1.5)
forged = {**honest, "sharpe_lo": -9.0, "ret_at_budget_lo": -1.0, "max_dd_mag_hi": 1.0, "trades_lo": 0.0,
          "trades_hi": 1e9, "p_pass_zero_edge": 0.05}
print(f"  honest band: sharpe_lo {honest['sharpe_lo']:.3f}, p_pass_zero_edge {honest['p_pass_zero_edge']:.3f}; keys "
      f"{sorted(k for k in honest if k.endswith(('_lo', '_hi')))}")
attempt("register_holdout_band(doctored band) at S5", lambda: ledger.register_holdout_band(
    study_id=sid, band=forged, reason="S5", ledger_dir=ld)["event"])
attempt("record_holdout_unlock with the doctored band", lambda: unlock(ld, sid, "forged", forged)["exam"])
zero = np.random.default_rng(5).normal(0.0, 0.006, honest["horizon_days"])
res = attempt("run_holdout_exam on a ZERO-edge holdout", lambda: G.run_holdout_exam(
    book="FBS", system="forged", study_id=sid, holdout_daily=zero, n_trades=0.4 * honest["horizon_days"],
    periods_per_year=260.0, ledger_dir=ld)["status"])
chk = S.holdout_check(honest, zero, 0.4 * honest["horizon_days"], 260.0)
print(f"  same holdout against the honest band: {chk['status']}")

print("== H3: family keying")
ld = TMP / "led_h3"
b = new(ld, "fbs-0042-a1", "probe", 42)
unlock(ld, "fbs-0042-a1", "probe", b)
r = G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1",
                       holdout_daily=np.full(b["horizon_days"], -0.001), n_trades=100, periods_per_year=260.0,
                       ledger_dir=ld)
print("  probe/42 exam:", r["status"])
for sid, name, iss in (("fbs-0042-a2", "Probe", 42), ("fbs-0042-a3", "probe_v2", 42),
                       ("fbs-0043-a1", "probe_v2", 43), ("fbs-0044-a1", "probe2", None),
                       ("fbs-0045-a1", "probe_fast", 45)):
    def go(sid=sid, name=name, iss=iss):
        b2 = new(ld, sid, name, iss)
        return unlock(ld, sid, name, b2)["exam"]
    attempt(f"after FAIL: {name!r} issue {iss}", go)
ld = TMP / "led_h3b"            # clean ledger: rename + NEW issue only
b = new(ld, "fbs-0042-a1", "probe", 42)
unlock(ld, "fbs-0042-a1", "probe", b)
G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1", holdout_daily=np.full(b["horizon_days"], -0.001),
                   n_trades=100, periods_per_year=260.0, ledger_dir=ld)
def go2():
    b2 = new(ld, "fbs-0043-a1", "probe_v2", 43)
    return unlock(ld, "fbs-0043-a1", "probe_v2", b2)["exam"]
attempt("clean ledger, after FAIL: 'probe_v2' on a NEW issue 43", go2)
# false positives
ld = TMP / "led_h3fp"
b = new(ld, "fbs-0100-a1", "trend_fx", 100)
unlock(ld, "fbs-0100-a1", "trend_fx", b)
G.run_holdout_exam(book="FBS", system="trend_fx", study_id="fbs-0100-a1",
                   holdout_daily=np.full(b["horizon_days"], -0.001), n_trades=100, periods_per_year=260.0, ledger_dir=ld)
ledger.create_study(ledger_dir=ld, study_id="b3-0100-a1", book="B3", system="trend_win", issue=100, attempt=1,
                    dev_window=["2016-05-02", "2025-04-28"], cost_model_version="t", cv_scheme="t")
stt = ledger.holdout_state("B3", "trend_win", ld, study_id="b3-0100-a1")
print(f"  B3 'trend_win' on the same issue 100 as FBS 'trend_fx' (FAILed): killed={stt['killed']} family={stt['systems']}")
ledger.create_study(ledger_dir=ld, study_id="fbs-0100-a9", book="FBS", system="carry_basket", issue=100, attempt=1,
                    dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
stt = ledger.holdout_state("FBS", "carry_basket", ld, study_id="fbs-0100-a9")
print(f"  unrelated FBS 'carry_basket' filed under issue 100: killed={stt['killed']}")
# a system that ever touched issue 100 drags ALL its other issues into the family
ledger.create_study(ledger_dir=ld, study_id="fbs-0200-a1", book="FBS", system="carry_basket", issue=200, attempt=1,
                    dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
stt = ledger.holdout_state("FBS", "carry_basket", ld, study_id="fbs-0200-a1")
print(f"  'carry_basket' study on its own issue 200 (one earlier study on 100): killed={stt['killed']}")
ledger.create_study(ledger_dir=ld, study_id="fbs-0200-a2", book="FBS", system="mr_gold", issue=200, attempt=1,
                    dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
stt = ledger.holdout_state("FBS", "mr_gold", ld, study_id="fbs-0200-a2")
print(f"  'mr_gold' on issue 200 (shared with carry_basket, not with trend_fx): killed={stt['killed']} family={stt['systems']}")
stt = ledger.holdout_state("FBS", "mr_gold", ld)
print(f"  same, holdout_state WITHOUT study_id (as data.load_bars / holdout_access call it): killed={stt['killed']}")

print("== H5: load_bars scope + impersonation (synthetic M1 files)")
from conftest import write_m1_parquet, catalog_row
config.LEDGER_DIR = TMP / "led_h5"


def m1_rows(start, end):
    rows, t = [], start
    while t < end:
        if t.weekday() < 5:
            rows.append((t, 1.1, 1.1001, 1.0999, 1.1, 1, 0, 10))
        t += timedelta(hours=1)
    return rows


cat_rows = []
for sym in ("EURUSD", "GBPUSD"):
    f = write_m1_parquet(TMP / f"{sym}_M1.parquet", m1_rows(datetime(2025, 4, 1), datetime(2026, 9, 30)))
    cat_rows.append(catalog_row(file=f, symbol=sym, start=datetime(2025, 4, 1), end=datetime(2026, 9, 29, 23), rows=1))
fixed = pl.DataFrame(cat_rows)
data.catalog = lambda book=None, *, include_ticks=False: (fixed if book is None else
                                                          fixed.filter(pl.col("book") == config.get_book(book).name))
b = new(None, "fbs-0050-a1", "passed_sys", 50)
unlock(None, "fbs-0050-a1", "passed_sys", b)
g = np.full(b["horizon_days"], 0.0012) + np.random.default_rng(1).normal(0, 1e-4, b["horizon_days"])
print("  passed_sys exam:", G.run_holdout_exam(book="FBS", system="passed_sys", study_id="fbs-0050-a1", holdout_daily=g,
                                              n_trades=0.4 * b["horizon_days"], periods_per_year=260.0)["status"],
      "horizon_end", b["horizon_end"])
b = new(None, "fbs-0051-a1", "pending_sys", 51)
unlock(None, "fbs-0051-a1", "pending_sys", b)
for key, sym in (("passed_sys", "EURUSD"), ("passed_sys", "GBPUSD"), ("pending_sys", "EURUSD"),
                 ("pending_sys", "GBPUSD"), ("PASSED-SYS", "EURUSD")):
    try:
        bb = data.load_bars(sym, "M1", end="2026-09-30", include_holdout=True, system=key)
        print(f"  load_bars({sym}, system={key!r}): served up to {bb['ts'].max()}")
    except contracts.HoldoutLocked as e:
        print(f"  load_bars({sym}, system={key!r}): HoldoutLocked ({str(e)[:70]})")
rd = ledger.holdout_reads()
print(f"  holdout_read events: {len(rd)}; systems logged {sorted({r['system'] for r in rd})}")
print("  (the `system` argument is a free string: any code may read passed_sys's symbols by naming it; the read is")
print("   logged under that name, and no code compares holdout_read events with the caller's own study)")

print("== LOG: holdout_read volume and append cost")
t0 = time.perf_counter()
n0 = len(ledger.holdout_reads())
for i in range(200):
    data.load_bars("EURUSD", "H1", start=f"2025-05-{15 + i % 10:02d}", end=f"2025-06-{1 + i % 25:02d}",
                   include_holdout=True, system="pending_sys")
dt = time.perf_counter() - t0
n1 = len(ledger.holdout_reads())
print(f"  200 load_bars calls with 250 distinct (start, end) pairs -> {n1 - n0} new holdout_read lines in {dt:.1f}s")
p = ledger._ledger_dir(None) / ledger.HOLDOUT_FILE
big = TMP / "led_big"
big.mkdir()
bp = big / ledger.HOLDOUT_FILE
for n in (1_000, 10_000, 50_000):
    lines = ledger._raw_lines(bp)
    prev = ledger._sha(lines[-1]) if lines else ledger.GENESIS
    with open(bp, "a") as fh:            # bulk-fill a valid chain (same row format as _append)
        for i in range(len(lines), n):
            ln = json.dumps({"seq": i, "prev": prev, "at": "2026-09-25T00:00:00Z", "event": "holdout_read",
                             "book": "FBS", "system": "x", "study_id": "s", "symbol": "EURUSD", "timeframe": "M1",
                             "start": "2025-05-15 00:00:00", "end": "2025-06-01 00:00:00"}, sort_keys=True)
            fh.write(ln + "\n")
            prev = ledger._sha(ln)
    t0 = time.perf_counter()
    for _ in range(20):
        ledger._append(bp, {"event": "holdout_read", "book": "FBS", "system": "x", "symbol": "EURUSD",
                            "timeframe": "M1", "start": "a", "end": "b"})
    t1 = time.perf_counter()
    ledger.holdout_state("FBS", "x", big)
    t2 = time.perf_counter()
    print(f"  holdout file with {n} lines: {(t1 - t0) / 20 * 1e3:.1f} ms per append; holdout_state {1e3 * (t2 - t1):.1f} ms; "
          f"size {bp.stat().st_size / 1e6:.1f} MB")
ledger.verify_chain(big)

print("== H6 / H7")
for h in (262, 392, 522, 652):
    tol = max(S.HOLDOUT_SPAN_TOL_DAYS, S.HOLDOUT_SPAN_TOL_FRAC * h)
    print(f"  band horizon {h} d: realised span accepted within +-{tol:.1f} d (~{tol / 5 * 7:.0f} calendar days)")
