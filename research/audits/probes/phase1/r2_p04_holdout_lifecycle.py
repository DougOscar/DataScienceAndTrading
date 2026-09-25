"""R2-P04 (48e696f) -- holdout life cycle attacks.  SYNTHETIC data only: data.catalog, the manifest,
config.LEDGER_DIR / CACHE_DIR and git_commit are patched to temp objects; the synthetic M1 files are
written to a temp dir; no real holdout bar is ever read.

 H1  NOT_DECISIVE loop: P(eventual PASS | zero-edge holdout) over exams at 1y, 1.5y, 2y, 2.5y (honest),
     vs one exam at the first decisive horizon, vs an author who can SEE the running holdout (another
     system's unlock key, H5) and unlocks at the first month-end where the full span passes a decisive band.
 H2  band re-roll at S5: repeated log_gates with different seed / tiny n_boot -> the latest band is
     "registered" and unlockable.  rebuild_holdout_band(seed=, n_boot=, symbols=) overrides.
 H3  FAIL -> killed; the same system renamed ('probe' -> 'Probe', 'probe_v2') gets a fresh unlock.
 H4  unlock never followed by a recorded exam: state stays 'pending' (not killed).
 H5  load_bars key: a PASSED system name reads ALL holdout data for ANY symbol; a pending system reads
     every symbol up to its horizon (incl. symbols of systems that were never unlocked).
 H6  manifest forged later than the real data: the exam tolerance max(10 d, 5 %).
 H7  horizon when only some symbols have newer data; conversion legs are not part of the horizon.
"""
import json, math, sys, tempfile, warnings
from datetime import date, datetime, timedelta
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
import polars as pl
from quantlab import config, data, ledger, gates as G, stats as S
from quantlab.ledger import LedgerError

TMP = Path(tempfile.mkdtemp(prefix="rt_r2_holdout_"))
config.CACHE_DIR = TMP / "cache"
ledger.git_commit = lambda: "abc1234"


class WfoStudy:
    def __init__(self, sr, n=1500, seed=0, sd=0.006):
        z = np.random.default_rng(seed).standard_normal(n)
        z = (z - z.mean()) / z.std(ddof=1)
        self.mu, self.sd = sr / math.sqrt(260) * sd, sd
        d = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
        self.wfo_oos = pl.DataFrame({"date": d, "ret": self.mu + sd * z})
        self.cpcv_paths = None


def band(sr, h, seed=3, n_boot=1500, n_power=800):
    return S.holdout_band(WfoStudy(sr, seed=seed), h, 260.0, n_boot=n_boot, n_power=n_power,
                          trades_per_day=0.4, seed=seed).as_dict()


print("== H1: NOT_DECISIVE loop on a ZERO-edge holdout (sd 0.6 %/day, 0.4 trades/day)")
H_EXAMS = (262, 392, 522, 652)
H_MONTHS = tuple(range(262, 653, 21))
rng = np.random.default_rng(11)
NSIM = 2000
HMAX = max(H_EXAMS + H_MONTHS)
paths = rng.standard_normal((NSIM, HMAX)) * 0.006
trades = np.cumsum(rng.poisson(0.4, (NSIM, HMAX)), axis=1)
for sr in (1.0, 1.2, 1.5, 1.9):
    bands = {h: band(sr, h) for h in sorted(set(H_EXAMS) | set(H_MONTHS))}
    p0 = {h: bands[h]["p_pass_zero_edge"] for h in H_EXAMS}
    crit = {h: np.array([S.holdout_check(bands[h], paths[i, :h], float(trades[i, h - 1]), 260.0,
                                         check_horizon=False)["criteria_pass"] for i in range(NSIM)]) for h in bands}
    dec = {h: bands[h]["p_pass_zero_edge"] <= 0.30 for h in bands}
    # honest chain
    res = np.full(NSIM, "PENDING", dtype=object)
    alive = np.ones(NSIM, bool)
    for h in H_EXAMS:
        f = alive & ~crit[h]
        res[f] = "FAIL"
        p = alive & crit[h] & dec[h]
        res[p] = "PASS"
        alive &= crit[h].astype(bool) & (not dec[h])
    first_dec = next((h for h in H_EXAMS if dec[h]), None)
    single = crit[first_dec].mean() if first_dec else float("nan")
    peek = np.zeros(NSIM, bool)
    for h in H_MONTHS:
        if dec[h]:
            peek |= crit[h]
    print(f"  dev SR {sr}: p_pass_zero_edge by exam horizon {[round(p0[h], 2) for h in H_EXAMS]}; "
          f"honest chain P(PASS) {np.mean(res == 'PASS'):.3f} (FAIL {np.mean(res == 'FAIL'):.3f}, pending "
          f"{np.mean(res == 'PENDING'):.3f}); one exam at first decisive horizon {first_dec}: {single:.3f}; "
          f"peeking author (monthly, {len(H_MONTHS)} looks) {peek.mean():.3f}")

print("== H2: band re-roll at S5 and rebuild overrides")
from test_gates import make_study, register
man = TMP / "manifest.json"
def set_manifest(entries):
    man.write_text(json.dumps({f"data/forex/{s}_M1_DOES_NOT_EXIST.parquet": {"file": f"data/forex/{s}_M1_DOES_NOT_EXIST.parquet", "market": "forex",
                    "broker": "x", "symbol": s, "timeframe": "M1", "kind": "bars", "rows": 1,
                    "start": "2016-05-02 00:00:00", "end": str(e)} for s, e in entries.items()}))
    config.MANIFEST_PATH = man
set_manifest({"EURUSD": datetime(2026, 5, 15, 10, 36), "GBPUSD": datetime(2026, 5, 15, 10, 36)})
study, trades_ = make_study(seed=4)
ld = register(study, TMP / "led_h2")
lo = []
for seed, nb in ((1, 2000), (2, 2000), (3, 2000), (4, 100), (5, 60), (6, 60), (7, 60)):
    rep = G.evaluate_gates(study, None, periods_per_year=260.0, selected_trades=trades_, n_boot=nb, seed=seed,
                           ledger_dir=ld, holdout_symbols="EURUSD")
    G.log_gates(rep, ledger_dir=ld)
    hb = rep.holdout_band
    lo.append((seed, nb, round(hb["sharpe_lo"], 3), round(hb["max_dd_mag_hi"], 4), round(hb["p_pass_zero_edge"], 3)))
print("  (seed, n_boot, sharpe_lo, max_dd_hi, p_pass_zero_edge) per gate run:", lo)
reg = ledger.registered_holdout_band(study.study_id, ledger_dir=ld)
print(f"  registered band = last gate run's (seed {reg['seed']}, n_boot {reg['n_boot']}); gate runs logged: "
      f"{len(ledger.study_events(study.study_id, 'gates', ledger_dir=ld))}")
try:
    row = ledger.record_holdout_unlock(book="FBS", system="toy", study_id=study.study_id, pass_band=reg,
                                       user_confirmation=ledger.unlock_phrase("FBS", "toy"), ledger_dir=ld)
    print(f"  unlock with the re-rolled n_boot={reg['n_boot']} band: ACCEPTED (exam {row['exam']})")
except LedgerError as e:
    print("  unlock with re-rolled band refused:", str(e)[:100])
ld2 = register(study, TMP / "led_h2b")
G.log_gates(G.evaluate_gates(study, None, periods_per_year=260.0, selected_trades=trades_, n_boot=2000,
                             ledger_dir=ld2, holdout_symbols="EURUSD"), ledger_dir=ld2)
set_manifest({"EURUSD": datetime(2026, 9, 30, 23, 59), "GBPUSD": datetime(2027, 3, 1, 23, 59)})
try:
    nb = G.rebuild_holdout_band(study, periods_per_year=260.0, reason="export", seed=999, n_boot=40,
                                symbols=["GBPUSD"], ledger_dir=ld2)
    print(f"  rebuild_holdout_band(seed=999, n_boot=40, symbols=['GBPUSD']) ACCEPTED: horizon_end {nb['horizon_end']} "
          f"(the system trades EURUSD, which ends 2026-09-30), n_boot {nb['n_boot']}, seed {nb['seed']}")
except Exception as e:
    print("  rebuild override refused:", type(e).__name__, str(e)[:100])

print("== H3: FAIL -> killed; renamed system")
set_manifest({"EURUSD": datetime(2026, 5, 15, 10, 36)})
ld3 = TMP / "led_h3"
def setup(sid, system, issue):
    ledger.create_study(ledger_dir=ld3, study_id=sid, book="FBS", system=system, issue=issue, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
    b = S.holdout_band(WfoStudy(2.8, seed=3), data.holdout_horizon("FBS", "EURUSD"), 260.0, n_boot=800, n_power=300,
                       trades_per_day=0.4, seed=3).as_dict()
    ledger.register_holdout_band(study_id=sid, band=b, reason="S5", ledger_dir=ld3)
    return b
b = setup("fbs-0042-a1", "probe", 42)
ledger.record_holdout_unlock(book="FBS", system="probe", study_id="fbs-0042-a1", pass_band=b,
                             user_confirmation=ledger.unlock_phrase("FBS", "probe"), ledger_dir=ld3)
r = G.run_holdout_exam(book="FBS", system="probe", study_id="fbs-0042-a1", holdout_daily=np.full(b["horizon_days"], -0.001),
                       n_trades=100, periods_per_year=260.0, ledger_dir=ld3)
print("  exam 1:", r["status"], "| state killed:", ledger.holdout_state("FBS", "probe", ld3)["killed"])
for name, iss in (("Probe", 42), ("probe_v2", 43)):
    sid = f"fbs-00{iss}-r{name}"
    b2 = setup(sid, name, iss)
    try:
        row = ledger.record_holdout_unlock(book="FBS", system=name, study_id=sid, pass_band=b2,
                                           user_confirmation=ledger.unlock_phrase("FBS", name), ledger_dir=ld3)
        print(f"  renamed system {name!r} (issue {iss}) after the FAIL: unlock ACCEPTED (exam {row['exam']}); "
              f"normalise_system equal: {ledger.normalise_system(name) == ledger.normalise_system('probe')}")
    except LedgerError as e:
        print(f"  renamed {name!r}: refused: {str(e)[:100]}")

print("== H4 + H5: never-recorded exam; load_bars keys (synthetic M1 parquet)")
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
REAL_CATALOG = data.catalog
data.catalog = lambda book=None, *, include_ticks=False: (fixed if book is None else
                                                          fixed.filter(pl.col("book") == config.get_book(book).name))
def mk(sid, system, issue, sr):
    ledger.create_study(study_id=sid, book="FBS", system=system, issue=issue, attempt=1,
                        dev_window=["2016-05-02", "2025-05-14"], cost_model_version="t", cv_scheme="t")
    b = S.holdout_band(WfoStudy(sr, seed=3), data.holdout_horizon("FBS", "EURUSD"), 260.0, n_boot=800, n_power=300,
                       trades_per_day=0.4, seed=3).as_dict()
    ledger.register_holdout_band(study_id=sid, band=b, reason="S5")
    ledger.record_holdout_unlock(book="FBS", system=system, study_id=sid, pass_band=b,
                                 user_confirmation=ledger.unlock_phrase("FBS", system))
    return b
bp = mk("fbs-0050-a1", "passed_sys", 50, 2.8)
g = np.full(bp["horizon_days"], 0.0012) + np.random.default_rng(1).normal(0, 1e-4, bp["horizon_days"])
print("  passed_sys exam:", G.run_holdout_exam(book="FBS", system="passed_sys", study_id="fbs-0050-a1", holdout_daily=g,
                                              n_trades=0.4 * bp["horizon_days"], periods_per_year=260.0)["status"])
mk("fbs-0051-a1", "pending_sys", 51, 2.8)       # unlocked, exam never recorded
print("  pending_sys state:", {k: v for k, v in ledger.holdout_state("FBS", "pending_sys").items()
                                if k in ("pending", "killed", "last_status")})
for key in ("passed_sys", "pending_sys", "never_unlocked_sys"):
    for sym in ("GBPUSD",):
        try:
            b_ = data.load_bars(sym, "M1", end="2026-09-30", include_holdout=True, system=key)
            print(f"  load_bars({sym}, system={key!r}, end=2026-09-30): served up to {b_['ts'].max()}")
        except Exception as e:
            print(f"  load_bars({sym}, system={key!r}): {type(e).__name__}")

print("== H6: forged manifest end vs exam tolerance")
for h in (262, 392, 522, 652):
    tol = max(10.0, 0.05 * h)
    print(f"  band horizon {h} d: a realised series up to {int(tol)} d shorter/longer is accepted "
          f"(= a manifest end up to {int(tol / 5 * 7)} calendar days off)")

print("== H7: newer data for some symbols only; conversion legs")
set_manifest({"EURJPY": datetime(2027, 5, 14, 23, 59), "USDJPY": datetime(2026, 5, 15, 10, 36),
              "EURUSD": datetime(2027, 5, 14, 23, 59)})
data.catalog = REAL_CATALOG
h1 = data.holdout_horizon("FBS", "EURJPY")
h2 = data.holdout_horizon("FBS", ["EURJPY", "USDJPY"])
print(f"  EURJPY alone: horizon_end {h1['horizon_end']} ({h1['horizon_days']} d); with its USDJPY conversion leg: "
      f"{h2['horizon_end']} ({h2['horizon_days']} d).  gates default symbols = evaluator.symbol only.")
